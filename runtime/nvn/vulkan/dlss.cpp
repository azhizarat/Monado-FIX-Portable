#ifdef _WIN32  // (Streamline is Windows-only: the Linux build gets the "unavailable" stubs at the end)
#define VK_USE_PLATFORM_WIN32_KHR
// [XCDE] NVIDIA DLSS Super Resolution through Streamline 2.x (third_party/streamline, MIT; sl.interposer.dll,
// sl.common.dll, sl.dlss.dll and nvngx_dlss.dll next to the exe), for temporal.c. Streamline's manual hooking mode:
// the Vulkan API is called directly; before the instance / device are created the DLSS plugin's extensions,
// features and queues are added (device.c), the device is then handed to Streamline (slSetVulkanInfo), and the
// swapchain calls it intercepts go through its proxies (dlss_proxy). Loaded at run time: without the DLLs, without an
// RTX GPU or with another upscaler chosen nothing of this runs (temporal.c falls back to FSR 3).
// Per frame (dlss_evaluate): constants (camera, jitter), DLSS options (quality mode, output size), the tagged inputs
// (colour, depth, motion vectors at render size; output at window size) and the evaluation into the command buffer.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <vulkan/vulkan.h>

#include <sl.h>
#include <sl_consts.h>
#include <sl_dlss.h>
#include <sl_hooks.h>
#include <sl_helpers_vk.h>
#include "../../host/nvidia_signature.h"
#include "gpu_scan.h"
#include "../../dlss_policy.h"

extern "C" {
#include "temporal.h"
#include "../../settings.h"

}
#define DLOG(...) fprintf(stderr, __VA_ARGS__)

static HMODULE sl_mod;
static bool sl_ok, dlss_ok, sl_initialized, sl_surface_ready;
static char dlss_note[256] = "No inicializado; se comprueba al iniciar";
static VkDevice sl_device;
static PFN_vkGetInstanceProcAddr p_proxyInstanceProcAddr;
static PFun_slShutdown *p_slShutdown;
extern "C" const char *dlss_status(void) { return dlss_note; }
static void dlss_note_set(const char *text) { snprintf(dlss_note, sizeof dlss_note, "%s", text); DLOG("[dlss] %s\n", text); }
static void dlss_shutdown(void) {
    if (sl_initialized && p_slShutdown) { if (sl_device) vkDeviceWaitIdle(sl_device); p_slShutdown(); }
    sl_initialized = sl_ok = dlss_ok = sl_surface_ready = false;
}
static PFun_slInit *p_slInit;
static PFun_slGetFeatureRequirements *p_slGetFeatureRequirements;
static PFun_slSetVulkanInfo *p_slSetVulkanInfo;
static PFun_slIsFeatureSupported *p_slIsFeatureSupported;
static PFun_slGetFeatureFunction *p_slGetFeatureFunction;
static PFun_slGetNewFrameToken *p_slGetNewFrameToken;
static PFun_slSetConstants *p_slSetConstants;
static PFun_slSetTagForFrame *p_slSetTagForFrame;
static PFun_slEvaluateFeature *p_slEvaluateFeature;
static PFun_slDLSSSetOptions *p_slDLSSSetOptions;
static PFN_vkGetDeviceProcAddr p_proxyDeviceProcAddr;
static sl::FeatureRequirements reqs;

static void sl_log(sl::LogType type, const char *msg) {
    if (type != sl::LogType::eInfo) DLOG("[dlss] %s", msg);
}

// Before the Vulkan instance: load Streamline with the DLSS plugin (only when DLSS is the chosen upscaler).
extern "C" bool dlss_init_early(void) {
    if ((!g_settings.upscaler_auto && g_settings.upscaler != UPSCALE_DLSS) || getenv("XCDE_TMP_OFF")) {
        dlss_note_set("DLSS no solicitado (elige Automático o DLSS y reinicia)"); return false;
    }
    uint32_t count = 0; char error[512];
    XcdeGpu *adapters = xcde_gpu_scan(&count, error, sizeof error);
    XcdeGpuInfo *infos = count ? (XcdeGpuInfo *)calloc(count, sizeof *infos) : nullptr;
    if (!adapters || !infos) { free(adapters); free(infos); dlss_note_set("No se pudo comprobar el adaptador Vulkan"); return false; }
    for (uint32_t i = 0; i < count; ++i) infos[i] = adapters[i].info;
    bool fallback;
    int selected = xcde_gpu_choose(infos, count, getenv("LARECOMP_GPU"), g_settings.gpu_uuid, g_settings.gpu_index, &fallback);
    int route = selected >= 0 ? xcde_vulkan_route(infos[selected].vulkan_routes,
        10 + (int)VK_API_VERSION_MINOR(xcde_vk_loader_version()), g_settings.vulkan_api, &fallback) : 0;
    bool nvidia = selected >= 0 && infos[selected].vendor == 0x10de;
    free(adapters); free(infos);
    if (!nvidia) { dlss_note_set("La GPU seleccionada no es NVIDIA RTX; DLSS desactivado"); return false; }
    if (route < 13) { dlss_note_set("DLSS requiere la ruta Vulkan 1.3 o 1.4"); return false; }
    wchar_t dir[MAX_PATH];
    DWORD n = GetModuleFileNameW(NULL, dir, MAX_PATH);
    if (!n || n >= MAX_PATH) { dlss_note_set("Ruta del ejecutable no disponible"); return false; }
    while (n && dir[n - 1] != L'\\' && dir[n - 1] != L'/') n--;
    dir[n] = 0;
    wchar_t path[MAX_PATH + 32];
    swprintf(path, MAX_PATH + 32, L"%lssl.interposer.dll", dir);
    if (!xcde_nvidia_signature(path)) { dlss_note_set("Falta el runtime oficial o su firma NVIDIA no es válida"); return false; }
    sl_mod = LoadLibraryExW(path, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!sl_mod) { snprintf(dlss_note, sizeof dlss_note, "No se pudo cargar DLSS (Win32 %lu). Comprueba DLL y Visual C++ x64", GetLastError()); DLOG("[dlss] %s\n", dlss_note); return false; }
#define GET(f) p_##f = (PFun_##f *)(void *)GetProcAddress(sl_mod, #f)
    GET(slShutdown); GET(slInit); GET(slGetFeatureRequirements); GET(slSetVulkanInfo); GET(slIsFeatureSupported); GET(slGetFeatureFunction);
    GET(slGetNewFrameToken); GET(slSetConstants); GET(slSetTagForFrame); GET(slEvaluateFeature);
#undef GET
    p_proxyDeviceProcAddr = (PFN_vkGetDeviceProcAddr)(void *)GetProcAddress(sl_mod, "vkGetDeviceProcAddr");
    p_proxyInstanceProcAddr = (PFN_vkGetInstanceProcAddr)(void *)GetProcAddress(sl_mod, "vkGetInstanceProcAddr");
    if (!p_slInit || !p_slShutdown || !p_slGetFeatureRequirements || !p_slSetVulkanInfo || !p_slEvaluateFeature ||
        !p_slIsFeatureSupported || !p_slGetFeatureFunction || !p_slGetNewFrameToken || !p_slSetConstants ||
        !p_slSetTagForFrame || !p_proxyDeviceProcAddr || !p_proxyInstanceProcAddr) {
        dlss_note_set("El runtime Streamline no contiene todas las funciones requeridas");
        return false;
    }
    static const wchar_t *paths[1];
    paths[0] = dir;
    static sl::Feature features[] = {sl::kFeatureDLSS};
    sl::Preferences pref{};
    pref.showConsole = false;
    pref.logLevel = getenv("XCDE_TMP_LOG") ? sl::LogLevel::eVerbose : sl::LogLevel::eDefault;
    pref.logMessageCallback = sl_log;
    pref.pathsToPlugins = paths;
    pref.numPathsToPlugins = 1;
    pref.pathToLogsAndData = nullptr;
    pref.flags = sl::PreferenceFlags::eDisableCLStateTracking | sl::PreferenceFlags::eUseManualHooking |
                 sl::PreferenceFlags::eUseFrameBasedResourceTagging;
    pref.featuresToLoad = features;
    pref.numFeaturesToLoad = 1;
    pref.engine = sl::EngineType::eCustom;
    pref.engineVersion = "xcde";
    pref.projectId = "5b7c1d8e-6a4f-4e1d-9c3a-2f8e7d6b5a41";
    pref.renderAPI = sl::RenderAPI::eVulkan;
    sl::Result r = p_slInit(pref, sl::kSDKVersion);
    if (r != sl::Result::eOk) { snprintf(dlss_note, sizeof dlss_note, "Streamline no pudo iniciarse (SDK %d)", (int)r); DLOG("[dlss] %s\n", dlss_note); return false; }
    sl_initialized = true; atexit(dlss_shutdown);
    r = p_slGetFeatureRequirements(sl::kFeatureDLSS, reqs);
    if (r != sl::Result::eOk) { snprintf(dlss_note, sizeof dlss_note, "No se pudo cargar el plugin DLSS (SDK %d)", (int)r); DLOG("[dlss] %s\n", dlss_note); return false; }
    sl_ok = true;
    dlss_note_set("Runtime oficial cargado; comprobando GPU/controlador RTX");
    DLOG("[dlss] Streamline loaded: %u instance / %u device extensions, %u + %u extra queues\n", reqs.vkNumInstanceExtensions,
         reqs.vkNumDeviceExtensions, reqs.vkNumGraphicsQueuesRequired, reqs.vkNumComputeQueuesRequired);
    return true;
}

extern "C" int dlss_instance_exts(const char **out, int max) {
    int n = 0;
    for (uint32_t i = 0; sl_ok && i < reqs.vkNumInstanceExtensions && n < max; i++) out[n++] = reqs.vkInstanceExtensions[i];
    return n;
}

// [XCDE] T250: optional DLSS requirements must never make an otherwise usable GPU fail.
extern "C" void dlss_validate_adapter(VkPhysicalDevice phys, uint32_t api, uint32_t qfam) {
    if (!sl_ok) return;
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(phys, &props);
    bool supported = api >= VK_API_VERSION_1_3 && props.vendorID == 0x10de;
    uint32_t n = 0;
    VkResult r = vkEnumerateDeviceExtensionProperties(phys, nullptr, &n, nullptr);
    auto *ext = n ? (VkExtensionProperties *)calloc(n, sizeof(VkExtensionProperties)) : nullptr;
    if (r != VK_SUCCESS || !ext || vkEnumerateDeviceExtensionProperties(phys, nullptr, &n, ext) != VK_SUCCESS) supported = false;
    for (uint32_t i = 0; supported && i < reqs.vkNumDeviceExtensions; ++i) {
        bool found = false;
        for (uint32_t j = 0; j < n; ++j) found |= !strcmp(reqs.vkDeviceExtensions[i], ext[j].extensionName);
        if (!found) supported = false;
    }
    free(ext);
    if (supported) {
        VkPhysicalDeviceVulkan13Features h13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceVulkan12Features h12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; h12.pNext = &h13;
        VkPhysicalDeviceFeatures2 have{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; have.pNext = &h12;
        vkGetPhysicalDeviceFeatures2(phys, &have);
        auto need12 = sl::getVkPhysicalDeviceVulkan12Features(reqs.vkNumFeatures12, reqs.vkFeatures12);
        auto need13 = sl::getVkPhysicalDeviceVulkan13Features(reqs.vkNumFeatures13, reqs.vkFeatures13);
        const VkBool32 *h = &h12.samplerMirrorClampToEdge, *need = &need12.samplerMirrorClampToEdge;
        for (size_t i = 0; i < (offsetof(VkPhysicalDeviceVulkan12Features, subgroupBroadcastDynamicId) + sizeof(VkBool32) - offsetof(VkPhysicalDeviceVulkan12Features, samplerMirrorClampToEdge)) / sizeof(VkBool32); ++i)
            if (need[i] && !h[i]) supported = false;
        h = &h13.robustImageAccess; need = &need13.robustImageAccess;
        for (size_t i = 0; i < (offsetof(VkPhysicalDeviceVulkan13Features, maintenance4) + sizeof(VkBool32) - offsetof(VkPhysicalDeviceVulkan13Features, robustImageAccess)) / sizeof(VkBool32); ++i)
            if (need[i] && !h[i]) supported = false;
        uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, nullptr);
        auto *queues = nq ? (VkQueueFamilyProperties *)calloc(nq, sizeof(VkQueueFamilyProperties)) : nullptr;
        if (queues) vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, queues);
        uint64_t required = 1ull + reqs.vkNumGraphicsQueuesRequired + reqs.vkNumComputeQueuesRequired;
        if (!queues || qfam >= nq || required > queues[qfam].queueCount || required > 16) supported = false;
        free(queues);
    }
    if (!supported) { sl_ok = dlss_ok = false; dlss_note_set("GPU/controlador sin los requisitos DLSS de Vulkan; se usa escalado alternativo"); }
}

// The device's extensions (appended to ext[*n]) and features (OR-ed into f12 / f13); extra queues it asks for.
extern "C" void dlss_device_requirements(const char **ext, uint32_t *n, uint32_t max, VkPhysicalDeviceVulkan12Features *f12,
                                         VkPhysicalDeviceVulkan13Features *f13, uint32_t *extra_gfx, uint32_t *extra_compute) {
    *extra_gfx = *extra_compute = 0;
    if (!sl_ok) return;
    uint32_t additional = 0;
    for (uint32_t i = 0; i < reqs.vkNumDeviceExtensions; ++i) {
        bool present = false;
        for (uint32_t j = 0; j < *n; ++j) present |= !strcmp(ext[j], reqs.vkDeviceExtensions[i]);
        if (!present) ++additional;
    }
    if (*n > max || additional > max - *n) { sl_ok = false; dlss_note_set("Demasiadas extensiones opcionales DLSS; se usa escalado alternativo"); return; }
    for (uint32_t i = 0; i < reqs.vkNumDeviceExtensions && *n < max; i++) {
        bool dup = false;
        for (uint32_t j = 0; j < *n; j++) dup |= !strcmp(ext[j], reqs.vkDeviceExtensions[i]);
        if (!dup) ext[(*n)++] = reqs.vkDeviceExtensions[i];
    }
    VkPhysicalDeviceVulkan12Features a = sl::getVkPhysicalDeviceVulkan12Features(reqs.vkNumFeatures12, reqs.vkFeatures12);
    VkPhysicalDeviceVulkan13Features b = sl::getVkPhysicalDeviceVulkan13Features(reqs.vkNumFeatures13, reqs.vkFeatures13);
    // (OR the VkBool32 fields: everything between pNext and the end of each structure)
    VkBool32 *d = &f12->samplerMirrorClampToEdge, *s = &a.samplerMirrorClampToEdge;
    for (size_t i = 0; i < (offsetof(VkPhysicalDeviceVulkan12Features, subgroupBroadcastDynamicId) + sizeof(VkBool32) - offsetof(VkPhysicalDeviceVulkan12Features, samplerMirrorClampToEdge)) / sizeof(VkBool32); i++) d[i] |= s[i];
    d = &f13->robustImageAccess; s = &b.robustImageAccess;
    for (size_t i = 0; i < (offsetof(VkPhysicalDeviceVulkan13Features, maintenance4) + sizeof(VkBool32) - offsetof(VkPhysicalDeviceVulkan13Features, robustImageAccess)) / sizeof(VkBool32); i++) d[i] |= s[i];
    *extra_gfx = reqs.vkNumGraphicsQueuesRequired;
    *extra_compute = reqs.vkNumComputeQueuesRequired;
}

// After vkCreateDevice: the device for Streamline (its queues start at index 1 of the host's family).
extern "C" void dlss_device_created(VkInstance inst, VkPhysicalDevice phys, VkDevice dev, uint32_t qfam) {
    if (!sl_ok) return;
    sl::VulkanInfo info{};
    info.device = dev;
    info.instance = inst;
    info.physicalDevice = phys;
    info.graphicsQueueFamily = qfam;
    uint32_t graphics_index, compute_index;
    xcde_dlss_queue_indices(reqs.vkNumGraphicsQueuesRequired, reqs.vkNumComputeQueuesRequired, &graphics_index, &compute_index);
    info.graphicsQueueIndex = graphics_index;
    info.computeQueueFamily = qfam;
    info.computeQueueIndex = compute_index;
    sl_device = dev;
    sl::Result r = p_slSetVulkanInfo(info);
    if (r != sl::Result::eOk) { snprintf(dlss_note, sizeof dlss_note, "No se pudo preparar DLSS en Vulkan (SDK %d)", (int)r); DLOG("[dlss] %s\n", dlss_note); sl_ok = false; return; }
    sl::AdapterInfo ai{};
    VkPhysicalDevice pd = phys;
    ai.vkPhysicalDevice = pd;
    r = p_slIsFeatureSupported(sl::kFeatureDLSS, ai);
    if (r != sl::Result::eOk) { sl_ok = false; snprintf(dlss_note, sizeof dlss_note, "DLSS no compatible con esta GPU/controlador (SDK %d)", (int)r); DLOG("[dlss] %s\n", dlss_note); return; }
    void *f = nullptr;
    if (p_slGetFeatureFunction(sl::kFeatureDLSS, "slDLSSSetOptions", f) != sl::Result::eOk || !f) {
        DLOG("[dlss] slDLSSSetOptions missing\n");
        return;
    }
    p_slDLSSSetOptions = (PFun_slDLSSSetOptions *)f;
    dlss_ok = true;
}

// T263: Streamline's surface wrapper uses the instance dispatch set up by slSetVulkanInfo.
extern "C" bool dlss_create_surface(void *sdl_window, VkInstance instance, VkPhysicalDevice phys,
                                    uint32_t qfam, VkSurfaceKHR *surface) {
    if (!dlss_ok || !p_proxyInstanceProcAddr) return false;
    auto create = (PFN_vkCreateWin32SurfaceKHR)p_proxyInstanceProcAddr(instance, "vkCreateWin32SurfaceKHR");
    auto destroy = (PFN_vkDestroySurfaceKHR)p_proxyInstanceProcAddr(instance, "vkDestroySurfaceKHR");
    HWND hwnd = (HWND)SDL_GetPointerProperty(SDL_GetWindowProperties((SDL_Window *)sdl_window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);
    if (!create || !destroy || !hwnd) {
        dlss_note_set("Falta el hook o la ventana para preparar la superficie DLSS");
        return false;
    }
    VkWin32SurfaceCreateInfoKHR info{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
    info.hinstance = (HINSTANCE)GetWindowLongPtrW(hwnd, GWLP_HINSTANCE);
    if (!info.hinstance) info.hinstance = GetModuleHandleW(nullptr);
    info.hwnd = hwnd;
    VkSurfaceKHR hooked = VK_NULL_HANDLE;
    VkResult r = create(instance, &info, nullptr, &hooked);
    if (r != VK_SUCCESS || !hooked) {
        snprintf(dlss_note, sizeof dlss_note, "Falló la superficie Streamline (Vulkan %d)", (int)r);
        DLOG("[dlss] %s\n", dlss_note);
        return false;
    }
    VkBool32 present = VK_FALSE;
    r = vkGetPhysicalDeviceSurfaceSupportKHR(phys, qfam, hooked, &present);
    if (r != VK_SUCCESS || !present) {
        destroy(instance, hooked, nullptr);
        snprintf(dlss_note, sizeof dlss_note, "La cola no puede presentar la superficie Streamline (Vulkan %d)", (int)r);
        DLOG("[dlss] %s\n", dlss_note);
        return false;
    }
    *surface = hooked;
    sl_surface_ready = true;
    dlss_note_set("DLSS Super Resolution disponible en la GPU activa");
    return true;
}

// Streamline's proxies for the swapchain calls it intercepts (NULL: call Vulkan directly).
extern "C" PFN_vkVoidFunction dlss_proxy(VkDevice dev, const char *name) {
    if (!sl_surface_ready || !p_proxyDeviceProcAddr) return nullptr;
    return p_proxyDeviceProcAddr(dev, name);
}

extern "C" bool dlss_available(void) { return dlss_ok && sl_surface_ready; }

static sl::Resource tex(const TexBackend *b, VkImageUsageFlags usage) {
    static const VkComponentMapping id = {};
    VkImageView v = vk_image_view((TexBackend *)b, VK_IMAGE_VIEW_TYPE_2D, b->format, 0, 1, 0, 1, id, VK_IMAGE_ASPECT_COLOR_BIT);
    sl::Resource r(sl::ResourceType::eTex2d, (void *)b->image, nullptr, (void *)v, (uint32_t)VK_IMAGE_LAYOUT_GENERAL);
    r.width = b->width;
    r.height = b->height;
    r.nativeFormat = (uint32_t)b->format;
    r.mipLevels = 1;
    r.arrayLayers = 1;
    r.flags = 0;
    r.usage = usage;
    return r;
}

static void rows_to_sl(const double m[16], sl::float4x4 &o) {  // column-vector matrix -> Streamline's row-vector one
    for (int r = 0; r < 4; r++) o.row[r] = sl::float4((float)m[0 * 4 + r], (float)m[1 * 4 + r], (float)m[2 * 4 + r], (float)m[3 * 4 + r]);
}
static bool inv4(const double m[16], double o[16]) {
    double a[4][8];
    for (int i = 0; i < 4; i++) for (int j = 0; j < 8; j++) a[i][j] = j < 4 ? m[i * 4 + j] : (j - 4 == i);
    for (int c = 0; c < 4; c++) {
        int p = c;
        for (int r = c + 1; r < 4; r++) if (fabs(a[r][c]) > fabs(a[p][c])) p = r;
        if (fabs(a[p][c]) < 1e-12) return false;
        if (p != c) for (int j = 0; j < 8; j++) { double t = a[c][j]; a[c][j] = a[p][j]; a[p][j] = t; }
        double d = a[c][c];
        for (int j = 0; j < 8; j++) a[c][j] /= d;
        for (int r = 0; r < 4; r++) if (r != c) { double f = a[r][c]; for (int j = 0; j < 8; j++) a[r][j] -= f * a[c][j]; }
    }
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) o[i * 4 + j] = a[i][j + 4];
    return true;
}

// reproj: the previous clip position from this frame's (column vectors), as temporal.c computed it.
extern "C" bool dlss_evaluate(VkCommandBuffer cb, const TexBackend *color, const TexBackend *mv, const TexBackend *depth,
                              const TexBackend *out, const TexBackend *hud, const float jitter[2], bool reset, const TmpCamera *cam,
                              const double reproj[16]) {
    if (!dlss_ok) return false;
    static uint32_t frame;
    sl::FrameToken *token = nullptr;
    uint32_t fi = frame++;
    if (p_slGetNewFrameToken(token, &fi) != sl::Result::eOk || !token) return false;
    sl::ViewportHandle vp(0);
    sl::Constants c{};
    double proj[16], iproj[16];
    for (int i = 0; i < 16; i++) proj[i] = cam->proj[i];
    if (!inv4(proj, iproj)) return false;
    rows_to_sl(proj, c.cameraViewToClip);
    rows_to_sl(iproj, c.clipToCameraView);
    double idm[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, back[16];
    rows_to_sl(idm, c.clipToLensClip);
    rows_to_sl(reproj, c.clipToPrevClip);
    if (!inv4(reproj, back)) memcpy(back, idm, sizeof back);
    rows_to_sl(back, c.prevClipToClip);
    c.jitterOffset = sl::float2(jitter[0], jitter[1]);
    c.mvecScale = sl::float2(1.0f, 1.0f);  // (motion vectors in UV units)
    c.cameraPinholeOffset = sl::float2(0, 0);
    c.cameraPos = sl::float3(0, 0, 0);
    c.cameraUp = sl::float3(cam->view[4], cam->view[5], cam->view[6]);
    c.cameraRight = sl::float3(cam->view[0], cam->view[1], cam->view[2]);
    c.cameraFwd = sl::float3(-cam->view[8], -cam->view[9], -cam->view[10]);
    c.cameraNear = fmaxf(fabsf(cam->proj[11]) * 0.5f, 0.001f);
    c.cameraFar = 100000.0f;
    c.cameraFOV = cam->proj[5] != 0 ? 2.0f * atanf(1.0f / fabsf(cam->proj[5])) : 0.785f;
    c.cameraAspectRatio = cam->proj[0] != 0 ? fabsf(cam->proj[5] / cam->proj[0]) : 1.778f;
    c.motionVectorsInvalidValue = sl::INVALID_FLOAT; // zero is a valid stationary motion vector
    c.motionVectorsJittered = sl::Boolean::eFalse;
    c.depthInverted = sl::Boolean::eFalse;
    c.cameraMotionIncluded = sl::Boolean::eTrue;
    c.motionVectors3D = sl::Boolean::eFalse;
    c.reset = reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    if (p_slSetConstants(c, *token, vp) != sl::Result::eOk) return false;
    sl::DLSSOptions o{};
    static const sl::DLSSMode modes[] = {sl::DLSSMode::eDLAA, sl::DLSSMode::eMaxQuality, sl::DLSSMode::eBalanced,
                                         sl::DLSSMode::eMaxPerformance, sl::DLSSMode::eUltraPerformance};
    int q = g_settings.upscale_quality;
    o.mode = modes[q >= 0 && q <= 4 ? q : 1];
    o.outputWidth = out->width;
    o.outputHeight = out->height;
    o.useAutoExposure = sl::Boolean::eTrue;
    o.colorBuffersHDR = sl::Boolean::eFalse;  // (the game image: gamma-encoded LDR)
    if (p_slDLSSSetOptions(vp, o) != sl::Result::eOk) return false;
    sl::Resource rc = tex(color, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    sl::Resource rd = tex(depth, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    sl::Resource rm = tex(mv, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    sl::Resource ro = tex(out, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    sl::Extent ein{0, 0, color->width, color->height}, eout{0, 0, out->width, out->height};
    sl::ResourceTag tags[5] = {sl::ResourceTag(&rc, sl::kBufferTypeScalingInputColor, sl::ResourceLifecycle::eOnlyValidNow, &ein),
                               sl::ResourceTag(&rd, sl::kBufferTypeDepth, sl::ResourceLifecycle::eOnlyValidNow, &ein),
                               sl::ResourceTag(&rm, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eOnlyValidNow, &ein),
                               sl::ResourceTag(&ro, sl::kBufferTypeScalingOutputColor, sl::ResourceLifecycle::eOnlyValidNow, &eout),
                               sl::ResourceTag()};
    uint32_t ntags = 4;
    sl::Resource rh;
    if (hud) {  // [XCDE] T213: the HUD's pixels (temporal.c tmp_hud_snapshot): DLSS takes the current colour there (no trails)
        rh = tex(hud, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        tags[ntags++] = sl::ResourceTag(&rh, sl::kBufferTypeBiasCurrentColorHint, sl::ResourceLifecycle::eOnlyValidNow, &ein);
    }
    if (p_slSetTagForFrame(*token, vp, tags, ntags, (sl::CommandBuffer *)cb) != sl::Result::eOk) { vk_external_state_changed(); return false; }
    const sl::BaseStructure *inputs[] = {&vp};
    sl::Result r = p_slEvaluateFeature(sl::kFeatureDLSS, *token, inputs, 1, (sl::CommandBuffer *)cb);
    vk_external_state_changed(); // SDK may change pipeline, descriptors and dynamic state
    if (r != sl::Result::eOk) { snprintf(dlss_note, sizeof dlss_note, "DLSS dispatch falló (SDK %d); frame con escalado alternativo", (int)r); static int n; if (n++ < 5) DLOG("[dlss] %s\n", dlss_note); return false; }
    return true;
}

#else  // Linux: no Streamline. temporal.c / device.c see DLSS as unavailable and use FSR.
#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan.h>
extern "C" {
#include "temporal.h"
}
extern "C" const char *dlss_status(void) { return "DLSS: Windows only"; }
extern "C" bool dlss_init_early(void) { return false; }
extern "C" int dlss_instance_exts(const char **out, int max) { (void)out, (void)max; return 0; }
extern "C" void dlss_validate_adapter(VkPhysicalDevice, uint32_t, uint32_t) {}
extern "C" void dlss_device_requirements(const char **, uint32_t *, uint32_t, VkPhysicalDeviceVulkan12Features *,
                                         VkPhysicalDeviceVulkan13Features *, uint32_t *extra_gfx, uint32_t *extra_compute) {
    *extra_gfx = *extra_compute = 0;
}
extern "C" void dlss_device_created(VkInstance, VkPhysicalDevice, VkDevice, uint32_t) {}
extern "C" bool dlss_create_surface(void *, VkInstance, VkPhysicalDevice, uint32_t, VkSurfaceKHR *) { return false; }
extern "C" PFN_vkVoidFunction dlss_proxy(VkDevice, const char *) { return nullptr; }
extern "C" bool dlss_available(void) { return false; }
extern "C" bool dlss_evaluate(VkCommandBuffer, const TexBackend *, const TexBackend *, const TexBackend *, const TexBackend *,
                              const TexBackend *, const float[2], bool, const TmpCamera *, const double[16]) { return false; }
#endif
