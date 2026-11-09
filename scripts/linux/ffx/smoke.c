// [XCDE] Smoke test of libamd_fidelityfx_vk.so (scripts/linux/ffx_build.sh smoke; runs in monado-lab on lavapipe):
// dlopen + the five entry points, the upscaler versions, an FSR 3.1 upscale context with a few dispatches (the output read
// back: a flat grey input must come out the same grey), a frame-generation context, and - when the loader offers
// VK_EXT_headless_surface and the device 3 queues (FFX wants game, present and acquire queues apart) - the frame-generation swapchain with a few presents.
//   smoke <path to libamd_fidelityfx_vk.so>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#define __declspec(x)
#include "ffx_api/ffx_api.h"
#include "ffx_api/ffx_framegeneration.h"
#include "ffx_api/ffx_upscale.h"

// (ffx_api/vk/ffx_api_vk.h is C++-only: as declared there)
#define FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK 0x0000003u
struct ffxCreateBackendVKDesc { ffxCreateContextDescHeader header; VkDevice vkDevice; VkPhysicalDevice vkPhysicalDevice; PFN_vkGetDeviceProcAddr vkDeviceProcAddr; };
#define FFX_API_CREATE_CONTEXT_DESC_TYPE_FGSWAPCHAIN_VK 0x40001u
#define FFX_API_QUERY_DESC_TYPE_FGSWAPCHAIN_FUNCTIONS_VK 0x40005u
typedef VkResult (*PFN_vkQueueSubmitFFXAPI)(uint32_t, const VkSubmitInfo *, VkFence);
struct VkQueueInfoFFXAPI { VkQueue queue; uint32_t familyIndex; PFN_vkQueueSubmitFFXAPI submitFunc; };
struct ffxCreateContextDescFrameGenerationSwapChainVK {
    ffxCreateContextDescHeader header; VkPhysicalDevice physicalDevice; VkDevice device; VkSwapchainKHR *swapchain;
    VkAllocationCallbacks *allocator; VkSwapchainCreateInfoKHR createInfo;
    struct VkQueueInfoFFXAPI gameQueue, asyncComputeQueue, presentQueue, imageAcquireQueue;
};
typedef VkResult (*PFN_vkCreateSwapchainFFXAPI)(VkDevice, const VkSwapchainCreateInfoKHR *, const VkAllocationCallbacks *, VkSwapchainKHR *, void *);
typedef void (*PFN_vkDestroySwapchainFFXAPI)(VkDevice, VkSwapchainKHR, const VkAllocationCallbacks *, void *);
typedef uint64_t (*PFN_getLastPresentCountFFXAPI)(VkSwapchainKHR);
struct ffxQueryDescSwapchainReplacementFunctionsVK {
    ffxQueryDescHeader header; PFN_vkCreateSwapchainFFXAPI pOutCreateSwapchainFFXAPI; PFN_vkDestroySwapchainFFXAPI pOutDestroySwapchainFFXAPI;
    PFN_vkGetSwapchainImagesKHR pOutGetSwapchainImagesKHR; PFN_vkAcquireNextImageKHR pOutAcquireNextImageKHR;
    PFN_vkQueuePresentKHR pOutQueuePresentKHR; PFN_vkSetHdrMetadataEXT pOutSetHdrMetadataEXT; PFN_getLastPresentCountFFXAPI pOutGetLastPresentCountFFXAPI;
};

static PfnFfxCreateContext ffx_create;
static PfnFfxDestroyContext ffx_destroy;
static PfnFfxConfigure ffx_configure;
static PfnFfxQuery ffx_query;
static PfnFfxDispatch ffx_dispatch;

static VkInstance inst;
static VkPhysicalDevice phys;
static VkDevice dev;
static VkQueue queue[4];
static uint32_t qfam, nqueues;
static VkCommandPool pool;
static int failures;

#define CHECK(what, cond) do { int ok_ = (cond); printf("%-58s %s\n", what, ok_ ? "ok" : "FAILED"); if (!ok_) failures++; } while (0)
#define VK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { printf("%s: VkResult %d\n", #x, r_); exit(1); } } while (0)

static void ffx_msg(uint32_t type, const wchar_t *m) { printf("  [ffx %s] %ls\n", type == FFX_API_MESSAGE_TYPE_ERROR ? "error" : "warning", m); }

// as temporal.c ffx_proc: the backend asks for some functions by their KHR names (vkGetBufferMemoryRequirements2KHR); the
// loader returns NULL for those unless the extension was enabled, the device has them as core functions
static PFN_vkVoidFunction VKAPI_PTR ffx_proc(VkDevice d, const char *name) {
    PFN_vkVoidFunction f = vkGetDeviceProcAddr(d, name);
    size_t n = strlen(name);
    char core[128];
    if (!f && n > 3 && n < sizeof core && !strcmp(name + n - 3, "KHR")) {
        memcpy(core, name, n - 3);
        core[n - 3] = 0;
        f = vkGetDeviceProcAddr(d, core);
    }
    return f;
}

static bool has_ext(const VkExtensionProperties *e, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++) if (!strcmp(e[i].extensionName, name)) return true;
    return false;
}

static void vk_init(bool *headless) {
    uint32_t n = 0;
    vkEnumerateInstanceExtensionProperties(NULL, &n, NULL);
    VkExtensionProperties *ie = calloc(n, sizeof *ie);
    vkEnumerateInstanceExtensionProperties(NULL, &n, ie);
    *headless = has_ext(ie, n, VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME) && has_ext(ie, n, VK_KHR_SURFACE_EXTENSION_NAME);
    free(ie);
    const char *iext[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME};
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "ffx-smoke", .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app,
                                .enabledExtensionCount = *headless ? 2 : 0, .ppEnabledExtensionNames = iext};
    VK(vkCreateInstance(&ici, NULL, &inst));
    n = 1;
    vkEnumeratePhysicalDevices(inst, &n, &phys);
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(phys, &p);
    printf("device: %s (Vulkan %u.%u.%u)\n", p.deviceName, VK_VERSION_MAJOR(p.apiVersion), VK_VERSION_MINOR(p.apiVersion), VK_VERSION_PATCH(p.apiVersion));
    VkQueueFamilyProperties qf[8];
    n = 8;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, qf);
    for (qfam = 0; qfam < n && !(qf[qfam].queueFlags & VK_QUEUE_GRAPHICS_BIT && qf[qfam].queueFlags & VK_QUEUE_COMPUTE_BIT); qfam++) {}
    nqueues = qf[qfam].queueCount < 4 ? qf[qfam].queueCount : 4;
    // every feature the device has (what a game typically enables; FSR needs a few of them, e.g. 16-bit storage)
    VkPhysicalDeviceVulkan13Features f13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features f12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &f13};
    VkPhysicalDeviceVulkan11Features f11 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, .pNext = &f12};
    VkPhysicalDeviceFeatures2 f = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f11};
    vkGetPhysicalDeviceFeatures2(phys, &f);
    n = 0;
    vkEnumerateDeviceExtensionProperties(phys, NULL, &n, NULL);
    VkExtensionProperties *de = calloc(n, sizeof *de);
    vkEnumerateDeviceExtensionProperties(phys, NULL, &n, de);
    const char *dext[2];
    uint32_t nd = 0;
    if (*headless && has_ext(de, n, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) dext[nd++] = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    else *headless = false;
    free(de);
    float prio[4] = {1, 1, 1, 1};
    VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = qfam, .queueCount = nqueues, .pQueuePriorities = prio};
    VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
                              .enabledExtensionCount = nd, .ppEnabledExtensionNames = dext};
    VK(vkCreateDevice(phys, &dci, NULL, &dev));
    for (uint32_t i = 0; i < nqueues; i++) vkGetDeviceQueue(dev, qfam, i, &queue[i]);
    printf("queue family %u: %u queue(s) used\n", qfam, nqueues);
    VkCommandPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = qfam};
    VK(vkCreateCommandPool(dev, &pci, NULL, &pool));
}

static uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits >> i & 1) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    exit(2);
}
typedef struct { VkImage image; VkDeviceMemory mem; uint32_t w, h; } Img;
static Img image(VkFormat f, uint32_t w, uint32_t h) {
    Img r = {.w = w, .h = h};
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = f, .extent = {w, h, 1},
                            .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
    VK(vkCreateImage(dev, &ci, NULL, &r.image));
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(dev, r.image, &mr);
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    VK(vkAllocateMemory(dev, &ai, NULL, &r.mem));
    VK(vkBindImageMemory(dev, r.image, r.mem, 0));
    return r;
}
static struct FfxApiResource res(const Img *d, uint32_t fmt, uint32_t usage) {
    struct FfxApiResource r;
    memset(&r, 0, sizeof r);
    r.resource = (void *)d->image;
    r.description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    r.description.format = fmt;
    r.description.width = d->w;
    r.description.height = d->h;
    r.description.depth = 1;
    r.description.mipCount = 1;
    r.description.usage = usage;
    r.state = FFX_API_RESOURCE_STATE_UNORDERED_ACCESS;  // (every image is in GENERAL)
    return r;
}
static void barrier(VkCommandBuffer cb) {
    VkMemoryBarrier b = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT, .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &b, 0, NULL, 0, NULL);
}
static void to_general(VkCommandBuffer cb, VkImage im) {
    VkImageMemoryBarrier b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                              .newLayout = VK_IMAGE_LAYOUT_GENERAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                              .image = im, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &b);
}
static void clear(VkCommandBuffer cb, VkImage im, float r, float g, float b, float a) {
    VkClearColorValue c = {.float32 = {r, g, b, a}};
    VkImageSubresourceRange sr = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cb, im, VK_IMAGE_LAYOUT_GENERAL, &c, 1, &sr);
}
static VkCommandBuffer cmd_begin(void) {
    VkCommandBuffer cb;
    VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool, .commandBufferCount = 1};
    VK(vkAllocateCommandBuffers(dev, &ai, &cb));
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VK(vkBeginCommandBuffer(cb, &bi));
    return cb;
}
static void cmd_submit_wait(VkCommandBuffer cb) {
    VK(vkEndCommandBuffer(cb));
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
    VK(vkQueueSubmit(queue[0], 1, &si, VK_NULL_HANDLE));
    VK(vkQueueWaitIdle(queue[0]));
    vkFreeCommandBuffers(dev, pool, 1, &cb);
}

// the centre pixel of a 4-byte-per-pixel image (in GENERAL)
static void read_centre(const Img *im, uint8_t px[4]) {
    VkBuffer buf;
    VkDeviceMemory bm;
    VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 4, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VK(vkCreateBuffer(dev, &bci, NULL, &buf));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, buf, &mr);
    VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
                                .memoryTypeIndex = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    VK(vkAllocateMemory(dev, &mai, NULL, &bm));
    VK(vkBindBufferMemory(dev, buf, bm, 0));
    VkCommandBuffer cb = cmd_begin();
    VkBufferImageCopy c = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageOffset = {(int32_t)im->w / 2, (int32_t)im->h / 2, 0},
                           .imageExtent = {1, 1, 1}};
    vkCmdCopyImageToBuffer(cb, im->image, VK_IMAGE_LAYOUT_GENERAL, buf, 1, &c);
    cmd_submit_wait(cb);
    void *m;
    VK(vkMapMemory(dev, bm, 0, VK_WHOLE_SIZE, 0, &m));
    memcpy(px, m, 4);
    vkUnmapMemory(dev, bm);
    vkDestroyBuffer(dev, buf, NULL);
    vkFreeMemory(dev, bm, NULL);
}

// ---------------------------------------------------------------- upscaler
static void test_upscaler(void) {
    struct ffxCreateBackendVKDesc be = {.header = {.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK}, .vkDevice = dev, .vkPhysicalDevice = phys,
                                        .vkDeviceProcAddr = ffx_proc};
    uint64_t count = 0;
    struct ffxQueryDescGetVersions qv = {.header = {.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS}, .createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE,
                                         .device = dev, .outputCount = &count};
    CHECK("ffxQuery GET_VERSIONS (count)", ffx_query(NULL, &qv.header) == FFX_API_RETURN_OK && count > 0);
    uint64_t ids[8];
    const char *names[8];
    count = count > 8 ? 8 : count;
    qv.versionIds = ids;
    qv.versionNames = names;
    ffx_query(NULL, &qv.header);
    for (uint64_t i = 0; i < count; i++) printf("  upscaler version: %s (id 0x%016llx)\n", names[i], (unsigned long long)ids[i]);

    const uint32_t rw = 640, rh = 360, ow = 1280, oh = 720;
    ffxContext ctx = NULL;
    struct ffxCreateContextDescUpscale cd = {.header = {.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE, .pNext = &be.header},
                                             .flags = FFX_UPSCALE_ENABLE_DEPTH_INFINITE | FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE |
                                                      FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION | FFX_UPSCALE_ENABLE_DEBUG_CHECKING,
                                             .maxRenderSize = {rw, rh}, .maxUpscaleSize = {ow, oh}, .fpMessage = ffx_msg};
    ffxReturnCode_t r = ffx_create(&ctx, &cd.header, NULL);
    printf("ffxCreateContext(upscale %ux%u -> %ux%u): %u\n", rw, rh, ow, oh, r);
    CHECK("FSR 3.1 upscale context", r == FFX_API_RETURN_OK);
    if (r) return;
    struct ffxQueryGetProviderVersion pv = {.header = {.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION}};
    ffx_query(&ctx, &pv.header);
    printf("  provider: %s (0x%016llx)\n", pv.versionName ? pv.versionName : "?", (unsigned long long)pv.versionId);
    int32_t phases = 0;
    struct ffxQueryDescUpscaleGetJitterPhaseCount jq = {.header = {.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT},
                                                        .renderWidth = rw, .displayWidth = ow, .pOutPhaseCount = &phases};
    CHECK("ffxQuery jitter phase count", ffx_query(&ctx, &jq.header) == FFX_API_RETURN_OK && phases > 0);
    printf("  jitter phases: %d\n", phases);

    // dispatches: flat grey colour, constant depth, no motion -> the output must be that grey
    Img color = image(VK_FORMAT_R8G8B8A8_UNORM, rw, rh), depth = image(VK_FORMAT_R32_SFLOAT, rw, rh),
        mv = image(VK_FORMAT_R16G16_SFLOAT, rw, rh), out = image(VK_FORMAT_R8G8B8A8_UNORM, ow, oh);
    VkCommandBuffer cb = cmd_begin();
    Img *all[] = {&color, &depth, &mv, &out};
    for (int i = 0; i < 4; i++) to_general(cb, all[i]->image);
    barrier(cb);
    clear(cb, color.image, 0.25f, 0.5f, 0.75f, 1);
    clear(cb, depth.image, 0.5f, 0, 0, 0);
    clear(cb, mv.image, 0, 0, 0, 0);
    clear(cb, out.image, 1, 0, 1, 1);
    barrier(cb);
    cmd_submit_wait(cb);
    const int frames = 8;
    bool ok = true;
    for (int f = 0; f < frames && ok; f++) {
        struct ffxQueryDescUpscaleGetJitterOffset jo = {.header = {.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET}, .index = f, .phaseCount = phases};
        float jx = 0, jy = 0;
        jo.pOutX = &jx;
        jo.pOutY = &jy;
        ffx_query(&ctx, &jo.header);
        struct ffxDispatchDescUpscale d;
        memset(&d, 0, sizeof d);
        d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
        cb = cmd_begin();
        d.commandList = (void *)cb;
        d.color = res(&color, FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM, FFX_API_RESOURCE_USAGE_READ_ONLY);
        d.depth = res(&depth, FFX_API_SURFACE_FORMAT_R32_FLOAT, FFX_API_RESOURCE_USAGE_READ_ONLY);
        d.motionVectors = res(&mv, FFX_API_SURFACE_FORMAT_R16G16_FLOAT, FFX_API_RESOURCE_USAGE_READ_ONLY);
        d.output = res(&out, FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM, FFX_API_RESOURCE_USAGE_UAV);
        d.jitterOffset.x = jx;
        d.jitterOffset.y = jy;
        d.motionVectorScale.x = (float)rw;
        d.motionVectorScale.y = (float)rh;
        d.renderSize.width = rw;
        d.renderSize.height = rh;
        d.upscaleSize.width = ow;
        d.upscaleSize.height = oh;
        d.enableSharpening = true;
        d.sharpness = 0.3f;
        d.frameTimeDelta = 16.7f;
        d.preExposure = 1;
        d.reset = f == 0;
        d.cameraNear = 0.1f;
        d.cameraFar = 3.402823466e+38f;
        d.cameraFovAngleVertical = 1.0f;
        d.viewSpaceToMetersFactor = 1;
        d.flags = FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB;
        r = ffx_dispatch(&ctx, &d.header);
        ok = r == FFX_API_RETURN_OK;
        barrier(cb);
        cmd_submit_wait(cb);
    }
    printf("ffxDispatch x%d: %u\n", frames, r);
    CHECK("FSR 3.1 upscale dispatches", ok);
    uint8_t p[4];
    read_centre(&out, p);
    printf("  output centre pixel: %u %u %u %u (input 64 128 191 255; the output was cleared to 255 0 255 255)\n", p[0], p[1], p[2], p[3]);
    CHECK("upscaled output = input grey (+-3)", abs(p[0] - 64) <= 3 && abs(p[1] - 128) <= 3 && abs(p[2] - 191) <= 3);
    CHECK("ffxDestroyContext(upscale)", ffx_destroy(&ctx, NULL) == FFX_API_RETURN_OK);
    for (int i = 0; i < 4; i++) { vkDestroyImage(dev, all[i]->image, NULL); vkFreeMemory(dev, all[i]->mem, NULL); }
}

// ---------------------------------------------------------------- frame generation
static VkResult fg_present_loop(PFN_vkCreateSwapchainFFXAPI, PFN_vkDestroySwapchainFFXAPI, struct ffxQueryDescSwapchainReplacementFunctionsVK *,
                                ffxContext *, VkSwapchainKHR, uint32_t, uint32_t);
static ffxReturnCode_t fg_dispatch_cb(ffxDispatchDescFrameGeneration *p, void *ctx) { return ffx_dispatch((ffxContext *)ctx, &p->header); }

// Without the swapchain (no 3 queues): frame generation driven by hand, as its swapchain would - configure with
// NO_SWAPCHAIN_CONTEXT_NOTIFY, prepare (depth + motion), then the generation dispatch itself (optical flow + frame
// interpolation) between two identical flat frames: the generated frame must be that colour.
static void fg_dispatch_direct(ffxContext *ctx, uint32_t w, uint32_t h) {
    Img depth = image(VK_FORMAT_R32_SFLOAT, w, h), mv = image(VK_FORMAT_R16G16_SFLOAT, w, h), color = image(VK_FORMAT_B8G8R8A8_UNORM, w, h),
        out = image(VK_FORMAT_B8G8R8A8_UNORM, w, h);
    Img *all[] = {&depth, &mv, &color, &out};
    VkCommandBuffer cb = cmd_begin();
    for (int i = 0; i < 4; i++) to_general(cb, all[i]->image);
    barrier(cb);
    clear(cb, depth.image, 0.5f, 0, 0, 0);
    clear(cb, mv.image, 0, 0, 0, 0);
    clear(cb, color.image, 0.25f, 0.5f, 0.75f, 1);
    clear(cb, out.image, 1, 0, 1, 1);
    barrier(cb);
    cmd_submit_wait(cb);
    bool ok = true;
    const int frames = 4;
    for (uint64_t f = 0; f < frames && ok; f++) {
        struct ffxConfigureDescFrameGeneration c;
        memset(&c, 0, sizeof c);
        c.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
        c.flags = FFX_FRAMEGENERATION_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY;
        c.frameGenerationEnabled = true;
        c.generationRect = (struct FfxApiRect2D){0, 0, (int32_t)w, (int32_t)h};
        c.frameID = f;
        ffxReturnCode_t r = ffx_configure(ctx, &c.header);
        if (r) { printf("  configure (no swapchain): %u\n", r); ok = false; break; }
        cb = cmd_begin();
        struct ffxDispatchDescFrameGenerationPrepare p;
        memset(&p, 0, sizeof p);
        p.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
        p.frameID = f;
        p.commandList = (void *)cb;
        p.renderSize = (struct FfxApiDimensions2D){w, h};
        p.motionVectorScale.x = (float)w;
        p.motionVectorScale.y = (float)h;
        p.frameTimeDelta = 16.7f;
        p.cameraNear = 0.1f;
        p.cameraFar = 3.402823466e+38f;
        p.cameraFovAngleVertical = 1.0f;
        p.viewSpaceToMetersFactor = 1;
        p.depth = res(&depth, FFX_API_SURFACE_FORMAT_R32_FLOAT, FFX_API_RESOURCE_USAGE_READ_ONLY);
        p.motionVectors = res(&mv, FFX_API_SURFACE_FORMAT_R16G16_FLOAT, FFX_API_RESOURCE_USAGE_READ_ONLY);
        r = ffx_dispatch(ctx, &p.header);
        if (r) { printf("  prepare: %u\n", r); ok = false; }
        barrier(cb);
        ffxDispatchDescFrameGeneration g;
        memset(&g, 0, sizeof g);
        g.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION;
        g.commandList = (void *)cb;
        g.presentColor = res(&color, FFX_API_SURFACE_FORMAT_B8G8R8A8_UNORM, FFX_API_RESOURCE_USAGE_READ_ONLY);
        g.outputs[0] = res(&out, FFX_API_SURFACE_FORMAT_B8G8R8A8_UNORM, FFX_API_RESOURCE_USAGE_UAV);
        g.numGeneratedFrames = 1;
        g.reset = f == 0;
        g.backbufferTransferFunction = FFX_API_BACKBUFFER_TRANSFER_FUNCTION_SRGB;
        g.minMaxLuminance[0] = 0;
        g.minMaxLuminance[1] = 1;
        g.generationRect = c.generationRect;
        g.frameID = f;
        if (ok && (r = ffx_dispatch(ctx, &g.header))) { printf("  generation dispatch: %u\n", r); ok = false; }
        barrier(cb);
        cmd_submit_wait(cb);
    }
    CHECK("frame generation prepare + generation dispatches (x4)", ok);
    uint8_t in[4], px[4];
    read_centre(&color, in);
    read_centre(&out, px);
    printf("  generated frame centre pixel: %u %u %u %u (input %u %u %u %u; the output was cleared to 255 0 255 255)\n", px[0], px[1], px[2], px[3],
           in[0], in[1], in[2], in[3]);
    CHECK("generated frame = input colour (+-3)", abs(px[0] - in[0]) <= 3 && abs(px[1] - in[1]) <= 3 && abs(px[2] - in[2]) <= 3);
    for (int i = 0; i < 4; i++) { vkDestroyImage(dev, all[i]->image, NULL); vkFreeMemory(dev, all[i]->mem, NULL); }
}

static void test_frame_generation(bool headless) {
    const uint32_t w = 640, h = 360;
    struct ffxCreateBackendVKDesc be = {.header = {.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK}, .vkDevice = dev, .vkPhysicalDevice = phys,
                                        .vkDeviceProcAddr = ffx_proc};
    // the swapchain first (as a game does: frame generation configures against it)
    ffxContext scctx = NULL;
    VkSurfaceKHR surf = VK_NULL_HANDLE;
    VkSwapchainKHR sc = VK_NULL_HANDLE;
    struct ffxQueryDescSwapchainReplacementFunctionsVK q;
    memset(&q, 0, sizeof q);
    if (headless && nqueues >= 3) {
        PFN_vkCreateHeadlessSurfaceEXT mk = (PFN_vkCreateHeadlessSurfaceEXT)vkGetInstanceProcAddr(inst, "vkCreateHeadlessSurfaceEXT");
        VkHeadlessSurfaceCreateInfoEXT hci = {.sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT};
        VK(mk(inst, &hci, NULL, &surf));
        VkSurfaceCapabilitiesKHR caps;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys, surf, &caps);
        struct ffxCreateContextDescFrameGenerationSwapChainVK d;
        memset(&d, 0, sizeof d);
        d.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FGSWAPCHAIN_VK;
        d.header.pNext = &be.header;
        d.physicalDevice = phys;
        d.device = dev;
        d.swapchain = &sc;
        d.createInfo = (VkSwapchainCreateInfoKHR){.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, .surface = surf,
            .minImageCount = caps.minImageCount < 3 ? 3 : caps.minImageCount, .imageFormat = VK_FORMAT_B8G8R8A8_UNORM,
            .imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, .imageExtent = {w, h}, .imageArrayLayers = 1,
            .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR, .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = VK_PRESENT_MODE_FIFO_KHR, .clipped = VK_TRUE};
        VK(vkCreateSwapchainKHR(dev, &d.createInfo, NULL, &sc));  // (the game's swapchain, which FFX replaces)
        d.gameQueue = (struct VkQueueInfoFFXAPI){queue[0], qfam, NULL};
        d.presentQueue = (struct VkQueueInfoFFXAPI){queue[1], qfam, NULL};
        d.imageAcquireQueue = (struct VkQueueInfoFFXAPI){queue[2], qfam, NULL};
        d.asyncComputeQueue = (struct VkQueueInfoFFXAPI){nqueues > 3 ? queue[3] : VK_NULL_HANDLE, qfam, NULL};  // (optional)
        ffxReturnCode_t r = ffx_create(&scctx, &d.header, NULL);
        printf("ffxCreateContext(frame generation swapchain, headless %ux%u): %u\n", w, h, r);
        CHECK("FSR 3.1 frame generation swapchain", r == FFX_API_RETURN_OK && sc);
        if (r) scctx = NULL;
        else {
            q.header.type = FFX_API_QUERY_DESC_TYPE_FGSWAPCHAIN_FUNCTIONS_VK;
            ffx_query(&scctx, &q.header);
            CHECK("swapchain replacement functions", q.pOutCreateSwapchainFFXAPI && q.pOutGetSwapchainImagesKHR && q.pOutAcquireNextImageKHR &&
                                                     q.pOutQueuePresentKHR && q.pOutGetLastPresentCountFFXAPI);
        }
    } else printf("frame generation swapchain: skipped (%s)\n", headless ? "fewer than 3 queues" : "no VK_EXT_headless_surface");

    ffxContext ctx = NULL;
    struct ffxCreateContextDescFrameGeneration cd = {.header = {.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION, .pNext = &be.header},
                                                     .flags = FFX_FRAMEGENERATION_ENABLE_DEPTH_INFINITE, .displaySize = {w, h},
                                                     .maxRenderSize = {w, h}, .backBufferFormat = FFX_API_SURFACE_FORMAT_B8G8R8A8_UNORM};
    ffxReturnCode_t r = ffx_create(&ctx, &cd.header, NULL);
    printf("ffxCreateContext(frame generation %ux%u): %u\n", w, h, r);
    CHECK("FSR 3.1 frame generation context", r == FFX_API_RETURN_OK);
    if (!r && scctx && q.pOutQueuePresentKHR) {
        VkResult vr = fg_present_loop(q.pOutCreateSwapchainFFXAPI, q.pOutDestroySwapchainFFXAPI, &q, &ctx, sc, w, h);
        uint64_t presents = q.pOutGetLastPresentCountFFXAPI(sc);
        printf("  10 frames presented with interpolation: VkResult %d, %llu presents (rendered + generated)\n", vr, (unsigned long long)presents);
        CHECK("frame generation presents (> 10)", vr == VK_SUCCESS && presents > 10);
    }
    if (!r && !scctx) fg_dispatch_direct(&ctx, w, h);
    if (!r) CHECK("ffxDestroyContext(frame generation)", ffx_destroy(&ctx, NULL) == FFX_API_RETURN_OK);
    if (scctx) {
        vkDeviceWaitIdle(dev);
        q.pOutDestroySwapchainFFXAPI(dev, sc, NULL, scctx);
        CHECK("ffxDestroyContext(frame generation swapchain)", ffx_destroy(&scctx, NULL) == FFX_API_RETURN_OK);
    }
    if (surf) vkDestroySurfaceKHR(inst, surf, NULL);
}

// a game's frame loop: acquire, draw (a clear that changes every frame), configure + prepare frame generation, present
static VkResult fg_present_loop(PFN_vkCreateSwapchainFFXAPI cr, PFN_vkDestroySwapchainFFXAPI ds, struct ffxQueryDescSwapchainReplacementFunctionsVK *q,
                                ffxContext *ctx, VkSwapchainKHR sc, uint32_t w, uint32_t h) {
    (void)cr; (void)ds;
    uint32_t n = 0;
    q->pOutGetSwapchainImagesKHR(dev, sc, &n, NULL);
    VkImage imgs[8];
    n = n > 8 ? 8 : n;
    q->pOutGetSwapchainImagesKHR(dev, sc, &n, imgs);
    Img depth = image(VK_FORMAT_R32_SFLOAT, w, h), mv = image(VK_FORMAT_R16G16_SFLOAT, w, h);
    VkCommandBuffer cb = cmd_begin();
    to_general(cb, depth.image);
    to_general(cb, mv.image);
    barrier(cb);
    clear(cb, depth.image, 0.5f, 0, 0, 0);
    clear(cb, mv.image, 0, 0, 0, 0);
    cmd_submit_wait(cb);
    VkSemaphoreCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkSemaphore acq, done;
    VK(vkCreateSemaphore(dev, &sci, NULL, &acq));
    VK(vkCreateSemaphore(dev, &sci, NULL, &done));
    VkResult vr = VK_SUCCESS;
    for (uint64_t f = 0; f < 10 && vr == VK_SUCCESS; f++) {
        uint32_t idx = 0;
        vr = q->pOutAcquireNextImageKHR(dev, sc, UINT64_MAX, acq, VK_NULL_HANDLE, &idx);
        if (vr != VK_SUCCESS) { printf("  acquire: %d\n", vr); break; }
        struct ffxConfigureDescFrameGeneration c;
        memset(&c, 0, sizeof c);
        c.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
        c.swapChain = (void *)sc;
        c.frameGenerationCallback = fg_dispatch_cb;
        c.frameGenerationCallbackUserContext = ctx;
        c.frameGenerationEnabled = true;
        c.generationRect = (struct FfxApiRect2D){0, 0, (int32_t)w, (int32_t)h};
        c.frameID = f;
        if (ffx_configure(ctx, &c.header) != FFX_API_RETURN_OK) { printf("  configure failed\n"); return VK_ERROR_UNKNOWN; }
        cb = cmd_begin();
        VkImageMemoryBarrier b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                  .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                  .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                  .image = imgs[idx], .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &b);
        clear(cb, imgs[idx], (float)(f % 2), 0.5f, 0.25f, 1);
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = 0;
        b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 0, NULL, 1, &b);
        struct ffxDispatchDescFrameGenerationPrepare p;
        memset(&p, 0, sizeof p);
        p.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
        p.frameID = f;
        p.commandList = (void *)cb;
        p.renderSize = (struct FfxApiDimensions2D){w, h};
        p.motionVectorScale.x = (float)w;
        p.motionVectorScale.y = (float)h;
        p.frameTimeDelta = 16.7f;
        p.cameraNear = 0.1f;
        p.cameraFar = 3.402823466e+38f;
        p.cameraFovAngleVertical = 1.0f;
        p.viewSpaceToMetersFactor = 1;
        p.depth = res(&depth, FFX_API_SURFACE_FORMAT_R32_FLOAT, FFX_API_RESOURCE_USAGE_READ_ONLY);
        p.motionVectors = res(&mv, FFX_API_SURFACE_FORMAT_R16G16_FLOAT, FFX_API_RESOURCE_USAGE_READ_ONLY);
        if (ffx_dispatch(ctx, &p.header) != FFX_API_RETURN_OK) { printf("  prepare failed\n"); return VK_ERROR_UNKNOWN; }
        VK(vkEndCommandBuffer(cb));
        VkPipelineStageFlags ws = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .waitSemaphoreCount = 1, .pWaitSemaphores = &acq, .pWaitDstStageMask = &ws,
                           .commandBufferCount = 1, .pCommandBuffers = &cb, .signalSemaphoreCount = 1, .pSignalSemaphores = &done};
        VK(vkQueueSubmit(queue[0], 1, &si, VK_NULL_HANDLE));
        VkPresentInfoKHR pi = {.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, .waitSemaphoreCount = 1, .pWaitSemaphores = &done,
                               .swapchainCount = 1, .pSwapchains = &sc, .pImageIndices = &idx};
        vr = q->pOutQueuePresentKHR(queue[0], &pi);
        if (vr != VK_SUCCESS) printf("  present: %d\n", vr);
        vkQueueWaitIdle(queue[0]);
    }
    vkDeviceWaitIdle(dev);
    vkDestroySemaphore(dev, acq, NULL);
    vkDestroySemaphore(dev, done, NULL);
    vkDestroyImage(dev, depth.image, NULL); vkFreeMemory(dev, depth.mem, NULL);
    vkDestroyImage(dev, mv.image, NULL); vkFreeMemory(dev, mv.mem, NULL);
    return vr;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const char *path = argc > 1 ? argv[1] : "./libamd_fidelityfx_vk.so";
    void *so = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    CHECK("dlopen", so != NULL);
    if (!so) { printf("  %s\n", dlerror()); return 1; }
    ffx_create = (PfnFfxCreateContext)dlsym(so, "ffxCreateContext");
    ffx_destroy = (PfnFfxDestroyContext)dlsym(so, "ffxDestroyContext");
    ffx_configure = (PfnFfxConfigure)dlsym(so, "ffxConfigure");
    ffx_query = (PfnFfxQuery)dlsym(so, "ffxQuery");
    ffx_dispatch = (PfnFfxDispatch)dlsym(so, "ffxDispatch");
    CHECK("the five entry points", ffx_create && ffx_destroy && ffx_configure && ffx_query && ffx_dispatch);
    if (failures) return 1;
    bool headless;
    vk_init(&headless);
    test_upscaler();
    test_frame_generation(headless);
    vkDestroyCommandPool(dev, pool, NULL);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);
    printf("%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures, failures == 1 ? "" : "s");
    return failures != 0;
}
