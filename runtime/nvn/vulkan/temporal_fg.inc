// [XCDE] FSR 3 frame generation (included at the end of temporal.c: it uses its FidelityFX loader, images and helpers).
// Optional (xcde.ini frame_gen = 1, with FSR 3.1 as the upscaler). AMD's frame-generation swapchain replaces ours
// (device.c create_swapchain calls tmp_fg_install once): it presents each rendered frame plus one interpolated between
// it and the previous one, paced by its own present thread on the queues device.c set aside (present, image acquire,
// async compute). Each frame tmp_upscale configures it and prepares the interpolation from the same depth and motion
// vectors the upscaler used. The HUD: exec.c vk_present_draw renders the frame once more without it (the upscaled scene
// before T213's HUD composite) into fg_hl, handed over as HUDLessColor - the swapchain takes the HUD as the difference
// and composites it over the generated frames instead of moving it with the scene's motion (user: HUD smeared sideways).
#include "ffx_api/ffx_framegeneration.h"

// (ffx_api/vk/ffx_api_vk.h is C++-only: the declarations used here, as there)
#define FFX_API_CREATE_CONTEXT_DESC_TYPE_FGSWAPCHAIN_VK 0x40001u
#define FFX_API_QUERY_DESC_TYPE_FGSWAPCHAIN_FUNCTIONS_VK 0x40005u
typedef VkResult (*PFN_vkQueueSubmitFFXAPI)(uint32_t submitCount, const VkSubmitInfo *pSubmits, VkFence fence);
struct VkQueueInfoFFXAPI { VkQueue queue; uint32_t familyIndex; PFN_vkQueueSubmitFFXAPI submitFunc; };
struct ffxCreateContextDescFrameGenerationSwapChainVK {
    ffxCreateContextDescHeader header;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkSwapchainKHR *swapchain;
    VkAllocationCallbacks *allocator;
    VkSwapchainCreateInfoKHR createInfo;
    struct VkQueueInfoFFXAPI gameQueue, asyncComputeQueue, presentQueue, imageAcquireQueue;
};
typedef VkResult (*PFN_vkCreateSwapchainFFXAPI)(VkDevice, const VkSwapchainCreateInfoKHR *, const VkAllocationCallbacks *, VkSwapchainKHR *, void *);
typedef void (*PFN_vkDestroySwapchainFFXAPI)(VkDevice, VkSwapchainKHR, const VkAllocationCallbacks *, void *);
typedef uint64_t (*PFN_getLastPresentCountFFXAPI)(VkSwapchainKHR);
struct ffxQueryDescSwapchainReplacementFunctionsVK {
    ffxQueryDescHeader header;
    PFN_vkCreateSwapchainFFXAPI pOutCreateSwapchainFFXAPI;
    PFN_vkDestroySwapchainFFXAPI pOutDestroySwapchainFFXAPI;
    PFN_vkGetSwapchainImagesKHR pOutGetSwapchainImagesKHR;
    PFN_vkAcquireNextImageKHR pOutAcquireNextImageKHR;
    PFN_vkQueuePresentKHR pOutQueuePresentKHR;
    PFN_vkSetHdrMetadataEXT pOutSetHdrMetadataEXT;
    PFN_getLastPresentCountFFXAPI pOutGetLastPresentCountFFXAPI;
};

static ffxContext fgsc_ctx, fg_ctx;
static PFN_vkCreateSwapchainFFXAPI fg_create_fn;
static PFN_vkDestroySwapchainFFXAPI fg_destroy_fn;
static PFN_getLastPresentCountFFXAPI fg_present_count;
static uint64_t fg_frame_id;
static Img fg_hl;      // the frame without the HUD (window size, swapchain format)
static bool fg_hl_ok;  // filled for the frame being configured
static uint32_t fg_w, fg_h;
static bool fg_active;
bool tmp_fg_active(void) { return fg_active; }

// [XCDE] T224: frame generation for the whole game (frame_gen) or only for cutscenes held at 30 (cutscenes = 30fg). The
// swapchain is made at start either way; fg_on() says whether this frame is generated.
static bool fg_on(void) {
    extern bool xcde_cutscene_30(void);
    return g_settings.frame_gen || (g_settings.cutscene_mode == 2 && xcde_cutscene_30());
}
bool tmp_frame_gen_wanted(void) {
    // [XCDE] with DLSS too: AMD's frame generation takes the same depth and motion vectors (prep) whichever upscaler
    // ran, and fg_frame follows either (user: frame generation with DLSS on an RTX GPU)
    return g_settings.gpu_profile != GPU_PROFILE_COMPATIBILITY && g_vk.api_version >= VK_API_VERSION_1_3 &&
           (g_settings.frame_gen || g_settings.cutscene_mode == 2) &&
           (g_settings.upscaler == UPSCALE_FSR3 || g_settings.upscaler == UPSCALE_DLSS) && !diag_deterministic() &&
           !env_int("XCDE_TMP_OFF") && ffx_load();
}

static VkResult VKAPI_PTR fg_create_sc(VkDevice d, const VkSwapchainCreateInfoKHR *ci, const VkAllocationCallbacks *a, VkSwapchainKHR *sc) {
    return fg_create_fn(d, ci, a, sc, fgsc_ctx);
}
static void VKAPI_PTR fg_destroy_sc(VkDevice d, VkSwapchainKHR sc, const VkAllocationCallbacks *a) { fg_destroy_fn(d, sc, a, fgsc_ctx); }

static bool fg_context(uint32_t w, uint32_t h) {
    if (fg_ctx && fg_w == w && fg_h == h) return true;
    if (fg_ctx) { vk_submit_drain(); vkDeviceWaitIdle(g_vk.dev); ffx_destroy(&fg_ctx, NULL); fg_ctx = NULL; }
    struct ffxCreateBackendVKDesc be = {.header = {.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK},
                                        .vkDevice = g_vk.dev, .vkPhysicalDevice = g_vk.phys, .vkDeviceProcAddr = ffx_proc};
    struct ffxCreateContextDescFrameGeneration cd = {.header = {.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION, .pNext = &be.header},
                                                     .flags = FFX_FRAMEGENERATION_ENABLE_DEPTH_INFINITE, .displaySize = {w, h},
                                                     .maxRenderSize = {w, h}, .backBufferFormat = FFX_API_SURFACE_FORMAT_B8G8R8A8_UNORM};
    ffxReturnCode_t r = ffx_create(&fg_ctx, &cd.header, NULL);
    if (r != FFX_API_RETURN_OK) { LOG("[tmp] frame generation context failed: %u\n", r); fg_ctx = NULL; return false; }
    fg_w = w; fg_h = h;
    LOG("[tmp] frame generation %ux%u\n", w, h);
    return true;
}

static PFN_vkVoidFunction VKAPI_PTR fgsc_proc(VkDevice d, const char *name) {
    extern PFN_vkVoidFunction dlss_proxy(VkDevice dev, const char *name);
    static const char *const sc[] = {"vkCreateSwapchainKHR", "vkDestroySwapchainKHR", "vkGetSwapchainImagesKHR",
                                     "vkAcquireNextImageKHR", "vkQueuePresentKHR", "vkDeviceWaitIdle"};
    for (size_t i = 0; i < sizeof sc / sizeof sc[0]; i++)
        if (!strcmp(name, sc[i])) { PFN_vkVoidFunction f = dlss_proxy(d, name); if (f) return f; }
    return ffx_proc(d, name);
}

// device.c, the first swapchain: hand it to AMD's frame-generation swapchain and take its replacement functions.
bool tmp_fg_install(VkSwapchainKHR *sc, const VkSwapchainCreateInfoKHR *ci, PFN_vkCreateSwapchainKHR *create,
                    PFN_vkDestroySwapchainKHR *destroy, PFN_vkGetSwapchainImagesKHR *images, PFN_vkAcquireNextImageKHR *acquire,
                    PFN_vkQueuePresentKHR *present) {
    if (!tmp_frame_gen_wanted() || g_vk.fg_fam[0] == UINT32_MAX) return false;
    struct ffxCreateContextDescFrameGenerationSwapChainVK d;
    memset(&d, 0, sizeof d);
    d.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FGSWAPCHAIN_VK;
    d.physicalDevice = g_vk.phys;
    d.device = g_vk.dev;
    d.swapchain = sc;
    d.createInfo = *ci;
    d.createInfo.oldSwapchain = VK_NULL_HANDLE;
    d.gameQueue = (struct VkQueueInfoFFXAPI){g_vk.queue, g_vk.qfam, NULL};
    d.presentQueue = (struct VkQueueInfoFFXAPI){g_vk.fg_queue[0], g_vk.fg_fam[0], NULL};
    d.imageAcquireQueue = (struct VkQueueInfoFFXAPI){g_vk.fg_queue[1], g_vk.fg_fam[1], NULL};
    d.asyncComputeQueue = (struct VkQueueInfoFFXAPI){g_vk.fg_queue[2], g_vk.fg_fam[2], NULL};
    // [XCDE] with DLSS loaded the swapchain calls go through Streamline's proxies (device.c sc_proxies): AMD's swapchain
    // gets them too (its device functions through this backend's proc address), else Streamline never sees the
    // swapchain or its presents and DLSS waited forever on its first frame
    struct ffxCreateBackendVKDesc be = {.header = {.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK},
                                        .vkDevice = g_vk.dev, .vkPhysicalDevice = g_vk.phys, .vkDeviceProcAddr = fgsc_proc};
    d.header.pNext = &be.header;
    ffxReturnCode_t r = ffx_create(&fgsc_ctx, &d.header, NULL);
    if (r != FFX_API_RETURN_OK) { LOG("[tmp] frame generation swapchain failed: %u\n", r); fgsc_ctx = NULL; return false; }
    struct ffxQueryDescSwapchainReplacementFunctionsVK q;
    memset(&q, 0, sizeof q);
    q.header.type = FFX_API_QUERY_DESC_TYPE_FGSWAPCHAIN_FUNCTIONS_VK;
    ffx_query(&fgsc_ctx, &q.header);
    if (!q.pOutCreateSwapchainFFXAPI || !q.pOutGetSwapchainImagesKHR || !q.pOutAcquireNextImageKHR || !q.pOutQueuePresentKHR) {
        LOG("[tmp] frame generation swapchain: replacement functions missing\n");
        return false;
    }
    fg_create_fn = q.pOutCreateSwapchainFFXAPI;
    fg_destroy_fn = q.pOutDestroySwapchainFFXAPI;
    fg_present_count = q.pOutGetLastPresentCountFFXAPI;
    *create = fg_create_sc;
    *destroy = fg_destroy_sc;
    *images = q.pOutGetSwapchainImagesKHR;
    *acquire = q.pOutAcquireNextImageKHR;
    *present = q.pOutQueuePresentKHR;
    fg_active = true;
    LOG("[tmp] FSR 3 frame generation swapchain (queues: present %u.%u, acquire %u.%u, compute %u.%u)\n", g_vk.fg_fam[0],
        g_vk.fg_idx[0], g_vk.fg_fam[1], g_vk.fg_idx[1], g_vk.fg_fam[2], g_vk.fg_idx[2]);
    return true;
}

static ffxReturnCode_t fg_dispatch_cb(ffxDispatchDescFrameGeneration *p, void *ctx) { return ffx_dispatch((ffxContext *)ctx, &p->header); }

// Every presented frame (with or without this frame's camera): the frame-generation configuration; with a camera, the
// interpolation inputs. enabled = false: the swapchain presents only rendered frames (menus, loading screens).
static void fg_frame(VkCommandBuffer cb, bool enabled, const TmpCamera *cam, const float jit[2], uint32_t rw, uint32_t rh, float dt,
                     bool reset) {
    if (!fg_active) return;
    if (!fg_context(g_vk.sc_extent.width, g_vk.sc_extent.height)) return;
    enabled = enabled && fg_on();  // (the F1 menu can turn it off; the swapchain stays)
    struct ffxConfigureDescFrameGeneration c;
    memset(&c, 0, sizeof c);
    c.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
    c.swapChain = (void *)g_vk.swapchain;
    c.frameGenerationCallback = fg_dispatch_cb;
    c.frameGenerationCallbackUserContext = &fg_ctx;
    c.frameGenerationEnabled = enabled;
    c.allowAsyncWorkloads = false;
    c.generationRect = (struct FfxApiRect2D){0, 0, (int32_t)g_vk.sc_extent.width, (int32_t)g_vk.sc_extent.height};
    c.frameID = fg_frame_id;
    if (fg_hl_ok && fg_hl.b.image && fg_hl.b.width == g_vk.sc_extent.width && fg_hl.b.height == g_vk.sc_extent.height)
        c.HUDLessColor = res(&fg_hl, FFX_API_SURFACE_FORMAT_B8G8R8A8_UNORM, FFX_API_RESOURCE_USAGE_READ_ONLY);
    ffxReturnCode_t r = ffx_configure(&fg_ctx, &c.header);
    if (r != FFX_API_RETURN_OK) { static int n; if (n++ < 5) LOG("[tmp] frame generation configure failed: %u\n", r); }
    if (enabled) {
        struct ffxDispatchDescFrameGenerationPrepare p;
        memset(&p, 0, sizeof p);
        p.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
        p.frameID = fg_frame_id;
        p.commandList = (void *)cb;
        p.renderSize = (struct FfxApiDimensions2D){rw, rh};
        p.jitterOffset.x = jit[0];
        p.jitterOffset.y = jit[1];
        p.motionVectorScale.x = (float)rw;
        p.motionVectorScale.y = (float)rh;
        p.frameTimeDelta = dt;
        p.unused_reset = reset;
        p.cameraNear = fabsf(cam->proj[11]) * 0.5f;
        p.cameraFar = FLT_MAX;
        p.cameraFovAngleVertical = cam->proj[5] != 0 ? 2.0f * atanf(1.0f / fabsf(cam->proj[5])) : 0.785f;
        p.viewSpaceToMetersFactor = 1.0f;
        p.depth = res(&img[2], FFX_API_SURFACE_FORMAT_R32_FLOAT, FFX_API_RESOURCE_USAGE_READ_ONLY);
        p.motionVectors = res(&img[1], FFX_API_SURFACE_FORMAT_R16G16_FLOAT, FFX_API_RESOURCE_USAGE_READ_ONLY);
        r = ffx_dispatch(&fg_ctx, &p.header);
        if (r != FFX_API_RETURN_OK) { static int n; if (n++ < 5) LOG("[tmp] frame generation prepare failed: %u\n", r); }
    }
    fg_frame_id++;
    if (env_int("XCDE_TMP_LOG") && fg_present_count && fg_frame_id % 120 == 0) {  // (presents: rendered + generated)
        static uint64_t last_p, last_f;
        uint64_t p = fg_present_count(g_vk.swapchain);
        LOG("[tmp] frame generation: %llu presents for %llu frames\n", (unsigned long long)(p - last_p), (unsigned long long)(fg_frame_id - last_f));
        last_p = p; last_f = fg_frame_id;
    }
}
void tmp_fg_idle(VkCommandBuffer cb) { fg_frame(cb, false, NULL, NULL, 0, 0, 0, false); }

// exec.c vk_present_draw: where to render this frame without its HUD (NULL: frame generation off, or
// XCDE_FG_NO_HUDLESS=1). The copy is used from the next configure on: frames keep the same HUD layout from one to the next.
VkImageView tmp_fg_hudless_target(void) {
    fg_hl_ok = false;
    if (!fg_active || !fg_on() || env_int("XCDE_FG_NO_HUDLESS")) return VK_NULL_HANDLE;
    img_make(&fg_hl, g_vk.sc_format, g_vk.sc_extent.width, g_vk.sc_extent.height,
             VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    fg_hl_ok = true;
    return fg_hl.v;
}
