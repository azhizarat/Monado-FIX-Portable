// Vulkan backend for the NVN layer (Windows; macOS uses runtime/nvn/metal.m + renderer.m).
// Files: device.c (instance, window, swapchain, present, screenshots, ImGui hooks), textures.c (images, uploads,
// views), exec.c (command execution: passes, draws, clears, copies, compute), shaders.c (GLSL -> SPIR-V, programs,
// pipelines). Mirrors the Metal backend's behaviour; differences are noted where they happen.
//
// Synchronisation model: every image lives in VK_IMAGE_LAYOUT_GENERAL, and a full memory barrier separates
// render passes, copies and dispatches (Metal tracks these hazards itself). Submissions are numbered; a timeline
// semaphore signals each number, which is what NVN fences (renderer_fence_seq / renderer_wait_seq) wait for.
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan.h>
#include "dispatch.h"
#include "gpu_probe.h"
#if __has_include(<vma/vk_mem_alloc.h>)
#include <vma/vk_mem_alloc.h>
#else
#include <vk_mem_alloc.h>  // (Debian / Ubuntu libvulkan-memory-allocator-dev)
#endif

#include "../nvn.h"
#include "../maxwell/shader.h"

void vk_device_fault_report(void);
#define VK_CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { if (r_ == VK_ERROR_DEVICE_LOST) vk_device_fault_report(); \
    FATAL("%s failed: %d (%s:%d)\n", #x, r_, __FILE__, __LINE__); } } while (0)

typedef struct {
    uint32_t api_version, loader_api_version;
    bool pipeline_cache_control;
    XcdeVkDispatch dispatch;
    VkInstance inst;
    VkPhysicalDevice phys;
    VkDevice dev;
    VkQueue queue;
    uint32_t qfam;
    uint32_t phys_index;  // index from vkEnumeratePhysicalDevices, for diagnostics and manual selection
    VmaAllocator vma;
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceMemoryProperties memory_props; // T260: immutable for the lifetime of this physical device
    VkPhysicalDeviceDriverProperties driver_props;
    VkPipelineCache pcache;
    VkSemaphore timeline;      // value = last completed submission number
    VkCommandPool pool;        // render thread only
    bool custom_border, divisor, host_import, robust2, logic_op, minmax;
    bool heap_fd_import;  // [XCDE] VK_KHR_external_memory_fd: the placed guest heap (heap_placed.c)
    bool gpl;  // [XCDE] T188: VK_EXT_graphics_pipeline_library with fast linking (exec.c build_pipeline)
    // [XCDE] FSR 3 frame generation's queues (device.c create_device): present, image acquire, async compute; fam UINT32_MAX: none
    VkQueue fg_queue[3];
    uint32_t fg_fam[3], fg_idx[3];
    // window
    VkSurfaceKHR surface;
    VkSwapchainKHR swapchain;
    VkFormat sc_format;
    VkExtent2D sc_extent;
    uint32_t sc_count;
    VkImage sc_images[8];
    VkImageView sc_views[8];
} VkG;
extern VkG g_vk;
extern XcdeGpu *g_vk_adapters;
extern uint32_t g_vk_adapter_count;

// [XCDE] T250: do not call core 1.3 loader stubs on 1.2 extension devices.
#define vkCmdBeginRendering g_vk.dispatch.begin_rendering
#define vkCmdEndRendering g_vk.dispatch.end_rendering
#define vkCmdPipelineBarrier2 g_vk.dispatch.barrier2
#define vkQueueSubmit2 g_vk.dispatch.submit2
#define vkCmdWriteTimestamp2 g_vk.dispatch.timestamp2

// ---- submissions (exec.c)
// Work is recorded into the current command buffer; uploads go into a separate one that runs first.
VkCommandBuffer vk_cmd(void);         // current command buffer (outside any rendering unless exec.c began one)
VkCommandBuffer vk_upload_cmd(void);  // upload command buffer of the pending submission
uint64_t vk_pending_seq(void);        // number the pending submission will get
uint64_t vk_completed_seq(void);
void vk_wait_seq(uint64_t seq);
void vk_full_barrier(VkCommandBuffer cb);
void vk_end_pass(void);
void vk_external_state_changed(void); // restore tracked state after a vendor SDK dispatch
double vk_native_gpu_ms(void); // latest complete native-frame GPU sum, -1 if unavailable/stale; never waits
void vk_metrics_frame(uint64_t frame, uint64_t now_ns, uint64_t present_ns);
void vk_submit_extra(VkSemaphore wait, VkSemaphore signal, VkCommandBuffer extra);  // submit + one more cb (present)

// Transient GPU-visible memory that lives until the submission it was allocated for completes.
typedef struct { VkBuffer buf; uint64_t off; void *ptr; } VkSpan;
VkSpan vk_transient(uint64_t size, uint64_t align);
VkSpan vk_transient_copy(const void *data, uint64_t size, uint64_t align);
VkSpan vk_readback(uint64_t size);  // host-cached GPU -> CPU staging (read right after the copy completes)
VkDescriptorSet vk_alloc_set(VkDescriptorSetLayout layout);

// ---- textures (textures.c)
typedef struct TexBackend {
    VkImage image;
    VmaAllocation mem;
    VkFormat format;
    VkImageViewType base_type;
    uint32_t width, height, depth, layers, levels, samples;
    VkImageAspectFlags aspect;
    bool gpu_owned, uploaded, dynamic;
    uint64_t data_hash, sample_hash, checked_frame, full_frame;
    uint64_t first_up_frame;  // [XCDE] glitch watch: the frame of the first upload
    bool first_up_zero;       // that upload's data was all zero bytes
    uint32_t wgen;  // page write tracking generation at the last upload check (textures.c)
    uint8_t scalable;  // 0 fixed size, 1 the 16:9 scene targets (sx, sy), 2 [XCDE] T182 portrait targets (uniform)
    bool presented;
    uint64_t touched_frame, frame_cleared;
    uint32_t level_gen;
    float sx, sy;
    void *pending_rgba;  // host-produced pixels (video) waiting to be uploaded on the render thread
    uint32_t pending_w, pending_h, pending_stride;
    uint8_t astc_bw, astc_bh;  // ASTC without GPU support: decoded to RGBA8 on upload (runtime/nvn/astc.c)
    bool astc_srgb;
    int tp;              // runtime/nvn/texpatch.c: 0 not checked yet, -1 none, else known UI texture + 1
    uint32_t tp_gen;     // texpatch_generation() it was last uploaded with
    int reg;  // slot in textures.c's live list + 1 (a reboot destroys them all)
    struct TexBackend *address_next; // T260: renderer-thread address index, same sharing rules as the live list
    unsigned address_bucket;
    // [XCDE] other NVN texture objects over the same storage (same address, size, compatible format - e.g. the
    // game's stencil-mode texture over its G-buffer depth): they share this image instead of an empty one of their own
    struct NvnTexture *share[8];
    int nshare;
    uint32_t w_epoch, w_att, r_epoch;  // [XCDE] T162: full-barrier epoch of the last attachment write, attachment-barrier
                                       // epoch of that write, full-barrier epoch of the last sampling (exec.c pass_barrier)
    bool has_storage;  // [XCDE] T192: created with STORAGE usage (SULD / SUST views)
    uint8_t nlean;      // [XCDE] spec_lowend_gpu R6b: > 0: the image may only be viewed in lean_fmt[0..nlean)
    VkFormat lean_fmt[3];
    bool storage_failed;  // [XCDE] T192: vk_tex_add_storage could not give it (don't retry every draw)
} TexBackend;

TexBackend *vk_tex(NvnTexture *t, bool for_render);  // backend image (uploads CPU contents unless for_render)
bool vk_tex_add_storage(NvnTexture *t, VkCommandBuffer cb);  // [XCDE] T192 (textures.c)
VkFormat vk_format(uint32_t nvn_format);
bool vk_format_alpha_one(uint32_t nvn_format);
// View of t (optionally through an NVN texture view) for rendering or sampling.
VkImageView vk_view(NvnTexture *t, const NvnTextureView *v, bool has_view, bool render, int want_textype);
VkImageView vk_storage_view(NvnTexture *t);  // [XCDE] SULD/SUST
VkImageView vk_dummy_storage_view(void);
VkImageView vk_image_view(TexBackend *b, VkImageViewType type, VkFormat fmt, uint32_t lv0, uint32_t nlv, uint32_t ly0,
                          uint32_t nly, VkComponentMapping sw, VkImageAspectFlags aspect);
VkImageView vk_dummy_view(int textype);
VkSampler vk_sampler(NvnSamplerDesc *d);
VkSampler vk_default_sampler(bool linear);
void vk_image_init_layout(VkCommandBuffer cb, VkImage img, VkImageAspectFlags aspect);
// Destruction once every submission that may use the object has completed.
void vk_defer_image(VkImage img, VmaAllocation mem);
void vk_defer_buffer(VkBuffer buf, VmaAllocation mem);
void vk_defer_import(VkBuffer buf, VkDeviceMemory mem);
void vk_defer_view(VkImageView v);
void vk_views_forget(TexBackend *b);  // drop (deferred) the cached views of an image being replaced
uint64_t vk_frame_counter(void);

// ---- shaders / pipelines (shaders.c)
VkShaderModule vk_compile_glsl(const char *src, int stage, const char *name);  // NULL on failure (logged)
VkShaderModule vk_builtin_shader(const char *name);  // internal shaders (clear, copy, present)

// ---- device / window (device.c)
void vk_save_png(const char *path, const uint8_t *rgba, uint32_t w, uint32_t h);
