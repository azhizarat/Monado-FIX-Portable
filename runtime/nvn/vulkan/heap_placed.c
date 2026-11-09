// [XCDE] Linux: the guest heap as GPU memory from the start (VK_EXT_map_memory_placed), instead of importing it later as
// host memory (VK_EXT_external_memory_host).
//
// Why: the host-memory import is an amdgpu "userptr" buffer, and the kernel re-validates every page of a userptr buffer
// on every queue submission (hmm_range_fault over the whole range). With the 3.25 GB heap that cost ~4.5 ms per
// vkQueueSubmit on a Steam Deck (render thread: submit 1283 ms of every 2 s, 44 fps uncapped; without the import:
// 34 ms, 60 fps). An ordinary GTT buffer has no such check.
//
// How: before the guest starts (main_boot, nothing is in the heap yet), a small Vulkan device allocates HEAP_MAX of
// host-visible, coherent, cached memory, maps it at HEAP_BASE (placed map: replaces the reserved range there) and
// exports it as an opaque fd. The renderer's device imports that fd (exec.c gmem_import): the same buffer object, so
// CPU writes are what the GPU reads. The early device stays alive (it owns the CPU mapping).
// The RAM is committed when the driver first uses the buffer (~3.25 GB, like a fully used heap).
// XCDE_HEAP_PLACED=0: off (the host-memory import as before).
#include "vk.h"
#include "../../diag/diag.h"
#include "../../rt.h"
#include <stdlib.h>
#include <string.h>

#if defined(__linux__)
#include <unistd.h>

static int heap_fd = -1;
static uint32_t heap_type;          // memory type index (same physical device: same list in the renderer's device)
static uint8_t heap_uuid[VK_UUID_SIZE];

bool vk_heap_place(void) {
    const char *e = getenv("XCDE_HEAP_PLACED");
    if (e && !atoi(e)) return false;
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "xcde-heap", .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkInstance inst;
    if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) return false;
    VkPhysicalDevice phys[8];
    uint32_t np = 8;
    if (vkEnumeratePhysicalDevices(inst, &np, phys) < 0 || !np) goto fail_inst;
    // the renderer picks the discrete GPU first, else the integrated one (gpu_policy.h); LARECOMP_GPU=<n> overrides.
    // The import checks the device UUID, so a wrong guess here only means the old path.
    VkPhysicalDevice pd = phys[0];
    const char *g = getenv("LARECOMP_GPU");
    if (g && (uint32_t)atoi(g) < np) pd = phys[atoi(g)];
    else for (uint32_t i = 0; i < np; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(phys[i], &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { pd = phys[i]; break; }
    }
    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, NULL);
    VkExtensionProperties *ep = calloc(ne ? ne : 1, sizeof *ep);
    vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, ep);
    int have = 0;
    for (uint32_t i = 0; i < ne; i++)
        have += !strcmp(ep[i].extensionName, VK_EXT_MAP_MEMORY_PLACED_EXTENSION_NAME) ||
                !strcmp(ep[i].extensionName, VK_KHR_MAP_MEMORY_2_EXTENSION_NAME) ||
                !strcmp(ep[i].extensionName, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    free(ep);
    if (have != 3) { LOG("[heap] placed map unavailable (extensions): host-memory import\n"); goto fail_inst; }
    VkPhysicalDeviceMapMemoryPlacedFeaturesEXT placed = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAP_MEMORY_PLACED_FEATURES_EXT};
    VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &placed};
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    if (!placed.memoryMapPlaced) { LOG("[heap] placed map unavailable (feature): host-memory import\n"); goto fail_inst; }
    // host-visible + coherent + CACHED: the whole CPU heap lives here (write-combined memory would make every CPU read slow)
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    int type = -1;
    for (uint32_t i = 0; i < mp.memoryTypeCount && type < 0; i++)
        if ((mp.memoryTypes[i].propertyFlags & want) == want) type = (int)i;
    if (type < 0) { LOG("[heap] no host-cached coherent memory type: host-memory import\n"); goto fail_inst; }
    float prio = 1;
    VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1, .pQueuePriorities = &prio};
    const char *ext[] = {VK_KHR_MAP_MEMORY_2_EXTENSION_NAME, VK_EXT_MAP_MEMORY_PLACED_EXTENSION_NAME, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME};
    placed.memoryMapRangePlaced = VK_FALSE;
    placed.memoryUnmapReserve = VK_FALSE;
    VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &placed, .queueCreateInfoCount = 1,
                              .pQueueCreateInfos = &qi, .enabledExtensionCount = 3, .ppEnabledExtensionNames = ext};
    VkDevice dev;
    if (vkCreateDevice(pd, &dci, NULL, &dev) != VK_SUCCESS) { LOG("[heap] device creation failed: host-memory import\n"); goto fail_inst; }
    VkExportMemoryAllocateInfo ex = {.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
                                     .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &ex, .allocationSize = HEAP_MAX,
                               .memoryTypeIndex = (uint32_t)type};
    VkDeviceMemory mem;
    if (vkAllocateMemory(dev, &ai, NULL, &mem) != VK_SUCCESS) { LOG("[heap] %llu MB allocation failed: host-memory import\n", (unsigned long long)(HEAP_MAX >> 20)); goto fail_dev; }
    PFN_vkMapMemory2KHR map2 = (PFN_vkMapMemory2KHR)vkGetDeviceProcAddr(dev, "vkMapMemory2KHR");
    PFN_vkGetMemoryFdKHR getfd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(dev, "vkGetMemoryFdKHR");
    VkMemoryMapPlacedInfoEXT pl = {.sType = VK_STRUCTURE_TYPE_MEMORY_MAP_PLACED_INFO_EXT, .pPlacedAddress = (void *)(uintptr_t)HEAP_BASE};
    VkMemoryMapInfoKHR mi = {.sType = VK_STRUCTURE_TYPE_MEMORY_MAP_INFO_KHR, .pNext = &pl, .flags = VK_MEMORY_MAP_PLACED_BIT_EXT,
                             .memory = mem, .offset = 0, .size = VK_WHOLE_SIZE};
    void *p = NULL;
    if (!map2 || !getfd || map2(dev, &mi, &p) != VK_SUCCESS || p != (void *)(uintptr_t)HEAP_BASE) {
        // (a failed placed map leaves the range unmapped: give the guest its anonymous memory back)
        plat_reset_guest(HEAP_BASE, HEAP_BASE + HEAP_MAX);
        LOG("[heap] placed map at %llx failed: host-memory import\n", (unsigned long long)HEAP_BASE);
        goto fail_mem;
    }
    VkMemoryGetFdInfoKHR gi = {.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, .memory = mem,
                               .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
    if (getfd(dev, &gi, &heap_fd) != VK_SUCCESS) {
        // the heap stays this memory (mapped, fine for the CPU); the renderer just can't share it: no gmem import
        heap_fd = -1;
        LOG("[heap] fd export failed: the heap is placed but not shared\n");
        return false;
    }
    VkPhysicalDeviceIDProperties id = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &id};
    vkGetPhysicalDeviceProperties2(pd, &p2);
    memcpy(heap_uuid, id.deviceUUID, VK_UUID_SIZE);
    heap_type = (uint32_t)type;
    LOG("[heap] guest heap placed in GPU memory: %llu MB at %llx (memory type %d, %s)\n", (unsigned long long)(HEAP_MAX >> 20),
        (unsigned long long)HEAP_BASE, type, p2.properties.deviceName);
    return true;  // (dev, mem and inst stay alive: they own the mapping)
fail_mem:
    vkFreeMemory(dev, mem, NULL);
fail_dev:
    vkDestroyDevice(dev, NULL);
fail_inst:
    vkDestroyInstance(inst, NULL);
    return false;
}

// For the renderer's device (same physical device, checked by UUID): a new fd for vkAllocateMemory to import (it takes
// ownership of it) and the memory type. -1: not placed / another GPU.
int vk_heap_fd(VkPhysicalDevice phys, uint32_t *type) {
    if (heap_fd < 0) return -1;
    VkPhysicalDeviceIDProperties id = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &id};
    vkGetPhysicalDeviceProperties2(phys, &p2);
    if (memcmp(id.deviceUUID, heap_uuid, VK_UUID_SIZE)) { LOG("[heap] the renderer uses another GPU: not shared\n"); return -1; }
    *type = heap_type;
    return dup(heap_fd);
}
#else
bool vk_heap_place(void) { return false; }
int vk_heap_fd(VkPhysicalDevice phys, uint32_t *type) { (void)phys, (void)type; return -1; }
#endif
