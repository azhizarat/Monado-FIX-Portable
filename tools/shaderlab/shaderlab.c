// Shader lab: compile every SPIR-V module in cache/spirv.pack with the real driver, without the game, so the driver's
// per-shader statistics (RADV_DEBUG=shaderstats: VGPRs, SGPRs, spills, scratch, instructions...) can be ranked.
// Each module becomes a VK_EXT_shader_object of its stage; descriptor set layouts are rebuilt from the SPIR-V itself.
// Before each compile a marker line goes to stderr ("@@ <pack hash> <stage> sm=<0|1> words=<n>"), so the driver's
// stats that follow belong to it. sm=1: the translator's branch state machine (an OpSwitch inside a loop).
//   MESA_SHADER_CACHE_DISABLE=true RADV_DEBUG=shaderstats ./shaderlab cache/spirv.pack [max] 2> stats.txt
//   python3 shaderlab_rank.py stats.txt
#define _GNU_SOURCE
#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint32_t set, binding; VkDescriptorType type; } Bind;

// ---- SPIR-V reflection: just what a set layout and the stage need
#define MAXID (1u << 20)
static uint32_t *ty_op, *ty_a, *ty_b, *ptr_sc, *ptr_ty, *dec_set, *dec_bind, *dec_flags;  // per id
static int reflect(const uint32_t *w, size_t n, VkShaderStageFlagBits *stage, Bind *b, int maxb, int *sm) {
    if (n < 5 || w[0] != 0x07230203u) return -1;
    uint32_t bound = w[3];
    if (bound >= MAXID) return -1;
    memset(ty_op, 0, bound * 4); memset(dec_set, 0xFF, bound * 4); memset(dec_bind, 0xFF, bound * 4); memset(dec_flags, 0, bound * 4);
    int nb = 0, has_switch = 0, has_loop = 0;
    *stage = 0;
    typedef struct { uint32_t id, type, sc; } Var;
    static Var vars[4096];
    int nv = 0;
    for (size_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xFFFF, len = w[i] >> 16;
        if (!len || i + len > n) return -1;
        const uint32_t *a = w + i + 1;
        switch (op) {
        case 15: {  // OpEntryPoint
            uint32_t m = a[0];
            *stage = m == 0 ? VK_SHADER_STAGE_VERTEX_BIT : m == 4 ? VK_SHADER_STAGE_FRAGMENT_BIT : m == 3 ? VK_SHADER_STAGE_GEOMETRY_BIT :
                     m == 5 ? VK_SHADER_STAGE_COMPUTE_BIT : 0;
            break;
        }
        case 71:  // OpDecorate target dec [literal]
            if (a[1] == 34) dec_set[a[0]] = a[2];
            else if (a[1] == 33) dec_bind[a[0]] = a[2];
            else if (a[1] == 2) dec_flags[a[0]] |= 1;  // Block
            else if (a[1] == 3) dec_flags[a[0]] |= 2;  // BufferBlock
            break;
        case 25: ty_op[a[0]] = 25; ty_a[a[0]] = a[2]; ty_b[a[0]] = a[6]; break;  // OpTypeImage: dim, sampled
        case 27: ty_op[a[0]] = 27; ty_a[a[0]] = a[1]; break;                    // OpTypeSampledImage: image
        case 26: ty_op[a[0]] = 26; break;                                        // OpTypeSampler
        case 28: case 29: ty_op[a[0]] = op; ty_a[a[0]] = a[1]; break;            // OpTypeArray / RuntimeArray: element
        case 30: ty_op[a[0]] = 30; break;                                        // OpTypeStruct
        case 32: ptr_sc[a[0]] = a[1]; ptr_ty[a[0]] = a[2]; break;                // OpTypePointer
        case 59: if (nv < 4096) vars[nv++] = (Var){a[1], a[0], a[2]}; break;     // OpVariable type id sc
        case 251: has_switch = 1; break;  // OpSwitch
        case 246: has_loop = 1; break;    // OpLoopMerge
        }
        i += len;
    }
    for (int k = 0; k < nv && nb < maxb; k++) {
        Var v = vars[k];
        if (v.sc != 0 && v.sc != 2 && v.sc != 12) continue;  // UniformConstant, Uniform, StorageBuffer
        if (dec_set[v.id] == 0xFFFFFFFFu || dec_bind[v.id] == 0xFFFFFFFFu) continue;
        uint32_t t = ptr_ty[v.type];
        while (ty_op[t] == 28 || ty_op[t] == 29) t = ty_a[t];  // (arrays: one descriptor is enough for a compile)
        VkDescriptorType dt;
        if (v.sc == 12 || (v.sc == 2 && (dec_flags[t] & 2))) dt = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        else if (v.sc == 2) dt = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        else if (ty_op[t] == 27) dt = ty_a[ty_a[t]] == 5 ? VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        else if (ty_op[t] == 25) dt = ty_b[t] == 2 ? (ty_a[t] == 5 ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                                                   : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        else if (ty_op[t] == 26) dt = VK_DESCRIPTOR_TYPE_SAMPLER;
        else continue;
        b[nb++] = (Bind){dec_set[v.id], dec_bind[v.id], dt};
    }
    *sm = has_switch && has_loop;
    return nb;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: shaderlab <spirv.pack> [max]\n"); return 2; }
    long max = argc > 2 ? atol(argv[2]) : 0;
    uint32_t **tabs[] = {&ty_op, &ty_a, &ty_b, &ptr_sc, &ptr_ty, &dec_set, &dec_bind, &dec_flags};
    for (size_t i = 0; i < sizeof tabs / sizeof *tabs; i++) *tabs[i] = calloc(MAXID, 4);

    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "shaderlab", .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkInstance inst;
    if (vkCreateInstance(&ici, NULL, &inst)) return 3;
    VkPhysicalDevice pd; uint32_t np = 1;
    vkEnumeratePhysicalDevices(inst, &np, &pd);
    // every feature the device has (the translated shaders use int64, subgroups, storage writes...) + shader objects
    VkPhysicalDeviceShaderObjectFeaturesEXT so = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};
    VkPhysicalDeviceVulkan13Features f13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, .pNext = &so};
    VkPhysicalDeviceVulkan12Features f12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &f13};
    VkPhysicalDeviceVulkan11Features f11 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, .pNext = &f12};
    VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f11};
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    if (!so.shaderObject) { fprintf(stderr, "no VK_EXT_shader_object\n"); return 4; }
    float prio = 1;
    VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1, .pQueuePriorities = &prio};
    const char *ext[] = {VK_EXT_SHADER_OBJECT_EXTENSION_NAME};
    VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f2, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi,
                              .enabledExtensionCount = 1, .ppEnabledExtensionNames = ext};
    VkDevice dev;
    if (vkCreateDevice(pd, &dci, NULL, &dev)) return 5;
    PFN_vkCreateShadersEXT create = (PFN_vkCreateShadersEXT)vkGetDeviceProcAddr(dev, "vkCreateShadersEXT");
    PFN_vkDestroyShaderEXT destroy = (PFN_vkDestroyShaderEXT)vkGetDeviceProcAddr(dev, "vkDestroyShaderEXT");

    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 6; }
    uint64_t *seen = calloc(1u << 22, 8);  // open addressing on the pack hash
    long done = 0, fails = 0;
    uint32_t hdr[4];
    static uint32_t words[4 << 20];
    while (fread(hdr, 4, 4, f) == 4) {
        if (hdr[0] != 0x56505358u) { fprintf(stderr, "bad record\n"); break; }  // 'XSPV'
        uint32_t bytes = hdr[1];
        uint64_t h = hdr[2] | (uint64_t)hdr[3] << 32;
        if (bytes > sizeof words || fread(words, 1, bytes, f) != bytes) break;
        uint64_t s = (h * 0x9E3779B97F4A7C15ull) & ((1u << 22) - 1);
        while (seen[s] && seen[s] != h) s = (s + 1) & ((1u << 22) - 1);
        if (seen[s] == h) continue;
        seen[s] = h;
        VkShaderStageFlagBits stage; Bind b[160]; int sm;
        int nb = reflect(words, bytes / 4, &stage, b, 160, &sm);
        if (nb < 0 || !stage || stage == VK_SHADER_STAGE_COMPUTE_BIT) continue;  // (compute: not in this lab yet)
        // set layouts: one per set index up to the highest used, empty where unused
        uint32_t nsets = 0;
        for (int k = 0; k < nb; k++) if (b[k].set + 1 > nsets) nsets = b[k].set + 1;
        if (nsets > 8) continue;
        VkDescriptorSetLayout sl[8];
        for (uint32_t st = 0; st < nsets; st++) {
            VkDescriptorSetLayoutBinding lb[160]; uint32_t n = 0;
            for (int k = 0; k < nb; k++)
                if (b[k].set == st) lb[n++] = (VkDescriptorSetLayoutBinding){b[k].binding, b[k].type, 1, VK_SHADER_STAGE_ALL, NULL};
            VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = n, .pBindings = lb};
            vkCreateDescriptorSetLayout(dev, &li, NULL, &sl[st]);
        }
        VkShaderCreateInfoEXT ci = {.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT, .stage = stage,
                                    .nextStage = stage == VK_SHADER_STAGE_VERTEX_BIT ? VK_SHADER_STAGE_FRAGMENT_BIT : 0,
                                    .codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT, .codeSize = bytes, .pCode = words, .pName = "main",
                                    .setLayoutCount = nsets, .pSetLayouts = sl};
        fprintf(stderr, "@@ %016llx %s sm=%d words=%u\n", (unsigned long long)h,
                stage == VK_SHADER_STAGE_VERTEX_BIT ? "vs" : stage == VK_SHADER_STAGE_FRAGMENT_BIT ? "fs" : "gs", sm, bytes / 4);
        fflush(stderr);
        VkShaderEXT sh;
        if (create(dev, 1, &ci, NULL, &sh) == VK_SUCCESS) destroy(dev, sh, NULL);
        else { fails++; fprintf(stderr, "@@ FAIL\n"); }
        for (uint32_t st = 0; st < nsets; st++) vkDestroyDescriptorSetLayout(dev, sl[st], NULL);
        if (++done % 500 == 0) printf("%ld shaders (%ld failed)\n", done, fails), fflush(stdout);
        if (max && done >= max) break;
    }
    printf("done: %ld shaders, %ld failed\n", done, fails);
    return 0;
}
