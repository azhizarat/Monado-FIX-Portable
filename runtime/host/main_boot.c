// Headless boot host (M3a): loads the modules, sets up the kernel and runs nn::init::Start, with strict logging of
// every service, SVC and import the game touches. No window, no renderer. Derived from LASRecomp runtime/main.c (MIT). [XCDE]
#include "../diag/diag.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

#include "../gamelib.h"
#include "../hle.h"
#include "../kernel.h"
#include "../rt.h"

const char *g_romfs_dir;

#if defined(XCDE_GFX) && defined(_WIN32)
// [XCDE] T194: laptops with two GPUs (NVIDIA Optimus / AMD switchable graphics) run a program on the integrated one
// unless the exe exports these; the driver reads them at start. (device.c then picks the discrete GPU among those
// Vulkan lists; LARECOMP_GPU=<n> overrides.)
__declspec(dllexport) DWORD NvOptimusEnablement = 1;
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
#endif

// T265: the same lifted camera code is also linked into the headless host.
uint32_t xcde_scene_aspect(uint32_t original, uint32_t camera) {
#if defined(XCDE_GFX)
    extern uint32_t vk_scene_aspect(uint32_t original, uint32_t camera);
    return vk_scene_aspect(original, camera);
#else
    (void)camera;
    return original;
#endif
}

static void run_array(Cpu *c, uint64_t arr, uint64_t size) {
    for (uint64_t i = 0; i < size / 8; i++) {
        uint64_t fn = ((uint64_t *)arr)[i];
        if (fn) guest_call(c, fn);
    }
}
static void module_init(Cpu *c, Module *m) {
    LOG("init %s (DT_INIT %llx, %llu init_array entries)\n", m->name, (unsigned long long)m->init,
        (unsigned long long)(m->init_array_sz / 8));
    if (m->init) guest_call(c, m->init);
    else run_array(c, m->init_array, m->init_array_sz);
}
static void rtld_init_sdk(Cpu *c) { module_init(c, &g_modules[1]); }
static void rtld_init_main(Cpu *c) { module_init(c, &g_modules[0]); }

__attribute__((constructor(101))) static void reserve_guest_space(void) {
    if (!plat_reserve_guest(GUEST_LO, GUEST_HI)) {
        // [XCDE] T135: something the loader placed at random (ASLR) sits in the guest range in ~1 of 5 starts. A new
        // process gets a new layout: start this one again (same command line, same console), up to 8 times.
        char n[16] = "0";
#ifdef _WIN32
        GetEnvironmentVariableA("XCDE_RELAUNCH", n, sizeof n);
#else
        if (getenv("XCDE_RELAUNCH")) snprintf(n, sizeof n, "%s", getenv("XCDE_RELAUNCH"));
#endif
        int tries = atoi(n);
        fprintf(stderr, "[boot] guest address range %llx-%llx is not free (try %d)%s\n", (unsigned long long)GUEST_LO,
                (unsigned long long)GUEST_HI, tries + 1, tries < 8 ? ": starting again" : "");
        if (tries < 8) {
            snprintf(n, sizeof n, "%d", tries + 1);
#ifndef _WIN32
            // (Linux: exec replaces this process, so nothing is left running; the new image gets a new layout)
            setenv("XCDE_RELAUNCH", n, 1);
            extern char **environ;
            static char cmd[4096];
            FILE *cf = fopen("/proc/self/cmdline", "rb");
            size_t len = cf ? fread(cmd, 1, sizeof cmd - 1, cf) : 0;
            if (cf) fclose(cf);
            char *av[64]; int ac = 0;
            for (size_t i = 0; i < len && ac < 63; i += strlen(cmd + i) + 1) av[ac++] = cmd + i;
            av[ac] = NULL;
            if (ac) execve("/proc/self/exe", av, environ);
        }
#else
            SetEnvironmentVariableA("XCDE_RELAUNCH", n);
            STARTUPINFOA si = {.cb = sizeof si};
            PROCESS_INFORMATION pi;
            char exe[MAX_PATH];
            GetModuleFileNameA(NULL, exe, sizeof exe);
            // the new process dies with this one (a test's timeout or a closed console must not leave it running)
            HANDLE job = CreateJobObjectA(NULL, NULL);
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION jl = {0};
            jl.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if (job) SetInformationJobObject(job, JobObjectExtendedLimitInformation, &jl, sizeof jl);
            if (CreateProcessA(exe, GetCommandLineA(), NULL, NULL, TRUE, CREATE_SUSPENDED, NULL, NULL, &si, &pi)) {
                if (job) AssignProcessToJobObject(job, pi.hProcess);
                ResumeThread(pi.hThread);
                WaitForSingleObject(pi.hProcess, INFINITE);
                DWORD code = 1;
                GetExitCodeProcess(pi.hProcess, &code);
                ExitProcess(code);
            }
        }
#endif
        fprintf(stderr, "FATAL: guest address range %llx-%llx is not free\n", (unsigned long long)GUEST_LO,
                (unsigned long long)GUEST_HI);
        exit(1);
    }
}

// [XCDE] Watchdog (T027, first version): every few seconds the state of every guest thread; XCDE_WATCHDOG=<seconds>.
static void *watchdog_main(void *p) {
    unsigned secs = (unsigned)(uintptr_t)p;
    for (;;) { plat_sleep_until_ns(plat_now_ns() + secs * 1000000000ull); LOG("[wd] ---- %u s ----"" %c", secs, 10); kernel_dump_threads(stderr); }
}


int main(int argc, char **argv) {
    if (argc >= 6 && !strcmp(argv[1], "--minidump")) return diag_minidump_helper(argc, argv);  // [XCDE] T028: crash helper process
#ifdef XCDE_GFX
    if (argc >= 3 && !strcmp(argv[1], "--precompile")) {  // [XCDE] T187: fill the shader cache from the RomFS, then exit
        extern int vk_precompile_main(const char *root);
        return vk_precompile_main(argv[2]);
    }
#endif
    plat_init();
    setvbuf(stderr, NULL, _IONBF, 0);
    if (argc < 2) { fprintf(stderr, "usage: xcde_boot <dump dir with exefs/ and romfs/> [seconds]\n"); return 2; }
    const char *root = argv[1];
    int seconds = argc > 2 ? atoi(argv[2]) : 20;
    char exefs[1024], romfs[1024], npdm_path[1024];
    snprintf(exefs, sizeof exefs, "%s/exefs", root);
    snprintf(romfs, sizeof romfs, "%s/romfs", root);
    snprintf(npdm_path, sizeof npdm_path, "%s/main.npdm", exefs);
    g_romfs_dir = strdup(romfs);

    diag_watchdog_start();
    if (getenv("XCDE_WATCHDOG")) {
        pthread_t wt;
        pthread_create(&wt, NULL, watchdog_main, (void *)(uintptr_t)atoi(getenv("XCDE_WATCHDOG")));
    }
    loader_load(exefs);
    kernel_init();
    dispatch_init();

    uint32_t stack_size = 0x100000;
    int prio = 44;
    FILE *f = fopen(npdm_path, "rb");
    if (f) {
        uint8_t meta[0x80];
        if (fread(meta, 1, sizeof meta, f) == sizeof meta && !memcmp(meta, "META", 4)) {
            prio = meta[0xE];
            stack_size = *(uint32_t *)(meta + 0x1C);
        }
        fclose(f);
    }
    LOG("main thread: prio %d, stack 0x%x\n", prio, stack_size);

    uint64_t start = loader_find_symbol("_ZN2nn4init5StartEmmPFvvES2_");
    if (!start) FATAL("nn::init::Start not found\n");
    uint64_t init_fn = hle_register("<rtld_init_sdk>", rtld_init_sdk);
    uint64_t fini_fn = hle_register("<rtld_init_main>", rtld_init_main);
    KThread *t = kernel_create_main_thread(start, stack_size, prio);
    kernel_run_main(t, t->handle, 0, init_fn, fini_fn);
#ifdef XCDE_GFX
    extern int ui_run(void);  // window + Vulkan presenter on the main thread (runtime/nvn/vulkan/device.c)
    LOG("[boot] running with the window\n");
    (void)seconds;
    return ui_run();
#else
    LOG("[boot] running; stopping after %d s\n", seconds);
    plat_sleep_until_ns(plat_now_ns() + (uint64_t)seconds * 1000000000ull);
    LOG("[boot] timeout reached, exiting\n");
    return 0;
#endif
}
