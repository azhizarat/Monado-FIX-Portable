// Stall watchdog (T027, FR-061): when no frame has been presented for XCDE_STALL seconds (default 5; 0 disables) it writes a
// complete report -- every guest thread (state, last SVC and the guest function that made it, mutex owners, deadlock notes)
// -- to stderr and to crash/stall_<n>.txt, and repeats every <s> seconds while the stall lasts. It takes no locks it could
// block on, so it also works when the hang is inside the kernel lock.
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/stat.h>
#include <time.h>
#ifdef _WIN32
#include <direct.h>
#endif

#include "../kernel.h"
#include "../rt.h"
#include "diag.h"

static void *stall_main(void *p) {
    unsigned secs = (unsigned)(uintptr_t)p, still = 0, reports = 0;
    uint64_t last = 0;
    for (;;) {
        plat_sleep_until_ns(plat_now_ns() + 1000000000ull);
        uint64_t n = diag_frame();
        if (n != last || n == 0) { last = n; still = 0; continue; }  // (loading before the first frame is not a stall)
        if (++still < secs || (still - secs) % secs) continue;
        LOG("[stall] no frame for %u s (last frame %llu): report %u\n", still, (unsigned long long)n, ++reports);
        kernel_dump_threads(stderr);
        extern void backend_stall_report(FILE *f);  // [XCDE] T240: the render worker too
        backend_stall_report(stderr);
#ifdef _WIN32
        _mkdir("crash");
#else
        mkdir("crash", 0755);
#endif
        char path[64];
        snprintf(path, sizeof path, "crash/stall_%u.txt", reports);
        FILE *f = fopen(path, "w");
        if (f) {
            fprintf(f, "no frame for %u s (last frame %llu)\n", still, (unsigned long long)n);
            kernel_dump_threads(f);
            backend_stall_report(f);
            fclose(f);
        }
    }
    return 0;
}

void diag_watchdog_start(void) {
    const char *e = getenv("XCDE_STALL");
    unsigned secs = e ? (unsigned)atoi(e) : 5;
    pthread_t t;
    if (secs) pthread_create(&t, NULL, stall_main, (void *)(uintptr_t)secs);
}
