// Linux host layer (runtime/platform.h). x86-64 and aarch64.
#define _GNU_SOURCE
#include <SDL3/SDL.h>
#include <dlfcn.h>
#include <errno.h>
#include <execinfo.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/ucontext.h>
#include <time.h>
#include <unistd.h>

#include "../platform.h"

#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif

// ---- guest address space
// Everything readable/writable from the start, like macOS: the guest maps at 4K granularity, and untouched pages
// cost nothing (MAP_NORESERVE: the 12 GB window doesn't count against overcommit).
bool plat_reserve_guest(uint64_t lo, uint64_t hi) {
    void *p = mmap((void *)(uintptr_t)lo, hi - lo, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
    if (p == MAP_FAILED) return false;
    if (p != (void *)(uintptr_t)lo) { munmap(p, hi - lo); return false; }  // (kernels before 4.17: just a hint)
    return true;
}

// Fresh zero pages over the whole window (the old ones are released at once).
bool plat_reset_guest(uint64_t lo, uint64_t hi) {
    return mmap((void *)(uintptr_t)lo, hi - lo, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED,
                -1, 0) == (void *)(uintptr_t)lo;
}

void plat_commit(uint64_t addr, uint64_t size) { (void)addr; (void)size; }

int plat_written_pages(uint64_t addr, uint64_t size, uint64_t *pages, int max, uint64_t *page) {  // (no write tracking)
    (void)addr; (void)size; (void)pages; (void)max; (void)page;
    return -1;
}

int plat_protect(uint64_t addr, uint64_t size, int prot) {
    int p = prot == PLAT_PROT_NONE ? PROT_NONE : prot == PLAT_PROT_READ ? PROT_READ : PROT_READ | PROT_WRITE;
    return mprotect((void *)(uintptr_t)addr, size, p);
}

// ---- time
uint64_t plat_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
}

void plat_sleep_until_ns(uint64_t t) {
    struct timespec ts = {.tv_sec = (time_t)(t / 1000000000u), .tv_nsec = (long)(t % 1000000000u)};
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL) == EINTR) {}
}

// Real-time scheduling where the user may have it (rtkit / limits.conf), else just a higher priority: keeps the
// vsync thread on time while the guest threads load every core.
void plat_thread_realtime(uint64_t period_ns) {
    (void)period_ns;
    struct sched_param sp = {.sched_priority = 1};
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) == 0) return;
    setpriority(PRIO_PROCESS, (id_t)gettid(), -10);  // (per thread on Linux; fails quietly without CAP_SYS_NICE)
}

// ---- process
// Started from a desktop (the AppImage) there is no terminal: stdout / stderr go to lasrecomp.log in the data folder
// (runtime/paths.h; rt_paths_init runs later, so the same rule is applied here). Processes started by this one (the
// start screen re-executing itself for the game) append to the same log.
static char g_log_path[1024];

static void log_to_file_without_terminal(void) {
    if (isatty(STDERR_FILENO)) return;
    struct stat se;  // [XCDE] already sent somewhere (play.sh: build/last_game.log, Steam: its pipe): keep it there
    if (fstat(STDERR_FILENO, &se) == 0 && (S_ISREG(se.st_mode) || S_ISFIFO(se.st_mode) || S_ISSOCK(se.st_mode))) return;
    char dir[900];
    const char *env = getenv("LARECOMP_DATA_ROOT"), *xdg = getenv("XDG_DATA_HOME"), *home = getenv("HOME");
    struct stat st;
    if (env && *env) snprintf(dir, sizeof dir, "%s", env);
    else if (stat("extracted", &st) == 0 && S_ISDIR(st.st_mode)) return;  // dev mode: wherever stderr goes already
    else if (xdg && *xdg == '/') snprintf(dir, sizeof dir, "%s/lasrecomp", xdg);
    else snprintf(dir, sizeof dir, "%s/.local/share/lasrecomp", home && *home ? home : "/tmp");
    for (char *c = dir + 1; *c; c++)
        if (*c == '/') { *c = 0; mkdir(dir, 0755); *c = '/'; }
    mkdir(dir, 0755);
    snprintf(g_log_path, sizeof g_log_path, "%s/lasrecomp.log", dir);
    bool append = getenv("LASRECOMP_LOG_APPEND") != NULL;
    if (!freopen(g_log_path, append ? "a" : "w", stderr)) { g_log_path[0] = 0; return; }
    dup2(STDERR_FILENO, STDOUT_FILENO);
    setenv("LASRECOMP_LOG_APPEND", "1", 1);
}

#if defined(__x86_64__)
#include <cpuid.h>
// [XCDE] The CPU level this binary was built for (scripts/plat.sh CPU: Linux x86-64-v3 = AVX2 / FMA / BMI2, else
// x86-64-v2 = SSE4.2 / POPCNT): say so instead of crashing with an illegal instruction.
// (cpuid directly: zig's compiler-rt answers __builtin_cpu_supports with zeros, which stopped a Zen 2 Steam Deck.)
static void require_cpu(void) {
    unsigned a, b, c, d, b7 = 0, c7 = 0, d7 = 0;
    bool v2 = __get_cpuid(1, &a, &b, &c, &d) && (c & bit_SSE4_2) && (c & bit_POPCNT);
#ifdef __AVX2__
    bool fma = c & bit_FMA;
    __get_cpuid_count(7, 0, &a, &b7, &c7, &d7);
    if (v2 && fma && (b7 & bit_AVX2) && (b7 & bit_BMI2)) return;
    fprintf(stderr, "FATAL: this Monado build needs a CPU with AVX2 (Intel Haswell / 2013 or newer, AMD Zen or newer).\n");
#else
    if (v2) return;
    fprintf(stderr, "FATAL: Monado needs a CPU with SSE4.2 (Intel Nehalem / 2008 or newer, AMD Bulldozer / 2011 or newer).\n");
#endif
    exit(1);
}
#else
static void require_cpu(void) {}
#endif

void plat_init(void) {
    log_to_file_without_terminal();
    require_cpu();
}

// (inside an AppImage: the binary in its mount, which stays mounted while this process lives - re-exec included)
bool plat_exe_path(char *buf, size_t size) {
    ssize_t n = readlink("/proc/self/exe", buf, size - 1);
    if (n <= 0) return false;
    buf[n] = 0;
    return true;
}

// Our own command line, from /proc/self/cmdline (NUL-separated).
static char **self_argv(void) {
    static char buf[32768];
    static char *argv[256];
    int fd = open("/proc/self/cmdline", O_RDONLY);
    if (fd < 0) return NULL;
    ssize_t n = read(fd, buf, sizeof buf - 1);
    close(fd);
    if (n <= 0) return NULL;
    buf[n] = 0;
    int k = 0;
    for (char *p = buf; p < buf + n && k < 255; p += strlen(p) + 1) argv[k++] = p;
    argv[k] = NULL;
    return argv;
}

void plat_exec(char *const argv[]) {
    if (argv) { execv(argv[0], argv); return; }
    char path[4096];
    char **av = self_argv();
    if (av && plat_exe_path(path, sizeof path)) execv(path, av);
}

// ---- diagnostics
// Exported symbols only (the Makefile links with -rdynamic); frames in the generated code are left to the caller,
// which maps them to guest functions (runtime/dispatch.c). Static runtime functions come out as the nearest guest
// function, so backtrace lines also give the offset in the executable (addr2line -fe lasrecomp <offset>).
const char *plat_symbol(uint64_t pc, uint64_t *offset) {
    Dl_info di = {0};
    if (!dladdr((void *)(uintptr_t)pc, &di) || !di.dli_sname) return NULL;
    if (offset) *offset = pc - (uint64_t)(uintptr_t)di.dli_saddr;
    return di.dli_sname;
}

// Fatal error without a terminal: say so in a message box rather than vanishing (the details are in the log).
void plat_fatal_notice(void) {
    if (!g_log_path[0]) return;
    fflush(stderr);
    char text[1200];
    snprintf(text, sizeof text, "LAS Recomp stopped because of an error.\n\nDetails: %s", g_log_path);
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "LAS Recomp", text, NULL);
}

const char *rt_host_fn_name(uint64_t pc, uint64_t *offset);  // runtime/dispatch.c

void plat_print_backtrace(void) {
    void *bt[64];
    int n = backtrace(bt, 64);
    Dl_info self = {0};
    dladdr((void *)plat_print_backtrace, &self);
    for (int i = 0; i < n; i++) {
        uint64_t pc = (uint64_t)(uintptr_t)bt[i], off = 0;
        Dl_info di = {0};
        if (!dladdr(bt[i], &di)) di.dli_fname = NULL;
        bool ours = di.dli_fbase && di.dli_fbase == self.dli_fbase;
        const char *s = plat_symbol(pc, &off);
        if (!s && ours) s = rt_host_fn_name(pc, &off);
        const char *mod = di.dli_fname ? (strrchr(di.dli_fname, '/') ? strrchr(di.dli_fname, '/') + 1 : di.dli_fname) : "?";
        uint64_t moff = pc - (uint64_t)(uintptr_t)di.dli_fbase;
        if (s && ours) fprintf(stderr, "  #%-2d %016llx %s+%llx (lasrecomp+%llx)\n", i, (unsigned long long)pc, s,
                               (unsigned long long)off, (unsigned long long)moff);
        else if (s) fprintf(stderr, "  #%-2d %016llx %s+%llx (%s)\n", i, (unsigned long long)pc, s, (unsigned long long)off, mod);
        else fprintf(stderr, "  #%-2d %016llx %s+%llx\n", i, (unsigned long long)pc, mod, (unsigned long long)moff);
    }
}

// ---- [XCDE] CPU/GPU overlap (exec.c XCDE_OVERLAP_STATS / XCDE_OVERLAP): the pages the GPU of a recorded frame reads
// (PROT_READ) or writes (PROT_NONE) are protected; a CPU access faults. Stats mode: counted (who, read / write, ms into
// the CPU's next frame) and let through. Wait mode (one frame in flight): the access first waits until the GPU has
// finished that frame (its submission sequence on the timeline, through the callback exec.c registers: a plain counter
// query, no lock), so the CPU never changes what the GPU still reads nor reads what it hasn't written yet.
// Two sets are live: the previous frame's (released once the game has waited for it) and the current one. Sets are
// sorted runs (the handler bisects) in a ring of three buffers, so a handler never reads one being rewritten.
#define OV_MAX 65536
typedef struct { uint64_t lo[OV_MAX], hi[OV_MAX]; uint8_t w[OV_MAX]; int n; uint64_t seq; } OvSet;
static OvSet ov_sets[3];
static OvSet *volatile ov_cur, *volatile ov_prev;
static uint64_t (*volatile ov_gpu_done)(void);
static void (*volatile ov_gpu_wait)(uint64_t);
static volatile uint64_t ov_t0;  // CLOCK_MONOTONIC ns of the CPU's frame start
static volatile pid_t ov_worker;
// counters: [gpu-writes page? 1:0][cpu write? 1:0][worker? 1:0][time bucket: <4, 4-10, 10-17, >=17 ms]
static volatile uint32_t ov_cnt[2][2][2][4], ov_sys, ov_frames, ov_pages_sum, ov_runs_sum, ov_waits;
static volatile uint64_t ov_prot_ns, ov_wait_ns;
static uint64_t ov_now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec; }
static int ov_find(const OvSet *o, uint64_t a) {
    int lo = 0, hi = o->n - 1;
    while (lo <= hi) {
        int m = (lo + hi) >> 1;
        if (a < o->lo[m]) hi = m - 1;
        else if (a >= o->hi[m]) lo = m + 1;
        else return m;
    }
    return -1;
}
static __thread const ucontext_t *ov_fault_uc;  // (the signal context of the fault being handled: XCDE_OVERLAP_WHO)
static bool ov_fault(uint64_t fa, bool write) {
    const OvSet *c = __atomic_load_n(&ov_cur, __ATOMIC_ACQUIRE), *p = __atomic_load_n(&ov_prev, __ATOMIC_ACQUIRE), *o = NULL;
    int r = -1;
    if (c && (r = ov_find(c, fa)) >= 0) o = c;  // (the current frame first: its sequence is the later one)
    else if (p && (r = ov_find(p, fa)) >= 0) o = p;
    if (!o) return false;
    uint64_t t = ov_now(), ms = (t - ov_t0) / 1000000ull;
    int bk = ms < 4 ? 0 : ms < 10 ? 1 : ms < 17 ? 2 : 3;
    __atomic_add_fetch(&ov_cnt[o->w[r]][write][gettid() == ov_worker][bk], 1, __ATOMIC_RELAXED);
    uint64_t (*done)(void) = ov_gpu_done;
    if (o->seq && done && done() < o->seq) {  // wait mode: until the GPU has finished that frame
        // [XCDE] perf: a blocking wait on the timeline. The yield loop kept a core busy for the whole wait - power the
        // Deck's shared 15 W budget then took from the GPU clock.
        void (*wait)(uint64_t) = ov_gpu_wait;
        if (wait) wait(o->seq);
        while (done() < o->seq) sched_yield();  // (no wait function, or it gave up)
        __atomic_add_fetch(&ov_waits, 1, __ATOMIC_RELAXED);
        __atomic_add_fetch(&ov_wait_ns, ov_now() - t, __ATOMIC_RELAXED);
        static int nlog = -1;  // XCDE_OVERLAP_WHO=<n>: the first n waits - page, who, and the faulting code (RIP + stack words)
        if (nlog < 0) nlog = getenv("XCDE_OVERLAP_WHO") ? atoi(getenv("XCDE_OVERLAP_WHO")) : 0;
        if (nlog > 0 && ov_fault_uc) {
            nlog--;
            extern char __executable_start[];
            const ucontext_t *uc = ov_fault_uc;
            const uint64_t *sp = (const uint64_t *)(uintptr_t)uc->uc_mcontext.gregs[REG_RSP];
            fprintf(stderr, "[overlap-who] %s %s page %llx (%s frame set, gpu-%s) %.2f ms in, waited %.2f ms; base %p rip %llx stack",
                    gettid() == ov_worker ? "worker" : "game", write ? "write" : "read", (unsigned long long)fa, o == c ? "current" : "previous",
                    o->w[r] ? "written" : "read", (t - ov_t0) / 1e6, (ov_now() - t) / 1e6, (void *)__executable_start,
                    (unsigned long long)uc->uc_mcontext.gregs[REG_RIP]);
            for (int i = 0; i < 24; i++) fprintf(stderr, " %llx", (unsigned long long)sp[i]);
            fputc(10, stderr);
        }
    }
    mprotect((void *)(uintptr_t)(fa & ~0xFFFull), 4096, PROT_READ | PROT_WRITE);
    return true;
}
// open o's pages except those keep (sorted runs) still covers
static void ov_open_except(const OvSet *o, const OvSet *keep) {
    int k = 0;
    for (int i = 0; i < o->n; i++) {
        uint64_t a = o->lo[i], e = o->hi[i];
        while (keep && k < keep->n && keep->hi[k] <= a) k++;
        for (int j = k; keep && j < keep->n && keep->lo[j] < e && a < e; j++) {
            if (keep->lo[j] > a) mprotect((void *)(uintptr_t)a, keep->lo[j] - a, PROT_READ | PROT_WRITE);
            if (keep->hi[j] > a) a = keep->hi[j];
        }
        if (a < e) mprotect((void *)(uintptr_t)a, e - a, PROT_READ | PROT_WRITE);
    }
}
void plat_ov_set_gpu_done(uint64_t (*fn)(void)) { ov_gpu_done = fn; }
void plat_ov_set_gpu_wait(void (*fn)(uint64_t seq)) { ov_gpu_wait = fn; }
// the previous frame's GPU work is done: its pages open, except those the current frame protects
void plat_ov_release_prev(void) {
    OvSet *p = ov_prev;
    if (!p) return;
    ov_open_except(p, ov_cur);
    __atomic_store_n(&ov_prev, NULL, __ATOMIC_RELEASE);
}
// seq 0: stats (the previous set is opened right away); else this frame's GPU work ends at seq and the previous set
// stays until plat_ov_release_prev
void plat_ov_protect(const uint64_t *lo, const uint64_t *hi, const uint8_t *w, int n, uint64_t seq) {
    uint64_t t0 = ov_now();
    OvSet *c = ov_cur, *p = ov_prev, *nw = NULL;
    for (int i = 0; i < 3 && !nw; i++) if (&ov_sets[i] != c && &ov_sets[i] != p) nw = &ov_sets[i];
    if (p) {  // (wait mode, a release missed) the oldest set goes now, keeping what the newer one covers
        ov_open_except(p, c);
        __atomic_store_n(&ov_prev, NULL, __ATOMIC_RELEASE);
    }
    if (n > OV_MAX) n = OV_MAX;
    uint64_t pages = 0;
    for (int i = 0; i < n; i++) { nw->lo[i] = lo[i]; nw->hi[i] = hi[i]; nw->w[i] = w[i]; pages += (hi[i] - lo[i]) >> 12; }
    nw->n = n;
    nw->seq = seq;
    __atomic_store_n(&ov_prev, c, __ATOMIC_RELEASE);
    __atomic_store_n(&ov_cur, nw, __ATOMIC_RELEASE);  // (published before its pages are protected: a fault finds its run)
    for (int i = 0; i < n; i++) mprotect((void *)(uintptr_t)lo[i], hi[i] - lo[i], w[i] ? PROT_NONE : PROT_READ);
    if (!seq) plat_ov_release_prev();
    ov_prot_ns += ov_now() - t0;
    ov_frames++; ov_pages_sum += (uint32_t)pages; ov_runs_sum += (uint32_t)n;
}
void plat_ov_frame_start(void) { ov_t0 = ov_now(); }
void plat_ov_guest_write(uint64_t addr, uint64_t size) {
    if (!size || (!ov_cur && !ov_prev)) return;
    for (uint64_t p = addr & ~0xFFFull; p < addr + size; p += 4096)
        if (ov_fault(p, true)) __atomic_add_fetch(&ov_sys, 1, __ATOMIC_RELAXED);  // (waits as an access would)
}
void plat_ov_set_worker(void) { ov_worker = gettid(); }
void plat_ov_report(void) {
    if (!ov_frames) return;
    fprintf(stderr, "[overlap] %u frames: %.0f pages in %.0f runs a frame, protect %.2f ms a frame; syscall opens %u; waits %u (%.2f ms a frame)\n",
            ov_frames, (double)ov_pages_sum / ov_frames, (double)ov_runs_sum / ov_frames, ov_prot_ns / 1e6 / ov_frames, ov_sys,
            ov_waits, ov_wait_ns / 1e6 / ov_frames);
    static const char *const kind[2] = {"GPU-read page", "GPU-written page"}, *acc[2] = {"read", "write"}, *who[2] = {"game", "worker"};
    for (int k = 0; k < 2; k++)
        for (int a = 0; a < 2; a++)
            for (int t = 0; t < 2; t++) {
                volatile uint32_t *c = ov_cnt[k][a][t];
                if (c[0] | c[1] | c[2] | c[3])
                    fprintf(stderr, "[overlap]   %-16s %-5s by %-6s: <4ms %u, 4-10 %u, 10-17 %u, >=17 %u\n", kind[k], acc[a], who[t], c[0], c[1], c[2], c[3]);
            }
    memset((void *)ov_cnt, 0, sizeof ov_cnt);
    ov_sys = ov_frames = ov_pages_sum = ov_runs_sum = ov_waits = 0;
    ov_prot_ns = ov_wait_ns = 0;
}

static void crash_handler(int sig, siginfo_t *si, void *ctx) {
    ucontext_t *uc = ctx;
#if defined(__x86_64__)
    ov_fault_uc = uc;
    bool ov_hit = sig == SIGSEGV && ov_fault((uint64_t)(uintptr_t)si->si_addr, (uc->uc_mcontext.gregs[REG_ERR] & 2) != 0);  // [XCDE]
    ov_fault_uc = NULL;
    if (ov_hit) return;
#endif
    static volatile int depth;
    if (sig == SIGABRT && depth++ > 0) { signal(SIGABRT, SIG_DFL); abort(); }  // (nested: just die)
    uint64_t fa = (uint64_t)(uintptr_t)si->si_addr;
#if defined(__aarch64__)
    uint64_t pc = uc->uc_mcontext.pc, lr = uc->uc_mcontext.regs[30];
#else
    // x86-64 has no link register: report the return address on top of the stack (valid at function entry)
    uint64_t pc = (uint64_t)uc->uc_mcontext.gregs[REG_RIP], sp = (uint64_t)uc->uc_mcontext.gregs[REG_RSP], lr = 0;
    if (sig != SIGSEGV || (fa & ~0xFFFull) != (sp & ~0xFFFull)) lr = *(uint64_t *)(uintptr_t)sp;
#endif
    if (rt_fault_watch(fa, pc)) return;
    char what[32];
    snprintf(what, sizeof what, "signal %d", sig);
    rt_fault_report(what, fa, pc, lr);
}

void plat_crash_handler_install(void) {
    stack_t ss = {.ss_sp = malloc(1 << 20), .ss_size = 1 << 20};
    sigaltstack(&ss, NULL);
    struct sigaction sa = {0};
    sa.sa_sigaction = crash_handler;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGILL, &sa, NULL);
    sigaction(SIGFPE, &sa, NULL);
    sigaction(SIGABRT, &sa, NULL);
}

// ---- child processes
#include <ftw.h>
#include <spawn.h>
#include <sys/wait.h>
extern char **environ;

PlatProc plat_spawn(const char *const argv[], const char *log_path, bool append_log) {
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 1, log_path ? log_path : "/dev/null",
                                     O_WRONLY | O_CREAT | (append_log ? O_APPEND : O_TRUNC), 0644);
    posix_spawn_file_actions_adddup2(&fa, 1, 2);
    posix_spawnattr_t at;  // (its own process group: plat_proc_kill ends what it starts, e.g. a compiler's own child)
    posix_spawnattr_init(&at);
    posix_spawnattr_setflags(&at, POSIX_SPAWN_SETPGROUP);
    posix_spawnattr_setpgroup(&at, 0);
    pid_t pid;
    int rc = posix_spawn(&pid, argv[0], &fa, &at, (char *const *)argv, environ);
    posix_spawnattr_destroy(&at);
    posix_spawn_file_actions_destroy(&fa);
    return rc ? 0 : (PlatProc)pid;
}

void plat_proc_kill(PlatProc p) {
    kill(-(pid_t)p, SIGKILL);
    waitpid((pid_t)p, NULL, 0);
}

int plat_proc_wait(PlatProc p, bool block) {
    int st = 0;
    pid_t r = waitpid((pid_t)p, &st, block ? 0 : WNOHANG);
    if (r == 0) return -1;
    if (r < 0) return 255;
    return WIFEXITED(st) ? WEXITSTATUS(st) : 255;
}

static int rm_entry(const char *path, const struct stat *sb, int flag, struct FTW *ftw) {
    (void)sb, (void)flag, (void)ftw;
    return remove(path);
}
bool plat_remove_tree(const char *dir) { return nftw(dir, rm_entry, 16, FTW_DEPTH | FTW_PHYS) == 0; }

// End the process at once, skipping the library teardown: as on Windows, the GPU driver's threads crashed in exit()
// while the game's threads still submitted work.
void plat_terminate(void) { _exit(0); }

// [XCDE] T122: not sampled on this platform
uint64_t plat_thread_pc(void *pthread_handle) { (void)pthread_handle; return 0; }
void plat_thread_name(const char *name) { (void)name; }
void plat_thread_background(void) {}
void plat_thread_audio(void) {}
int plat_cpu_count(void) { long n = sysconf(_SC_NPROCESSORS_ONLN); return n > 0 ? (int)n : 4; }

// [XCDE] T122: not available on this platform
int plat_thread_backtrace(void *pthread_handle, uint64_t *pcs, int max) { (void)pthread_handle; (void)pcs; (void)max; return 0; }
int plat_guest_find(const unsigned char *pat, unsigned long n, unsigned long long *out, int max) { (void)pat; (void)n; (void)out; (void)max; return 0; }  // [XCDE] (not wired)
unsigned long long plat_guest_dump(const char *path) { (void)path; return 0; }  // [XCDE] (not wired)

// [VEGA-FIX] no crash guard on this platform: just run it
bool plat_try_call(void (*fn)(void *), void *arg) { fn(arg); return true; }

// ---- memory report (exec.c, XCDE_MEM_MAP): private = RssAnon, working set = VmRSS from /proc/self/status.
// ponytail: guest_commit is not measured (0); walk /proc/self/smaps over the guest window if it is ever needed here.
void plat_mem_info(uint64_t *priv, uint64_t *ws, uint64_t *guest_commit) {
    *priv = *ws = *guest_commit = 0;
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return;
    char line[256];
    unsigned long long kb;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "VmRSS: %llu kB", &kb) == 1) *ws = kb << 10;
        else if (sscanf(line, "RssAnon: %llu kB", &kb) == 1) *priv = kb << 10;
    }
    fclose(f);
}

void plat_mem_map(void) { fprintf(stderr, "[memmap] not implemented on Linux (see /proc/<pid>/smaps)\n"); }
