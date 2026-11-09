// Windows host layer (runtime/platform.h).
#include <windows.h>
#include <dbghelp.h>
#include <stdbool.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <shellapi.h>
#include <timeapi.h>
#include <io.h>
#include <signal.h>

#include "../platform.h"

// ---- guest address space
// The window is reserved up front and committed per region (plat_commit from the loader / kernel). Anything
// the guest touches outside a committed region is committed on its first access by the exception handler
// below, so the window behaves like macOS's all-RW reservation. The OS itself (ReadFile into a guest buffer)
// does not fault: such buffers must be inside committed regions.
static uint64_t g_lo, g_hi;
#define COMMIT_CHUNK 0x10000ull

static bool lazy_commit(uint64_t a) {
    if (a < g_lo || a >= g_hi) return false;
    MEMORY_BASIC_INFORMATION mi;
    if (!VirtualQuery((void *)(uintptr_t)a, &mi, sizeof mi)) return false;
    // Two threads touching the same uncommitted chunk: the second faulted before the first committed it and now finds
    // it committed - retry the access. (A page protected on purpose, plat_protect / watches, isn't plain read-write
    // and still goes on to the fault watch.) Fix from upstream LASRecomp 2332e74 (itself from letsgo-recomp).
    if (mi.State == MEM_COMMIT) return (mi.Protect & 0xFF) == PAGE_READWRITE && !(mi.Protect & PAGE_GUARD);
    if (mi.State != MEM_RESERVE) return false;
    uint64_t base = a & ~(COMMIT_CHUNK - 1);
    return VirtualAlloc((void *)(uintptr_t)base, COMMIT_CHUNK, MEM_COMMIT, PAGE_READWRITE) != NULL;
}

static bool g_report_crashes;

// [VEGA-FIX] plat_try_call: the context captured before the call; the handler below restores it on a divide by zero.
static __thread CONTEXT *tl_guard_ctx;
static __thread volatile int *tl_guard_hit;

bool plat_try_call(void (*fn)(void *), void *arg) {
    CONTEXT ctx;
    volatile int hit = 0;
    RtlCaptureContext(&ctx);
    if (hit) { tl_guard_ctx = NULL; return false; }  // (back here from the handler)
    tl_guard_hit = &hit;
    tl_guard_ctx = &ctx;
    fn(arg);
    tl_guard_ctx = NULL;
    return true;
}

static LONG CALLBACK vectored_handler(EXCEPTION_POINTERS *ep) {
    EXCEPTION_RECORD *er = ep->ExceptionRecord;
    if (tl_guard_ctx && er->ExceptionCode == EXCEPTION_INT_DIVIDE_BY_ZERO) {
        CONTEXT *c = tl_guard_ctx;
        *tl_guard_hit = 1;
        RtlRestoreContext(c, NULL);  // (does not return)
    }
    if (er->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || er->NumberParameters < 2) return EXCEPTION_CONTINUE_SEARCH;
    uint64_t fa = er->ExceptionInformation[1];
    // (not for an execute fault: on a read-write page, retrying it would fault again forever)
    if (er->ExceptionInformation[0] != EXCEPTION_EXECUTE_FAULT && lazy_commit(fa)) return EXCEPTION_CONTINUE_EXECUTION;
    if (g_report_crashes && rt_fault_watch(fa, ep->ContextRecord->Rip)) return EXCEPTION_CONTINUE_EXECUTION;
    return EXCEPTION_CONTINUE_SEARCH;
}

// The window keeps write tracking (MEM_WRITE_WATCH: the Vulkan renderer's vertex / index cache re-copies only pages the
// CPU wrote, runtime/nvn/vulkan/exec.c). Without it (refused): no tracking, the renderer reads guest memory in place.
static bool g_write_watch;

bool plat_reserve_guest(uint64_t lo, uint64_t hi) {
    void *p = VirtualAlloc((void *)(uintptr_t)lo, hi - lo, MEM_RESERVE | MEM_WRITE_WATCH, PAGE_READWRITE);
    if (p == (void *)(uintptr_t)lo) g_write_watch = true;
    else p = VirtualAlloc((void *)(uintptr_t)lo, hi - lo, MEM_RESERVE, PAGE_READWRITE);
    if (p != (void *)(uintptr_t)lo) return false;
    g_lo = lo;
    g_hi = hi;
    AddVectoredExceptionHandler(1, vectored_handler);
    return true;
}

int plat_written_pages(uint64_t addr, uint64_t size, uint64_t *pages, int max, uint64_t *page) {
    if (!g_write_watch || addr < g_lo || addr + size > g_hi || max <= 0) return -1;
    ULONG_PTR n = (ULONG_PTR)max;
    DWORD gran = 4096;
    if (GetWriteWatch(WRITE_WATCH_FLAG_RESET, (void *)(uintptr_t)addr, (SIZE_T)size, (PVOID *)pages, &n, &gran) != 0) return -1;
    *page = gran;
    if (n >= (ULONG_PTR)max) ResetWriteWatch((void *)(uintptr_t)addr, (SIZE_T)size);  // (more than fit: caller takes all)
    return (int)n;
}

// Decommit everything: the range stays reserved, and pages come back zeroed when committed again (lazily).
bool plat_reset_guest(uint64_t lo, uint64_t hi) {
    return VirtualFree((void *)(uintptr_t)lo, hi - lo, MEM_DECOMMIT) != 0;
}

void plat_commit(uint64_t addr, uint64_t size) {
    if (!size) return;
    uint64_t a = addr & ~0xFFFull, e = (addr + size + 0xFFF) & ~0xFFFull;
    if (!VirtualAlloc((void *)(uintptr_t)a, e - a, MEM_COMMIT, PAGE_READWRITE))
        fprintf(stderr, "[plat] commit %llx+%llx failed (%lu)\n", (unsigned long long)addr,
                (unsigned long long)size, GetLastError());
}

void plat_ov_protect(const uint64_t *lo, const uint64_t *hi, const uint8_t *w, int n, uint64_t seq) { (void)lo, (void)hi, (void)w, (void)n, (void)seq; }
void plat_ov_release_prev(void) {}
void plat_ov_set_gpu_done(uint64_t (*fn)(void)) { (void)fn; }
void plat_ov_set_gpu_wait(void (*fn)(uint64_t seq)) { (void)fn; }
void plat_ov_set_heap(uint64_t base, uint64_t size) { (void)base, (void)size; }
bool plat_ov_hot(uint64_t addr, uint64_t size) { (void)addr, (void)size; return false; }
void plat_ov_frame_start(void) {}
void plat_ov_guest_write(uint64_t addr, uint64_t size) { (void)addr, (void)size; }
void plat_ov_set_worker(void) {}
void plat_ov_report(void) {}

int plat_protect(uint64_t addr, uint64_t size, int prot) {
    DWORD p = prot == PLAT_PROT_NONE ? PAGE_NOACCESS : prot == PLAT_PROT_READ ? PAGE_READONLY : PAGE_READWRITE, old;
    if (addr >= g_lo && addr < g_hi) plat_commit(addr, size);
    return VirtualProtect((void *)(uintptr_t)addr, size, p, &old) ? 0 : -1;
}

// ---- time
static LARGE_INTEGER qpf;

static uint64_t qpc_ns(void);

// [XCDE] T190: on PCs where Windows times with the HPET (a common "tweak") a QueryPerformanceCounter read costs ~1 us
// instead of 10-20 ns, and the game reads the system tick constantly (its frame limiter, timers): measured by another
// Switch recomp (WindowSmash) as 40 % of a main thread. When a read is slow and the CPU's time stamp counter is
// invariant, the clock is the TSC scaled to nanoseconds: calibrated against QPC over 20 ms at start, then again at
// about 2 s, 30 s and 10 min, each time continuing from the clock's current value (monotonic, no jump).
// XCDE_CLOCK=qpc|tsc overrides the choice.
static int g_clock = -1;  // 0 QPC, 1 TSC
static struct { uint64_t tsc0, ns0, mul; } g_tc;  // ns = ns0 + (tsc - tsc0) * mul >> 32
static volatile LONG g_tc_lock, g_tc_seq, g_pick_once;  // (g_tc_seq: odd while g_tc is being rewritten)
static uint64_t g_tc_cal_tsc, g_tc_cal_qpc, g_tc_next;
static int g_tc_stage;

static uint64_t rdtscp_(void) { unsigned aux; return __builtin_ia32_rdtscp(&aux); }

static bool tsc_invariant(void) {
    unsigned a, b, c, d;
    __asm__ volatile("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(0x80000000u), "c"(0));
    if (a < 0x80000007u) return false;
    __asm__ volatile("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(0x80000007u), "c"(0));
    return (d >> 8) & 1;
}

static void clock_pick(void) {
    const char *e = getenv("XCDE_CLOCK");
    uint64_t t0 = rdtscp_();
    uint64_t q0 = qpc_ns();
    for (int i = 0; i < 199; i++) (void)qpc_ns();
    uint64_t q1 = qpc_ns();
    double per_read = (q1 - q0) / 200.0;
    bool slow = per_read > 250.0, inv = tsc_invariant();
    int c = (e && !strcmp(e, "tsc") && inv) ? 1 : (e && !strcmp(e, "qpc")) ? 0 : (slow && inv) ? 1 : 0;
    if (c) {  // first calibration: 20 ms against QPC
        uint64_t qa = qpc_ns(), ta = rdtscp_();
        Sleep(20);
        uint64_t qb = qpc_ns(), tb = rdtscp_();
        g_tc.mul = (uint64_t)(((unsigned __int128)(qb - qa) << 32) / (tb - ta ? tb - ta : 1));
        g_tc.tsc0 = tb; g_tc.ns0 = qb;
        g_tc_cal_tsc = tb; g_tc_cal_qpc = qb;
        g_tc_next = qb + 2000000000ull;
    }
    (void)t0;
    fprintf(stderr, "[clock] QPC read %.0f ns, invariant TSC %s: using %s\n", per_read, inv ? "yes" : "no", c ? "the TSC" : "QPC");
    g_clock = c;
}

// TSC -> ns; recalibrates now and then (one thread at a time, the others keep the current scale)
static uint64_t tsc_ns(void) {
    uint64_t t, ns;
    for (;;) {
        LONG s0 = g_tc_seq;
        __asm__ volatile("" ::: "memory");  // (x86 keeps loads in order: a compiler barrier is enough)
        t = rdtscp_();
        ns = g_tc.ns0 + (uint64_t)(((unsigned __int128)(t - g_tc.tsc0) * g_tc.mul) >> 32);
        __asm__ volatile("" ::: "memory");  // (x86 keeps loads in order: a compiler barrier is enough)
        if (!(s0 & 1) && s0 == g_tc_seq) break;
    }
    if (ns >= g_tc_next && g_tc_stage < 3 && !InterlockedExchange(&g_tc_lock, 1)) {
        if (ns >= g_tc_next) {
            uint64_t q = qpc_ns(), t2 = rdtscp_();
            uint64_t mul = (uint64_t)(((unsigned __int128)(q - g_tc_cal_qpc) << 32) / (t2 - g_tc_cal_tsc ? t2 - g_tc_cal_tsc : 1));
            uint64_t now = g_tc.ns0 + (uint64_t)(((unsigned __int128)(t2 - g_tc.tsc0) * g_tc.mul) >> 32);
            InterlockedIncrement(&g_tc_seq);
            g_tc.tsc0 = t2; g_tc.ns0 = now; g_tc.mul = mul;  // (continue from the current value with the new rate)
            InterlockedIncrement(&g_tc_seq);
            static const uint64_t next[3] = {30000000000ull, 600000000000ull, ~0ull};
            g_tc_next = next[g_tc_stage] == ~0ull ? ~0ull : g_tc_cal_qpc + next[g_tc_stage];
            g_tc_stage++;
        }
        InterlockedExchange(&g_tc_lock, 0);
    }
    return ns;
}

uint64_t plat_now_ns(void) {
    if (g_clock < 0) {
        if (!InterlockedExchange(&g_pick_once, 1)) clock_pick();
        else while (g_clock < 0) YieldProcessor();  // (another thread is picking)
    }
    if (g_clock == 1) return tsc_ns();
    return qpc_ns();
}

static uint64_t qpc_ns(void) {
    if (!qpf.QuadPart) QueryPerformanceFrequency(&qpf);
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    // [XCDE] perf: same result as t * 1e9 / f in 128 bits, without the software 128-bit division (compiler-rt udivmod
    // was the hottest runtime function: the game polls the clock in its frame limiter). Exact: q*f*1e9/f = q*1e9.
    uint64_t f = (uint64_t)qpf.QuadPart, c = (uint64_t)t.QuadPart;
    if (f > 18000000000ull) return (uint64_t)((unsigned __int128)c * 1000000000u / f);  // (r * 1e9 would overflow)
    return c / f * 1000000000ull + c % f * 1000000000ull / f;
}

#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x2
#endif

void plat_sleep_until_ns(uint64_t t) {
    static __thread HANDLE timer;
    if (!timer) {
        timer = CreateWaitableTimerExW(NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        if (!timer) timer = CreateWaitableTimerExW(NULL, NULL, 0, TIMER_ALL_ACCESS);
    }
    for (;;) {
        uint64_t now = plat_now_ns();
        if (now >= t) return;
        uint64_t left = t - now;
        if (left < 200000) { YieldProcessor(); continue; }  // last 0.2 ms: spin (timer slack)
        LARGE_INTEGER due = {.QuadPart = -(LONGLONG)((left - 100000) / 100)};  // relative, 100 ns units
        if (!timer || !SetWaitableTimer(timer, &due, 0, NULL, NULL, FALSE)) { Sleep(1); continue; }
        WaitForSingleObject(timer, INFINITE);
    }
}

void plat_thread_realtime(uint64_t period_ns) {
    (void)period_ns;
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
}

// ---- process
// Release builds are GUI-subsystem executables (no console window): stdout / stderr go to lasrecomp.log next to the
// exe, and a fatal error shows a message box (plat_fatal_notice) instead of vanishing. Processes started by this one
// (the start screen re-executing itself for the game) append to the same log.
static char g_log_path[MAX_PATH];

static void log_to_file_without_console(void) {
    if (GetStdHandle(STD_ERROR_HANDLE) && GetStdHandle(STD_ERROR_HANDLE) != INVALID_HANDLE_VALUE) return;  // console / pipe
    char exe[MAX_PATH];
    DWORD n = GetModuleFileNameA(NULL, exe, sizeof exe);
    if (!n || n >= sizeof exe) return;
    char *slash = strrchr(exe, '\\');
    if (slash) slash[1] = 0;
    snprintf(g_log_path, sizeof g_log_path, "%slasrecomp.log", exe);
    bool append = getenv("LASRECOMP_LOG_APPEND") != NULL;
    if (!append) {  // [XCDE] the previous run's log is kept as lasrecomp.prev.log (a crash report needs the run that crashed)
        char prev[MAX_PATH];
        snprintf(prev, sizeof prev, "%slasrecomp.prev.log", exe);
        MoveFileExA(g_log_path, prev, MOVEFILE_REPLACE_EXISTING);
    }
    if (!freopen(g_log_path, append ? "ab" : "wb", stderr)) { g_log_path[0] = 0; return; }
    setvbuf(stderr, NULL, _IONBF, 0);
    _dup2(_fileno(stderr), _fileno(stdout));
    _putenv("LASRECOMP_LOG_APPEND=1");
}

// [XCDE] The CPU level this build targets (scripts/plat.sh CPU: x86-64-v3 = AVX2 / FMA / BMI2 by default, x86-64-v2 =
// SSE4.2 / POPCNT with XCDE_CPU): say so instead of crashing with an illegal instruction on an older CPU.
// (Uses CPUID directly: __builtin_cpu_init needs compiler-rt's cpu model.)
#include <cpuid.h>
static void require_cpu(void) {
    unsigned a, b, c, d;
    bool v2 = __get_cpuid(1, &a, &b, &c, &d) && (c & (1u << 20)) /* SSE4.2 */ && (c & (1u << 23)) /* POPCNT */;
#ifdef __AVX2__
    bool fma = c & (1u << 12);
    unsigned b7 = 0, c7, d7;
    __get_cpuid_count(7, 0, &a, &b7, &c7, &d7);
    if (v2 && fma && (b7 & (1u << 5)) /* AVX2 */ && (b7 & (1u << 8)) /* BMI2 */) return;
    MessageBoxA(NULL, "This Monado build needs a CPU with AVX2 (Intel Haswell / 2013 or newer, AMD Zen or newer).\n"
                "Build with XCDE_CPU=\"-O1 -march=x86_64_v2\" for older CPUs.", "Monado", MB_ICONERROR | MB_OK);
#else
    if (v2) return;
    MessageBoxA(NULL, "Monado needs a CPU with SSE4.2 (Intel Nehalem / 2008 or newer, AMD Bulldozer / 2011 or newer).",
                "Monado", MB_ICONERROR | MB_OK);
#endif
    ExitProcess(1);
}

// Fatal error (rt_fatal_trace): the last FATAL / exception line of the log in a message box, when there is no console.
void plat_fatal_notice(void) {
    if (!g_log_path[0]) return;
    fflush(stderr);
    char text[1024] = "LAS Recomp stopped because of an error.";
    FILE *f = fopen(g_log_path, "rb");
    if (f) {
        static char tail[16384];
        fseek(f, 0, SEEK_END);
        long sz = ftell(f), from = sz > (long)sizeof tail - 1 ? sz - (long)sizeof tail + 1 : 0;
        fseek(f, from, SEEK_SET);
        size_t got = fread(tail, 1, sizeof tail - 1, f);
        tail[got] = 0;
        fclose(f);
        char *line = NULL;
        for (char *p = tail; (p = strstr(p, "FATAL: ")) != NULL; p++) line = p;
        if (!line) for (char *p = tail; (p = strstr(p, "*** ")) != NULL; p++) line = p;
        if (line) {
            char *end = strchr(line, '\n');
            if (end) *end = 0;
            snprintf(text, sizeof text, "LAS Recomp stopped because of an error:\n\n%.600s", line);
            if (strstr(line, "Vulkan") || strstr(line, "VK_") || strstr(line, "vk")) {
                size_t l = strlen(text);
                snprintf(text + l, sizeof text - l, "\n\nLAS Recomp needs a graphics driver with Vulkan 1.3: install the "
                         "latest driver for your GPU (NVIDIA, AMD or Intel).");
            }
        }
    }
    size_t l = strlen(text);
    snprintf(text + l, sizeof text - l, "\n\nDetails: %s", g_log_path);
    MessageBoxA(NULL, text, "LAS Recomp", MB_ICONERROR | MB_OK);
}

// [XCDE] Intel hybrid CPUs (P-cores + E-cores, and the low-power E-cores of Core Ultra): the game's threads (game,
// render, audio, guest threads) default to the fastest class of cores; the shader workers may use every core
// (plat_thread_background). One class of cores (AMD, older Intel): nothing changes. XCDE_PCORES=0: off. (Idea:
// LASRecomp's P-core placement.)
static ULONG *g_all_sets, g_nall_sets;
static void pcores_init(void) {
    const char *e = getenv("XCDE_PCORES");
    if ((e && atoi(e) == 0) || getenv("XCDE_LOWSPEC_CPUS")) return;
    ULONG len = 0;
    GetSystemCpuSetInformation(NULL, 0, &len, GetCurrentProcess(), 0);
    if (!len) return;
    uint8_t *buf = malloc(len);
    if (!buf || !GetSystemCpuSetInformation((PSYSTEM_CPU_SET_INFORMATION)buf, len, &len, GetCurrentProcess(), 0)) { free(buf); return; }
    int lo = 255, hi = -1;
    ULONG n = 0;
    for (ULONG off = 0; off < len; off += ((PSYSTEM_CPU_SET_INFORMATION)(buf + off))->Size) {
        PSYSTEM_CPU_SET_INFORMATION c = (PSYSTEM_CPU_SET_INFORMATION)(buf + off);
        if (c->Type != CpuSetInformation) continue;
        int ec = c->CpuSet.EfficiencyClass;
        if (ec < lo) lo = ec;
        if (ec > hi) hi = ec;
        n++;
    }
    if (getenv("XCDE_PCORES_LOG")) fprintf(stderr, "[cpu] %lu logical CPUs, efficiency classes %d..%d\n", n, lo, hi);
    if (hi <= lo) { free(buf); return; }  // (one class of cores)
    ULONG *fast = malloc(n * sizeof *fast), nfast = 0;
    g_all_sets = malloc(n * sizeof *g_all_sets);
    for (ULONG off = 0; off < len; off += ((PSYSTEM_CPU_SET_INFORMATION)(buf + off))->Size) {
        PSYSTEM_CPU_SET_INFORMATION c = (PSYSTEM_CPU_SET_INFORMATION)(buf + off);
        if (c->Type != CpuSetInformation) continue;
        g_all_sets[g_nall_sets++] = c->CpuSet.Id;
        if (c->CpuSet.EfficiencyClass == hi) fast[nfast++] = c->CpuSet.Id;
    }
    // (only with enough fast logical CPUs for the game's busy threads: a Core Ultra 5 125U has 2 P-cores = 4 of them,
    // too few for the game, render, job and our render threads, which then queue behind each other)
    if (nfast < 8) {
        fprintf(stderr, "[cpu] hybrid CPU: only %lu fastest logical CPUs - game threads stay on every core\n", nfast);
        g_nall_sets = 0;
    } else if (SetProcessDefaultCpuSets(GetCurrentProcess(), fast, nfast))
        fprintf(stderr, "[cpu] hybrid CPU: game threads on the %lu fastest logical CPUs (efficiency class %d), shader workers on all %lu\n",
                nfast, hi, g_nall_sets);
    else g_nall_sets = 0;
    free(fast);
    free(buf);
}

void plat_init(void) {
    log_to_file_without_console();
    require_cpu();
    _fmode = _O_BINARY;  // fopen/open default to binary (the runtime's "r"/"w" files are binary or LF text)
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stderr), _O_BINARY);
    SetConsoleOutputCP(CP_UTF8);
    timeBeginPeriod(1);  // 1 ms Sleep / condition-variable timeouts (default 15.6 ms)
    // Opt out of EcoQoS and of Windows 11 ignoring the timer resolution while the window is occluded.
    PROCESS_POWER_THROTTLING_STATE pt = {
        .Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        .ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED | PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION,
        .StateMask = 0,
    };
    SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &pt, sizeof pt);
    // [XCDE] low-spec test mode: XCDE_LOWSPEC_CPUS=<n> runs the process on n physical cores (one logical CPU of each
    // SMT pair: 0, 2, 4, ...), e.g. 4 for the minimum-spec target (Ryzen 3 3200G: 4 cores, 4 threads).
    const char *ls = getenv("XCDE_LOWSPEC_CPUS");
    if (ls && atoi(ls) > 0) {
        DWORD_PTR proc = 0, sys = 0, mask = 0;
        GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys);
        int want = atoi(ls), got = 0;
        for (int i = 0; i < 64 && got < want; i += 2) if (sys >> i & 1) { mask |= (DWORD_PTR)1 << i; got++; }
        if (mask && SetProcessAffinityMask(GetCurrentProcess(), mask)) fprintf(stderr, "[lowspec] process on %d cores (mask %#llx)\n", got, (unsigned long long)mask);
    }
    pcores_init();
}

// [XCDE] T169: memory of the process (private bytes, working set) and of the guest window (committed bytes), for the
// VK_PROF memory report. Walking the window is ~1 ms (a few thousand regions): call it every few seconds, not per frame.
#include <psapi.h>
void plat_mem_info(uint64_t *priv, uint64_t *ws, uint64_t *guest_commit) {
    PROCESS_MEMORY_COUNTERS_EX pm = {.cb = sizeof pm};
    *priv = *ws = 0;
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&pm, sizeof pm)) { *priv = pm.PrivateUsage; *ws = pm.WorkingSetSize; }
    uint64_t c = 0;
    for (uint64_t a = g_lo; a < g_hi;) {
        MEMORY_BASIC_INFORMATION mi;
        if (!VirtualQuery((void *)(uintptr_t)a, &mi, sizeof mi) || !mi.RegionSize) break;
        if (mi.State == MEM_COMMIT) c += mi.RegionSize;
        a = (uint64_t)(uintptr_t)mi.BaseAddress + mi.RegionSize;
    }
    *guest_commit = c;
}

// [XCDE] T169: XCDE_MEM_MAP - committed private memory outside the guest window, grouped by allocation (the largest 24
// logged with their base and the module that owns the return address of nothing - just sizes, enough to tell a
// 256 MB block from heap growth). Called from the memory report.
void plat_mem_map(void) {
    typedef struct { uint64_t base, bytes; } Grp;
    static Grp g[4096];
    int n = 0;
    uint64_t total = 0, heapish = 0;
    for (uint64_t a = 0x10000; a < 0x7FFFFFFF0000ull;) {
        MEMORY_BASIC_INFORMATION mi;
        if (!VirtualQuery((void *)(uintptr_t)a, &mi, sizeof mi) || !mi.RegionSize) break;
        uint64_t next = (uint64_t)(uintptr_t)mi.BaseAddress + mi.RegionSize;
        bool in_guest = a >= g_lo && a < g_hi;
        if (!in_guest && mi.State == MEM_COMMIT && mi.Type == MEM_PRIVATE) {
            uint64_t b = (uint64_t)(uintptr_t)mi.AllocationBase;
            int i = 0;
            while (i < n && g[i].base != b) i++;
            if (i == n && n < 4096) { g[n].base = b; g[n].bytes = 0; n++; }
            if (i < 4096) g[i].bytes += mi.RegionSize;
            total += mi.RegionSize;
            if (mi.RegionSize < (1u << 20)) heapish += mi.RegionSize;
        }
        a = next;
    }
    for (int i = 0; i < n; i++)  // (largest first)
        for (int j = i + 1; j < n; j++) if (g[j].bytes > g[i].bytes) { Grp t = g[i]; g[i] = g[j]; g[j] = t; }
    fprintf(stderr, "[memmap] private committed outside the guest window: %.0f MB in %d allocations (%.0f MB in regions under 1 MB)\n",
            total / 1048576.0, n, heapish / 1048576.0);
    for (int i = 0; i < n && i < 24; i++) fprintf(stderr, "[memmap]   %#llx  %8.1f MB\n", (unsigned long long)g[i].base, g[i].bytes / 1048576.0);
    {   // [XCDE] T242: the Win32 heaps (malloc / the CRT is the process heap; DLLs may have their own) vs the rest
        HANDLE hs[64];
        DWORD nh = GetProcessHeaps(64, hs);
        HANDLE def = GetProcessHeap();
        uint64_t heaps = 0;
        for (DWORD i = 0; i < nh && i < 64; i++) {
            HEAP_SUMMARY sm = {.cb = sizeof sm};
            if (!HeapSummary(hs[i], 0, &sm)) continue;
            heaps += sm.cbCommitted;
            if (sm.cbCommitted >= (16u << 20) || hs[i] == def)
                fprintf(stderr, "[memmap] heap %p%s: committed %.0f MB, allocated %.0f MB\n", hs[i], hs[i] == def ? " (process heap: malloc)" : "",
                        sm.cbCommitted / 1048576.0, sm.cbAllocated / 1048576.0);
        }
        fprintf(stderr, "[memmap] %lu heaps: %.0f MB committed; outside heaps %.0f MB\n", nh, heaps / 1048576.0,
                (total > heaps ? total - heaps : 0) / 1048576.0);
    }
}

// [XCDE] logical CPUs this process may run on (its affinity mask; the low-spec mode narrows it)
int plat_cpu_count(void) {
    DWORD_PTR proc = 0, sys = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys) || !proc) return 4;
    return __builtin_popcountll((unsigned long long)proc);
}

bool plat_exe_path(char *buf, size_t size) {
    DWORD n = GetModuleFileNameA(NULL, buf, (DWORD)size);
    return n > 0 && n < size;
}

// Quote one argument the way CommandLineToArgvW / the CRT parse it back.
static void append_arg(char *out, size_t cap, const char *a) {
    size_t n = strlen(out);
    bool q = !*a || strpbrk(a, " \t\"") != NULL;
    if (n && n + 1 < cap) out[n++] = ' ';
    if (q && n + 1 < cap) out[n++] = '"';
    for (const char *p = a; *p; p++) {
        int bs = 0;
        while (*p == '\\') { bs++; p++; }
        int reps = !*p ? (q ? 2 * bs : bs) : *p == '"' ? 2 * bs + 1 : bs;
        for (int i = 0; i < reps && n + 1 < cap; i++) out[n++] = '\\';
        if (!*p) break;
        if (n + 1 < cap) out[n++] = *p;
    }
    if (q && n + 1 < cap) out[n++] = '"';
    out[n] = 0;
}

// No exec() on Windows: start the new process with the same console / standard handles, then exit.
void plat_exec(char *const argv[]) {
    STARTUPINFOW si = {.cb = sizeof si};
    PROCESS_INFORMATION pi;
    BOOL ok;
    fflush(stdout);
    fflush(stderr);
    if (!argv) {
        ok = CreateProcessW(NULL, GetCommandLineW(), NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi);
    } else {
        static char cmd[32768];
        cmd[0] = 0;
        for (int i = 0; argv[i]; i++) append_arg(cmd, sizeof cmd, argv[i]);
        static wchar_t wcmd[32768];
        MultiByteToWideChar(CP_UTF8, 0, cmd, -1, wcmd, 32768);
        ok = CreateProcessW(NULL, wcmd, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi);
    }
    if (!ok) return;
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    ExitProcess(0);
}

// ---- diagnostics
// DLL frames are named module+offset; frames in our own executable are left to the caller, which maps
// them to guest functions (runtime/dispatch.c) - mingw builds carry DWARF, not PDB symbols.
const char *plat_symbol(uint64_t pc, uint64_t *offset) {
    static __thread char name[MAX_PATH];
    HMODULE m;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCSTR)(uintptr_t)pc, &m))
        return NULL;
    if (m == GetModuleHandleA(NULL)) {  // [XCDE] T122: the exe's own functions by name, from its PDB (dbghelp)
        static SRWLOCK lock = SRWLOCK_INIT;
        static int inited;
        bool ok = false;
        AcquireSRWLockExclusive(&lock);
        if (!inited) {
            // the exe names its PDB without a directory ("xcde.pdb"): search next to the exe, not only the current one
            inited = 1;
            char dir[MAX_PATH];
            DWORD n = GetModuleFileNameA(NULL, dir, sizeof dir);
            char *sl = n ? strrchr(dir, '\\') : NULL;
            if (sl) *sl = 0;
            SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
            SymInitialize(GetCurrentProcess(), sl ? dir : NULL, TRUE);
        }
        union { SYMBOL_INFO s; char buf[sizeof(SYMBOL_INFO) + 256]; } u = {0};
        u.s.SizeOfStruct = sizeof(SYMBOL_INFO);
        u.s.MaxNameLen = 255;
        DWORD64 disp = 0;
        if (SymFromAddr(GetCurrentProcess(), pc, &disp, &u.s)) {
            snprintf(name, sizeof name, "%s", u.s.Name);
            if (offset) *offset = disp;
            ok = true;
        }
        ReleaseSRWLockExclusive(&lock);
        return ok ? name : NULL;
    }
    char path[MAX_PATH];
    if (!GetModuleFileNameA(m, path, sizeof path)) return NULL;
    const char *b = strrchr(path, '\\');
    snprintf(name, sizeof name, "%s", b ? b + 1 : path);
    if (offset) *offset = pc - (uint64_t)(uintptr_t)m;
    return name;
}

const char *rt_host_fn_name(uint64_t pc, uint64_t *offset);  // runtime/dispatch.c

void plat_print_backtrace(void) {
    void *bt[64];
    USHORT n = RtlCaptureStackBackTrace(0, 64, bt, NULL);
    uint64_t exe = (uint64_t)(uintptr_t)GetModuleHandleA(NULL);
    for (int i = 0; i < n; i++) {
        uint64_t pc = (uint64_t)(uintptr_t)bt[i], off = 0;
        const char *s = plat_symbol(pc, &off);
        if (!s) s = rt_host_fn_name(pc, &off);
        if (s) fprintf(stderr, "  #%-2d %016llx %s+%llx\n", i, (unsigned long long)pc, s, (unsigned long long)off);
        else fprintf(stderr, "  #%-2d %016llx lasrecomp+%llx\n", i, (unsigned long long)pc,
                     (unsigned long long)(pc - exe));
    }
}

static LONG WINAPI unhandled_filter(EXCEPTION_POINTERS *ep) {
    EXCEPTION_RECORD *er = ep->ExceptionRecord;
    uint64_t fa = er->NumberParameters >= 2 ? er->ExceptionInformation[1] : 0;
    char what[48];
    snprintf(what, sizeof what, "exception %08lx", er->ExceptionCode);
    // x64 has no link register: report the return address on top of the stack (valid at function entry).
    uint64_t lr = 0;
    MEMORY_BASIC_INFORMATION mi;
    if (VirtualQuery((void *)(uintptr_t)ep->ContextRecord->Rsp, &mi, sizeof mi) && mi.State == MEM_COMMIT)
        lr = *(uint64_t *)(uintptr_t)ep->ContextRecord->Rsp;
    { extern void diag_crash_report(EXCEPTION_POINTERS *); diag_crash_report(ep); }  // [XCDE] T028: text report + out-of-process minidump
    rt_fault_report(what, fa, ep->ContextRecord->Rip, lr);
}

// [XCDE] exits that skip the exception filter: abort() (a library's assert / std::terminate) and the UCRT's invalid
// parameter check (close(-1), a bad format... whose default is __fastfail: the process just vanishes). Both leave a line
// and a backtrace in the log; an invalid parameter then returns EINVAL to the caller instead of killing the game.
static void abort_handler(int sig) {
    (void)sig;
    signal(SIGABRT, SIG_DFL);
    fprintf(stderr, "[crash] abort() called\n");
    plat_print_backtrace();
    fflush(stderr);
}
static void invalid_parameter(const wchar_t *expr, const wchar_t *fn, const wchar_t *file, unsigned line, uintptr_t r) {
    (void)expr; (void)fn; (void)file; (void)line; (void)r;
    static int n;
    if (n++ >= 5) return;
    fprintf(stderr, "[crash] C runtime invalid parameter (the call fails with EINVAL; logged 5 times at most)\n");
    plat_print_backtrace();
    fflush(stderr);
}

void plat_crash_handler_install(void) {
    ULONG guarantee = 64 << 10;  // room for the report after a stack overflow
    SetThreadStackGuarantee(&guarantee);
    if (g_report_crashes) return;
    g_report_crashes = true;
    SetUnhandledExceptionFilter(unhandled_filter);
    signal(SIGABRT, abort_handler);
    _set_invalid_parameter_handler(invalid_parameter);
}

// ---- POSIX shims (win_compat.h)
static ssize_t prw(int fd, void *buf, size_t n, off_t off, bool write) {
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) return -1;
    size_t done = 0;
    while (done < n) {
        OVERLAPPED ov = {0};
        uint64_t o = (uint64_t)off + done;
        ov.Offset = (DWORD)o;
        ov.OffsetHigh = (DWORD)(o >> 32);
        DWORD chunk = n - done > 0x40000000 ? 0x40000000 : (DWORD)(n - done), got = 0;
        BOOL ok = write ? WriteFile(h, (const char *)buf + done, chunk, &got, &ov)
                        : ReadFile(h, (char *)buf + done, chunk, &got, &ov);
        if (!ok) {
            if (GetLastError() == ERROR_HANDLE_EOF) break;
            return done ? (ssize_t)done : -1;
        }
        if (!got) break;
        done += got;
    }
    return (ssize_t)done;
}
ssize_t pread(int fd, void *buf, size_t n, off_t off) { return prw(fd, buf, n, off, false); }
ssize_t pwrite(int fd, const void *buf, size_t n, off_t off) { return prw(fd, (void *)buf, n, off, true); }

// ---- child processes
PlatProc plat_spawn(const char *const argv[], const char *log_path, bool append_log) {
    static __thread char cmd[32768];
    cmd[0] = 0;
    for (int i = 0; argv[i]; i++) append_arg(cmd, sizeof cmd, argv[i]);
    static __thread wchar_t wcmd[32768], wlog[MAX_PATH];
    MultiByteToWideChar(CP_UTF8, 0, cmd, -1, wcmd, 32768);
    MultiByteToWideChar(CP_UTF8, 0, log_path ? log_path : "NUL", -1, wlog, MAX_PATH);
    SECURITY_ATTRIBUTES sa = {.nLength = sizeof sa, .bInheritHandle = TRUE};
    HANDLE log = CreateFileW(wlog, append_log ? FILE_APPEND_DATA : GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                             &sa, append_log ? OPEN_ALWAYS : CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (log == INVALID_HANDLE_VALUE) return 0;
    STARTUPINFOW si = {.cb = sizeof si, .dwFlags = STARTF_USESTDHANDLES, .hStdInput = NULL, .hStdOutput = log,
                       .hStdError = log};
    PROCESS_INFORMATION pi;
    // CreateProcess doesn't resolve relative paths written with '/' (e.g. a helper under tools/): pass the program
    // as a full path, like posix_spawn takes it as given
    static __thread wchar_t wprog[MAX_PATH], wfull[MAX_PATH];
    MultiByteToWideChar(CP_UTF8, 0, argv[0], -1, wprog, MAX_PATH);
    for (wchar_t *p = wprog; *p; p++) if (*p == L'/') *p = L'\\';
    const wchar_t *app = GetFullPathNameW(wprog, MAX_PATH, wfull, NULL) ? wfull : NULL;
    BOOL ok = CreateProcessW(app, wcmd, NULL, NULL, TRUE, CREATE_NO_WINDOW | CREATE_SUSPENDED, NULL, NULL, &si, &pi);
    if (!ok) fprintf(stderr, "[plat] could not start %s (error %lu)\n", argv[0], GetLastError());
    CloseHandle(log);
    if (!ok) return 0;
    // every child in one job that ends with this process (a compiler's own children too: no leftovers on quit)
    static HANDLE job;
    if (!job && (job = CreateJobObjectW(NULL, NULL))) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION li = {0};
        li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &li, sizeof li);
    }
    if (job) AssignProcessToJobObject(job, pi.hProcess);
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);
    return (PlatProc)pi.hProcess;
}

void plat_proc_kill(PlatProc p) {  // (what it started ends with this process: the job above)
    TerminateProcess((HANDLE)p, 255);
    WaitForSingleObject((HANDLE)p, INFINITE);
    CloseHandle((HANDLE)p);
}

int plat_proc_wait(PlatProc p, bool block) {
    HANDLE h = (HANDLE)p;
    DWORD w = WaitForSingleObject(h, block ? INFINITE : 0);
    if (w == WAIT_TIMEOUT) return -1;
    if (w != WAIT_OBJECT_0) return 255;  // (not a process we can wait for: ended as far as the caller is concerned)
    DWORD code = 255;
    GetExitCodeProcess(h, &code);
    CloseHandle(h);
    return code > 255 ? 255 : (int)code;  // (a crash's NTSTATUS, e.g. 0xC0000409, is not "still running" (-1))
}

bool plat_remove_tree(const char *dir) {
    // SHFileOperation wants a double-NUL-terminated path with backslashes
    static __thread wchar_t w[MAX_PATH + 2];
    int n = MultiByteToWideChar(CP_UTF8, 0, dir, -1, w, MAX_PATH);
    if (n <= 1) return false;
    w[n] = 0;
    for (wchar_t *p = w; *p; p++) if (*p == L'/') *p = L'\\';
    SHFILEOPSTRUCTW op = {.wFunc = FO_DELETE, .pFrom = w, .fFlags = FOF_NO_UI};
    return SHFileOperationW(&op) == 0 && !op.fAnyOperationsAborted;
}

// End the process at once, skipping DLL teardown (from an atexit handler, after flushing). Exit status 0.
void plat_terminate(void) { TerminateProcess(GetCurrentProcess(), 0); }

// [XCDE] T122: sampled instruction pointer of a host thread (watchdog: what a thread that makes no SVCs is running)
#include <pthread.h>
uint64_t plat_thread_pc(void *pthread_handle) {
    HANDLE h = (HANDLE)pthread_gethandle(*(pthread_t *)pthread_handle);
    if (!h || SuspendThread(h) == (DWORD)-1) return 0;
    CONTEXT ctx = {.ContextFlags = CONTEXT_CONTROL};
    uint64_t pc = GetThreadContext(h, &ctx) ? ctx.Rip : 0;
    ResumeThread(h);
    return pc;
}

// [XCDE] T122: host call stack of another thread (suspended while unwound); returns the number of frames
int plat_thread_backtrace(void *pthread_handle, uint64_t *pcs, int max) {
    HANDLE h = (HANDLE)pthread_gethandle(*(pthread_t *)pthread_handle);
    if (!h || SuspendThread(h) == (DWORD)-1) return 0;
    CONTEXT ctx = {.ContextFlags = CONTEXT_FULL};
    int n = 0;
    if (GetThreadContext(h, &ctx)) {
        while (n < max && ctx.Rip) {
            pcs[n++] = ctx.Rip;
            DWORD64 base = 0;
            PRUNTIME_FUNCTION fn = RtlLookupFunctionEntry(ctx.Rip, &base, NULL);
            if (!fn) {  // leaf: the return address is at rsp
                ctx.Rip = *(DWORD64 *)ctx.Rsp;
                ctx.Rsp += 8;
                continue;
            }
            PVOID hd = NULL;
            DWORD64 est = 0;
            RtlVirtualUnwind(UNW_FLAG_NHANDLER, base, ctx.Rip, fn, &ctx, &hd, &est, NULL);
        }
    }
    ResumeThread(h);
    return n;
}

// [XCDE] T160: background work (shader / pipeline builds): below the game, render and audio threads, so a burst of
// builds on entering an area doesn't take CPU time from the frame. A thread waiting for a build sleeps, so the
// worker still gets the core it needs. LARECOMP_WORKER_PRIO=0: normal priority.
void plat_thread_background(void) {
    if (g_nall_sets) SetThreadSelectedCpuSets(GetCurrentThread(), g_all_sets, g_nall_sets);  // (every core: pcores_init)
    const char *e = getenv("LARECOMP_WORKER_PRIO");
    if (e && atoi(e) == 0) return;
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
}

// [XCDE] audio feeders (audout / audren threads): they sleep 1 ms at a time and need little CPU, but a late wake-up is
// an audible gap - with 4 cores busy with game threads, one 5-minute run had a device underrun at normal priority.
// LARECOMP_AUDIO_PRIO=0: normal priority.
void plat_thread_audio(void) {
    const char *e = getenv("LARECOMP_AUDIO_PRIO");
    if (e && atoi(e) == 0) return;
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
}

// [XCDE] T112: names the calling thread (shown by the sampling profiler, debuggers and crash dumps)
void plat_thread_name(const char *name) {
    wchar_t w[64];
    if (MultiByteToWideChar(CP_UTF8, 0, name, -1, w, 64)) SetThreadDescription(GetCurrentThread(), w);
}

// [XCDE] diag.c XCDE_DUMP_HEAP: the committed, readable guest heap pages to a file (u64 address, u64 size, bytes)
uint64_t plat_guest_dump(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    uint64_t total = 0;
    for (uint64_t a = 0x7240000000ull; a < 0x7240000000ull + 0xD0000000ull;) {
        MEMORY_BASIC_INFORMATION mi;
        if (!VirtualQuery((void *)(uintptr_t)a, &mi, sizeof mi) || !mi.RegionSize) break;
        uint64_t base = (uint64_t)(uintptr_t)mi.BaseAddress, size = mi.RegionSize;
        if (mi.State == MEM_COMMIT && !(mi.Protect & (PAGE_NOACCESS | PAGE_GUARD))) {
            fwrite(&base, 8, 1, f);
            fwrite(&size, 8, 1, f);
            fwrite((const void *)(uintptr_t)base, 1, size, f);
            total += size;
        }
        a = base + size;
    }
    fclose(f);
    return total;
}

// [XCDE] diag.c XCDE_FIND: addresses of a byte pattern in the committed guest memory (main image .. end of the heap)
int plat_guest_find(const uint8_t *pat, size_t n, uint64_t *out, int max) {
    int k = 0;
    for (uint64_t a = 0x7000000000ull; a < 0x7400000000ull && k < max;) {
        MEMORY_BASIC_INFORMATION mi;
        if (!VirtualQuery((void *)(uintptr_t)a, &mi, sizeof mi) || !mi.RegionSize) break;
        uint64_t base = (uint64_t)(uintptr_t)mi.BaseAddress, end = base + mi.RegionSize;
        if (mi.State == MEM_COMMIT && !(mi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) && end - base >= n) {
            const uint8_t *m = (const uint8_t *)(uintptr_t)base;
            for (uint64_t i = 0; i + n <= end - base && k < max; i++)
                if (m[i] == pat[0] && !memcmp(m + i, pat, n)) out[k++] = base + i;
        }
        a = end;
    }
    return k;
}
