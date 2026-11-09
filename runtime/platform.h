// Host OS layer: everything the runtime needs that differs between macOS and Windows.
// Implementations: runtime/platform/macos.c, runtime/platform/windows.c (the Makefile picks one).
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#include "platform/win_compat.h"
#elif defined(__linux__)
#include "platform/linux_compat.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

// [VEGA-FIX] Runs fn(arg); returns false if it died with an integer divide by zero meanwhile (a graphics driver compiler
// bug: amdvlk64.dll on Vega iGPUs) - the thread then continues after the call instead of taking the process down.
bool plat_try_call(void (*fn)(void *), void *arg);
// ---- guest address space
// Claim [lo, hi) for the guest at exactly that address, readable/writable (pages may be backed lazily).
bool plat_reserve_guest(uint64_t lo, uint64_t hi);
// Make sure [addr, addr+size) inside the guest window is backed (before the OS itself writes there, e.g.
// file reads); a no-op where the whole window is backed from the start.
void plat_commit(uint64_t addr, uint64_t size);
// In-place reboot: every page of [lo, hi) back to zero (and unbacked), the window still claimed.
bool plat_reset_guest(uint64_t lo, uint64_t hi);
// Pages of [addr, addr+size) in the guest window the CPU wrote since the last call over them (write tracking: Windows
// MEM_WRITE_WATCH), their addresses into pages[] (at most max), and the tracking reset for the range. Returns the
// count (max: maybe more), or -1 where there is no tracking (macOS, Linux). Page size: *page.
int plat_written_pages(uint64_t addr, uint64_t size, uint64_t *pages, int max, uint64_t *page);
enum { PLAT_PROT_NONE = 0, PLAT_PROT_READ = 1, PLAT_PROT_RW = 3 };
int plat_protect(uint64_t addr, uint64_t size, int prot);  // 0 on success
// [XCDE] CPU/GPU overlap study (Linux; no-ops elsewhere): page ranges the GPU of the last recorded frame reads (prot
// PLAT_PROT_READ) or writes (PLAT_PROT_NONE) are protected; CPU accesses to them fault, are counted and let through.
void plat_ov_protect(const uint64_t *lo, const uint64_t *hi, const uint8_t *gpu_writes, int n, uint64_t seq);  // seq 0: stats
void plat_ov_release_prev(void);                       // (wait mode) the previous frame's GPU work is done
void plat_ov_set_gpu_done(uint64_t (*fn)(void));       // the timeline's completed sequence (wait mode)
void plat_ov_set_gpu_wait(void (*fn)(uint64_t seq));   // blocks until the timeline reaches seq (wait mode; else a yield loop)
void plat_ov_frame_start(void);                        // the CPU starts its next frame now (fault times count from here)
void plat_ov_guest_write(uint64_t addr, uint64_t size);  // a syscall is about to write guest memory: open those pages
void plat_ov_set_worker(void);                         // the calling thread is the render worker
void plat_ov_report(void);                             // logs the counters of the last report period
#define PLAT_PAGE 0x4000ull  // largest page size we deal with (macOS 16K; multiple of Windows' 4K)

// ---- time
uint64_t plat_now_ns(void);                // monotonic
void plat_sleep_until_ns(uint64_t t);      // precise wait for plat_now_ns() >= t
void plat_thread_realtime(uint64_t period_ns);  // periodic timing thread (vsync): don't coalesce its timers

// ---- process
void plat_init(void);                                // first thing in main()
bool plat_exe_path(char *buf, size_t size);
void plat_exec(char *const argv[]);  // replace the process (NULL = same command line); returns only on failure
#if defined(_WIN32) || defined(__linux__)
void plat_terminate(void);  // end the process now, skipping DLL teardown (exit status 0)
#endif

// Child processes (seed generator, game-file extraction). stdout + stderr go to log_path (NULL = /dev/null).
typedef intptr_t PlatProc;  // 0 = failed to start
PlatProc plat_spawn(const char *const argv[], const char *log_path, bool append_log);
int plat_proc_wait(PlatProc p, bool block);  // exit code, or -1 while still running (block = false)
void plat_proc_kill(PlatProc p);             // end it and what it started (then it's gone: no wait)
bool plat_remove_tree(const char *dir);      // rm -rf
#ifdef _WIN32
#define PLAT_EXE_SUFFIX ".exe"
#define PLAT_VENV_PYTHON "venv/Scripts/python.exe"  // relative to the venv's parent
#else
#define PLAT_EXE_SUFFIX ""
#define PLAT_VENV_PYTHON "venv/bin/python"
#endif

// ---- diagnostics
// Name of the host function containing pc (guest functions are f_<addr>), NULL if unknown.
const char *plat_symbol(uint64_t pc, uint64_t *offset);
void plat_print_backtrace(void);
// [XCDE] T122: the instruction pointer of another host thread (suspended for the read); 0 if unknown
uint64_t plat_thread_pc(void *pthread_handle);
int plat_cpu_count(void);  // [XCDE] logical CPUs the process may use (affinity)
void plat_thread_background(void);  // [XCDE] T160: the calling thread runs below normal priority (build workers)
void plat_thread_audio(void);  // [XCDE] the calling thread feeds the audio device: above the game threads
void plat_thread_name(const char *name);  // names the calling thread (profiler, debugger)
int plat_thread_backtrace(void *pthread_handle, uint64_t *pcs, int max);  // (same, the whole host call stack)
void plat_fatal_notice(void);  // a fatal error is about to end the process: tell the user (Windows release: message box)
void plat_crash_handler_install(void);  // per thread where the OS needs it (alternate signal stack)
// Called by the crash handler (runtime/dispatch.c): true = fault handled (watchpoint), resume.
bool rt_fault_watch(uint64_t fault_addr, uint64_t host_pc);
void rt_fault_report(const char *what, uint64_t fault_addr, uint64_t host_pc, uint64_t host_lr) __attribute__((noreturn));

#ifdef __cplusplus
}
#endif
