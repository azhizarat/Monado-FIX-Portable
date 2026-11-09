// nn::os fibers over host (Windows) fibers (T032b, FR-022).  [XCDE]
//
// The SDK switches fibers by saving/restoring registers and `ret`-ing into the other context. Recompiled code runs on
// the host stack, so it cannot do that: each guest fiber gets a host fiber (its own host stack) and its own guest
// register file (Cpu). The guest stack (given to InitializeFiber) stays guest memory.
//
//   InitializeFiber(FiberType*, FiberFunction, void *arg, void *stack, size_t stack_size, int flags)
//   SwitchToFiber(FiberType*)      FinalizeFiber(FiberType*)      GetCurrentFiber() -> FiberType*
//   FiberFunction: FiberType *(*)(void *arg) -- returns the fiber to switch to when it ends (null: back to the thread)
//
// Lessons applied (research R10): a fiber may be resumed on another host thread than the one that suspended it, so
// its guest thread block (tpidr/tpidrro/thread) is handed over from the switching thread on every switch (ssbu #43);
// the per-thread "current fiber" is read through non-inlined accessors, because a compiler may keep a thread-local
// address across a switch that moves the code to another thread.
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
// Linux: the few Win32 fiber calls this file uses, over ucontext (each fiber: its own mmap'd stack).
// ponytail: swapcontext also saves/restores the signal mask (one syscall per switch); a hand-written
// register switch is the upgrade if fiber switches ever show up in a profile.
#include <pthread.h>
#include <sys/mman.h>
#include <ucontext.h>
#define CALLBACK
typedef struct { ucontext_t uc; void *stack; size_t size; void (*fn)(void *); void *arg; } PosixFiber;
static __thread PosixFiber *t_host;  // host fiber running on this thread (read through the accessors: see below)
__attribute__((noinline)) static PosixFiber *host_get(void) { __asm__ volatile("" ::: "memory"); return t_host; }
__attribute__((noinline)) static void host_set(PosixFiber *f) { t_host = f; __asm__ volatile("" ::: "memory"); }
static int IsThreadAFiber(void) { return host_get() != NULL; }
static void *GetCurrentFiber(void) { return host_get(); }
static void *ConvertThreadToFiber(void *p) { (void)p; PosixFiber *f = calloc(1, sizeof *f); host_set(f); return f; }
static void fiber_entry(void) { PosixFiber *f = host_get(); f->fn(f->arg); abort(); }  // (fiber_main never returns)
static void *CreateFiberEx(size_t commit, size_t reserve, int flags, void (*fn)(void *), void *arg) {
    (void)commit; (void)flags;
    PosixFiber *f = calloc(1, sizeof *f);
    f->stack = mmap(NULL, reserve, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_STACK, -1, 0);
    if (f->stack == MAP_FAILED) { free(f); return NULL; }
    f->size = reserve, f->fn = fn, f->arg = arg;
    getcontext(&f->uc);
    f->uc.uc_stack.ss_sp = f->stack;
    f->uc.uc_stack.ss_size = reserve;
    f->uc.uc_link = NULL;
    makecontext(&f->uc, fiber_entry, 0);
    return f;
}
static void SwitchToFiber(void *to) {
    PosixFiber *from = host_get();
    host_set(to);
    swapcontext(&from->uc, &((PosixFiber *)to)->uc);
}
static void DeleteFiber(void *p) { PosixFiber *f = p; if (f->stack) munmap(f->stack, f->size); free(f); }
static unsigned long GetLastError(void) { return 0; }
typedef size_t SIZE_T;
typedef pthread_mutex_t CRITICAL_SECTION;
#endif

#include "../rt.h"

typedef struct HFiber {
    uint64_t key;                 // guest FiberType*
    uint64_t fn, arg, stack, stack_size;
    void *host;                   // the host fiber
    Cpu cpu;                      // this fiber's guest registers
    uint64_t tpidr, tpidrro;      // thread block of the host thread that runs it now (set by whoever switches to it)
    void *thread;
    struct HFiber *next;          // hash chain
} HFiber;

#define NBUCKETS 4096
static HFiber *g_buckets[NBUCKETS];
#ifdef _WIN32
static CRITICAL_SECTION g_lock;
static INIT_ONCE g_once = INIT_ONCE_STATIC_INIT;

static BOOL CALLBACK init_lock(PINIT_ONCE o, PVOID p, PVOID *ctx) { (void)o; (void)p; (void)ctx; InitializeCriticalSection(&g_lock); return TRUE; }
static void lock(void) { InitOnceExecuteOnce(&g_once, init_lock, NULL, NULL); EnterCriticalSection(&g_lock); }
static void unlock(void) { LeaveCriticalSection(&g_lock); }
#else
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static void lock(void) { pthread_mutex_lock(&g_lock); }
static void unlock(void) { pthread_mutex_unlock(&g_lock); }
#endif

static unsigned bucket(uint64_t key) { return (unsigned)((key >> 4) * 0x9E3779B1u) % NBUCKETS; }
static HFiber *find(uint64_t key) {
    lock();
    HFiber *f = g_buckets[bucket(key)];
    while (f && f->key != key) f = f->next;
    unlock();
    return f;
}

// ---- per host thread state, through non-inlined accessors
static __thread uint64_t t_fiber;       // guest FiberType* running on this host thread now (0: the thread itself)
static __thread void *t_root;           // the host fiber of the thread itself
__attribute__((noinline)) static uint64_t cur_get(void) { __asm__ volatile("" ::: "memory"); return t_fiber; }
__attribute__((noinline)) static void cur_set(uint64_t k) { t_fiber = k; __asm__ volatile("" ::: "memory"); }
__attribute__((noinline)) static void *root_get(void) { __asm__ volatile("" ::: "memory"); return t_root; }
__attribute__((noinline)) static void root_set(void *r) { t_root = r; __asm__ volatile("" ::: "memory"); }

static void *ensure_root(void) {
    void *r = root_get();
    if (!r) {
        r = IsThreadAFiber() ? GetCurrentFiber() : ConvertThreadToFiber(NULL);
        root_set(r);
    }
    return r;
}

// hand the caller's thread block to the fiber about to run on this host thread
static void hand_over(HFiber *to, const Cpu *from) {
    to->tpidr = from->tpidr;
    to->tpidrro = from->tpidrro;
    to->thread = from->thread;
}
static void take_over(Cpu *c, const HFiber *f) {
    c->tpidr = f->tpidr;
    c->tpidrro = f->tpidrro;
    c->thread = f->thread;
}

static void CALLBACK fiber_main(void *p) {
    HFiber *f = p;
    for (;;) {
        Cpu *c = &f->cpu;
        take_over(c, f);
        c->x[0] = f->arg;
        c->sp = (f->stack + f->stack_size) & ~15ull;
        c->excl_addr = ~0ull;
        guest_call(c, f->fn);                 // returns the fiber to switch to next, or null
        uint64_t next = c->x[0];
        HFiber *n = next ? find(next) : NULL;
        if (next && !n) FATAL("[fiber] function of fiber %llx returned unknown fiber %llx\n", (unsigned long long)f->key, (unsigned long long)next);
        if (n) hand_over(n, c);
        cur_set(next);
        SwitchToFiber(n ? n->host : root_get());
        // only reached if a finished fiber is switched to again: it starts over with its function
    }
}

// ---- the HLE entry points (names fixed: gen.py --hle binds the SDK functions to hle_<mangled>)
void hle__ZN2nn2os15InitializeFiberEPNS0_9FiberTypeEPFS2_PvES3_S3_mi(Cpu *c) {
    uint64_t key = c->x[0];
    lock();
    HFiber *f = g_buckets[bucket(key)];
    while (f && f->key != key) f = f->next;
    if (!f) {
        f = calloc(1, sizeof *f);
        f->key = key;
        f->next = g_buckets[bucket(key)];
        g_buckets[bucket(key)] = f;
    } else if (f->host) {
        DeleteFiber(f->host);  // re-initialized without Finalize: start fresh
        f->host = NULL;
    }
    unlock();
    f->fn = c->x[1], f->arg = c->x[2], f->stack = c->x[3], f->stack_size = c->x[4];
    memset(&f->cpu, 0, sizeof f->cpu);
    f->cpu.excl_addr = ~0ull;
    f->host = CreateFiberEx(256 * 1024, (SIZE_T)64 << 20, 0, fiber_main, f);
    if (!f->host) FATAL("[fiber] CreateFiber failed (%lu)\n", GetLastError());
}

void hle__ZN2nn2os13FinalizeFiberEPNS0_9FiberTypeE(Cpu *c) {
    uint64_t key = c->x[0];
    lock();
    HFiber **pp = &g_buckets[bucket(key)];
    while (*pp && (*pp)->key != key) pp = &(*pp)->next;
    HFiber *f = *pp;
    if (f) *pp = f->next;
    unlock();
    if (f) {
        if (f->host) DeleteFiber(f->host);
        free(f);
    }
}

void hle__ZN2nn2os13SwitchToFiberEPNS0_9FiberTypeE(Cpu *c) {
    uint64_t key = c->x[0];
    void *root = ensure_root();
    HFiber *n = key ? find(key) : NULL;
    if (key && !n) FATAL("[fiber] SwitchToFiber to unknown fiber %llx\n", (unsigned long long)key);
    uint64_t self = cur_get();
    if (n) hand_over(n, c);
    cur_set(key);
    SwitchToFiber(n ? n->host : root);
    // resumed: possibly on another host thread than the one that suspended us. The one that switched to us wrote
    // its thread block into our record (or, for the thread's own context, nothing changed).
    cur_set(self);
    if (self) {
        HFiber *me = find(self);
        if (me) take_over(c, me);
    }
}

void hle__ZN2nn2os15GetCurrentFiberEv(Cpu *c) { c->x[0] = cur_get(); }
