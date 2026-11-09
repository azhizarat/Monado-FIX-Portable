// [XCDE] Win32 shims for building AMD FidelityFX SDK v1.1.4 (ffx-api, Vulkan backend, FSR 3.1 upscaler + frame generation
// + its swapchain proxy) on Linux. Force-included (-include) into every translation unit by scripts/linux/ffx_build.sh;
// only what the compiled sources use. The source edits a header cannot do are in linux.patch.
#pragma once
#ifndef __cplusplus
#error "C++ only"
#endif
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <locale>  // (ffx_vk.cpp's non-Windows name conversion uses std::wstring_convert)
#include <mutex>
#include <pthread.h>
#include <time.h>

// ffx_api.h marks the five entry points __declspec(dllexport); everything else is hidden (-fvisibility=hidden + the
// version script).
#define __declspec(x) __attribute__((visibility("default")))
#define _countof(a) (sizeof(a) / sizeof((a)[0]))

// ---------------------------------------------------------------- MSVC "secure" CRT
inline int wcscpy_s(wchar_t *d, size_t n, const wchar_t *s) {
    if (!d || !n) return 22;
    size_t l = s ? wcslen(s) : 0;
    if (l >= n) l = n - 1;  // (MSVC would abort; truncating is kinder for debug names)
    if (l) memcpy(d, s, l * sizeof(wchar_t));
    d[l] = 0;
    return 0;
}
template <size_t N> inline int wcscpy_s(wchar_t (&d)[N], const wchar_t *s) { return wcscpy_s(d, N, s); }
inline int strcpy_s(char *d, size_t n, const char *s) {
    if (!d || !n) return 22;
    snprintf(d, n, "%s", s ? s : "");
    return 0;
}
template <size_t N> inline int strcpy_s(char (&d)[N], const char *s) { return strcpy_s(d, N, s); }
#define sprintf_s snprintf
#define swprintf_s swprintf
// MSVC: *ret = converted characters + the terminator
inline int wcstombs_s(size_t *ret, char *d, size_t n, const wchar_t *s, size_t count) {
    size_t lim = count < n ? count : n - 1, i = 0;
    for (; s && i < lim && s[i]; i++) d[i] = (s[i] < 0x80) ? (char)s[i] : '?';  // (debug names: ASCII is enough)
    d[i] = 0;
    if (ret) *ret = i + 1;
    return 0;
}

// ---------------------------------------------------------------- types
typedef int BOOL;
typedef uint32_t DWORD;
typedef uint64_t UINT64;
typedef void *LPVOID;
typedef void *HANDLE;
#define WINAPI
#define TRUE 1
#define FALSE 0
#define INFINITE 0xFFFFFFFFu
#define TEXT(s) s
#define THREAD_PRIORITY_HIGHEST 2
typedef union {
    struct { uint32_t LowPart; int32_t HighPart; };
    int64_t QuadPart;
} LARGE_INTEGER;

// ---------------------------------------------------------------- time
inline BOOL QueryPerformanceCounter(LARGE_INTEGER *c) {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    c->QuadPart = (int64_t)t.tv_sec * 1000000000 + t.tv_nsec;
    return TRUE;
}
inline BOOL QueryPerformanceFrequency(LARGE_INTEGER *f) { f->QuadPart = 1000000000; return TRUE; }

// ---------------------------------------------------------------- CRITICAL_SECTION (re-entrant, as on Windows)
typedef struct { pthread_mutex_t m; } CRITICAL_SECTION;
inline void InitializeCriticalSection(CRITICAL_SECTION *cs) {
    pthread_mutexattr_t a;
    pthread_mutexattr_init(&a);
    pthread_mutexattr_settype(&a, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&cs->m, &a);
    pthread_mutexattr_destroy(&a);
}
inline void EnterCriticalSection(CRITICAL_SECTION *cs) { pthread_mutex_lock(&cs->m); }
inline void LeaveCriticalSection(CRITICAL_SECTION *cs) { pthread_mutex_unlock(&cs->m); }
inline void DeleteCriticalSection(CRITICAL_SECTION *cs) { pthread_mutex_destroy(&cs->m); }

// ---------------------------------------------------------------- events and threads (HANDLE = either)
struct FfxLinuxHandle {
    bool is_thread = false;
    // event
    std::mutex m;
    std::condition_variable cv;
    bool signaled = false, manual = false;
    // thread
    pthread_t th{};
    DWORD (*fn)(LPVOID) = nullptr;
    LPVOID arg = nullptr;
    bool joined = false;
};
inline HANDLE CreateEvent(void *, BOOL manualReset, BOOL initialState, const char *) {
    FfxLinuxHandle *h = new FfxLinuxHandle;
    h->manual = manualReset;
    h->signaled = initialState;
    return h;
}
inline BOOL SetEvent(HANDLE hh) {
    FfxLinuxHandle *h = (FfxLinuxHandle *)hh;
    { std::lock_guard<std::mutex> l(h->m); h->signaled = true; }
    if (h->manual) h->cv.notify_all(); else h->cv.notify_one();
    return TRUE;
}
inline BOOL ResetEvent(HANDLE hh) {
    FfxLinuxHandle *h = (FfxLinuxHandle *)hh;
    std::lock_guard<std::mutex> l(h->m);
    h->signaled = false;
    return TRUE;
}
inline void *ffx_linux_thread_main(void *p) {
    FfxLinuxHandle *h = (FfxLinuxHandle *)p;
    h->fn(h->arg);
    return nullptr;
}
inline HANDLE CreateThread(void *, size_t, DWORD (*fn)(LPVOID), LPVOID arg, DWORD, DWORD *id) {
    FfxLinuxHandle *h = new FfxLinuxHandle;
    h->is_thread = true;
    h->fn = fn;
    h->arg = arg;
    if (pthread_create(&h->th, nullptr, ffx_linux_thread_main, h)) { delete h; return nullptr; }
    if (id) *id = 0;
    return h;
}
#define WAIT_OBJECT_0 0u
// (only INFINITE waits are used)
inline DWORD WaitForSingleObject(HANDLE hh, DWORD) {
    FfxLinuxHandle *h = (FfxLinuxHandle *)hh;
    if (h->is_thread) {
        if (!h->joined) { pthread_join(h->th, nullptr); h->joined = true; }
        return WAIT_OBJECT_0;
    }
    std::unique_lock<std::mutex> l(h->m);
    h->cv.wait(l, [h] { return h->signaled; });
    if (!h->manual) h->signaled = false;
    return WAIT_OBJECT_0;
}
inline BOOL CloseHandle(HANDLE hh) {
    FfxLinuxHandle *h = (FfxLinuxHandle *)hh;
    if (h->is_thread && !h->joined) pthread_detach(h->th);
    delete h;
    return TRUE;
}
inline BOOL SetThreadPriority(HANDLE, int) { return TRUE; }  // (needs privileges on Linux; the threads keep the default)
inline long SetThreadDescription(HANDLE hh, const wchar_t *name) {
    FfxLinuxHandle *h = (FfxLinuxHandle *)hh;
    char n[16];
    size_t i = 0;
    for (; i < 15 && name[i]; i++) n[i] = (char)name[i];
    n[i] = 0;
    pthread_setname_np(h->th, n);
    return 0;
}

#ifdef FFX_LINUX_COMPAT_SELFTEST  // (ffx_build.sh compiles and runs this first: the shims' Win32 semantics)
#include <cassert>
static CRITICAL_SECTION st_cs;
static DWORD st_waiter(LPVOID e) { WaitForSingleObject(e, INFINITE); return 0; }
static DWORD st_locker(LPVOID) { EnterCriticalSection(&st_cs); LeaveCriticalSection(&st_cs); return 0; }
int main() {
    HANDLE a = CreateEvent(nullptr, FALSE, TRUE, "auto");  // auto-reset, initially signaled
    WaitForSingleObject(a, INFINITE);                       // takes the signal
    assert(!((FfxLinuxHandle *)a)->signaled);
    HANDLE t = CreateThread(nullptr, 0, st_waiter, a, 0, nullptr);
    SetEvent(a);                                            // releases the waiter, which takes the signal
    WaitForSingleObject(t, INFINITE);                       // = join
    CloseHandle(t);
    assert(!((FfxLinuxHandle *)a)->signaled);
    HANDLE m = CreateEvent(nullptr, TRUE, FALSE, "manual");
    SetEvent(m);
    WaitForSingleObject(m, INFINITE);
    assert(((FfxLinuxHandle *)m)->signaled);                // manual-reset stays signaled
    CloseHandle(a);
    CloseHandle(m);
    InitializeCriticalSection(&st_cs);
    EnterCriticalSection(&st_cs);
    EnterCriticalSection(&st_cs);                           // re-entrant
    LeaveCriticalSection(&st_cs);
    LeaveCriticalSection(&st_cs);
    t = CreateThread(nullptr, 0, st_locker, nullptr, 0, nullptr);  // (would hang if still held)
    WaitForSingleObject(t, INFINITE);
    CloseHandle(t);
    DeleteCriticalSection(&st_cs);
    wchar_t w[4];
    wcscpy_s(w, L"abcdef");
    assert(!wcscmp(w, L"abc"));
    wchar_t name[32];
    swprintf(name, 32, L"%ls%d", L"Res", 2);
    assert(!wcscmp(name, L"Res2"));
    char n[8];
    size_t r = 0;
    wcstombs_s(&r, n, sizeof n, L"Res2", sizeof n);
    assert(!strcmp(n, "Res2") && r == 5);
    LARGE_INTEGER c0, c1, f;
    QueryPerformanceCounter(&c0);
    QueryPerformanceCounter(&c1);
    QueryPerformanceFrequency(&f);
    assert(c1.QuadPart >= c0.QuadPart && f.QuadPart == 1000000000);
    puts("ffx_linux_compat self-test ok");
    return 0;
}
#endif
