#include "../../staging_copy.h"
#include "../../render_policy.h"
#include "../../descriptor_cache.h"
#include "../../resource_policy.h"
// Executes recorded NVN command lists with Vulkan (port of renderer.m; same structure, same behaviour).
#include "../../diag/diag.h"
#include "../../kernel.h"
#include "../../platform.h"
#include <SDL3/SDL_cpuinfo.h>
#include <SDL3/SDL_timer.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../../kernel.h"
#include "shaders.h"
#include "vk.h"
#include "temporal.h"
void vk_watch(VkImage dst, const char *site);  // [XCDE] debugging: log writes to a watched image
static VkImage g_watch_img;
#include "../renderer.h"
#include "../../settings.h"
#include "../../env_cache.h"  // (last: cached getenv)

static void fb_stats_flush(void);  // (T161, below)
static uint64_t stat_build_waits, stat_build_wait_ns;  // (T174)
static void prewarm_pump(void);  // (T163, below)
static uint64_t stat_prewarm_progs, stat_prewarm_pipes;
static int pw_npending;
static uint64_t stat_prewarm_libs;  // [XCDE] T211 (stats line: "libs prewarmed")
#define PL_MAX 4096  // [XCDE] T218: (was 1024; an area registers ~1,000 programs at once)
static void *pl_pending[PL_MAX];  // [XCDE] T211: programs whose shader libraries are prebuilt once translated
static int pl_npending;
static int lv_npending;  // [XCDE] T212: variant library jobs waiting for their program (lv_pending, below)
// [XCDE] T136: paused transform feedback captures, by control buffer address (where NVN keeps the write position)
static struct { uint64_t ctrl, written; } xfb_ctrl[16];
static int n_xfb_ctrl;
static unsigned xfb_ctrl_rr;
static int xfb_ctrl_find(uint64_t ctrl) { for (int i = 0; i < n_xfb_ctrl; i++) if (xfb_ctrl[i].ctrl == ctrl) return i; return -1; }
static uint64_t g_draws_total;  // [XCDE] T240 (XCDE_TRACE_PRESENT: draws in each presented frame)
uint64_t vk_draws_total(void) { return g_draws_total; }
static uint64_t stat_draws, stat_skipped, stat_passes, stat_dispatches, stat_wait_ns, stat_flushes, stat_feedback_copies;
// [XCDE] Which skip site drops draws: counts per source line, the first few occurrences are logged.
static void skip_note(int line) {
    static int counts[8192];
    if (line >= 0 && line < 8192) {
        int n = ++counts[line];
        if ((n & (n - 1)) == 0) LOG("[vk] draw skipped at exec.c:%d (%d so far)""%c", line, n, 10);
    }
}
#define SKIPPED() (stat_skipped++, skip_note(__LINE__))
static uint64_t stat_upload_bytes;
static XcdeDescriptorEntry descriptor_cache[XCDE_DESC_SLOTS];
static uint64_t metric_desc_hits, metric_desc_updates, metric_gmem_upload, metric_gmem_readback;
// LARECOMP_VK_PROF=1: render-thread time per category in the fps log (exec = whole command lists, draw = draw(),
// sets = descriptor sets incl. cbuf copies, vtx = index scan + vertex copies, submit = vkQueueSubmit2)
static int prof = -1;
static uint64_t prof_exec, prof_draw, prof_sets, prof_vtx, prof_submit, prof_pre, prof_key, prof_dyn, prof_tex, prof_up, prof_gpu;
// [XCDE] (LARECOMP_VK_PROF) "pre" in parts: 0 prewarm + shader key + program, 1 fs hash / skips / debug, 2 T192 + feedback +
// snapshot invalidation, 3 barriers / epochs, 4 command buffer (begin_pass) + topology
static uint64_t prof_pre_part[5], prof_pp_, stat_pipekey_memo;
static uint64_t prof_ds_alloc, prof_ds_upd, stat_ds_writes, prof_bp, prof_bp_submit;  // [XCDE] (stage_set: set allocation / vkUpdateDescriptorSets ns, writes)  // (+ draws that reused the previous pipeline)
#define PRE_MARK(i) do { if (prof) { uint64_t n_ = SDL_GetTicksNS(); prof_pre_part[i] += n_ - prof_pp_; prof_pp_ = n_; } } while (0)
static VkQueryPool prof_qp;
static uint64_t prof_same_pipe, prof_same_prog;  // (profiler: draws repeating the previous draw's program / pipeline)
static _Atomic uint64_t stat_gpl_cached, stat_gpl_linked, stat_gpl_libs, stat_gpl_opt;  // [XCDE] T188 (stats line)
static uint32_t pipe_count(void);  // (stats line)
static uint64_t cum_exec, cum_wait, cum_draws, cum_submits;  // (profiling) never reset: hitch log
static uint32_t prof_end_line[16384];  // passes ended, by the exec.c line that ended them (0: submit / present)
static uint32_t prof_dirty_line[16384], rt_dirty_line;  // [XCDE] passes restarted by begin_pass, by the line that set rt_dirty
#define RT_DIRTY() (st.rt_dirty = true, rt_dirty_line = __LINE__)
uint64_t prof_views_created;  // textures.c vk_image_view cache misses
uint64_t prof_up_ns = ~0ull;  // textures.c upload checks (~0: profiling off)
#define PROF_T0() uint64_t prof_t0_ = prof ? SDL_GetTicksNS() : 0
#define PROF_ADD(v) do { if (prof) v += SDL_GetTicksNS() - prof_t0_; } while (0)
static _Atomic int stat_compiles;
// [XCDE] hitch log (device.c): cumulative pipeline-build and wait times, read at each present
static _Atomic uint64_t hc_cached_ns, hc_link_ns, hc_whole_ns, hc_cached_n, hc_link_n, hc_whole_n;
static uint64_t hc_wait_ns, hc_wait_n, hc_upload;
static uint64_t op_counts[CMD_COUNT];
bool g_debug_frame;
static char stats_text[2048];
static double vc_bytes_mb(void);  // [XCDE] T169 (memory report)
static int ring_chunks(void);
void vk_collect_dead(void);

// ================================================================== submissions
// A submission = optional upload command buffer + the main one, signalling the timeline with its number.
#define SUBS 8
typedef struct {
    VkCommandBuffer main, upload;
    bool main_begun, upload_begun, ts, upload_ts;  // ts: GPU timestamps written (profiling)
    uint64_t seq;
    uint64_t gpu_frame; // T260: native frame owning this timestamp pair
    VkDescriptorPool pools[16];
    int npools, cur_pool;
} Sub;
uint64_t renderer_present_count(void);
static Sub subs[SUBS];
static struct { uint64_t tag, ns; } gpu_frames[32];
static uint64_t gpu_last_tag;
static double gpu_last_ms=-1;
static unsigned gpu_timestamp_bits;
static bool compile_jobs_pending(void);
static void resolve_gpu_time(Sub *s, int index) {
    if (!s->ts) return;
    uint64_t t[4];
    if (vkGetQueryPoolResults(g_vk.dev, prof_qp, (uint32_t)index*4, s->upload_ts ? 4 : 2, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT)!=VK_SUCCESS) return;
    uint64_t ticks=xcde_timestamp_delta(t[0],t[1],gpu_timestamp_bits);
    if (s->upload_ts) ticks+=xcde_timestamp_delta(t[2],t[3],gpu_timestamp_bits);
    uint64_t ns=(uint64_t)(ticks*g_vk.props.limits.timestampPeriod);
    unsigned slot=(unsigned)(s->gpu_frame%32);
    if (gpu_frames[slot].tag!=s->gpu_frame+1) gpu_frames[slot].tag=s->gpu_frame+1, gpu_frames[slot].ns=0;
    gpu_frames[slot].ns+=ns;
    if (prof) prof_gpu+=ns;
    s->ts=false;
}
double vk_native_gpu_ms(void) {
    // T260: only completed submissions; no WAIT flag or extra timeline wait. Sum every piece of a native frame.
    if (!prof && g_settings.res_mode!=RES_DYNAMIC) return -1;
    if (compile_jobs_pending()) return -2; // T260: resolution cannot fix first-use compiler stalls
    if (!prof_qp) return -1;
    uint64_t done=vk_completed_seq(), frame=renderer_present_count();
    for (int i=0;i<SUBS;i++) if (subs[i].ts && subs[i].seq<=done) resolve_gpu_time(&subs[i],i);
    for (unsigned j=0;j<32;j++) {
        uint64_t tag=gpu_frames[j].tag;
        if (!tag || tag<=gpu_last_tag || tag>frame) continue;
        bool pending=false;
        for (int i=0;i<SUBS;i++) if (subs[i].ts && subs[i].gpu_frame+1==tag) pending=true;
        if (!pending) { gpu_last_tag=tag; gpu_last_ms=gpu_frames[j].ns/1e6; }
    }
    return gpu_last_tag && frame+1-gpu_last_tag<=12 ? gpu_last_ms : -1;
}
static int cur_sub;
static void ss_forget(void);
static void dyn_defaults(VkCommandBuffer cb);
static uint64_t submit_seq;  // last submitted
static pthread_mutex_t seq_lock = PTHREAD_MUTEX_INITIALIZER;
static bool in_pass;
static uint64_t tmp_scene_frame = ~0ull;  // [XCDE] T213: the last frame with a scene draw (jittered)
static TexBackend *pass_dbg_rt;  // [XCDE] colour 0 of the open pass (the debug dump must save the pass's own target, not st.rt's)

uint64_t vk_completed_seq(void) {
    uint64_t v = 0;
    VK_CHECK(vkGetSemaphoreCounterValue(g_vk.dev, g_vk.timeline, &v));  // [XCDE] T128 (upstream #127): not an unusable value
    return v;
}
// [XCDE] (overlap wait mode, from the page fault handler) sleeps until the timeline reaches seq - submitted work only
// (vk_ov_frame clamps the frame's sequence to the last submission). Gives up on any error: the caller re-checks.
static void vk_block_seq(uint64_t seq) {
    VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1, .pSemaphores = &g_vk.timeline,
                              .pValues = &seq};
    while (vkWaitSemaphores(g_vk.dev, &wi, 100000000ull) == VK_TIMEOUT) {}
}

// (LARECOMP_VK_PROF) waits on the GPU by cause: the caller names it in wait_why just before waiting
static __thread const char *wait_why;
// [XCDE] T240: where the render worker is (the stall watchdog prints it: the rare freeze while loading a save had the
// game waiting for the worker, which never answered, and the dump listed guest threads only)
const char *volatile g_rq_where = "not started";
volatile uint64_t g_rq_where_ns;
static __thread bool tl_rq_worker;  // (only the worker's own steps: game threads wait on the GPU too)
void vk_rq_at(const char *where) { if (tl_rq_worker) g_rq_where = where, g_rq_where_ns = SDL_GetTicksNS(); }
static struct { const char *why; uint64_t ns, n; } wait_tab[12];
static pthread_mutex_t wait_tab_lock = PTHREAD_MUTEX_INITIALIZER;
static void wait_account(uint64_t ns) {
    const char *why = wait_why ? wait_why : "other";
    wait_why = NULL;
    pthread_mutex_lock(&wait_tab_lock);
    for (int i = 0; i < 12; i++)
        if (!wait_tab[i].why || wait_tab[i].why == why) { wait_tab[i].why = why; wait_tab[i].ns += ns; wait_tab[i].n++; break; }
    pthread_mutex_unlock(&wait_tab_lock);
}

static void cq_resolve_done(uint64_t done);

// T252: the render thread owns readback memory; other waiters must not touch its cache state.
static void readback_sync(uint64_t completed);
static void ring_flush(uint64_t pending);
// [PERF-V5] the host-read barrier is only needed on submissions that really carry a GPU->CPU readback / guest-heap store
static atomic_bool host_read_pending;
void vk_wait_seq(uint64_t seq) {
    if (!seq) { wait_why = NULL; return; }
    VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1, .pSemaphores = &g_vk.timeline,
                              .pValues = &seq};
    uint64_t t0 = SDL_GetTicksNS();
    const char *was = g_rq_where;
    vk_rq_at(wait_why ? wait_why : "GPU wait");
    VkResult r = vkWaitSemaphores(g_vk.dev, &wi, 10ull * 1000000000ull);
    if (tl_rq_worker) g_rq_where = was;
    stat_wait_ns += SDL_GetTicksNS() - t0;
    cum_wait += SDL_GetTicksNS() - t0;
    wait_account(SDL_GetTicksNS() - t0);
    // [XCDE] T128 (upstream LASRecomp #127, hjqcan): a timed-out wait is fatal - the callers go on to reset descriptor
    // pools and command buffers the GPU may still be using
    if (r == VK_TIMEOUT) {
        extern void vk_devlost_dump(void);
        vk_devlost_dump();
        FATAL("[vk] GPU did not finish submission %llu within 10 s (hang or device lost)\n", (unsigned long long)seq);
    } else if (r == VK_SUCCESS) { readback_sync(seq); cq_resolve_done(seq); }  // (the game reads its timing counters after waiting)
    else {
        extern void vk_devlost_dump(void);
        vk_devlost_dump();
        vk_device_fault_report();
        FATAL("vkWaitSemaphores: %d (device lost)\n", r);
    }
}

uint64_t vk_pending_seq(void) { return submit_seq + 1; }

// ---- the game's GPU timing counters (nvnCommandBufferReportCounter, two per frame)
// The game times its GPU work with these and drops to 30 fps when frames look slow. They used to get the host clock
// at translation time, so our command translation counted as GPU time: busy scenes (outdoors, ~400 draws) looked slow
// and the game locked itself at 30 even on fast GPUs, at any resolution. Now each counter also records a GPU
// timestamp at its place in the command stream; once that submission is done (the game waits on its fence before
// reading - on the Switch the GPU writes them too) the GPU time replaces the provisional host value, in the host
// clock's terms (offset calibrated at each resolve: a timestamp can't be later than the moment its submission is seen
// done). LARECOMP_VK_COUNTER=cpu: host clock only; =frozen: one value per presented frame (test); =legacy: the host
// clock in ns instead of GPU ticks, as before (test: reproduces the locked 30).
#define CQ_MAX 64
typedef struct { VkQueryPool pool; uint32_t n; uint64_t seq; uint64_t *out[CQ_MAX]; } CounterQ;
static CounterQ cq[SUBS];
static pthread_mutex_t cq_lock = PTHREAD_MUTEX_INITIALIZER;
static int64_t cq_offset;  // host ns - GPU ns (the smallest seen)
// Counter timestamps are in the Switch GPU's ticks (614.4 MHz): the game converts the difference to ns itself
// (x 625 / 384). Writing ns made every GPU time look 1.63x as long.
static uint64_t ns_to_gpu_ticks(uint64_t ns) { return (uint64_t)((unsigned __int128)ns * 384 / 625); }
static bool cq_calibrated;
static int counter_mode = -1;  // 0 cpu, 1 frozen, 2 gpu
static uint64_t stat_counters_gpu;

static int vk_counter_mode(void) {
    if (counter_mode < 0 && diag_deterministic()) counter_mode = 1;  // [XCDE] T029: one virtual timestamp per frame
    if (counter_mode < 0) {
        const char *e = getenv("LARECOMP_VK_COUNTER");
        counter_mode = e && !strcmp(e, "cpu") ? 0 : e && !strcmp(e, "frozen") ? 1 : e && !strcmp(e, "legacy") ? 3 : 2;
        if (counter_mode == 2 && !g_vk.props.limits.timestampComputeAndGraphics) counter_mode = 0;
    }
    return counter_mode;
}

// ---- frame pacing: the game's own 30 fps mode when we can't hold 60
// The game advances its world one fixed step per frame in its 60 fps mode, and two steps per frame in its 30 fps mode
// (measured: Link walks twice as far per frame there) - so a 60-mode frame that takes 20 ms plays in slow motion, while
// the 30 mode plays at full speed. The game picks its mode from these counters (GPU time per frame), as on the Switch;
// but our frames can be slow for reasons its GPU time doesn't show (command translation, a slower CPU). So each
// presented frame's busy time is measured (present to present, minus the time the presenting thread was blocked in
// kernel waits and sleeps: idle - the game waits for vsync through its own threads); when
// frames don't fit 60 (12 of the last 60 over 16.2 ms), the counters report a frame of at least PACE_SLOW_NS: the game
// takes its 30 fps mode, at full speed. Back to the real times after 2 s with headroom (busy under 13 ms on average,
// at most 2 of the last 60 frames over). The game itself also tries 60 again now and then.
// LARECOMP_PACE=0: off.
#define PACE_SLOW_NS 20000000ull
extern __thread uint64_t g_blocked_ns;  // kernel.c: this thread's time blocked in waits / sleeps
static bool pace_slow;
static uint64_t stat_pace_over;

void vk_pace_frame(void) {
    if (diag_deterministic()) return;  // [XCDE] T029: real-time measurements must not steer the game
    static int on = -1;
    // (only for the game's own 30 fps: with the FPS option above 30 the frame interval is forced every frame, and a
    // "slow" counter would only make the game's internals believe the GPU is slow)
    if (on < 0) { const char *e = getenv("LARECOMP_PACE"); on = e ? atoi(e) != 0 : g_settings.fps == 30 && !getenv("XCDE_FPS_INTERVAL"); }
    static uint64_t last, last_wait;
    static uint8_t over[60];
    static int idx, nover, calm;
    static double avg;  // busy ms, moving average (~20 frames)
    uint64_t now = plat_now_ns(), w = g_blocked_ns;
    uint64_t p = now - last, wt = w - last_wait;
    bool first = !last;
    last = now, last_wait = w;
    if (first || p > 100000000ull) return;  // (loading, a hitch: not a pace)
    double busy = (p > wt ? p - wt : 0) / 1e6;
    avg = avg ? avg + (busy - avg) * 0.05 : busy;
    static int log = -1;  // LARECOMP_LOG_PACE=1: each frame's period, vsync wait and busy time
    if (log < 0) log = getenv("LARECOMP_LOG_PACE") != NULL;
    if (log) LOG("[pace] period %.2f blocked %.2f busy %.2f ms\n", p / 1e6, wt / 1e6, busy);
    int o = busy > 16.2;
    stat_pace_over += o;
    nover += o - over[idx];
    over[idx] = (uint8_t)o;
    idx = (idx + 1) % 60;
    if (!on) return;  // (measured and counted all the same: LARECOMP_LOG_PACE, the stats)
    if (!pace_slow && nover >= 12) {
        pace_slow = true, calm = 0;
        LOG("[pace] frames don't fit 60 fps (%d of the last 60 over 16.2 ms busy): the game's 30 fps mode\n", nover);
    } else if (pace_slow) {
        // headroom: busy well under a 60 fps frame on average, hardly a frame over (single spikes are normal play)
        calm = avg < 13.0 && nover <= 2 ? calm + 1 : 0;
        if (calm >= 120) {
            pace_slow = false, nover = 0;
            memset(over, 0, sizeof over);
            LOG("[pace] 2 s with headroom: back to the game's own choice\n");
        }
    }
}

// a pair's end counter (the 16 bytes after its start) is at least PACE_SLOW_NS after it while pace_slow
static uint64_t pace_end_counter(const uint64_t *start_entry, const uint64_t *entry, uint64_t value) {
    // [XCDE] test: XCDE_TEST_GPU_SLOW=<factor> scales each counter pair as if the GPU were that much slower (> 1: the
    // game lowers its dynamic resolution, as on slow GPUs - Intel iGPUs - which a fast GPU never shows) or faster
    // (< 1: a slow GPU keeps the full resolution; tells whether a slow GPU's problem comes from the resolution changes)
    static double slow = -1;
    if (slow < 0) slow = getenv("XCDE_TEST_GPU_SLOW") ? atof(getenv("XCDE_TEST_GPU_SLOW")) : 0;
    if (slow > 0 && slow != 1 && start_entry && (const uint8_t *)entry - (const uint8_t *)start_entry == 16 && value > start_entry[1])
        value = start_entry[1] + (uint64_t)((value - start_entry[1]) * slow);
    if (!pace_slow || !start_entry || (const uint8_t *)entry - (const uint8_t *)start_entry != 16) return value;
    uint64_t min = start_entry[1] + ns_to_gpu_ticks(PACE_SLOW_NS);
    return value < min ? min : value;
}

static void cq_resolve(int i) {  // (cq_lock held) submission slot i is done: its counters get their GPU times
    CounterQ *q = &cq[i];
    if (!q->n) return;
    uint64_t t[CQ_MAX];
    if (vkGetQueryPoolResults(g_vk.dev, q->pool, 0, q->n, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
        double per = g_vk.props.limits.timestampPeriod;
        int64_t cand = (int64_t)plat_now_ns() - (int64_t)((double)t[q->n - 1] * per);
        if (!cq_calibrated || cand < cq_offset) cq_offset = cand, cq_calibrated = true;
        static int log = -1;  // LARECOMP_LOG_COUNTER=1: the host (provisional) and GPU times of each counter
        if (log < 0) log = getenv("LARECOMP_LOG_COUNTER") != NULL;
        for (uint32_t k = 0; k < q->n; k++) {
            uint64_t gpu = ns_to_gpu_ticks((uint64_t)((int64_t)((double)t[k] * per) + cq_offset));
            if (log) LOG("[counter] seq %llu #%u at %llx host %.3f gpu %.3f ms\n", (unsigned long long)q->seq, k,
                         (unsigned long long)(uintptr_t)q->out[k], q->out[k][1] * 625.0 / 384 / 1e6, gpu * 625.0 / 384 / 1e6);
            q->out[k][1] = pace_end_counter(k ? q->out[k - 1] : NULL, q->out[k], gpu);
        }
        stat_counters_gpu += q->n;
    }
    q->n = 0;
}

// ---- [XCDE] T139: samples-passed counters (ResetCounter / ReportCounter type 1) as Vulkan occlusion queries.
// XCDE culls field characters and enemies with them (a bounding box drawn between a reset and a report, the count read
// back by the CPU); reported as 0, nobody was drawn in gameplay (T137). The count runs from a reset across draws, and a
// Vulkan query can't outlive its render pass, so each pass while counting holds one query (a "segment"); a report
// sums the segments since the reset. Resolved like the timing counters once the submission is done (the game waits
// on its fence before reading); until then - and when the segments can't be told (pool full, a reset in an earlier
// submission) - the counter says "visible". Deterministic mode (T029): always "visible" (GPU results must not steer
// the game); XCDE_OCCLUSION=0: always "visible" too.
#define OQ_MAX 4096
#define OQ_REPORTS 1024
#define OQ_VISIBLE (1ull << 20)
typedef struct { uint64_t *out; uint32_t from, to; } OccRep;
typedef struct { VkQueryPool pool; uint32_t n, nrep; uint64_t seq; OccRep rep[OQ_REPORTS]; } OccQ;
static OccQ oq[SUBS];
static bool oq_counting;  // a samples counter was reset (counting continues past reports)
static int oq_open = -1;  // the query open in the current render pass
static int oq_slot = -1;  // submission slot of the reset
static uint32_t oq_from;  // its first segment
static uint64_t stat_occ_reports, stat_occ_hidden;

static bool oq_enabled(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("XCDE_OCCLUSION"); on = !diag_deterministic() && !(e && atoi(e) == 0); }
    return on;
}

static void oq_begin(VkCommandBuffer cb) {  // (in a render pass, counting) open this pass's segment
    OccQ *q = &oq[cur_sub];
    if (oq_open >= 0 || !q->pool || q->n >= OQ_MAX) return;
    vkCmdBeginQuery(cb, q->pool, q->n, 0);
    oq_open = (int)q->n++;
}
static void oq_end(VkCommandBuffer cb) {  // (before the render pass ends)
    if (oq_open < 0) return;
    vkCmdEndQuery(cb, oq[cur_sub].pool, (uint32_t)oq_open);
    oq_open = -1;
}

static void oq_resolve(int i) {  // (cq_lock held) slot i is done: each report gets the sum of its segments
    OccQ *q = &oq[i];
    static uint64_t r[OQ_MAX];
    if (q->nrep && q->n && vkGetQueryPoolResults(g_vk.dev, q->pool, 0, q->n, sizeof(uint64_t) * q->n, r, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
        for (uint32_t k = 0; k < q->nrep; k++) {
            uint64_t sum = 0;
            for (uint32_t s = q->rep[k].from; s < q->rep[k].to; s++) sum += r[s];
            q->rep[k].out[0] = sum;
            stat_occ_hidden += !sum;
        }
        stat_occ_reports += q->nrep;
    }
    q->nrep = q->n = 0;
}

// ---- [XCDE] T143: XCDE_CHECK_EQ=1 - automatic detector for the black-grass class of bug (T134): a draw testing depth
// EQUAL against a pre-pass draws nothing when the two disagree. Each such draw gets an occlusion query (only when the
// game's own occlusion queries are off: deterministic runs); programs whose EQUAL draws never pass a sample are
// logged as "[check-eq]" with the frame. (Some may be legitimately hidden: a program seen in many frames with zero
// samples every time is the signal.)
#define CK_MAX 4096
#define CK_PROGS 1024
typedef struct { VkQueryPool pool; uint32_t n; uint64_t seq; uint16_t prog[CK_MAX]; } CheckQ;
static CheckQ ckq[SUBS];
static struct { uint64_t vs, fs, draws, zero, frames_seen, last_frame; bool told; } ck_prog[CK_PROGS];
static int ck_nprog;
static bool oq_enabled(void);
static bool ck_on(void) {
    static int on = -1;
    if (on < 0) on = getenv("XCDE_CHECK_EQ") && !oq_enabled();
    return on;
}
static void ck_resolve(int i) {  // (cq_lock held)
    CheckQ *q = &ckq[i];
    static uint64_t r[CK_MAX];
    extern uint64_t vk_frame_counter(void);
    if (q->n && vkGetQueryPoolResults(g_vk.dev, q->pool, 0, q->n, sizeof(uint64_t) * q->n, r, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
        for (uint32_t k = 0; k < q->n; k++) {
            typeof(ck_prog[0]) *p = &ck_prog[q->prog[k]];
            p->draws++;
            p->zero += !r[k];
            if (p->last_frame != vk_frame_counter()) { p->last_frame = vk_frame_counter(); p->frames_seen++; }
            if (!p->told && p->frames_seen >= 30 && p->zero == p->draws) {
                p->told = true;
                LOG("[check-eq] frame %llu: program vs %llx fs %llx - %llu EQUAL depth draws over %llu frames, none drew a pixel\n",
                    (unsigned long long)vk_frame_counter(), (unsigned long long)p->vs, (unsigned long long)p->fs,
                    (unsigned long long)p->draws, (unsigned long long)p->frames_seen);
            }
        }
    q->n = 0;
}

static void cq_resolve_done(uint64_t done) {  // every slot whose submission is done
    pthread_mutex_lock(&cq_lock);
    for (int i = 0; i < SUBS; i++) if (cq[i].n && cq[i].seq <= done) cq_resolve(i);
    for (int i = 0; i < SUBS; i++) if (oq[i].nrep && oq[i].seq <= done) oq_resolve(i);
    for (int i = 0; i < SUBS; i++) if (ckq[i].n && ckq[i].seq <= done) ck_resolve(i);
    pthread_mutex_unlock(&cq_lock);
}

static Sub *sub(void) {
    Sub *s = &subs[cur_sub];
    if (s->seq != submit_seq + 1) {  // slot reused: its previous submission must be done
        wait_why = "ring slot";
        vk_wait_seq(s->seq);
        cq_resolve_done(s->seq);
        resolve_gpu_time(s,cur_sub);
        for (int i = 0; i < s->npools; i++) vkResetDescriptorPool(g_vk.dev, s->pools[i], 0);
        s->cur_pool = 0;
        s->seq = submit_seq + 1;
        s->main_begun = s->upload_begun = s->upload_ts = false;
    }
    return s;
}

static void begin_cb(VkCommandBuffer *cb, bool *begun) {
    if (*begun) return;
    if (!*cb) {
        VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = g_vk.pool,
                                          .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
        VK_CHECK(vkAllocateCommandBuffers(g_vk.dev, &ai, cb));
    }
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VK_CHECK(vkBeginCommandBuffer(*cb, &bi));
    *begun = true;
}

void vk_external_state_changed(void) { ss_forget(); dyn_defaults(vk_cmd()); }

static void gp_mark(VkCommandBuffer cb, const char *fmt, ...);
// [XCDE] perf: a slot's query pools are reset from the CPU when the host-reset feature is there. vk_cmd runs after
// sub() waited for the slot's previous submission and its results were read, so nothing uses the queries. On RADV a
// command-buffer reset is a fill shader plus a cache flush - 14 times a frame for the 4096-query occlusion pool alone.
// XCDE_HOST_QUERY_RESET=0: reset in the command buffer, as before.
static void reset_query_pool(VkCommandBuffer cb, VkQueryPool pool, uint32_t n) {
    static int host = -1;
    if (host < 0) { const char *e = getenv("XCDE_HOST_QUERY_RESET"); host = g_vk.host_query_reset && !(e && atoi(e) == 0); }
    if (host) vkResetQueryPool(g_vk.dev, pool, 0, n);
    else vkCmdResetQueryPool(cb, pool, 0, n);
}
VkCommandBuffer vk_cmd(void) {
    Sub *s = sub();
    if (!s->main_begun) {
        begin_cb(&s->main, &s->main_begun);
        ss_forget();  // (a new command buffer starts with no state)
        // [XCDE] T240: every dynamic state the pipelines declare gets a defined value up front - draws with the stencil
        // test off never set the stencil op / masks, and a buffer without a pipeline clear left them unset.
        // NVIDIA tolerates it; on a Steam Deck the same pattern hung the GPU (PokeZ-ARecomp 13b5280).
        dyn_defaults(s->main);
        if (vk_counter_mode() == 2) {  // this slot's timing counter queries (reset outside any render pass)
            CounterQ *q = &cq[cur_sub];
            if (!q->pool) {
                VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                            .queryCount = CQ_MAX};
                if (vkCreateQueryPool(g_vk.dev, &qi, NULL, &q->pool) != VK_SUCCESS) q->pool = VK_NULL_HANDLE;
            }
            pthread_mutex_lock(&cq_lock);
            q->n = 0;  // (resolved when the slot was freed: sub())
            q->seq = s->seq;
            pthread_mutex_unlock(&cq_lock);
            if (q->pool) reset_query_pool(s->main, q->pool, CQ_MAX);
        }
        if (oq_enabled()) {  // [XCDE] T139: this slot's occlusion queries
            OccQ *q = &oq[cur_sub];
            if (!q->pool) {
                VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_OCCLUSION,
                                            .queryCount = OQ_MAX};
                if (vkCreateQueryPool(g_vk.dev, &qi, NULL, &q->pool) != VK_SUCCESS) q->pool = VK_NULL_HANDLE;
            }
            pthread_mutex_lock(&cq_lock);
            q->n = q->nrep = 0;  // (resolved when the slot was freed: sub())
            q->seq = s->seq;
            pthread_mutex_unlock(&cq_lock);
            if (q->pool) reset_query_pool(s->main, q->pool, OQ_MAX);
        }
        if (ck_on()) {  // [XCDE] T143: this slot's EQUAL-draw checks
            CheckQ *q = &ckq[cur_sub];
            if (!q->pool) {
                VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_OCCLUSION,
                                            .queryCount = CK_MAX};
                if (vkCreateQueryPool(g_vk.dev, &qi, NULL, &q->pool) != VK_SUCCESS) q->pool = VK_NULL_HANDLE;
            }
            pthread_mutex_lock(&cq_lock);
            q->n = 0;
            q->seq = s->seq;
            pthread_mutex_unlock(&cq_lock);
            if (q->pool) reset_query_pool(s->main, q->pool, CK_MAX);
        }
        gp_mark(s->main, "(submission start; includes GPU idle before it)");
        if (prof_qp && (prof || g_settings.res_mode==RES_DYNAMIC)) {
            vkCmdResetQueryPool(s->main, prof_qp, (uint32_t)cur_sub * 4, 2);
            vkCmdWriteTimestamp2(s->main, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, prof_qp, (uint32_t)cur_sub * 4);
            s->gpu_frame=renderer_present_count();
            s->ts = true;
        }
    }
    return s->main;
}

VkCommandBuffer vk_upload_cmd(void) {
    Sub *s = sub();
    if (!s->upload_begun) {
        begin_cb(&s->upload, &s->upload_begun);
        if (prof_qp && (prof || g_settings.res_mode==RES_DYNAMIC)) {
            vkCmdResetQueryPool(s->upload,prof_qp,(uint32_t)cur_sub*4+2,2);
            vkCmdWriteTimestamp2(s->upload,VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,prof_qp,(uint32_t)cur_sub*4+2);
            s->upload_ts=true;
        }
        // after everything submitted before it: a submission's commands may start while the previous one's still run,
        // and these copies overwrite what those may read (textures being replaced, vertex cache pages)
        vk_full_barrier(s->upload);
    }
    return s->upload;
}

void vk_full_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

static void ring_retire(void);
void vc_flush(void);

// ---- LARECOMP_GPUPROF_AT=<frame>: GPU time per pass / dispatch / copy of that frame, logged as "[gpuprof]".
// A timestamp (ALL_COMMANDS: after everything before it) at each mark; a row = the work since the previous mark.
bool g_gpuprof_frame;
static VkQueryPool gp_qp;
static int gp_n;
#define GP_MAX 8192  // [XCDE] (was 512: XCDE_GPUPROF_DRAWS marks every draw)
static char gp_label[GP_MAX][112];
static uint32_t gp_draws[GP_MAX];
// LARECOMP_VK_DEVLOST_TRACE=1: the last 64 GPU operations, dumped on device loss (with LARECOMP_SYNC_SUBMIT=1 the
// dump ends with the command list that faulted).
static char dl_ring[64][128];
static unsigned dl_n;
static int dl_on = -1;
static void dl_note(const char *fmt, ...) {
    if (dl_on < 0) dl_on = getenv("LARECOMP_VK_DEVLOST_TRACE") ? atoi(getenv("LARECOMP_VK_DEVLOST_TRACE")) : 0;
    if (!dl_on) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(dl_ring[dl_n++ & 63], sizeof dl_ring[0], fmt, ap);
    va_end(ap);
}
void vk_dl_note_upload(void *img, uint32_t w, uint32_t h, uint32_t lv) { dl_note("  upload %p %ux%u lv%u", img, w, h, lv); }
void vk_devlost_dump(void) {
    if (!dl_on) return;
    LOG("[vk] last GPU operations before the device loss (oldest first):\n");
    for (unsigned i = dl_n > 64 ? dl_n - 64 : 0; i < dl_n; i++) LOG("[vk]   %s\n", dl_ring[i & 63]);
}
static void gp_mark(VkCommandBuffer cb, const char *fmt, ...) {
    if (!g_gpuprof_frame || gp_n >= GP_MAX - 1) return;
    if (!gp_qp) {
        VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = GP_MAX};
        if (vkCreateQueryPool(g_vk.dev, &qi, NULL, &gp_qp) != VK_SUCCESS) { g_gpuprof_frame = false; return; }
    }
    if (!gp_n) vkCmdResetQueryPool(cb, gp_qp, 0, GP_MAX);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, gp_qp, (uint32_t)gp_n);
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(gp_label[gp_n], sizeof gp_label[0], fmt, ap);
    va_end(ap);
    gp_draws[gp_n] = 0;
    gp_n++;
}
void renderer_flush(void);
void vk_gpuprof_report(void) {
    if (!gp_n) return;
    renderer_flush();
    static uint64_t t[GP_MAX];
    if (vkGetQueryPoolResults(g_vk.dev, gp_qp, 0, (uint32_t)gp_n, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
        double per = g_vk.props.limits.timestampPeriod / 1e6, total = (t[gp_n - 1] - t[0]) * per;
        LOG("[gpuprof] %d marks, %.2f ms first to last (rows: the work after each mark, until the next)\n", gp_n, total);
        for (int i = 0; i + 1 < gp_n; i++) {
            double ms = (t[i + 1] - t[i]) * per;
            LOG("[gpuprof] %6.3f ms %5.1f%%  %s%s", ms, 100 * ms / (total > 0 ? total : 1), gp_label[i], gp_draws[i] ? "" : "\n");
            if (gp_draws[i]) LOG(", %u draws\n", gp_draws[i]);
        }
    }
    gp_n = 0;
}

// ---- [XCDE] perf: submission thread. vkQueueSubmit2 cost the render worker ~67 us each (14 a frame on the Deck: the
// kernel's buffer-list validation), ~5 % of its time. The worker now ends its command buffers and queues them here, in
// order, with their timeline value already assigned (waiting for a timeline value whose submission is still queued is
// legal: it just waits longer). Everything else that uses the queue - present (binary semaphores need their signal
// submitted first), vkDeviceWaitIdle - calls vk_submit_drain first. XCDE_SUBMIT_THREAD=0: submit inline, as before
// (default off outside Linux for now).
typedef struct { VkCommandBufferSubmitInfo cbs[2]; int n; VkSemaphore wait, signal; uint64_t value; } SubJob;
#define SQ_N 32
static SubJob sq[SQ_N];
static uint64_t sq_head, sq_tail;  // (sq_lock) queued / submitted
static pthread_mutex_t sq_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t sq_more = PTHREAD_COND_INITIALIZER, sq_done = PTHREAD_COND_INITIALIZER;
static void sq_submit(const SubJob *j) {
    VkSemaphoreSubmitInfo sig[2] = {{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = g_vk.timeline,
                                     .value = j->value, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT},
                                    {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = j->signal,
                                     .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT}};
    VkSemaphoreSubmitInfo wt = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = j->wait,
                                .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSubmitInfo2 si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .commandBufferInfoCount = (uint32_t)j->n, .pCommandBufferInfos = j->cbs,
                        .signalSemaphoreInfoCount = j->signal ? 2 : 1, .pSignalSemaphoreInfos = sig,
                        .waitSemaphoreInfoCount = j->wait ? 1 : 0, .pWaitSemaphoreInfos = &wt};
    VkResult sr = vkQueueSubmit2(g_vk.queue, 1, &si, VK_NULL_HANDLE);
    if (sr == VK_ERROR_DEVICE_LOST) vk_devlost_dump();
    VK_CHECK(sr);
}
static void *sq_thread(void *arg) {
    (void)arg;
    pthread_mutex_lock(&sq_lock);
    for (;;) {
        while (sq_tail == sq_head) pthread_cond_wait(&sq_more, &sq_lock);
        SubJob j = sq[sq_tail % SQ_N];
        pthread_mutex_unlock(&sq_lock);
        sq_submit(&j);
        pthread_mutex_lock(&sq_lock);
        sq_tail++;
        pthread_cond_broadcast(&sq_done);
    }
    return NULL;
}
static bool sq_on(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("XCDE_SUBMIT_THREAD");
#ifdef __linux__
        on = !e || atoi(e) != 0;
#else
        on = e && atoi(e) != 0;
#endif
        pthread_t t;
        if (on && pthread_create(&t, NULL, sq_thread, NULL) != 0) on = 0;
        if (on) { pthread_detach(t); LOG("[vk] submission thread on\n"); }
    }
    return on;
}
// every queued submission handed to the driver (before any other use of the queue)
void vk_submit_drain(void) {
    if (!sq_on()) return;
    pthread_mutex_lock(&sq_lock);
    while (sq_tail != sq_head) pthread_cond_wait(&sq_done, &sq_lock);
    pthread_mutex_unlock(&sq_lock);
}

// Submit the pending work (if any). wait/signal: binary semaphores for presentation (may be null).
void vk_submit_extra(VkSemaphore wait, VkSemaphore signal, VkCommandBuffer extra) {
    (void)extra;
    vk_end_pass();
    Sub *s = sub();
    if (!s->main_begun && !s->upload_begun && !wait && !signal) return;
    VkCommandBufferSubmitInfo cbs[2] = {0};
    int n = 0;
    vc_flush();
    ring_flush(submit_seq + 1);
    if (s->upload_begun) {
        vk_full_barrier(s->upload);  // uploads complete before anything in the main buffer reads them
        if (s->upload_ts) vkCmdWriteTimestamp2(s->upload,VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,prof_qp,(uint32_t)cur_sub*4+3);
        VK_CHECK(vkEndCommandBuffer(s->upload));
        cbs[n++] = (VkCommandBufferSubmitInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = s->upload};
    }
    vk_cmd();  // (an empty main buffer still carries the semaphores)
    if (s->ts) vkCmdWriteTimestamp2(s->main, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, prof_qp, (uint32_t)cur_sub * 4 + 1);
    // T252: make shader/transfer writes visible to the host after the timeline wait.
    // [PERF-V5] was unconditional on every submit (a full ALL_COMMANDS -> HOST flush); now only when this submission has host reads.
    if (atomic_exchange(&host_read_pending, false)) {
        VkMemoryBarrier2 host_read = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
        VkDependencyInfo host_dep = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &host_read};
        vkCmdPipelineBarrier2(s->main, &host_dep);
    }
    VK_CHECK(vkEndCommandBuffer(s->main));
    cbs[n++] = (VkCommandBufferSubmitInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = s->main};
    {
        PROF_T0();
        dl_note("submit %llu", (unsigned long long)submit_seq + 1);
        SubJob j = {.cbs = {cbs[0], cbs[1]}, .n = n, .wait = wait, .signal = signal, .value = submit_seq + 1};
        if (sq_on()) {
            pthread_mutex_lock(&sq_lock);
            while (sq_head - sq_tail == SQ_N) pthread_cond_wait(&sq_done, &sq_lock);
            sq[sq_head % SQ_N] = j;
            sq_head++;
            pthread_cond_signal(&sq_more);
            pthread_mutex_unlock(&sq_lock);
        } else {
            sq_submit(&j);
        }
        PROF_ADD(prof_submit);
    }
    pthread_mutex_lock(&seq_lock);  // (sequence and ring slot published together: renderer_fence_seq reads both)
    submit_seq++;
    cur_sub = (cur_sub + 1) % SUBS;
    pthread_mutex_unlock(&seq_lock);
    stat_flushes++;
    cum_submits++;
    ring_retire();
    vk_collect_dead();
}

void renderer_submit(void) { vk_submit_extra(VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE); }

void renderer_flush(void) {
    renderer_submit();
    if (!wait_why) wait_why = "flush";
    vk_wait_seq(submit_seq);
}

void renderer_sync(void) {
    pthread_mutex_lock(&seq_lock);
    uint64_t s = submit_seq;
    pthread_mutex_unlock(&seq_lock);
    wait_why = "QueueFinish / sync all";
    vk_wait_seq(s);
}

uint64_t renderer_fence_seq(void) {
    pthread_mutex_lock(&seq_lock);
    // [XCDE] T128 (upstream LASRecomp #108, hjqcan): the slot after a submission keeps its previous generation's
    // begun flags until sub() resets it - only a slot that belongs to the next submission counts as pending work
    const Sub *s = &subs[cur_sub];
    uint64_t q = submit_seq + (s->seq == submit_seq + 1 && (s->main_begun || s->upload_begun) ? 1 : 0);
    pthread_mutex_unlock(&seq_lock);
    return q;
}

void renderer_wait_seq(uint64_t seq) {
    if (!seq) return;
    pthread_mutex_lock(&seq_lock);
    uint64_t s = submit_seq;
    pthread_mutex_unlock(&seq_lock);
    wait_why = seq > s ? "SyncWait (fence not submitted yet)" : "SyncWait";
    static int lg = -1;  // (diagnostics) XCDE_LOG_SYNCWAIT=1: which submission the game waits for, against the newest, every 60th
    if (lg < 0) lg = getenv("XCDE_LOG_SYNCWAIT") != NULL;
    uint64_t t0 = lg ? plat_now_ns() : 0, done0 = lg ? vk_completed_seq() : 0;
    vk_wait_seq(seq > s ? s : seq);  // not submitted yet (other thread): everything submitted so far
    if (lg) {
        static uint64_t n, sum_ns, behind;
        sum_ns += plat_now_ns() - t0; behind += s - (seq > s ? s : seq); n++;
        if (n % 60 == 0) {
            LOG("[syncwait] waits for submission %llu, newest %llu (%llu behind), done before %llu; last 60: avg %.2f ms, avg %.1f behind\n",
                (unsigned long long)seq, (unsigned long long)s, (unsigned long long)(s - (seq > s ? s : seq)), (unsigned long long)done0,
                sum_ns / 60 / 1e6, behind / 60.0);
            sum_ns = behind = 0;
        }
    }
}

VkDescriptorSet vk_alloc_set(VkDescriptorSetLayout layout) {
    Sub *s = sub();
    for (;;) {
        if (s->cur_pool == s->npools) {
            if (s->npools == 16) FATAL("descriptor pools exhausted\n");
            VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 32768}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096},
                                            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 32768},
                                            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_SAMPLER, 1024},
                                            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024}};  // [XCDE] storage images (SULD/SUST)
            VkDescriptorPoolCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 8192,
                                             .poolSizeCount = 6, .pPoolSizes = sizes};
            VK_CHECK(vkCreateDescriptorPool(g_vk.dev, &ci, NULL, &s->pools[s->npools++]));
        }
        VkDescriptorSetAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                          .descriptorPool = s->pools[s->cur_pool], .descriptorSetCount = 1, .pSetLayouts = &layout};
        VkDescriptorSet set;
        VkResult r = vkAllocateDescriptorSets(g_vk.dev, &ai, &set);
        if (r == VK_SUCCESS) return set;
        s->cur_pool++;
    }
}

// ================================================================== transient memory ring
// 64 MB host-visible chunks (device-local when the GPU exposes resizable BAR); a chunk is reused once every
// submission that allocated from it has completed.
#define CHUNK_SIZE (64ull << 20)
typedef struct { VkBuffer buf; VmaAllocation mem; uint8_t *ptr; uint64_t size, used, seq; } Chunk;
static Chunk *chunks;
static int nchunks, active = -1;
static _Thread_local VmaAllocation readback_allocation;
static _Thread_local uint64_t readback_seq;
static void ring_flush(uint64_t pending) {
    for (int i = 0; i < nchunks; ++i)
        if (chunks[i].seq == pending && chunks[i].used)
            VK_CHECK(vmaFlushAllocation(g_vk.vma, chunks[i].mem, 0, chunks[i].used));
}
static void readback_sync(uint64_t completed) {
    if (readback_allocation && readback_seq && readback_seq <= completed) {
        VK_CHECK(vmaInvalidateAllocation(g_vk.vma, readback_allocation, 0, VK_WHOLE_SIZE));
        readback_seq = 0;
    }
}
static int ring_chunks(void) { return nchunks; }

static int new_chunk(uint64_t size) {
    Chunk *expanded = realloc(chunks, (nchunks + 1) * sizeof *chunks);
    if (!expanded) FATAL("[vk] cannot grow transient allocation tracking\n");
    chunks = expanded;
    Chunk *c = &chunks[nchunks];
    memset(c, 0, sizeof *c);
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
                             .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    if (getenv("LARECOMP_VK_RING_SYSMEM")) ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;  // (test: a GPU without resizable BAR)
    VmaAllocationInfo info;
    uint64_t t0 = SDL_GetTicksNS();
    VK_CHECK(vmaCreateBuffer(g_vk.vma, &bi, &ai, &c->buf, &c->mem, &info));
    c->ptr = info.pMappedData;
    c->size = size;
    LOG("[vk] transient chunk %d: %llu MB in %.1f ms\n", nchunks, (unsigned long long)(size >> 20), (SDL_GetTicksNS() - t0) / 1e6);
    if (!nchunks) {  // where the ring lives: device-local (resizable BAR) or system memory read across PCIe
        VkMemoryPropertyFlags f;
        vmaGetMemoryTypeProperties(g_vk.vma, info.memoryType, &f);
        LOG("[vk] transient ring: memory type %u (%s%s%s)\n", info.memoryType, f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT ? "device-local " : "system memory ",
            f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ? "host-visible " : "", f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT ? "cached" : "");
    }
    return nchunks++;
}

static void ring_retire(void) {}  // chunks are recycled lazily in vk_transient

VkSpan vk_transient(uint64_t size, uint64_t align) {
    if (align < 16) align = 16;
    uint64_t pending = vk_pending_seq();
    if (active >= 0) {
        Chunk *c = &chunks[active];
        uint64_t off = (c->used + align - 1) & ~(align - 1);
        if (off + size <= c->size) {
            c->used = off + size;
            c->seq = pending;
            stat_upload_bytes += size; hc_upload += size;
            return (VkSpan){c->buf, off, c->ptr + off};
        }
    }
    uint64_t done = vk_completed_seq();
    int pick = -1;
    for (int i = 0; i < nchunks; i++)
        if (i != active && chunks[i].seq <= done && chunks[i].size >= size) { pick = i; break; }
    if (pick < 0) pick = new_chunk(size > CHUNK_SIZE ? size : CHUNK_SIZE);
    active = pick;
    chunks[pick].used = size;
    chunks[pick].seq = pending;
    stat_upload_bytes += size; hc_upload += size;
    return (VkSpan){chunks[pick].buf, 0, chunks[pick].ptr};
}

// GPU -> CPU copies: the ring is write-combined (often device-local over the BAR), where CPU reads crawl (a
// 1280x720 screenshot took 7 s). Readbacks use one host-cached buffer instead. Every user waits for the GPU and
// reads the data before the next readback (render thread), so a single grow-only buffer suffices.
VkSpan vk_readback(uint64_t size) {
    static VkBuffer buf;
    static VmaAllocation mem;
    static uint8_t *ptr;
    static uint64_t cap;
    if (size > cap) {
        if (buf) { vk_wait_seq(submit_seq); vmaDestroyBuffer(g_vk.vma, buf, mem); }
        cap = (size + (16 << 20) - 1) & ~((16ull << 20) - 1);
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = cap, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
        VmaAllocationInfo info;
        VK_CHECK(vmaCreateBuffer(g_vk.vma, &bi, &ai, &buf, &mem, &info));
        ptr = info.pMappedData;
    }
    readback_allocation = mem; readback_seq = vk_pending_seq();
    atomic_store(&host_read_pending, true);
    return (VkSpan){buf, 0, ptr};
}

VkSpan vk_transient_copy(const void *data, uint64_t size, uint64_t align) {
    VkSpan s = vk_transient(size ? size : 16, align);
    if (size) memcpy(s.ptr, data, size);
    return s;
}

// Per-draw constants that usually repeat the previous draw's (scale table, tint, driver constants): reuse that copy
// while it's identical, within one submission. slot: one per kind of data.
enum { DEDUPE_RSCALE_V, DEDUPE_RSCALE_F, DEDUPE_TINT, DEDUPE_DRV_V, DEDUPE_DRV_F, DEDUPE_DRV_C, DEDUPE_SLOTS };
static VkSpan copy_dedupe(int slot, const void *data, uint64_t size, uint64_t align) {
    static struct { uint64_t seq, size; uint8_t data[0x800]; VkSpan span; } last[DEDUPE_SLOTS];
    uint64_t seq = vk_pending_seq();
    if (size > sizeof last[0].data) return vk_transient_copy(data, size, align);
    if (last[slot].seq == seq && last[slot].size == size && !memcmp(last[slot].data, data, size)) return last[slot].span;
    VkSpan s = vk_transient_copy(data, size, align);
    last[slot].seq = seq; last[slot].size = size; memcpy(last[slot].data, data, size); last[slot].span = s;
    return s;
}

// [XCDE] T240 XCDE_WATCH_MEM=<hex addr>:<hex size>: GPU commands writing into that guest range are logged (inline
// uniform updates, buffer copies), and device.c logs its first words at each present (who wrote a stale constant buffer)
static uint64_t wm_addr, wm_size;
static void watch_mem_note(const char *what, uint64_t dst, uint64_t n) {
    static int init;
    if (!init) { init = 1; const char *e = getenv("XCDE_WATCH_MEM"); if (e) sscanf(e, "%llx:%llx", (unsigned long long *)&wm_addr, (unsigned long long *)&wm_size); }
    if (!wm_size || dst >= wm_addr + wm_size || dst + n <= wm_addr) return;
    extern uint64_t vk_frame_counter(void);
    LOG("[watchmem] frame %llu: %s writes %llx +%llx%c", (unsigned long long)vk_frame_counter(), what, (unsigned long long)dst, (unsigned long long)n, 10);
}
// Guest memory copied once per submission: draws binding the same range share the copy.
#define GCOPY_TAB 4096
static struct { uint64_t addr, size, seq; VkSpan span; } gcopy[GCOPY_TAB];
static uint64_t gcopy_gen;  // bumped to drop every shared copy (guest memory written mid-list)
static void guest_copy_flush(void) { gcopy_gen++; }
static VkSpan guest_copy(uint64_t addr, uint64_t size) {
    uint64_t seq = vk_pending_seq() + (gcopy_gen << 40), h = (addr * 0x9E3779B97F4A7C15ull) ^ (size * 0xC2B2AE3D27D4EB4Full);
    uint32_t slot = (uint32_t)(h ^ (h >> 32)) & (GCOPY_TAB - 1);
    if (gcopy[slot].seq == seq && gcopy[slot].addr == addr && gcopy[slot].size == size) return gcopy[slot].span;
    VkSpan s = vk_transient_copy((const void *)(uintptr_t)addr, size, 256);
    gcopy[slot].addr = addr; gcopy[slot].size = size; gcopy[slot].seq = seq; gcopy[slot].span = s;
    return s;
}

// ================================================================== worker threads (shader / pipeline builds)
// First-use compiles run off the render thread; draws needing them are skipped until ready (as on Metal).
// LARECOMP_SYNC_SHADERS=1 builds on first use, blocking (deterministic tests).
typedef struct Job { void (*fn)(void *); void *arg; struct Job *next; } Job;
static Job *jobs_head, *jobs_tail;
static Job *ljobs_head, *ljobs_tail;  // [XCDE] T188: low-priority jobs (optimized pipelines replacing fast-linked ones)
static pthread_mutex_t jobs_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t jobs_cond = PTHREAD_COND_INITIALIZER;

// [XCDE] shader / pipeline build threads get a large stack: deep compiler recursion (a driver compile inside a pipeline build)
// can overflow the default 1 MB Windows thread stack. (Idea from WindowSmash's notes: its workers use 64 MiB stacks.)
// [XCDE] On Windows an explicit pthread stack size is committed memory: those threads take the executable's default
// stack instead, a 256 MB reservation committed on use (scripts/link_gfx.sh -Wl,--stack; see thread_start in kernel.c).
static int big_stack_thread(pthread_t *t, void *(*fn)(void *)) {
    pthread_attr_t at;
    pthread_attr_init(&at);
#ifndef _WIN32
    pthread_attr_setstacksize(&at, 64ull << 20);
#endif
    int r = pthread_create(t, &at, fn, NULL);
    pthread_attr_destroy(&at);
    return r;
}

// [XCDE] T240: build work not done yet (vk_prewarm_hold): normal jobs queued / running, translations queued / running
static _Atomic int g_jobs_queued, g_jobs_busy, g_xlat_queued;
static bool compile_jobs_pending(void) {
    return atomic_load(&g_jobs_queued)>0 || atomic_load(&g_jobs_busy)>0 || atomic_load(&g_xlat_queued)>0;
}
static void *worker(void *a) {
    (void)a;
    { extern void plat_thread_name(const char *), plat_thread_background(void); plat_thread_name("shader worker"); plat_thread_background(); }
    for (;;) {
        pthread_mutex_lock(&jobs_lock);
        while (!jobs_head && !ljobs_head) pthread_cond_wait(&jobs_cond, &jobs_lock);
        Job *j;
        bool normal = jobs_head != NULL;
        if (jobs_head) {
            j = jobs_head;
            jobs_head = j->next;
            if (!jobs_head) jobs_tail = NULL;
        } else {  // [XCDE] T188: background work only when nothing else waits
            j = ljobs_head;
            ljobs_head = j->next;
            if (!ljobs_head) ljobs_tail = NULL;
        }
        if (normal) atomic_fetch_add(&g_jobs_busy, 1), atomic_fetch_sub(&g_jobs_queued, 1);
        pthread_mutex_unlock(&jobs_lock);
        j->fn(j->arg);
        if (normal) atomic_fetch_sub(&g_jobs_busy, 1);
        free(j);
    }
    return NULL;
}

static bool sync_shaders(void) {
    static int v = -1;
    if (v < 0) v = getenv("LARECOMP_SYNC_SHADERS") != NULL || diag_deterministic();  // [XCDE] T029: builds on first use
    return v;
}

// Shader translation: one dedicated thread (the translator isn't re-entrant - translate.c's RSC2 string ring - and
// translating on the game thread cost up to ~90 ms per frame on entering an area). Compiles go on to the workers.
static pthread_mutex_t translate_lock = PTHREAD_MUTEX_INITIALIZER;  // (compute shaders translate on the game thread)
// [XCDE] A thread that needs a translation now (a draw waiting for its program, a compute dispatch) goes before the
// translator thread's queue. The lock isn't fair: on entering an area the game registers hundreds of programs, the
// translator thread translates their variants back to back (T163 prewarm, ~0.1-0.5 ms each) and re-took the lock
// each time before the waiting draw got it - measured 98 ms waits for the lock with 0.1 ms of translation of its own
// (frame drops to ~50 fps when entering a new part of Colony 9). The translator now lets urgent callers in first.
static _Atomic int xlat_urgent;
static _Thread_local bool tl_translator;
static void xlat_lock(void) {
    if (tl_translator) { while (atomic_load(&xlat_urgent)) sched_yield(); pthread_mutex_lock(&translate_lock); return; }
    atomic_fetch_add(&xlat_urgent, 1);
    pthread_mutex_lock(&translate_lock);
    atomic_fetch_sub(&xlat_urgent, 1);
}
static Job *tjobs_head, *tjobs_tail;
static pthread_mutex_t tjobs_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t tjobs_cond = PTHREAD_COND_INITIALIZER;
static void *translator(void *a) {
    (void)a;
    tl_translator = true;
    for (;;) {
        pthread_mutex_lock(&tjobs_lock);
        while (!tjobs_head) pthread_cond_wait(&tjobs_cond, &tjobs_lock);
        Job *j = tjobs_head;
        tjobs_head = j->next;
        if (!tjobs_head) tjobs_tail = NULL;
        pthread_mutex_unlock(&tjobs_lock);
        j->fn(j->arg);
        atomic_fetch_sub(&g_xlat_queued, 1);
        free(j);
    }
    return NULL;
}
static bool sync_shaders(void);
static void run_translate(void (*fn)(void *), void *arg) {
    if (sync_shaders()) { fn(arg); return; }
    static bool started;
    if (!started) { started = true; pthread_t t; big_stack_thread(&t, translator); }
    Job *j = malloc(sizeof *j);
    if (!j) FATAL("[vk] cannot allocate shader worker job\n");
    j->fn = fn; j->arg = arg; j->next = NULL;
    atomic_fetch_add(&g_xlat_queued, 1);
    pthread_mutex_lock(&tjobs_lock);
    if (tjobs_tail) tjobs_tail->next = j; else tjobs_head = j;
    tjobs_tail = j;
    pthread_cond_signal(&tjobs_cond);
    pthread_mutex_unlock(&tjobs_lock);
}

static __thread bool tl_inline_jobs;  // [XCDE] T187: this thread runs its jobs itself (precompilation threads)
static void run_job(void (*fn)(void *), void *arg) {
    if (sync_shaders() || tl_inline_jobs) { fn(arg); return; }
    static bool started;
    if (!started) {
        started = true;
        // first entry into an area builds dozens of pipelines at once: use most of the CPU (draws wait for them)
        extern int plat_cpu_count(void);
        int cpus = plat_cpu_count();
        int n = xcde_shader_workers(cpus, SDL_GetSystemRAM(),
                  g_vk.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, g_settings.gpu_performance);
        static int logged;
        if (!logged++) LOG("[vk] %d shader workers (%d logical CPUs)\n", n, cpus);
        if (n > 16) n = 16;
        for (int i = 0; i < n; i++) { pthread_t t; big_stack_thread(&t, worker); }
    }
    Job *j = malloc(sizeof *j);
    if (!j) FATAL("[vk] cannot allocate shader worker job\n");
    j->fn = fn; j->arg = arg; j->next = NULL;
    atomic_fetch_add(&g_jobs_queued, 1);
    pthread_mutex_lock(&jobs_lock);
    if (jobs_tail) jobs_tail->next = j; else jobs_head = j;
    jobs_tail = j;
    pthread_cond_signal(&jobs_cond);
    pthread_mutex_unlock(&jobs_lock);
}

// [XCDE] T188: a job the workers take only when the normal queue is empty (nobody waits for it). Queued only from a
// worker's own job, so the workers exist.
static void run_job_low(void (*fn)(void *), void *arg) {
    Job *j = malloc(sizeof *j);
    if (!j) FATAL("[vk] cannot allocate shader worker job\n");
    j->fn = fn; j->arg = arg; j->next = NULL;
    pthread_mutex_lock(&jobs_lock);
    if (ljobs_tail) ljobs_tail->next = j; else ljobs_head = j;
    ljobs_tail = j;
    pthread_cond_signal(&jobs_cond);
    pthread_mutex_unlock(&jobs_lock);
}

// [XCDE] T174: a draw whose program or pipeline is still being built waits for it instead of being dropped. Dropped,
// an object's colour pass went missing while its depth prepass had drawn: the object flashed black for a frame (user
// report: grass on the title screen, characters and enemies the first time they show up; ~18,000 draws dropped in 6
// minutes of field play). The job is taken out of its queue and run here if no worker has started it yet, else this
// thread waits for that worker. The disk pipeline cache makes most builds a cache hit; the cost is a first-time hitch,
// which shader precompilation (T163) will remove. LARECOMP_ASYNC_SKIP=1: drop the draw as before.
static Job *take_job(Job **head, Job **tail, pthread_mutex_t *lk, void *arg) {
    pthread_mutex_lock(lk);
    Job *prev = NULL, *j = *head;
    while (j && j->arg != arg) prev = j, j = j->next;
    if (j) {
        if (prev) prev->next = j->next; else *head = j->next;
        if (*tail == j) *tail = prev;
    }
    pthread_mutex_unlock(lk);
    return j;
}
static uint64_t g_last_wait_ns;  // (the last wait_ready that waited: build-wait log)
static bool wait_ready(_Atomic int *state, void *arg) {
    static int skip = -1;
    if (skip < 0) skip = getenv("LARECOMP_ASYNC_SKIP") != NULL;
    g_last_wait_ns = 0;
    if (atomic_load(state) != 0 || skip) return atomic_load(state) == 1;
    uint64_t t0 = SDL_GetTicksNS();
    const char *was = g_rq_where;
    vk_rq_at("waiting for a shader / pipeline build");
    while (atomic_load(state) == 0) {
        Job *j = take_job(&tjobs_head, &tjobs_tail, &tjobs_lock, arg);
        bool xl = j != NULL;
        if (!j && (j = take_job(&jobs_head, &jobs_tail, &jobs_lock, arg))) atomic_fetch_sub(&g_jobs_queued, 1);
        if (j) { j->fn(j->arg); if (xl) atomic_fetch_sub(&g_xlat_queued, 1); free(j); continue; }
        if (pw_npending || pl_npending || lv_npending) prewarm_pump();  // (T163: a prewarmed pipeline waits there until its program is ready)
        sched_yield();
    }
    if (tl_rq_worker) g_rq_where = was;
    stat_build_waits++;
    g_last_wait_ns = SDL_GetTicksNS() - t0;
    stat_build_wait_ns += g_last_wait_ns;
    hc_wait_ns += g_last_wait_ns; hc_wait_n++;
    return atomic_load(state) == 1;
}

// [PIPE-BUDGET] A draw waits for its pipeline build (above). On a GPU where the driver compiles whole pipelines slowly
// (AMD integrated: the pipeline libraries are off, see device.c) a big effect needs dozens of new pipelines at once and
// the render thread waited for all of them: 1 fps and the PC frozen. Per frame, after XCDE_PIPE_WAIT_MS of waiting
// (default 40 on AMD iGPU, 0 = no limit elsewhere) the draws whose pipeline isn't ready are skipped instead: the
// workers keep building them and they appear a few frames later. XCDE_PIPE_WAIT_MS=0 gives the old behaviour.
uint64_t renderer_present_count(void);
static uint64_t pb_frame, pb_used_ns;
// [AMD-IGPU-FIX] Three fragment shaders (hashes 53329ef60c533b43, 36580ad1389195c3 -> small 224..360 targets, ~80-120 ms
// each, and 071492ad778cf10c -> a 1208x680 target, ~1000 ms) are each drawn as a single 36-vertex box (volume effects, only
// in some areas). On AMD integrated GPUs (Ryzen Vega, Beelink EQR5) they cost ~1.2-1.4 us per pixel and dropped the game
// from 30 fps to 5 fps or even 0.4 fps (GPU profiles). Skipping them restored 30 fps and the missing effects were not
// noticeable. Off for everything that is not an AMD integrated GPU. XCDE_AMD_KEEP_HEAVY=1 draws them anyway.
// [FIX-TEX] Una textura nula, vacia o de un tipo que el shader no puede leer se sustituye por una textura "dummy". Antes solo se
// avisaba con XCDE_GLITCH_WATCH=1; ahora cada combinacion distinta (shader + ranura) queda en last_game.log una vez, para
// saber que shader dibuja con texturas sustitutas. Cuesta una busqueda en una tabla de 128 entradas, solo en ese caso raro.
static void dummy_report(const char *why, int stage, int slot, uint32_t tic, int shader_type, uint64_t fs, uint32_t target, uint32_t fmt, uint32_t w, uint32_t h) {
    static uint64_t seen[128];
    static int n;
    uint64_t key = (fs * 0x9E3779B97F4A7C15ull) ^ ((uint64_t)(stage + 1) << 56) ^ ((uint64_t)(slot + 1) << 48) ^ ((uint64_t)tic << 8) ^ (uint64_t)shader_type;
    if (!key) key = 1;
    for (int i = 0; i < n; i++) if (seen[i] == key) return;
    if (n >= 128) return;
    seen[n++] = key;
    LOG("[tex-fix] %s: stage %d unit %d pool slot %u, shader type %d, texture target %u fmt %#x %ux%u, fs %llx%c", why, stage, slot, tic, shader_type,
        target, fmt, w, h, (unsigned long long)fs, 10);
}
// [HUNT] "modo caceria" (XCDE_HUNT=1, o si existe skip_fs.txt junto al .exe): sirve para encontrar el fragment shader que pinta
// algo mal (p. ej. los toldos azules) y dejarlo saltado para siempre.
//   F6 / F7: shader anterior / siguiente (Shift: de 10 en 10) - se SALTA ese shader, mira si desaparece lo azul
//   F8: lo encontre -> guarda el hash en skip_fs.txt (se salta siempre, tambien sin XCDE_HUNT)
//   F5: dejar de saltar
// Los shaders son los distintos fragment shaders dibujados en el frame anterior, ordenados por hash.
volatile int g_hunt_step, g_hunt_save, g_hunt_reset;
static uint64_t g_skipfs[64];
static int g_nskipfs = -1;
static uint64_t hunt_cur[512], hunt_prev[512];
static int hunt_ncur, hunt_nprev, hunt_idx = -1;
static uint64_t hunt_frame = ~0ull, hunt_target, hunt_last;
static int hunt_on(void) {
    static int v = -1;
    if (v < 0) {
        v = getenv("XCDE_HUNT") != NULL;
        FILE *f = fopen("skip_fs.txt", "r");
        if (f) { v = 1; fclose(f); }
    }
    return v;
}
static void hunt_load_file(void) {
    FILE *f = fopen("skip_fs.txt", "r");
    if (!f) return;
    char ln[128];
    while (fgets(ln, sizeof ln, f) && g_nskipfs < 64) {
        uint64_t h = strtoull(ln, NULL, 16);
        if (h) { g_skipfs[g_nskipfs++] = h; LOG("[hunt] skip_fs.txt: fragment shader %016llx se salta siempre%c", (unsigned long long)h, 10); }
    }
    fclose(f);
}
static int hunt_cmp(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}
static bool hunt_draw(uint64_t h) {
    uint64_t f = renderer_present_count();
    if (f != hunt_frame) {  // new frame: last frame's distinct shaders become the list to step through
        hunt_frame = f;
        memcpy(hunt_prev, hunt_cur, (size_t)hunt_ncur * sizeof hunt_cur[0]);
        hunt_nprev = hunt_ncur; hunt_ncur = 0; hunt_last = 0;
        qsort(hunt_prev, (size_t)hunt_nprev, sizeof hunt_prev[0], hunt_cmp);
        if (g_hunt_reset) { g_hunt_reset = 0; hunt_idx = -1; hunt_target = 0; LOG("[hunt] saltos desactivados%c", 10); }
        if (g_hunt_step && hunt_nprev) {
            int st = g_hunt_step; g_hunt_step = 0;
            hunt_idx = hunt_idx < 0 ? (st > 0 ? st - 1 : hunt_nprev + st) : hunt_idx + st;
            if (hunt_idx < 0) hunt_idx = 0;
            if (hunt_idx >= hunt_nprev) hunt_idx = hunt_nprev - 1;
            hunt_target = hunt_prev[hunt_idx];
            LOG("[hunt] %d/%d: se salta el fragment shader %016llx%c", hunt_idx + 1, hunt_nprev, (unsigned long long)hunt_target, 10);
        }
        if (g_hunt_save) {
            g_hunt_save = 0;
            if (hunt_target && g_nskipfs < 64) {
                FILE *o = fopen("skip_fs.txt", "a");
                if (o) { fprintf(o, "%016llx\n", (unsigned long long)hunt_target); fclose(o); }
                g_skipfs[g_nskipfs++] = hunt_target;
                LOG("[hunt] GUARDADO en skip_fs.txt: fragment shader %016llx%c", (unsigned long long)hunt_target, 10);
                hunt_target = 0; hunt_idx = -1;
            }
        }
    }
    if (h && h != hunt_last) {
        hunt_last = h;
        int i = 0;
        while (i < hunt_ncur && hunt_cur[i] != h) i++;
        if (i == hunt_ncur && hunt_ncur < 512) hunt_cur[hunt_ncur++] = h;
    }
    return hunt_target && h == hunt_target;
}

static int amd_igpu_heavy_skip(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("XCDE_AMD_KEEP_HEAVY");
        v = !(e ? atoi(e) : g_settings.gpu_keep_heavy) && g_vk.props.vendorID == 0x1002 && g_vk.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    }
    return v;
}
static bool pipe_wait_over_budget(void) {
    static int64_t lim_ms = -1;
    if (lim_ms < 0) {
        const char *e = getenv("XCDE_PIPE_WAIT_MS");
        lim_ms = e ? atoi(e) : 0; // T260: correctness by default; never hide geometry to satisfy a compile budget.
        if (lim_ms < 0) lim_ms = 0;
        if (lim_ms) LOG("[vk] pipeline wait budget: %lld ms per frame (XCDE_PIPE_WAIT_MS=0: off)%c", (long long)lim_ms, 10);
    }
    if (!lim_ms) return false;
    uint64_t f = renderer_present_count();
    if (f != pb_frame) { pb_frame = f; pb_used_ns = 0; }
    return pb_used_ns >= (uint64_t)lim_ms * 1000000ull;
}

// ================================================================== state
typedef struct {
    NvnTexture *color[8];
    NvnTextureView cview[8];
    uint8_t has_cview[8];
    int ncolor;
    NvnTexture *depth;
    NvnTextureView dview;
    uint8_t has_dview;
} RtState;

typedef struct { uint64_t addr, size; } Range;

static struct {
    RtState rt;
    bool rt_dirty;
    NvnProgram *program;
    uint8_t blend[8][8];
    uint32_t channel_mask;
    uint8_t color_state[4], ds[8], ms[0x18], poly[4];
    float poly_offset[3];
    float alpha_ref;  // [XCDE] T191 (color_state[2]: the alpha test compare)
    uint32_t attrib[16];
    int nattrib;
    uint32_t stream[16][2];
    int nstream;
    Range vbuf[16], ubuf[6][18], sbuf[6][16];
    uint64_t tex[6][32], img[6][8];
    NvnDescPool *tpool, *spool;
    float viewport[16][4];
    int32_t scissor[16][4];
    float depth_range[16][2];
    float blend_color[4];
    uint32_t stencil_ref[2], stencil_mask[2], stencil_vmask[2];
    bool prim_restart;
    uint32_t restart_index;
    uint64_t copy_row_stride, copy_image_stride;
    Range xfb_buf[4];      // [XCDE] T134: transform feedback buffers, capture active, bytes captured into buffer 0
    bool xfb_on, raster_discard;
    uint64_t xfb_written;
} st;
static uint32_t g_xfbp[8];  // [XCDE] T134: the current draw's XfbP uniform (shader.h XFB_PARAM_INDEX; set in draw_)

uint32_t nvn_window_origin(void);
uint32_t nvn_depth_mode(void);
bool renderer_origin_lower_left(void) { return nvn_window_origin() == 0; }

// (LARECOMP_VK_PROF) called once per presented frame: frames over 50 ms are logged with where their time went.
// present_ns: time in backend_present's acquire/blit/present.
void vk_prof_frame(uint64_t frame, uint64_t present_ns) {
    if (!prof) return;
    static uint64_t last_t, l_exec, l_wait, l_draws, l_subs;
    uint64_t now = SDL_GetTicksNS();
    if (last_t && now - last_t > 50000000) {
        uint64_t dt = now - last_t, ex = cum_exec - l_exec, wt = cum_wait - l_wait;
        LOG("[hitch] frame %llu: %.0f ms = backend exec %.1f (of which GPU waits %.1f) + present %.1f + other %.1f; %llu draws, "
            "%llu submits, compiles %d pending\n", (unsigned long long)frame, dt / 1e6, ex / 1e6, wt / 1e6, present_ns / 1e6,
            ((double)dt - ex - present_ns) / 1e6, (unsigned long long)(cum_draws - l_draws), (unsigned long long)(cum_submits - l_subs),
            atomic_load(&stat_compiles));
    }
    last_t = now; l_exec = cum_exec; l_wait = cum_wait; l_draws = cum_draws; l_subs = cum_submits;
}

const char *renderer_stats(void) {
    static int dumps;
    if (dumps++ < 3) {
        LOG("[vk] command counts:");
        for (int i = 0; i < CMD_COUNT; i++) if (op_counts[i]) LOG(" %d:%llu", i, (unsigned long long)op_counts[i]);
        LOG("\n");
    }
    extern uint64_t g_tex_bytes, g_tex_astc_bytes;  // (textures.c)
    snprintf(stats_text, sizeof stats_text,
             "%llu draws, %llu skipped, %llu passes, %llu dispatches, %llu flushes, wait %.0fms, compile %d, fbcopy %llu, upload %.1fMB, occlusion %llu (%llu hidden), build waits %llu (%.1fms), prewarm %llu progs %llu pipes, gpl cached %llu linked %llu libs %llu optimized %llu, textures %.0f MB (astc %.0f MB), pipelines %u",
             (unsigned long long)stat_draws, (unsigned long long)stat_skipped, (unsigned long long)stat_passes,
             (unsigned long long)stat_dispatches, (unsigned long long)stat_flushes, (double)stat_wait_ns / 1e6,
             atomic_exchange(&stat_compiles, 0), (unsigned long long)stat_feedback_copies, (double)stat_upload_bytes / 1048576.0,
             (unsigned long long)stat_occ_reports, (unsigned long long)stat_occ_hidden, (unsigned long long)stat_build_waits,
             stat_build_wait_ns / 1e6, (unsigned long long)stat_prewarm_progs, (unsigned long long)stat_prewarm_pipes,
             (unsigned long long)atomic_exchange(&stat_gpl_cached, 0), (unsigned long long)atomic_exchange(&stat_gpl_linked, 0),
             (unsigned long long)atomic_exchange(&stat_gpl_libs, 0), (unsigned long long)atomic_exchange(&stat_gpl_opt, 0),
             g_tex_bytes / 1048576.0, g_tex_astc_bytes / 1048576.0, pipe_count());
    stat_feedback_copies = stat_upload_bytes = stat_occ_reports = stat_occ_hidden = stat_build_waits = stat_build_wait_ns = 0;
    fb_stats_flush();
    {
        extern void vk_alias_stats(void);
        vk_alias_stats();
    }
    if (prof) {
        prof_up = prof_up_ns;
        prof_up_ns = 0;
        {   // [XCDE] T169: memory report every ~10 s ("[mem]"): process, guest window, GPU heaps, our own caches
            static int nm;
            if (nm++ % 5 == 0) {
                extern void plat_mem_info(uint64_t *priv, uint64_t *ws, uint64_t *guest_commit);
                extern uint64_t kernel_heap_size(void);
                uint64_t pv, ws, gc;
                plat_mem_info(&pv, &ws, &gc);
                VmaBudget bud[VK_MAX_MEMORY_HEAPS];
                vmaGetHeapBudgets(g_vk.vma, bud);
                const VkPhysicalDeviceMemoryProperties *mp;
                vmaGetMemoryProperties(g_vk.vma, &mp);
                char heaps[512] = "";
                for (uint32_t h = 0; h < mp->memoryHeapCount && h < 4; h++) {
                    size_t l = strlen(heaps);
                    snprintf(heaps + l, sizeof heaps - l, " heap%u(%s) ours %.0f MB of %.0f, process %.0f MB, budget %.0f MB", h,
                             mp->memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT ? "device" : "host",
                             bud[h].statistics.allocationBytes / 1048576.0, bud[h].statistics.blockBytes / 1048576.0, bud[h].usage / 1048576.0,
                             bud[h].budget / 1048576.0);
                }
                { static int mm = -1; if (mm < 0) mm = getenv("XCDE_MEM_MAP") != NULL; if (mm && nm % 30 == 1) { extern void plat_mem_map(void); plat_mem_map(); } }
                LOG("[mem] private %.0f MB, working set %.0f MB, guest committed %.0f MB (heap %.0f MB); vertex cache %.0f MB, ring %d x 64 MB;%s\n",
                    pv / 1048576.0, ws / 1048576.0, gc / 1048576.0, kernel_heap_size() / 1048576.0, vc_bytes_mb(), ring_chunks(), heaps);
            }
        }
        size_t n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | gpu %.0f cpu exec %.0f draw %.0f (pre %.0f key %.0f dyn %.0f sets %.0f tex %.0f (upload %.0f) vtx %.0f) submit %.0f ms",
                 prof_gpu / 1e6, prof_exec / 1e6, prof_draw / 1e6, prof_pre / 1e6, prof_key / 1e6, prof_dyn / 1e6, prof_sets / 1e6,
                 prof_tex / 1e6, prof_up / 1e6, prof_vtx / 1e6, prof_submit / 1e6);
        n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | pre: prog %.0f fs %.0f fb %.0f barrier %.0f pass %.0f | pipeline key memo %llu",
                 prof_pre_part[0] / 1e6, prof_pre_part[1] / 1e6, prof_pre_part[2] / 1e6, prof_pre_part[3] / 1e6, prof_pre_part[4] / 1e6,
                 (unsigned long long)stat_pipekey_memo);
        stat_pipekey_memo = 0;
        n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | sets: alloc %.0f update %.0f ms, %llu writes | begin_pass %.0f (submit %.0f) ms",
                 prof_ds_alloc / 1e6, prof_ds_upd / 1e6, (unsigned long long)stat_ds_writes, prof_bp / 1e6, prof_bp_submit / 1e6);
        prof_ds_alloc = prof_ds_upd = stat_ds_writes = prof_bp = prof_bp_submit = 0;
        extern uint64_t prof_tex_checks, prof_tex_samples, prof_tex_full, prof_tex_full_bytes, prof_tex_dyn;
        n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | tex checks %llu samples %llu full %llu (%.1f MB, dynamic %llu)",
                 (unsigned long long)prof_tex_checks, (unsigned long long)prof_tex_samples, (unsigned long long)prof_tex_full,
                 prof_tex_full_bytes / 1048576.0, (unsigned long long)prof_tex_dyn);
        n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | pass ends:");
        for (int l = 0; l < 16384; l++)
            if (prof_end_line[l] > 20) { n = strlen(stats_text); snprintf(stats_text + n, sizeof stats_text - n, " %d:%u", l, prof_end_line[l]); }
        memset(prof_end_line, 0, sizeof prof_end_line);
        n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | restarts by:");
        for (int l = 0; l < 16384; l++)
            if (prof_dirty_line[l] > 20) { n = strlen(stats_text); snprintf(stats_text + n, sizeof stats_text - n, " %d:%u", l, prof_dirty_line[l]); }
        memset(prof_dirty_line, 0, sizeof prof_dirty_line);
        n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | views created %llu | counters gpu %llu | pace %s, %llu frames over", (unsigned long long)prof_views_created,
                 (unsigned long long)stat_counters_gpu, pace_slow ? "30" : "60", (unsigned long long)stat_pace_over);
        stat_counters_gpu = stat_pace_over = 0;
        {
            extern uint64_t prof_vc_hits, prof_vc_bypass, prof_vc_copied, prof_vc_queries, prof_vc_scan_pages, prof_vc_scan_ns;
            n = strlen(stats_text);
            snprintf(stats_text + n, sizeof stats_text - n,  " | vcache %llu hit %llu in place %.1f MB copied, %llu queries %.2f ms, %llu pages written",
                     (unsigned long long)prof_vc_hits, (unsigned long long)prof_vc_bypass, prof_vc_copied / 1048576.0,
                     (unsigned long long)prof_vc_queries, prof_vc_scan_ns / 1e6, (unsigned long long)prof_vc_scan_pages);
            prof_vc_hits = prof_vc_bypass = prof_vc_copied = prof_vc_queries = prof_vc_scan_pages = prof_vc_scan_ns = 0;
            extern uint64_t prof_tex_watch_skips, stat_bar_full, stat_bar_att, stat_bar_none, stat_bar_why[6], stat_bar_mem, stat_bar_gfx;
            n = strlen(stats_text);
            snprintf(stats_text + n, sizeof stats_text - n, " | tex unwritten %llu | same prog %llu pipe %llu | pass barriers full %llu (ended %llu dirty %llu hazard %llu alias %llu forced %llu war %llu) graphics %llu memory %llu attachment %llu none %llu", (unsigned long long)prof_tex_watch_skips,
                     (unsigned long long)prof_same_prog, (unsigned long long)prof_same_pipe, (unsigned long long)stat_bar_full,
                     (unsigned long long)stat_bar_why[0], (unsigned long long)stat_bar_why[1], (unsigned long long)stat_bar_why[2],
                     (unsigned long long)stat_bar_why[3], (unsigned long long)stat_bar_why[4], (unsigned long long)stat_bar_why[5],
                     (unsigned long long)stat_bar_gfx, (unsigned long long)stat_bar_mem, (unsigned long long)stat_bar_att, (unsigned long long)stat_bar_none);
            memset(stat_bar_why, 0, sizeof stat_bar_why);
            prof_same_prog = prof_same_pipe = 0; stat_bar_full = stat_bar_att = stat_bar_none = stat_bar_mem = stat_bar_gfx = 0;
            prof_tex_watch_skips = 0;
        }
        pthread_mutex_lock(&wait_tab_lock);
        for (int i = 0; i < 12 && wait_tab[i].why; i++) {
            n = strlen(stats_text);
            snprintf(stats_text + n, sizeof stats_text - n, "%s%s %llux %.0f ms", i ? "," : " | waits:", wait_tab[i].why,
                     (unsigned long long)wait_tab[i].n, wait_tab[i].ns / 1e6);
        }
        memset(wait_tab, 0, sizeof wait_tab);
        pthread_mutex_unlock(&wait_tab_lock);
        prof_views_created = 0;
        prof_tex_checks = prof_tex_samples = prof_tex_full = prof_tex_full_bytes = prof_tex_dyn = 0;
        prof_exec = prof_draw = prof_sets = prof_vtx = prof_submit = prof_pre = prof_key = prof_dyn = prof_tex = prof_up = prof_gpu = 0;
        memset(prof_pre_part, 0, sizeof prof_pre_part);
    }
    stat_draws = stat_skipped = stat_passes = stat_dispatches = stat_flushes = stat_wait_ns = 0;
    return stats_text;
}

// [XCDE] perf: generation of the NVN state the pipeline key reads (bumped by every command except the few known not
// to touch it, by st_defaults and by vk_exec_reset) and of the render pass (begin_pass): a draw with the same program,
// topology class and both generations as the previous one reuses its pipeline without building and hashing the key.
static uint32_t g_pipe_state_gen, g_pass_gen, g_pipe_bump_op;  // (g_pipe_bump_op: the command that last bumped it, XCDE_MEMO_WHY)
static uint32_t g_tex_bind_gen[6], g_sbuf_bind_gen[6];  // ([XCDE] g_sbuf_bind_gen: CMD_BIND_STORAGE_BUFFERS, cbuf_info memo)  // [XCDE] per stage: bumped by CMD_BIND_TEXTURES (and st_defaults / vk_exec_reset)
static bool cmd_keeps_pipe_key(uint32_t op) {
    switch (op) {
    case CMD_END: case CMD_JUMP: case CMD_CALL: case CMD_BIND_DEPTH_STENCIL: case CMD_BIND_VERTEX_BUFFERS:
    case CMD_BIND_PROGRAM: case CMD_BIND_BLEND: case CMD_BIND_CHANNEL_MASK: case CMD_BIND_COLOR: case CMD_BIND_MULTISAMPLE:
    case CMD_BIND_POLYGON: case CMD_BIND_VERTEX_ATTRIB: case CMD_BIND_VERTEX_STREAM: case CMD_SET_RENDER_TARGETS:
    case CMD_CLEAR_COLOR: case CMD_CLEAR_DEPTH_STENCIL:  // (clears read the targets, change nothing the key reads)
        // (these bump it themselves, on a change)
    case CMD_BIND_UNIFORM_BUFFERS: case CMD_BIND_STORAGE_BUFFERS: case CMD_BIND_TEXTURES: case CMD_BIND_IMAGES:
    case CMD_SET_TEXTURE_POOL: case CMD_SET_SAMPLER_POOL: case CMD_SET_VIEWPORTS: case CMD_SET_SCISSORS:
    case CMD_SET_DEPTH_RANGES: case CMD_SET_BLEND_COLOR: case CMD_SET_STENCIL: case CMD_SET_PRIMITIVE_RESTART:
    case CMD_SET_ALPHA_REF: case CMD_SET_DEPTH_BOUNDS: case CMD_SET_POLYGON_OFFSET: case CMD_SET_LINE_WIDTH:
    case CMD_SET_POINT_SIZE: case CMD_DRAW_ARRAYS: case CMD_DRAW_ELEMENTS: case CMD_UPDATE_UNIFORM_BUFFER:
    case CMD_FENCE_SYNC: case CMD_WAIT_SYNC: case CMD_REPORT_COUNTER: case CMD_SET_COPY_STRIDES: case CMD_BARRIER:
        return true;
    default:
        return false;
    }
}
static void st_defaults(void) {
    g_pipe_state_gen++;
    for (int i = 0; i < 6; i++) g_tex_bind_gen[i]++, g_sbuf_bind_gen[i]++;
    memset(&st, 0, sizeof st);
    for (int i = 0; i < 16; i++) {
        st.viewport[i][2] = 1280; st.viewport[i][3] = 720;
        st.scissor[i][2] = 16384; st.scissor[i][3] = 16384;
        st.depth_range[i][1] = 1.0f;
    }
    st.channel_mask = 0xFFFFFFFF;
    st.stencil_mask[0] = st.stencil_mask[1] = 0xFF;
    st.stencil_vmask[0] = st.stencil_vmask[1] = 0xFF;
    st.ds[1] = 1; st.ds[2] = 2;
    st.poly[1] = 1; st.poly[2] = 2;
    for (int i = 0; i < 8; i++) { st.blend[i][1] = 2; st.blend[i][2] = 1; st.blend[i][3] = 2; st.blend[i][4] = 1; st.blend[i][5] = 1; st.blend[i][6] = 1; }
}

void vk_exec_init(void) {
    prof = getenv("LARECOMP_VK_PROF") != NULL || getenv("XCDE_METRICS") != NULL;
    uint32_t nq=0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_vk.phys,&nq,NULL);
    VkQueueFamilyProperties *queues=calloc(nq,sizeof *queues);
    if (queues) {
        vkGetPhysicalDeviceQueueFamilyProperties(g_vk.phys,&nq,queues);
        if (g_vk.qfam<nq) gpu_timestamp_bits=queues[g_vk.qfam].timestampValidBits;
        free(queues);
    }
    if (gpu_timestamp_bits && !(getenv("XCDE_GPU_TIMING") && !atoi(getenv("XCDE_GPU_TIMING")))) {
        prof_up_ns = 0;
        VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = SUBS * 4};
        if (vkCreateQueryPool(g_vk.dev, &qi, NULL, &prof_qp) != VK_SUCCESS) prof_qp = VK_NULL_HANDLE;
    }
    st_defaults();
}

void renderer_texture_scale(NvnTexture *t, float *sx, float *sy);
bool renderer_texture_presented(NvnTexture *t);
bool renderer_texture_first_write(NvnTexture *t);
bool renderer_texture_level_fresh(NvnTexture *t);
bool renderer_texture_frame_fresh(NvnTexture *t);
void renderer_texture_mark_presented(NvnTexture *t);
uint64_t renderer_present_count(void);
void renderer_save_texture_tex(TexBackend *b, const char *path);
static float pass_sx = 1, pass_sy = 1;
static float vp_sx = 1, vp_sy = 1, vp_ox = 0, vp_oy = 0;
static float tex_sc[2][64][2];
static uint8_t tex_d16[2][64];  // [XCDE] T243: the texture bound to the unit is D16 (rscale[1 + unit].z: the shader rounds a
static bool ti_d16;             // compare reference to 16 bits, as Maxwell does); ti_d16: texture_info's last answer
static uint32_t tex_wh[2][64][2];

// ================================================================== render passes
static VkFormat pass_color[8], pass_depth, pass_stencil;
static uint32_t pass_ncolor, pass_samples = 1, pass_w, pass_h;
static int debug_pass_no;

// Redundant-state filter: what draw() last set in the current command buffer, so unchanged pipeline binds, dynamic
// state and vertex / index buffer binds are not recorded again (a draw used to issue ~20 vkCmd* calls; most repeat the
// previous draw's). Forgotten (ss_forget) whenever something else may have recorded state: a new command buffer, a new
// render pass (present, ImGui, post-processing and copies all run between game passes) and the clear pipeline.
// Descriptor sets are not filtered: every draw writes new ones (sharing identical sets within a submission was tried:
// a third matched, but comparing them cost more than allocating and writing them on NVIDIA).
// LARECOMP_VK_STATE_FILTER=0: every draw sets everything, as before.
enum { SS_PIPE = 1 << 0, SS_DT = 1 << 1, SS_DW = 1 << 2, SS_DOP = 1 << 3, SS_STEN = 1 << 4, SS_SFACE0 = 1 << 5, SS_SFACE1 = 1 << 6,
       SS_BC = 1 << 7, SS_CULL = 1 << 8, SS_FF = 1 << 9, SS_TOPO = 1 << 10, SS_PRE = 1 << 11, SS_DBE = 1 << 12, SS_DB = 1 << 13,
       SS_VP = 1 << 14, SS_SC = 1 << 15, SS_IB = 1 << 16 };
static struct {
    uint32_t known;  // SS_* bits: the value below is what the command buffer has
    uint32_t vb_known;  // per vertex binding
    VkPipeline pipe;
    VkBool32 dt, dw, sten, pre, dbe;
    VkCompareOp dop;
    uint32_t sface[2][7];  // fail, pass, depth fail, compare, compare mask, write mask, reference
    float bc[4], db[3];
    VkCullModeFlags cull;
    VkFrontFace ff;
    VkPrimitiveTopology topo;
    VkViewport vp;
    VkRect2D sc;
    VkBuffer ib; VkDeviceSize io; VkIndexType it;
    VkBuffer vb[17]; VkDeviceSize vo[17], vsz[17];
    VkPipelineLayout descriptor_layout;
    VkDescriptorSet descriptor_sets[3];
    unsigned descriptor_known;
} ss;
static int ss_on = -1;
static void ss_forget(void) { ss.known = 0; ss.vb_known = 0; ss.descriptor_known=0; }
static void vk_bind_graphics_sets(VkCommandBuffer cb, VkPipelineBindPoint point, VkPipelineLayout layout,
                                 uint32_t first, uint32_t count, const VkDescriptorSet *sets,
                                 uint32_t dynamic_count, const uint32_t *offsets) {
    bool same=ss.descriptor_layout==layout && first+count<=3 && !dynamic_count;
    for (unsigned i=0;same && i<count;i++)
        same=(ss.descriptor_known&(1u<<(first+i))) && ss.descriptor_sets[first+i]==sets[i];
    if (same) return;
    vkCmdBindDescriptorSets(cb,point,layout,first,count,sets,dynamic_count,offsets);
    if (ss.descriptor_layout!=layout || dynamic_count) ss.descriptor_known=0;
    ss.descriptor_layout=layout;
    if (!dynamic_count) for (unsigned i=0;i<count && first+i<3;i++) {
        ss.descriptor_sets[first+i]=sets[i]; ss.descriptor_known|=1u<<(first+i);
    }
}
static bool ss_enabled(void) {
    if (ss_on < 0) { const char *e = getenv("LARECOMP_VK_STATE_FILTER"); ss_on = !e || atoi(e) != 0; }
    return ss_on;
}
// true: the state must be recorded (and is remembered as known)
static inline bool ss_need(uint32_t bit, bool same) {
    if (ss_on && (ss.known & bit) && same) return false;
    ss.known |= bit;
    return true;
}

// [XCDE] T152: XCDE_CHECK_NAN=<N> - every N-th presented frame, each floating-point colour attachment is read back when
// its pass ends and scanned for NaN / Inf ("[check-nan]" lines: frame, pass, target, counts, first texel). NaN in
// lighting / post buffers turns into black or white blotches downstream.
static TexBackend *pass_tb[8];
static VkFormat pass_tb_fmt[8];
static int pass_no_in_frame;
static uint64_t pass_frame_seen = ~0ull;
static bool nan_check_now(void) {
    static long every = -2;
    if (every == -2) every = getenv("XCDE_CHECK_NAN") ? atol(getenv("XCDE_CHECK_NAN")) : 0;
    return every > 0 && renderer_present_count() % (uint64_t)every == 0;
}
static uint64_t half_bad(uint16_t h) { return (h & 0x7C00) == 0x7C00; }
static void nan_scan(TexBackend *b, VkFormat f, int slot) {
    uint32_t bpp = f == VK_FORMAT_R16G16B16A16_SFLOAT ? 8 : f == VK_FORMAT_R32G32B32A32_SFLOAT ? 16 : f == VK_FORMAT_R32G32_SFLOAT ? 8 :
                   f == VK_FORMAT_R16G16_SFLOAT || f == VK_FORMAT_R32_SFLOAT || f == VK_FORMAT_B10G11R11_UFLOAT_PACK32 ? 4 :
                   f == VK_FORMAT_R16_SFLOAT ? 2 : 0;
    if (!bpp || !b || !b->image || (b->aspect & ~VK_IMAGE_ASPECT_COLOR_BIT)) return;
    uint32_t w = b->width, h = b->height;
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    VkSpan out = vk_readback((uint64_t)w * h * bpp);
    VkBufferImageCopy c = {.bufferOffset = out.off, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {w, h, 1}};
    vkCmdCopyImageToBuffer(cb, b->image, VK_IMAGE_LAYOUT_GENERAL, out.buf, 1, &c);
    renderer_flush();
    uint64_t bad = 0, first = ~0ull;
    const uint8_t *p = out.ptr;
    for (uint64_t i = 0; i < (uint64_t)w * h; i++) {
        bool x = false;
        if (bpp == 8 && f == VK_FORMAT_R16G16B16A16_SFLOAT) { const uint16_t *q = (const uint16_t *)(p + i * 8); x = half_bad(q[0]) | half_bad(q[1]) | half_bad(q[2]) | half_bad(q[3]); }
        else if (f == VK_FORMAT_R16G16_SFLOAT) { const uint16_t *q = (const uint16_t *)(p + i * 4); x = half_bad(q[0]) | half_bad(q[1]); }
        else if (f == VK_FORMAT_R16_SFLOAT) x = half_bad(*(const uint16_t *)(p + i * 2));
        else if (f == VK_FORMAT_B10G11R11_UFLOAT_PACK32) {
            uint32_t v = *(const uint32_t *)(p + i * 4);
            x = ((v >> 6) & 31) == 31 || ((v >> 17) & 31) == 31 || ((v >> 27) & 31) == 31;
        } else {
            const uint32_t *q = (const uint32_t *)(p + i * bpp);
            for (uint32_t k = 0; k < bpp / 4; k++) x |= (q[k] & 0x7F800000u) == 0x7F800000u;
        }
        if (x) { bad++; if (first == ~0ull) first = i; }
    }
    static int lines;
    if (bad && lines++ < 400)
        LOG("[check-nan] frame %llu pass %d target %d %ux%u vkfmt %d: %llu NaN/Inf texels (%.3f%%), first at %llu,%llu\n",
            (unsigned long long)renderer_present_count(), pass_no_in_frame, slot, w, h, (int)f, (unsigned long long)bad,
            100.0 * bad / ((double)w * h), (unsigned long long)(first % w), (unsigned long long)(first / w));
}

static void vk_end_pass_at(int line);
void vk_end_pass(void) { vk_end_pass_at(0); }
#define vk_end_pass() vk_end_pass_at(__LINE__)
static bool pass_untracked;  // (T162 part 2, below)
static void vk_end_pass_at(int line) {
    pass_untracked = true;  // [XCDE] T162 part 2 (the two tracked enders clear it after the call)
    if (!in_pass) return;
    if (prof && line < 16384) prof_end_line[line]++;
    oq_end(vk_cmd());  // [XCDE] T139
    vkCmdEndRendering(vk_cmd());
    in_pass = false;
    // [XCDE] T213: the first pass this frame that drew into the presented image (the scene composite: the HUD follows)
    // (only once this frame's scene was drawn: the game writes the window image earlier in the frame too - a snapshot
    // there turned the jitter off for the whole scene and DLSS / FSR got no sub-pixel detail: jagged edges)
    if (pass_dbg_rt && pass_dbg_rt->presented && tmp_enabled() && tmp_scene_frame == renderer_present_count())
        tmp_hud_snapshot(vk_cmd(), pass_dbg_rt);
    if (renderer_present_count() != pass_frame_seen) { pass_frame_seen = renderer_present_count(); pass_no_in_frame = 0; }
    pass_no_in_frame++;
    if (nan_check_now())  // [XCDE] T152
        for (int i = 0; i < 8; i++) if (pass_tb[i]) nan_scan(pass_tb[i], pass_tb_fmt[i], i);
    if (g_debug_frame && pass_dbg_rt) {
        char path[256];
        static uint64_t debug_pass_frame = ~0ull;
        static int multi = -1;
        if (multi < 0) multi = getenv("LARECOMP_DEBUG_FRAMES") != NULL;
        if (multi && renderer_present_count() != debug_pass_frame) { debug_pass_frame = renderer_present_count(); debug_pass_no = 0; }
        if (multi) snprintf(path, sizeof path, "build/pass_%llu_%03d.png", (unsigned long long)debug_pass_frame, debug_pass_no++);
        else snprintf(path, sizeof path, "build/pass_%03d.png", debug_pass_no++);
        renderer_save_texture_tex(pass_dbg_rt, path);
    }
    {   // [XCDE] T119: XCDE_PASS_HASH=<from>:<to>: hash of each pass's colour target in those presented frames
        static long from = -2, to;
        static uint64_t ph_frame = ~0ull;
        static int ph_no;
        if (from == -2) { from = -1; const char *e = getenv("XCDE_PASS_HASH"); if (e) sscanf(e, "%ld:%ld", &from, &to); }
        uint64_t f = renderer_present_count();
        if (from >= 0 && (long)f >= from && (long)f <= to && pass_dbg_rt) {
            if (f != ph_frame) { ph_frame = f; ph_no = 0; }
            char label[64];
            snprintf(label, sizeof label, "#f%llu p%03d", (unsigned long long)f, ph_no++);
            renderer_save_texture_tex(pass_dbg_rt, label);
        }
    }
}

static uint32_t level_extent(uint32_t v, uint32_t lv) { v >>= lv; return v ? v : 1; }

// [XCDE] T162: barriers between render passes only where something depends on what came before. Every pass used to
// start with a full barrier (all commands -> all commands): ~155 a frame in the field, each one draining the GPU
// (measured: without them GPU busy -35 %). Now each image remembers the barrier epoch of its last write as an attachment
// and its last sampling. A pass that directly follows another pass gets:
//   - a full barrier when anything untracked happened since the last one (the pass was ended by a copy, clear, dispatch,
//     memory-writing draw, the game's own barrier, a new command buffer...: pass_dirty), when aliased textures exist,
//     when a draw samples an image written since the last barrier (pass_hazard), or when an attachment was sampled
//     since then (write after read);
//   - an attachment-only barrier (attachment output -> attachment load / tests) when it reuses an attachment written
//     since the last barrier (e.g. the same depth buffer in the next pass): vertex work may overlap;
//   - none when its attachments and textures are untouched since the last barrier.
// LARECOMP_VK_PASS_BARRIER=full: a full barrier before every pass as before.
static int naliases;  // (defined with the alias table below)
static uint32_t pass_epoch = 1, att_epoch = 1;  // bumped by full barriers / by any barrier between passes
static bool pass_dirty = true, pass_hazard;
// [XCDE] T162 part 2: guest memory (the imported heap) on the GPU. ~80 % of the full barriers left were passes ended by
// a memory-writing draw (grass transform feedback / skinning-palette writers, ~54 a frame) or by the game's own
// barrier (~77 a frame). Those two are tracked instead of "anything may have happened": stores since the last barrier
// (mem_wr: stages that wrote), reads (mem_rd: vertex / index fetch always, shader stages that read the heap), and the
// stages of a memory-writing draw about to start (pre_wr: write after read, T136). The barrier then covers only those
// stages (plus attachments as before), so fragment work of the previous passes keeps running under the next vertex
// work. Everything else that ends a pass (copies, clears, dispatches, submissions...) still means a full barrier.
static VkPipelineStageFlags2 mem_rd, mem_wr, pre_wr;
static bool pass_untracked = true;  // the open pass was ended by something other than the two cases above
// XCDE_MEM_BARRIER=<bits> (debugging; default 7): 1 passes ended by memory-writing draws tracked, 2 passes ended by the
// game's barrier tracked, 4 narrow barrier before memory-writing draws; 0 = as T162 part 1
static int mem_track_mode(void) {
    static int m = -1;
    if (m < 0) { const char *e = getenv("XCDE_MEM_BARRIER"); m = e ? atoi(e) : 7; }
    return m;
}
uint64_t stat_bar_mem;
uint64_t stat_bar_full, stat_bar_att, stat_bar_none;  // (VK_PROF: barriers at pass starts by kind)
uint64_t stat_bar_why[6];  // (full ones: pass ended by something else, dirty, sampling hazard, aliases, forced, write after read)

static void att_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                           VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                           .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                           VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                           .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                                            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

#define MEM_READ_STAGES (VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT | \
                         VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | \
                         VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT)

// guest memory written since the last barrier -> any later read or write of it; reads since then -> the memory-writing
// draw that starts next (write after read); with_att: the attachment barrier too
static void mem_barrier(VkCommandBuffer cb, bool with_att) {
    VkMemoryBarrier2 mb[2];
    int n = 0;
    VkPipelineStageFlags2 src = mem_wr | (pre_wr ? mem_rd : 0), dst = (mem_wr ? MEM_READ_STAGES : 0) | pre_wr;
    mb[n++] = (VkMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = src,
                                 .srcAccessMask = mem_wr ? VK_ACCESS_2_MEMORY_WRITE_BIT : 0, .dstStageMask = dst,
                                 .dstAccessMask = mem_wr ? VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT : 0};
    if (with_att)
        mb[n++] = (VkMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                                     .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                                     VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                                     .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                     .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                                     VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                                     .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                                                      VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = (uint32_t)n, .pMemoryBarriers = mb};
    vkCmdPipelineBarrier2(cb, &di);
    if (pre_wr) mem_rd = 0;  // (ordered before everything after: dst covers every reading stage when mem_wr, else pre_wr)
    mem_wr = 0;
}

// [XCDE] T162 part 3: a sampling hazard (an image written as an attachment since the last full barrier is sampled
// now) or a write after read (an attachment sampled since then is written now) when nothing untracked happened:
// graphics stages only - attachment output and the shader stages (plus guest memory as in mem_barrier). Transfers,
// compute and host work are untouched (anything of theirs since the last barrier would have meant a full barrier).
static void gfx_barrier(VkCommandBuffer cb) {
    const VkPipelineStageFlags2 att = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                      VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                                sh = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT |
                                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = att | sh | mem_wr | mem_rd,
                           .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | (mem_wr ? VK_ACCESS_2_MEMORY_WRITE_BIT : 0),
                           .dstStageMask = att | sh | MEM_READ_STAGES,
                           .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT |
                                            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                                            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                                            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                            (mem_wr ? VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT : 0)};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}
uint64_t stat_bar_gfx;

static void pass_barrier(VkCommandBuffer cb, bool was_in_pass) {
    static int full_always = -1, gfx_on = -1;
    if (full_always < 0) { const char *e = getenv("LARECOMP_VK_PASS_BARRIER"); full_always = e && !strcmp(e, "full"); }
    if (gfx_on < 0) { const char *e = getenv("XCDE_GFX_BARRIER"); gfx_on = !(e && *e == '0'); }  // (0: full ones there, as part 2)
    int mt = mem_track_mode();
    bool ended = !was_in_pass && pass_untracked;
    bool untracked = full_always || pass_dirty || ended || naliases || (!(mt & 4) && pre_wr);
    bool war = false, reuse = false;
    if (!untracked) {
        TexBackend *a[9];
        int na = 0;
        for (int i = 0; i < st.rt.ncolor && i < 8; i++) if (st.rt.color[i] && st.rt.color[i]->backend) a[na++] = st.rt.color[i]->backend;
        if (st.rt.depth && st.rt.depth->backend) a[na++] = st.rt.depth->backend;
        for (int i = 0; i < na; i++) {
            if (a[i]->r_epoch == pass_epoch) war = true;          // sampled since the last full barrier, now written
            else if (a[i]->w_att == att_epoch) reuse = true;     // written by an earlier pass since the last barrier, written again
        }
    }
    int kind = 0;  // 0 none, 1 attachments, 2 full, 3 graphics (sampling hazard / write after read)
    if (untracked || ((pass_hazard || war) && !gfx_on)) {
        kind = 2;
        stat_bar_why[ended ? 0 : pass_dirty ? 1 : pass_hazard ? 2 : naliases ? 3 : war ? 5 : 4]++;
    } else if (pass_hazard || war) kind = 3;
    else if (reuse) kind = 1;
    if (kind == 2) { vk_full_barrier(cb); stat_bar_full++; }
    else if (kind == 3) { gfx_barrier(cb); stat_bar_gfx++; }
    else if (mem_wr || pre_wr) { mem_barrier(cb, kind == 1); stat_bar_mem++; }
    else if (kind == 1) { att_barrier(cb); stat_bar_att++; }
    else stat_bar_none++;
    if (kind) att_epoch++;  // (any barrier orders earlier attachment writes for attachment use)
    if (kind >= 2) { pass_epoch++; pass_dirty = false; mem_rd = mem_wr = 0; }  // (full / graphics: visible to sampling)
    pre_wr = 0;
    pass_hazard = false;
    pass_untracked = true;  // (until the two known enders say otherwise)
}

// this pass's attachments: written from now on (after begin_pass resolved them: a first use creates the image there)
static void pass_mark_attachments(void) {
    for (int i = 0; i < st.rt.ncolor && i < 8; i++)
        if (st.rt.color[i] && st.rt.color[i]->backend) {
            TexBackend *b = st.rt.color[i]->backend;
            b->w_epoch = pass_epoch; b->w_att = att_epoch;
        }
    if (st.rt.depth && st.rt.depth->backend) { TexBackend *b = st.rt.depth->backend; b->w_epoch = pass_epoch; b->w_att = att_epoch; }
}

static bool split_pending;  // (split_submit: submit at the next pass boundary)
static uint32_t split_since;
static uint64_t pass_seq_no;  // [XCDE] T220: counts begun passes
static void begin_pass(void) {
    if (in_pass && !st.rt_dirty) return;
    pass_seq_no++;
    bool was_in_pass = in_pass, untracked = pass_untracked;
    if (prof && in_pass && rt_dirty_line < 16384) prof_dirty_line[rt_dirty_line]++;
    vk_end_pass();
    if (!was_in_pass) pass_untracked = untracked;  // (that call ended nothing: keep how the last pass ended)
    if (split_pending) {  // [XCDE] perf: the split submission goes here, where the game's pass ends anyway
        split_pending = false;
        split_since = 0;
        uint64_t ps0 = prof ? SDL_GetTicksNS() : 0;
        renderer_submit();
        if (prof) prof_bp_submit += SDL_GetTicksNS() - ps0;
        was_in_pass = false;  // (a new command buffer: full barrier, as at any submission start)
    }
    if (dl_on == 2) renderer_flush();  // (LARECOMP_VK_DEVLOST_TRACE=2: each pass its own submission)
    st.rt_dirty = false;
    VkCommandBuffer cb = vk_cmd();
    pass_barrier(cb, was_in_pass);  // [XCDE] T162 (was: a full barrier before every pass)
    {
        NvnTexture *t0n = st.rt.ncolor ? st.rt.color[0] : NULL;
        dl_note("pass %d color %ux%u fmt %#x depth %p", st.rt.ncolor, t0n ? t0n->width : 0, t0n ? t0n->height : 0, t0n ? t0n->format : 0,
                (void *)st.rt.depth);
    }
    if (g_gpuprof_frame) {
        NvnTexture *t0 = st.rt.ncolor ? st.rt.color[0] : NULL;
        gp_mark(cb, "pass: %d color %ux%u fmt %#x, depth %s", st.rt.ncolor, t0 ? t0->width : 0, t0 ? t0->height : 0,
                t0 ? t0->format : 0, st.rt.depth ? "yes" : "no");
    }
    VkRenderingAttachmentInfo ca[8], da = {0}, sa = {0};
    uint32_t w = 0, minw = UINT32_MAX, minh = UINT32_MAX;  // w: first attachment seen
    pass_ncolor = (uint32_t)st.rt.ncolor;
    pass_samples = 1;
    memset(pass_tb, 0, sizeof pass_tb);  // [XCDE] T152
    for (int i = 0; i < st.rt.ncolor; i++) {
        NvnTexture *t = st.rt.color[i];
        ca[i] = (VkRenderingAttachmentInfo){.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        pass_color[i] = VK_FORMAT_UNDEFINED;
        if (!t) continue;
        if (i < 8 && t->backend) { pass_tb[i] = t->backend; pass_tb_fmt[i] = ((TexBackend *)t->backend)->format; }
        { extern void tmp_note_target(TexBackend *b); tmp_note_target(t->backend); }  // [XCDE] T228 (the game's velocity buffer)
        renderer_mark_gpu_owned(t);
        VkImageView v = vk_view(t, &st.rt.cview[i], st.rt.has_cview[i], true, TEXTYPE_2D);
        TexBackend *b = t->backend;
        if (!v) continue;
        ca[i].imageView = v;
        ca[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        ca[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        bool fresh = (int)renderer_texture_level_fresh(t) | (int)renderer_texture_frame_fresh(t);  // both: each marks its state
        if (renderer_texture_first_write(t) || fresh) {
            ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            ca[i].clearValue.color.float32[3] = fresh ? 0.0f : 1.0f;
        }
        pass_color[i] = st.rt.has_cview[i] && (st.rt.cview[i].setmask & 4) && vk_format(t->format) == b->format
                      ? vk_format(st.rt.cview[i].format) : b->format;
        uint32_t lv = st.rt.has_cview[i] && (st.rt.cview[i].setmask & 1) ? st.rt.cview[i].base_level : 0;
        uint32_t ew = level_extent(b->width, lv), eh = level_extent(b->height, lv);
        if (!w) { renderer_texture_scale(t, &pass_sx, &pass_sy); w = ew; }
        if (ew < minw) minw = ew;
        if (eh < minh) minh = eh;
        pass_samples = b->samples;
    }
    {   // [XCDE] R1 (docs/spec_lowend_gpu.md): trailing colour slots without an image are left out of the pass. They went
        // in as VK_NULL_HANDLE attachments - legal, but Intel's Windows driver crashes in vkCmdBeginRendering on them
        // (LASRecomp f4cb039 / letsgo-recomp 431f4da). Pipelines (k.ncolor) and clears follow pass_ncolor; an empty slot
        // before a used one stays (logged once). XCDE_KEEP_EMPTY_SLOTS=1: as before.
        static int keep = -1;
        if (keep < 0) keep = getenv("XCDE_KEEP_EMPTY_SLOTS") && atoi(getenv("XCDE_KEEP_EMPTY_SLOTS")) != 0;
        if (!keep) {
            uint32_t n0 = pass_ncolor;
            while (pass_ncolor && !ca[pass_ncolor - 1].imageView) pass_ncolor--;
            static int logged, trimmed;
            if (pass_ncolor != n0 && !trimmed++) LOG("[vk] trailing empty colour slots left out (first: %u -> %u)\n", n0, pass_ncolor);
            for (uint32_t i = 0; i < pass_ncolor && !logged; i++)
                if (!ca[i].imageView) { logged = 1; LOG("[vk] pass with an empty colour slot %u before a used one (kept)\n", i); }
        }
    }
    pass_depth = pass_stencil = VK_FORMAT_UNDEFINED;
    bool has_d = false, has_s = false;
    if (st.rt.depth) {
        renderer_mark_gpu_owned(st.rt.depth);
        VkImageView v = vk_view(st.rt.depth, &st.rt.dview, st.rt.has_dview, true, TEXTYPE_2D);
        TexBackend *b = st.rt.depth->backend;
        if (v && renderer_format_is_depth(st.rt.depth->format)) {
            VkRenderingAttachmentInfo a = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = v,
                                           .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                                           .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
            if (b->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) { da = a; has_d = true; pass_depth = b->format; }
            if (b->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) { sa = a; has_s = true; pass_stencil = b->format; }
            uint32_t lv = st.rt.has_dview && (st.rt.dview.setmask & 1) ? st.rt.dview.base_level : 0;
            uint32_t ew = level_extent(b->width, lv), eh = level_extent(b->height, lv);
            if (!w) { renderer_texture_scale(st.rt.depth, &pass_sx, &pass_sy); w = ew; }
            if (ew < minw) minw = ew;
            if (eh < minh) minh = eh;
            if (!pass_ncolor) pass_samples = b->samples;
        }
    }
    if (!w) { pass_sx = pass_sy = 1; minw = minh = 16; }
    pass_w = minw; pass_h = minh;
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {minw, minh}}, .layerCount = 1,
                          .colorAttachmentCount = pass_ncolor, .pColorAttachments = ca,
                          .pDepthAttachment = has_d ? &da : NULL, .pStencilAttachment = has_s ? &sa : NULL};
    if (g_debug_frame && getenv("XCDE_TRACE_VP")) LOG("[pass] begin renderArea %ux%u depth %p img %p view %p (%ux%u) ncolor %u\n", minw, minh, (void *)st.rt.depth, st.rt.depth ? (void *)((TexBackend *)st.rt.depth->backend)->image : NULL, (void *)da.imageView, st.rt.depth ? ((TexBackend *)st.rt.depth->backend)->width : 0, st.rt.depth ? ((TexBackend *)st.rt.depth->backend)->height : 0, pass_ncolor);
    if (g_watch_img && has_d && st.rt.depth && ((TexBackend *)st.rt.depth->backend)->image == g_watch_img) vk_watch(g_watch_img, "begin_pass (depth attachment)");
    for (uint32_t i = 0; i < pass_ncolor; i++) if (g_watch_img && st.rt.color[i] && st.rt.color[i]->backend && ((TexBackend *)st.rt.color[i]->backend)->image == g_watch_img) vk_watch(g_watch_img, "begin_pass (COLOR attachment)");
    pass_mark_attachments();  // [XCDE] T162
    vkCmdBeginRendering(cb, &ri);
    {   // [XCDE] perf: the cached state survives a pass change - Vulkan keeps the bound pipeline, dynamic state, vertex
        // buffers and sets across render passes of a command buffer. Forgetting it at every pass (~400 a frame) made the
        // next draw rebind all of it. It is forgotten when something else began a pass of its own since our last one
        // (post / present / save / temporal / window passes set their own pipeline and state; the clear pipeline, DLSS
        // and a new command buffer forget it themselves). XCDE_PASS_SS_FORGET=1: forget at every pass, as before.
        static uint32_t ours;
        static int every = -1;
        if (every < 0) every = getenv("XCDE_PASS_SS_FORGET") && atoi(getenv("XCDE_PASS_SS_FORGET")) != 0;
        if (every || g_vk_begin_count != ours + 1) ss_forget();
        ours = g_vk_begin_count;
    }
    in_pass = true;
    if (oq_counting) oq_begin(cb);  // [XCDE] T139: this pass's samples count toward the open counter
    pass_dbg_rt = st.rt.ncolor && st.rt.color[0] ? (TexBackend *)st.rt.color[0]->backend : NULL;
    vp_sx = pass_sx; vp_sy = pass_sy; vp_ox = 0; vp_oy = 0;
    stat_passes++;
}

static void pass_size(int *w, int *h) {
    *w = *h = 0;
    for (int i = 0; i < st.rt.ncolor && !*w; i++)
        if (st.rt.color[i]) { *w = (int)st.rt.color[i]->width; *h = (int)st.rt.color[i]->height; }
    if (!*w && st.rt.depth) { *w = (int)st.rt.depth->width; *h = (int)st.rt.depth->height; }
    *w = (int)lroundf(*w * pass_sx);
    *h = (int)lroundf(*h * pass_sy);
}

static VkRect2D scissor_rect(void) {
    int w, h;
    pass_size(&w, &h);
    if (!w) return (VkRect2D){{0, 0}, {pass_w, pass_h}};
    int x = st.scissor[0][0], y = st.scissor[0][1], sw = st.scissor[0][2], sh = st.scissor[0][3];
    if (vp_sx != 1 || vp_sy != 1 || vp_ox != 0 || vp_oy != 0) {
        double x0 = floor(x * (double)vp_sx + vp_ox), y0 = floor(y * (double)vp_sy + vp_oy);
        double x1 = ceil(((double)x + sw) * vp_sx + vp_ox), y1 = ceil(((double)y + sh) * vp_sy + vp_oy);
        if (sw >= 16384) x1 = 1e9;
        if (sh >= 16384) y1 = 1e9;
        if (x1 > 1e9) x1 = 1e9;
        if (y1 > 1e9) y1 = 1e9;
        x = (int)x0; y = (int)y0; sw = (int)(x1 - x0); sh = (int)(y1 - y0);
    }
    if (x < 0) { sw += x; x = 0; }
    if (y < 0) { sh += y; y = 0; }
    if (w > (int)pass_w) w = (int)pass_w;
    if (h > (int)pass_h) h = (int)pass_h;
    if (x >= w || y >= h) { x = y = 0; sw = sh = 0; }
    if (x + sw > w) sw = w - x;
    if (y + sh > h) sh = h - y;
    if (sw < 0) sw = 0;
    if (sh < 0) sh = 0;
    return (VkRect2D){{x, y}, {(uint32_t)sw, (uint32_t)sh}};
}

static void apply_scissor(void) {
    VkRect2D r = scissor_rect();
    if (!ss_need(SS_SC, !memcmp(&r, &ss.sc, sizeof r))) return;
    ss.sc = r;
    vkCmdSetScissor(vk_cmd(), 0, 1, &r);
}

// Metal convention (y up in NDC, origin top-left) through a negative-height viewport.
static void set_viewport(float x, float y, float w, float h, float zn, float zf) {
    if (w <= 0) w = 1;
    if (h == 0) h = 1;
    VkViewport vp = {x, y + h, w, -h, fminf(fmaxf(zn, 0), 1), fminf(fmaxf(zf, 0), 1)};
    if (!ss_need(SS_VP, !memcmp(&vp, &ss.vp, sizeof vp))) return;
    ss.vp = vp;
    if (g_debug_frame && getenv("XCDE_TRACE_VP")) LOG("[vp] set %.1f %.1f %.1f %.1f z %.2f..%.2f\n", vp.x, vp.y, vp.width, vp.height, vp.minDepth, vp.maxDepth);
    vkCmdSetViewport(vk_cmd(), 0, 1, &vp);
}

// ================================================================== clears
static VkPipelineLayout pc_layout(void) {  // push constants only (clears)
    static VkPipelineLayout l;
    if (!l) {
        VkPushConstantRange r = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32};
        VkPipelineLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1,
                                         .pPushConstantRanges = &r};
        VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &l));
    }
    return l;
}

static const VkDynamicState DYN[] = {
    VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS, VK_DYNAMIC_STATE_BLEND_CONSTANTS,
    VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE,
    VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE, VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY,
    VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE, VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
    VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE, VK_DYNAMIC_STATE_STENCIL_OP, VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE,
    VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE};
#define NDYN (sizeof DYN / sizeof DYN[0])

// Fixed-function state that isn't part of a pipeline is set explicitly on every bind (it persists across pipelines).
static void dyn_defaults(VkCommandBuffer cb) {
    vkCmdSetDepthBias(cb, 0, 0, 0);
    vkCmdSetDepthBiasEnable(cb, VK_FALSE);
    float bc[4] = {0, 0, 0, 0};
    vkCmdSetBlendConstants(cb, bc);
    vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0xFF);
    vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0xFF);
    vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0);
    vkCmdSetCullMode(cb, VK_CULL_MODE_NONE);
    vkCmdSetFrontFace(cb, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    vkCmdSetPrimitiveTopology(cb, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    vkCmdSetDepthTestEnable(cb, VK_FALSE);
    vkCmdSetDepthWriteEnable(cb, VK_FALSE);
    vkCmdSetDepthCompareOp(cb, VK_COMPARE_OP_ALWAYS);
    vkCmdSetStencilTestEnable(cb, VK_FALSE);
    vkCmdSetStencilOp(cb, VK_STENCIL_FACE_FRONT_AND_BACK, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP,
                      VK_COMPARE_OP_ALWAYS);
    vkCmdSetPrimitiveRestartEnable(cb, VK_FALSE);
}

typedef struct { uint64_t key; VkPipeline pipe; } PipeEnt;
#define CLEAR_TAB 256
static PipeEnt clear_tab[CLEAR_TAB];

static VkPipeline clear_pipeline(int index, int kind, uint32_t mask, bool depth, bool stencil) {
    uint64_t key = 0xcbf29ce484222325ull;
#define MIX(v) do { uint64_t v_ = (uint64_t)(v); for (int b_ = 0; b_ < 8; b_++) { key ^= (v_ >> (b_ * 8)) & 0xFF; key *= 0x100000001B3ull; } } while (0)
    MIX(index); MIX(kind); MIX(mask); MIX(depth); MIX(stencil); MIX(pass_ncolor); MIX(pass_depth); MIX(pass_stencil); MIX(pass_samples);
    for (uint32_t i = 0; i < pass_ncolor; i++) MIX(pass_color[i]);
    uint32_t slot = (uint32_t)key & (CLEAR_TAB - 1);
    for (int p = 0; p < CLEAR_TAB; p++, slot = (slot + 1) & (CLEAR_TAB - 1)) {
        if (clear_tab[slot].key == key) return clear_tab[slot].pipe;
        if (!clear_tab[slot].key) break;
    }
    char fsname[32];
    if (index >= 0) snprintf(fsname, sizeof fsname, "clear.frag.%c%d", kind == 1 ? 'i' : kind == 2 ? 'u' : 'f', index);
    else snprintf(fsname, sizeof fsname, "clear.frag.none");
    VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vk_builtin_shader("clear.vert"), .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = vk_builtin_shader(fsname), .pName = "main"}};
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineViewportStateCreateInfo vs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = (VkSampleCountFlagBits)pass_samples};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState cba[8] = {0};
    for (uint32_t i = 0; i < pass_ncolor; i++)
        if ((int)i == index) cba[i].colorWriteMask = mask & 0xF;
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                              .attachmentCount = pass_ncolor, .pAttachments = cba};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                           .dynamicStateCount = NDYN, .pDynamicStates = DYN};
    VkPipelineRenderingCreateInfo rinfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = pass_ncolor,
                                           .pColorAttachmentFormats = pass_color, .depthAttachmentFormat = pass_depth,
                                           .stencilAttachmentFormat = pass_stencil};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &rinfo, .stageCount = 2,
                                       .pStages = stages, .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vs,
                                       .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &ds,
                                       .pColorBlendState = &cb, .pDynamicState = &dy, .layout = pc_layout()};
    VkPipeline p = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, &pi, NULL, &p) != VK_SUCCESS) LOG("[vk] clear pipeline failed\n");
    clear_tab[slot].key = key;
    clear_tab[slot].pipe = p;
    return p;
}

static void clear(int index, int kind, const void *color, uint32_t mask, bool depth, float dval, bool stencil, uint32_t sval,
                  uint32_t smask) {
    begin_pass();
    VkCommandBuffer cb = vk_cmd();
    vp_sx = pass_sx; vp_sy = pass_sy; vp_ox = 0; vp_oy = 0;
    VkRect2D rect = scissor_rect();
    if (!rect.extent.width || !rect.extent.height) return;
    if (index >= 0 && (index >= (int)pass_ncolor || pass_color[index] == VK_FORMAT_UNDEFINED)) return;
    bool simple = (index < 0 || (mask & 0xF) == 0xF) && (!stencil || (smask & 0xFF) == 0xFF);
    if (simple) {
        VkClearAttachment a[2];
        uint32_t n = 0;
        if (index >= 0) {
            a[n] = (VkClearAttachment){.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .colorAttachment = (uint32_t)index};
            memcpy(a[n].clearValue.color.float32, color, 16);
            n++;
        }
        VkImageAspectFlags dsa = (depth && pass_depth ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) | (stencil && pass_stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
        if (dsa) a[n++] = (VkClearAttachment){.aspectMask = dsa, .clearValue.depthStencil = {dval, sval}};
        VkClearRect cr = {rect, 0, 1};
        if (n) vkCmdClearAttachments(cb, n, a, 1, &cr);
        return;
    }
    VkPipeline p = clear_pipeline(index, kind, mask, depth, stencil);
    if (!p) return;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    ss_forget();  // (its own pipeline and state: the next draw sets everything again)
    dyn_defaults(cb);
    if (depth) { vkCmdSetDepthTestEnable(cb, VK_TRUE); vkCmdSetDepthWriteEnable(cb, VK_TRUE); }
    if (stencil) {
        vkCmdSetStencilTestEnable(cb, VK_TRUE);
        vkCmdSetStencilOp(cb, VK_STENCIL_FACE_FRONT_AND_BACK, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_REPLACE,
                          VK_COMPARE_OP_ALWAYS);
        vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, smask);
        vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, sval);
    }
    int w, h;
    pass_size(&w, &h);
    VkViewport vp = {0, 0, (float)(w ? w : (int)pass_w), (float)(h ? h : (int)pass_h), 0, 1};
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdSetScissor(cb, 0, 1, &rect);
    float pc[8] = {0};
    if (color) memcpy(pc, color, 16);
    pc[4] = dval;
    vkCmdPushConstants(cb, pc_layout(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32, pc);
    vkCmdDraw(cb, 3, 1, 0, 0);
}

// ================================================================== programs
#define MAX_BIND 96
typedef struct {
    ShaderInfo *vi, *fi;
    uint64_t code[2], cmax[2];  // translation inputs (copied: translated on the translator thread)
    uint64_t src[2], gsrc;      // [XCDE] T240: where the translator reads them (objects.c code_snapshot; code[] = the game's address, for logs)
    ShaderKey vkey, fkey;
    VkShaderModule vs, fs;
    ShaderInfo *gi;             // [XCDE] geometry stage (NULL without one): its key, code, module (descriptor set 2)
    ShaderKey gkey;
    uint64_t gcode, gcmax;
    VkShaderModule gs;
    bool no_fs;                 // [XCDE] the program has no fragment stage (depth-only passes): an empty one, no colour writes
    VkBindingInfo binds[MAX_BIND];
    int nbinds;
    VkDescriptorSetLayout set_layout[3];
    VkPipelineLayout layout;
    bool vs_uses_vid;
    _Atomic int state;  // 0 building, 1 ready, 2 failed
    // [XCDE] build-wait log: when it was asked for, its translation job started / got the translator / ended, its
    // compile ended (ns), the SPIR-V the disk cache didn't have, whether prewarm asked for it
    uint64_t t_new, t_x0, t_xl, t_x1, t_c1;
    uint8_t spv_new, prewarmed;
} Prog;

static VkPipelineLayout make_layout(Prog *p, int nsets, VkShaderStageFlags s0, VkShaderStageFlags s1) {
    p->set_layout[0] = vk_set_layout(p->binds, p->nbinds, 0, s0);
    if (nsets > 1) p->set_layout[1] = vk_set_layout(p->binds, p->nbinds, 1, s1);
    if (nsets > 2) p->set_layout[2] = vk_set_layout(p->binds, p->nbinds, 2, VK_SHADER_STAGE_GEOMETRY_BIT);  // [XCDE]
    VkPipelineLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = (uint32_t)nsets,
                                     .pSetLayouts = p->set_layout};
    VkPipelineLayout l;
    VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &l));
    return l;
}

static uint64_t hmix(uint64_t h, const void *p, size_t n);
static uint64_t fs_code_hash(uint64_t fs);
static uint64_t hash_words(uint64_t h, const void *p, size_t n);
static void save_glsl(const char *src, uint64_t addr, const char *suffix) {
    if (!getenv("LARECOMP_SAVE_MSL")) return;
    mkdir("build/glsl", 0755);
    char path[256];
    snprintf(path, sizeof path, "build/glsl/%llx_%s.glsl", (unsigned long long)addr, suffix);
    FILE *f = fopen(path, "w");
    if (f) { fputs(src, f); fclose(f); }
}

static void compile_program(void *arg) {
    Prog *p = arg;
    size_t nv = 0, nf = 0;
    const char *fs_src = p->fi->msl;
    char *fs_over = NULL;
    if (getenv("XCDE_GLSL_DIR")) {  // [XCDE] debugging: a hand-edited fragment shader (<dir>/<code addr>_fs.glsl) replaces the translated one
        char path[512];
        snprintf(path, sizeof path, "%s/%llx_fs.glsl", getenv("XCDE_GLSL_DIR"), (unsigned long long)p->code[1]);
        FILE *f = fopen(path, "rb");
        if (f) {
            fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
            fs_over = malloc((size_t)n + 1);
            if (fs_over && fread(fs_over, 1, (size_t)n, f) == (size_t)n) { fs_over[n] = 0; fs_src = fs_over; LOG("[vk] fragment shader %llx replaced from %s\n", (unsigned long long)p->code[1], path); }
            fclose(f);
        }
    }
    const char *vs_src = p->vi->msl;
    char *vs_over = NULL;
    if (getenv("XCDE_GLSL_DIR")) {  // [XCDE] the same for vertex shaders (<dir>/<code addr>_vs.glsl)
        char path[512];
        snprintf(path, sizeof path, "%s/%llx_vs.glsl", getenv("XCDE_GLSL_DIR"), (unsigned long long)p->code[0]);
        FILE *f = fopen(path, "rb");
        if (f) {
            fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
            vs_over = malloc((size_t)n + 1);
            if (vs_over && fread(vs_over, 1, (size_t)n, f) == (size_t)n) { vs_over[n] = 0; vs_src = vs_over; LOG("[vk] vertex shader %llx replaced from %s\n", (unsigned long long)p->code[0], path); }
            fclose(f);
        }
    }
    unsigned miss0 = vk_spirv_tl_misses;
    uint32_t *sv = vk_glsl_to_spirv(vs_src, STAGE_VERTEX, "vs", &nv);
    uint32_t *sf = sv ? vk_glsl_to_spirv(fs_src, STAGE_FRAGMENT, "fs", &nf) : NULL;
    free(fs_over); free(vs_over);
    if (!sv || !sf) { free(sv); free(sf); atomic_store(&p->state, 2); return; }
    p->nbinds = vk_reflect(sv, nv, p->binds, MAX_BIND);
    p->nbinds += vk_reflect(sf, nf, p->binds + p->nbinds, MAX_BIND - p->nbinds);
    if (p->gi) {  // [XCDE] geometry stage
        size_t ng = 0;
        uint32_t *sg = vk_glsl_to_spirv(p->gi->msl, STAGE_GEOMETRY, "gs", &ng);
        if (!sg) { free(sv); free(sf); atomic_store(&p->state, 2); return; }
        p->nbinds += vk_reflect(sg, ng, p->binds + p->nbinds, MAX_BIND - p->nbinds);
        p->gs = vk_module(sg, ng);
        free(sg);
    }
    p->vs = vk_module(sv, nv);
    p->fs = vk_module(sf, nf);
    free(sv);
    free(sf);
    p->layout = make_layout(p, p->gi ? 3 : 2, VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT);
    p->spv_new = (uint8_t)(vk_spirv_tl_misses - miss0);
    p->t_c1 = SDL_GetTicksNS();
    if (p->spv_new && !getenv("XCDE_PRECOMPILING")) {  // [XCDE] T211: a variant the precompilation didn't make - how it
        static int nl;                                // differs from its guess (float inputs / targets / textures)
        if (nl++ < 400) {
            uint32_t ai = 0, au = 0, rt = 0;
            for (int i = 0; i < 32; i++) { if (p->vkey.attr_kind[i] == 1) ai |= 1u << i; if (p->vkey.attr_kind[i] == 2) au |= 1u << i; }
            for (int i = 0; i < 8; i++) if (p->fkey.rt_kind[i]) rt |= (uint32_t)p->fkey.rt_kind[i] << (i * 2);
            LOG("[variant-miss] vs %llx fs %llx | in int %x uint %x | rt %x | tex_uint %x | xfb %d gs %d atest %d dm1 %d flip %d%d | no_fs %d | %d new\n",
                (unsigned long long)p->code[0], (unsigned long long)p->code[1], ai, au, rt, p->fkey.tex_uint, p->vkey.xfb_n,
                p->gi != NULL, p->fkey.alpha_func, p->vkey.depth_minus_one, p->vkey.flip_x, p->vkey.flip_y, p->no_fs, p->spv_new);
        }
    }
    atomic_fetch_add(&stat_compiles, 1);
    atomic_store(&p->state, p->vs && p->fs && (!p->gi || p->gs) ? 1 : 2);
}

static void translate_program(void *arg) {
    Prog *p = arg;
    maxwell_gmem_base = HEAP_BASE;
    maxwell_gmem_size = HEAP_MAX;
    p->t_x0 = SDL_GetTicksNS();
    xlat_lock();
    p->t_xl = SDL_GetTicksNS();
    static int trace_tr = -1;  // [XCDE] XCDE_TRACE_TRANSLATE=1: start / end of every translation (finds a translation that never ends)
    if (trace_tr < 0) trace_tr = getenv("XCDE_TRACE_TRANSLATE") != NULL;
    if (trace_tr) LOG("[xlat] begin vs=%llx fs=%llx""%c", (unsigned long long)p->code[0], (unsigned long long)p->code[1], 10);
    p->vi = maxwell_translate((const uint8_t *)(uintptr_t)p->src[0], p->cmax[0], STAGE_VERTEX, &p->vkey);
    if (p->vi->msl && p->gcode) {  // [XCDE] geometry stage: fed by the vertex outputs, feeds the fragment inputs
        p->gkey.vs_out_mask = p->vi->out_mask;
        p->gi = maxwell_translate((const uint8_t *)(uintptr_t)p->gsrc, p->gcmax, STAGE_GEOMETRY, &p->gkey);
        if (!p->gi->msl) LOG("[shader] GS translate failed: %s (code %llx)\n", p->gi->error, (unsigned long long)p->gcode);
    }
    if (p->vi->msl && (!p->gcode || p->gi->msl) && p->no_fs) {  // [XCDE] depth-only: an empty fragment shader
        p->fi = calloc(1, sizeof *p->fi);
        p->fi->stage = STAGE_FRAGMENT;
        p->fi->msl = strdup("#version 460\nvoid main() {}\n");
    } else if (p->vi->msl && (!p->gcode || p->gi->msl)) {
        p->fkey.vs_out_mask = p->gi ? p->gi->out_mask : p->vi->out_mask;
        p->fi = maxwell_translate((const uint8_t *)(uintptr_t)p->src[1], p->cmax[1], STAGE_FRAGMENT, &p->fkey);
    }
    if (trace_tr) LOG("[xlat] end   vs=%llx fs=%llx""%c", (unsigned long long)p->code[0], (unsigned long long)p->code[1], 10);
    pthread_mutex_unlock(&translate_lock);
    p->t_x1 = SDL_GetTicksNS();
    if (!p->vi->msl) {
        LOG("[shader] VS translate failed: %s (code %llx)\n", p->vi->error, (unsigned long long)p->code[0]);
        atomic_store(&p->state, 2);
        return;
    }
    if (!p->fi || !p->fi->msl) {
        if (p->fi) LOG("[shader] FS translate failed: %s\n", p->fi->error);
        atomic_store(&p->state, 2);
        return;
    }
    p->vs_uses_vid = strstr(p->vi->msl, "gl_VertexIndex") != NULL;
    save_glsl(p->vi->msl, p->code[0], "vs");
    save_glsl(p->fi->msl, p->code[1], "fs");
    {   // [FIX-TEX] XCDE_DUMP_FS_HASH=<hash>: writes the translated GLSL of the fragment shader with that hash (and the vertex shader
        // of the same program) to build/glsl/<hash>_fs.glsl / _vs.glsl. The hash is the one the [hunt] log lines / skip_fs.txt show.
        const char *dh = getenv("XCDE_DUMP_FS_HASH");
        if (dh && *dh && p->code[1]) {
            uint64_t got = fs_code_hash(p->code[1]), want = 0;  // [XCDE] a comma-separated list of hashes
            for (const char *q = dh; *q && !want;) { uint64_t w = strtoull(q, (char **)&q, 16); if (w == got) want = w; while (*q == ',') q++; if (!w) break; }
            if (want && got == want) {
                mkdir("build/glsl", 0755);
                const char *srcs[2] = {p->fi->msl, p->vi->msl}, *suf[2] = {"fs", "vs"};
                for (int k = 0; k < 2; k++) {
                    char path[256];
                    snprintf(path, sizeof path, "build/glsl/%016llx_%s.glsl", (unsigned long long)got, suf[k]);
                    FILE *f = fopen(path, "w");
                    if (f) { fputs(srcs[k], f); fclose(f); }
                }
                LOG("[tex-fix] dumped GLSL of fragment shader %016llx (vs %llx fs %llx) to build/glsl/%c", (unsigned long long)got,
                    (unsigned long long)p->code[0], (unsigned long long)p->code[1], 10);
            }
        }
    }
    if (p->gi) save_glsl(p->gi->msl, p->gcode, "gs");
    run_job(compile_program, p);
}

// [XCDE] T185: program and pipeline tables (render thread only) grow. Fixed at 4096 / 8192 entries with a probe loop that
// never ended once a table was full, the render thread spun forever after ~69 min of real-time play (soak run
// 2026-10-20: frame 89,121, render thread in get_program, no frame for the remaining 52 min). Entries are never removed
// (streamed shaders reuse guest addresses with new contents: new keys), so the tables must hold the whole session.
typedef struct { uint64_t key; void *val; } HEnt;
typedef struct { HEnt *e; uint32_t cap, n; const char *name; } HTab;
static HEnt *htab_slot(HTab *t, uint64_t h) {  // the entry of key h, or the empty one it would take (h != 0)
    if (!t->cap || (t->n + 1) * 2 > t->cap) {  // (grow at half full: short probes, and an empty slot always exists)
        HTab o = *t;
        t->cap = o.cap ? o.cap * 2 : 4096;
        t->e = calloc(t->cap, sizeof *t->e);
        t->n = 0;
        for (uint32_t i = 0; i < o.cap; i++) if (o.e[i].key) { *htab_slot(t, o.e[i].key) = o.e[i]; t->n++; }
        free(o.e);
        if (o.cap) LOG("[vk] %s table: %u entries, grown to %u%c", t->name, t->n, t->cap, 10);
    }
    uint32_t i = (uint32_t)(h ^ (h >> 31)) & (t->cap - 1);
    while (t->e[i].key && t->e[i].key != h) i = (i + 1) & (t->cap - 1);
    return &t->e[i];
}
static HTab prog_tab = {.name = "program"};

// [XCDE] Texture units of `stage` whose bound texture is a depth-stencil image read in STENCIL mode (NVN depth-stencil mode
// of the view if it sets one, else of the texture): the shader samples those as unsigned integers (ShaderKey.tex_uint).
// NVN: 0 depth, 1 stencil (seen in the game: the stencil-reading shader samples a mode-1 D24S8).
// [XCDE] perf: generation of everything stencil_unit_mask reads besides the bindings - texture pool entries
// (pool_write_apply) and texture objects (nvnTextureInitialize, objects.c). Its result is remembered per stage
// for the same pool, bindings and generation (it walked 32 units x 2 stages a draw: ~4 % of the render thread).
static _Atomic uint32_t g_tex_desc_gen;
void vk_tex_desc_changed(void) { atomic_fetch_add_explicit(&g_tex_desc_gen, 1, memory_order_relaxed); }
static uint32_t stencil_unit_mask_(int stage);
static uint32_t stencil_unit_mask(int stage) {
    static struct { uint32_t gen, bind_gen, mask; const void *tpool; bool valid; } memo[6];
    uint32_t g = atomic_load_explicit(&g_tex_desc_gen, memory_order_relaxed);
    typeof(memo[0]) *m = &memo[stage];
    if (m->valid && m->gen == g && m->bind_gen == g_tex_bind_gen[stage] && m->tpool == st.tpool) return m->mask;
    m->mask = stencil_unit_mask_(stage);
    m->gen = g; m->bind_gen = g_tex_bind_gen[stage]; m->tpool = st.tpool; m->valid = true;
    return m->mask;
}
static uint32_t stencil_unit_mask_(int stage) {
    if (!st.tpool) return 0;
    uint32_t mask = 0;
    for (int u = 0; u < 32; u++) {
        uint64_t h = st.tex[stage][u];
        uint32_t tic = h & 0xFFFFF;
        if (!h || tic >= (uint32_t)st.tpool->count) continue;
        const NvnTexEntry *e = &((NvnTexEntry *)st.tpool->entries)[tic];
        if (!e->tex) continue;
        {   // [XCDE] T151: an unsigned-integer texture (R8_UINT...) returns raw integers on Maxwell; a float sampler on it is
            // undefined in Vulkan (validation VUID-vkCmdDrawIndexed-format-07753): declare it usampler, raw bits
            uint32_t f = e->has_view && (e->view.setmask & 4) ? e->view.format : e->tex->format;
            switch (vk_format(f)) {
            case VK_FORMAT_R8_UINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_R16_UINT:
            case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16B16A16_UINT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R32G32_UINT:
            case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_A2B10G10R10_UINT_PACK32:
                mask |= 1u << u; continue;
            default: break;
            }
        }
        if (!renderer_format_has_stencil(e->tex->format)) continue;
        uint32_t mode = e->has_view && (e->view.setmask & 16) ? e->view.ds_mode : e->tex->ds_mode;
        if (getenv("XCDE_TRACE_DS")) {
            static uint32_t seen[8]; static int n;
            uint32_t sig = e->tex->format << 8 | mode;
            bool dup = false;
            for (int i = 0; i < n; i++) dup |= seen[i] == sig;
            if (!dup && n < 8) { seen[n++] = sig; LOG("[ds] stage %d unit %d fmt %#x view %d mode %u (texture %u)\n", stage, u, e->tex->format, e->has_view ? (int)e->view.ds_mode : -1, mode, e->tex->ds_mode); }
        }
        if (mode == 1) mask |= 1u << u;
    }
    return mask;
}

static uint8_t g_draw_gs_in = 1;  // [XCDE] input primitive class of the draw being set up (geometry shaders): 1 points, 2 lines, 3 triangles
static uint32_t prog_tab_gen;  // (bumped when prog_tab is cleared: get_program's last-call memo)
static bool g_prewarming;      // [XCDE] (build-wait log) prewarm_drain is asking for programs / pipelines
static Prog *get_program_(NvnProgram *np, ShaderKey *vkey, ShaderKey *fkey);

// [XCDE] T191: NVN alpha test as a fragment shader variant. XCDE_ALPHA_TEST=0: ignored (the old behaviour).
static bool alpha_test_on(void) {
    static int on = -1;
    if (on < 0) on = !getenv("XCDE_ALPHA_TEST") || atoi(getenv("XCDE_ALPHA_TEST")) != 0;
    return on;
}
// float -> half-float bits, round to nearest even (the alpha reference in the shader key)
static uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u, e = (x >> 23) & 0xFF, m = x & 0x7FFFFFu;
    if (e == 0xFF) return (uint16_t)(sign | 0x7C00u | (m ? 0x200u : 0));
    int he = (int)e - 127 + 15;
    if (he >= 31) return (uint16_t)(sign | 0x7C00u);
    if (he <= 0) {  // subnormal half (or zero)
        if (he < -10) return (uint16_t)sign;
        m |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - he), v = m >> shift, rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (v & 1))) v++;
        return (uint16_t)(sign | v);
    }
    uint32_t v = ((uint32_t)he << 10) | (m >> 13), rem = m & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (v & 1))) v++;  // (a carry into the exponent is still correct)
    return (uint16_t)(sign | v);
}
// [XCDE] perf: ~81 % of draws use the previous draw's program with the same keys: the byte-wise key hash is skipped then
static Prog *get_program(NvnProgram *np, ShaderKey *vkey, ShaderKey *fkey) {
    static NvnProgram *l_np;
    static ShaderKey l_v, l_f;
    static uint64_t l_code[3], l_hash[3];
    static uint8_t l_gs_in, l_mask;
    static uint32_t l_gen = ~0u;
    static Prog *l_p;
    if (l_p && np == l_np && l_gen == prog_tab_gen && l_gs_in == g_draw_gs_in && l_mask == np->stage_mask &&
        l_code[0] == np->stages[0].code_addr && l_code[1] == np->stages[1].code_addr && l_code[2] == np->stages[2].code_addr &&
        l_hash[0] == np->stages[0].code_hash && l_hash[1] == np->stages[1].code_hash && l_hash[2] == np->stages[2].code_hash &&
        !memcmp(vkey, &l_v, sizeof l_v) && !memcmp(fkey, &l_f, sizeof l_f))
        return l_p;
    Prog *p = get_program_(np, vkey, fkey);
    l_np = np; l_v = *vkey; l_f = *fkey; l_gs_in = g_draw_gs_in; l_mask = np->stage_mask; l_gen = prog_tab_gen; l_p = p;
    for (int i = 0; i < 3; i++) { l_code[i] = np->stages[i].code_addr; l_hash[i] = np->stages[i].code_hash; }
    return p;
}
static Prog *get_program_(NvnProgram *np, ShaderKey *vkey, ShaderKey *fkey) {
    if (!(np->stage_mask & 1)) return NULL;  // [XCDE] a program without a fragment stage gets an empty one (no_fs)
    // [XCDE] T185: keyed by content only (T125 hashes), not by guest address - the same shaders streamed in again at
    // another address made new programs and pipelines (~10,000 program builds in 69 min of soak play, private memory
    // +0.8 GB/h). Translation reads only the code and the key, so equal content gives the same program.
    uint64_t h = 0xcbf29ce484222325ull;
    for (int i = 0; i < 2; i++) { h ^= (uint64_t)(np->stages[i].code_addr != 0); h *= 0x100000001B3ull; }
    for (int i = 0; i < 3; i++) { h ^= np->stages[i].code_hash; h *= 0x100000001B3ull; }
    h = hash_words(h, vkey, sizeof *vkey);  // [XCDE] perf (was a byte at a time; the program table is in memory only)
    for (int i = 0; i < 8; i++) { h ^= fkey->rt_kind[i]; h *= 0x100000001B3ull; }
    for (int k = 0; k < 4; k++) { h ^= (fkey->tex_uint >> (k * 8)) & 0xFF; h *= 0x100000001B3ull; }  // [XCDE]
    if (fkey->alpha_func) { h ^= fkey->alpha_func | (uint64_t)fkey->alpha_ref << 8; h *= 0x100000001B3ull; }  // (T191: only then)
    ShaderKey gkey = {0}, vk2 = *vkey;  // [XCDE] geometry stage: it applies the clip-space epilogue (flip, depth remap), not the VS
    bool has_gs = (np->stage_mask & 4) && np->stages[2].code_addr;
    if (has_gs) {
        gkey.glsl = 1; gkey.flip_y = vkey->flip_y; gkey.flip_x = vkey->flip_x; gkey.depth_minus_one = vkey->depth_minus_one;
        gkey.gs_in = g_draw_gs_in; gkey.tex_uint = stencil_unit_mask(2);
        vk2.flip_y = vk2.flip_x = false; vk2.depth_minus_one = false;
        h ^= gkey.gs_in; h *= 0x100000001B3ull;
        for (int k = 0; k < 4; k++) { h ^= (gkey.tex_uint >> (k * 8)) & 0xFF; h *= 0x100000001B3ull; }
    }
    if (!h) h = 1;
    HEnt *slot = htab_slot(&prog_tab, h);
    if (slot->key) return slot->val;
    Prog *p = calloc(1, sizeof *p);
    slot->key = h;
    slot->val = p;
    prog_tab.n++;
    for (int i = 0; i < 2; i++) {
        p->code[i] = np->stages[i].code_addr;
        p->src[i] = nvn_stage_code(&np->stages[i]);
        p->cmax[i] = np->stages[i].const_offset ? np->stages[i].const_offset : 0x10000;
    }
    p->vkey = has_gs ? vk2 : *vkey;
    p->fkey = *fkey;
    p->no_fs = !(np->stage_mask & 2) || !np->stages[1].code_addr;
    if (has_gs) {
        p->gkey = gkey;
        p->gcode = np->stages[2].code_addr;
        p->gsrc = nvn_stage_code(&np->stages[2]);
        p->gcmax = np->stages[2].const_offset ? np->stages[2].const_offset : 0x10000;
    }
    p->t_new = SDL_GetTicksNS();
    p->prewarmed = g_prewarming;
    run_translate(translate_program, p);
    return p;
}

// ================================================================== vertex formats / fixed function tables
static int attr_kind(uint32_t fmt) {
    switch (fmt) {
    case 0x03: case 0x08: case 0x0B: case 0x0F: case 0x14: case 0x17: case 0x1B: case 0x20: case 0x23:
    case 0x27: case 0x2C: case 0x2F: case 0x3E:
        return 2;
    case 0x04: case 0x09: case 0x0C: case 0x10: case 0x15: case 0x18: case 0x1C: case 0x21: case 0x24:
    case 0x28: case 0x2D: case 0x30:
        return 1;
    default: return 0;
    }
}

static VkFormat vertex_format(uint32_t f) {
    switch (f) {
    case 0x01: return VK_FORMAT_R8_UNORM;
    case 0x02: return VK_FORMAT_R8_SNORM;
    case 0x03: return VK_FORMAT_R8_UINT;
    case 0x04: return VK_FORMAT_R8_SINT;
    case 0x05: return VK_FORMAT_R16_SFLOAT;
    case 0x06: return VK_FORMAT_R16_UNORM;
    case 0x07: return VK_FORMAT_R16_SNORM;
    case 0x08: return VK_FORMAT_R16_UINT;
    case 0x09: return VK_FORMAT_R16_SINT;
    case 0x0A: return VK_FORMAT_R32_SFLOAT;
    case 0x0B: return VK_FORMAT_R32_UINT;
    case 0x0C: return VK_FORMAT_R32_SINT;
    case 0x0D: return VK_FORMAT_R8G8_UNORM;
    case 0x0E: return VK_FORMAT_R8G8_SNORM;
    case 0x0F: return VK_FORMAT_R8G8_UINT;
    case 0x10: return VK_FORMAT_R8G8_SINT;
    case 0x11: return VK_FORMAT_R16G16_SFLOAT;
    case 0x12: return VK_FORMAT_R16G16_UNORM;
    case 0x13: return VK_FORMAT_R16G16_SNORM;
    case 0x14: return VK_FORMAT_R16G16_UINT;
    case 0x15: return VK_FORMAT_R16G16_SINT;
    case 0x16: return VK_FORMAT_R32G32_SFLOAT;
    case 0x17: return VK_FORMAT_R32G32_UINT;
    case 0x18: return VK_FORMAT_R32G32_SINT;
    case 0x19: return VK_FORMAT_R8G8B8_UNORM;
    case 0x1A: return VK_FORMAT_R8G8B8_SNORM;
    case 0x1B: return VK_FORMAT_R8G8B8_UINT;
    case 0x1C: return VK_FORMAT_R8G8B8_SINT;
    case 0x1D: return VK_FORMAT_R16G16B16_SFLOAT;
    case 0x1E: return VK_FORMAT_R16G16B16_UNORM;
    case 0x1F: return VK_FORMAT_R16G16B16_SNORM;
    case 0x20: return VK_FORMAT_R16G16B16_UINT;
    case 0x21: return VK_FORMAT_R16G16B16_SINT;
    case 0x22: return VK_FORMAT_R32G32B32_SFLOAT;
    case 0x23: return VK_FORMAT_R32G32B32_UINT;
    case 0x24: return VK_FORMAT_R32G32B32_SINT;
    case 0x25: return VK_FORMAT_R8G8B8A8_UNORM;
    case 0x26: return VK_FORMAT_R8G8B8A8_SNORM;
    case 0x27: return VK_FORMAT_R8G8B8A8_UINT;
    case 0x28: return VK_FORMAT_R8G8B8A8_SINT;
    case 0x29: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case 0x2A: return VK_FORMAT_R16G16B16A16_UNORM;
    case 0x2B: return VK_FORMAT_R16G16B16A16_SNORM;
    case 0x2C: return VK_FORMAT_R16G16B16A16_UINT;
    case 0x2D: return VK_FORMAT_R16G16B16A16_SINT;
    case 0x2E: return VK_FORMAT_R32G32B32A32_SFLOAT;
    case 0x2F: return VK_FORMAT_R32G32B32A32_UINT;
    case 0x30: return VK_FORMAT_R32G32B32A32_SINT;
    case 0x3D: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case 0x69: return VK_FORMAT_A2B10G10R10_SNORM_PACK32;
    case 0x3F: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case 0x40: return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
    case 0x51: return VK_FORMAT_R8_UINT;
    default: {  // [FIX-TEX] unknown NVN vertex format: read as 4 floats (garbage colours); say so once per format
        static uint8_t seen_f[256];
        if (f < 256 && !seen_f[f]++) LOG("[tex-fix] unknown NVN vertex format %#x: read as R32G32B32A32_SFLOAT%c", f, 10);
        return VK_FORMAT_R32G32B32A32_SFLOAT;
    }
    }
}

// 3-component 8/16-bit vertex formats aren't universally supported: read 4 components instead.
static VkFormat vertex_format_supported(VkFormat f) {
    static uint8_t known[256];  // 1 ok, 2 not
    uint32_t i = f < 256 ? f : 0;
    if (!known[i]) {
        VkFormatProperties p;
        vkGetPhysicalDeviceFormatProperties(g_vk.phys, f, &p);
        known[i] = (p.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT) ? 1 : 2;
        // [XCDE] spec_lowend_gpu N2: XCDE_VFMT_SIM=amd treats the 3-component 8 / 16-bit formats AMD's drivers lack as
        // unsupported, to test the fallback on another GPU
        static int sim = -1;
        if (sim < 0) sim = getenv("XCDE_VFMT_SIM") && !strcmp(getenv("XCDE_VFMT_SIM"), "amd");
        if (sim && ((f >= VK_FORMAT_R8G8B8_UNORM && f <= VK_FORMAT_R8G8B8_SRGB) || (f >= VK_FORMAT_R16G16B16_UNORM && f <= VK_FORMAT_R16G16B16_SFLOAT)))
            known[i] = 2;
        LOG("[vk] vertex format %d in use%s\n", f, known[i] == 1 ? "" : " (not supported)");
    }
    if (known[i] == 1) return f;
    { static uint8_t told[256]; if (!told[i]++) LOG("[vk] vertex format %d unsupported: read as 4 components\n", f); }
    switch (f) {
    case VK_FORMAT_R8G8B8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_R8G8B8_SNORM: return VK_FORMAT_R8G8B8A8_SNORM;
    case VK_FORMAT_R8G8B8_UINT: return VK_FORMAT_R8G8B8A8_UINT;
    case VK_FORMAT_R8G8B8_SINT: return VK_FORMAT_R8G8B8A8_SINT;
    case VK_FORMAT_R16G16B16_SFLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case VK_FORMAT_R16G16B16_UNORM: return VK_FORMAT_R16G16B16A16_UNORM;
    case VK_FORMAT_R16G16B16_SNORM: return VK_FORMAT_R16G16B16A16_SNORM;
    case VK_FORMAT_R16G16B16_UINT: return VK_FORMAT_R16G16B16A16_UINT;
    case VK_FORMAT_R16G16B16_SINT: return VK_FORMAT_R16G16B16A16_SINT;
    default: return VK_FORMAT_R32G32B32A32_SFLOAT;
    }
}

static VkBlendFactor blend_factor(uint8_t f) {
    switch (f) {
    case 1: return VK_BLEND_FACTOR_ZERO;
    case 2: return VK_BLEND_FACTOR_ONE;
    case 3: return VK_BLEND_FACTOR_SRC_COLOR;
    case 4: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case 5: return VK_BLEND_FACTOR_SRC_ALPHA;
    case 6: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case 7: return VK_BLEND_FACTOR_DST_ALPHA;
    case 8: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case 9: return VK_BLEND_FACTOR_DST_COLOR;
    case 10: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case 11: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    case 0x10: return VK_BLEND_FACTOR_SRC1_COLOR;
    case 0x11: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
    case 0x12: return VK_BLEND_FACTOR_SRC1_ALPHA;
    case 0x13: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
    case 0x61: return VK_BLEND_FACTOR_CONSTANT_COLOR;
    case 0x62: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    case 0x63: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
    case 0x64: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    default: {
        static uint8_t warned[256];
        if (!warned[f]++) LOG("[vk] unknown blend factor %#x\n", f);
        return VK_BLEND_FACTOR_ONE;
    }
    }
}

static VkBlendOp blend_op(uint8_t e) {
    switch (e) {
    case 2: return VK_BLEND_OP_SUBTRACT;
    case 3: return VK_BLEND_OP_REVERSE_SUBTRACT;
    case 4: return VK_BLEND_OP_MIN;
    case 5: return VK_BLEND_OP_MAX;
    default: return VK_BLEND_OP_ADD;
    }
}

static VkCompareOp compare_fn(uint8_t f) {
    switch (f) {
    case 1: return VK_COMPARE_OP_NEVER;
    case 2: return VK_COMPARE_OP_LESS;
    case 3: return VK_COMPARE_OP_EQUAL;
    case 4: return VK_COMPARE_OP_LESS_OR_EQUAL;
    case 5: return VK_COMPARE_OP_GREATER;
    case 6: return VK_COMPARE_OP_NOT_EQUAL;
    case 7: return VK_COMPARE_OP_GREATER_OR_EQUAL;
    default: return VK_COMPARE_OP_ALWAYS;
    }
}

static VkStencilOp stencil_op(uint8_t o) {
    switch (o) {
    case 2: return VK_STENCIL_OP_ZERO;
    case 3: return VK_STENCIL_OP_REPLACE;
    case 4: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
    case 5: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
    case 6: return VK_STENCIL_OP_INVERT;
    case 7: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
    case 8: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
    default: return VK_STENCIL_OP_KEEP;
    }
}

static int rt_kind(uint32_t f) {
    switch (f) {
    case 0x03: case 0x08: case 0x0B: case 0x0F: case 0x14: case 0x17: case 0x27: case 0x2C: case 0x2F: case 0x3E: return 2;
    case 0x04: case 0x09: case 0x0C: case 0x10: case 0x15: case 0x18: case 0x28: case 0x2D: case 0x30: return 1;
    default: return 0;
    }
}

// ================================================================== graphics pipelines
// Everything a pipeline bakes in, captured by value so it can be built on a worker thread.
#define ZERO_BINDING 16
typedef struct {
    Prog *prog;
    VkFormat color[8], depth, stencil;
    uint32_t ncolor, samples;
    uint32_t attr_fmt[32], attr_off[32], attr_bind[32];
    uint32_t attr_mask;
    uint32_t stride[17], divisor[17], used_streams;
    uint8_t blend[8][8];
    uint8_t blend_on, a2c, topo_class, poly_fill;
    uint32_t write_mask[8];
    uint8_t logic_op;  // [XCDE] T176: 0 off, else VkLogicOp + 1 (last: it sits in the old tail padding, so the
                       // variant records in cache/variants.bin keep their size and read as "off")
} PipeKey;

typedef struct {
    PipeKey k; uint64_t hash; VkPipeline pipe; _Atomic int state;  // state: 0 building, 1 ready, 2 failed
    uint64_t t_new, t_b0, t_b1;  // [XCDE] build-wait log: asked for / build started / ended (ns)
    uint8_t how, libs_new;       // 1 whole from the disk cache, 2 linked, 3 whole compile; libraries built for it (bit = kind)
                                 // (how: + 0x80 prewarmed, + 0x40 its optimized pipeline waits for its first draw - T218)
    _Atomic int jobs;            // [XCDE] build / optimize jobs queued for it (eviction leaves it alone meanwhile)
    uint64_t last_used;          // [XCDE] frame of its last draw (eviction: the least recently drawn go first)
    void *libs[4];               // [XCDE] T242: a linked pipeline's libraries (GplLib *, referenced until it is destroyed)
} Pipe;

// [XCDE] A draw that waited over 8 ms for its program or pipeline (a frame-time hitch): what it waited for. Program: ms
// from being asked for to its translation job starting (queue), getting the translator (lock), translating, compiling
// (SPIR-V the disk cache didn't have: new = a variant precompilation never produced); its shaders and key. Pipeline:
// queue, build, how and the libraries built for it (vi / pre = vertex shader native compile / fs = fragment shader
// native compile / fo). At most 400 lines a run. XCDE_BUILD_WAIT_MS: the threshold.
static uint64_t build_wait_min(void) {  // XCDE_BUILD_WAIT_MS=<ms>: the threshold (default 8)
    static uint64_t v;
    if (!v) { const char *e = getenv("XCDE_BUILD_WAIT_MS"); v = (uint64_t)((e ? atof(e) : 8.0) * 1e6); if (!v) v = 1; }
    return v;
}
static void build_wait_log(const Prog *p, const Pipe *pp, uint64_t waited) {
    static int n;
    if (n++ >= (getenv("XCDE_BUILD_WAIT_MS") ? 5000 : 400)) return;
    double ms = waited / 1e6;
    #define BW_MS(a, b) ((a) && (b) && (b) >= (a) ? ((b) - (a)) / 1e6 : -1.0)
    if (p) {
        uint32_t ai = 0, au = 0, rt = 0;
        for (int i = 0; i < 32; i++) { if (p->vkey.attr_kind[i] == 1) ai |= 1u << i; if (p->vkey.attr_kind[i] == 2) au |= 1u << i; }
        for (int i = 0; i < 8; i++) if (p->fkey.rt_kind[i]) rt |= (uint32_t)p->fkey.rt_kind[i] << (i * 2);
        LOG("[build-wait] program %.1f ms%s: queue %.1f lock %.1f translate %.1f compile %.1f (%d SPIR-V new) | vs %llx fs %llx gs %d |"
            " vkey int %x uint %x dm1 %d flip %d%d xfb %d | fkey rt %x tex_uint %x atest %d%c",
            ms, p->prewarmed ? " (prewarmed)" : "", BW_MS(p->t_new, p->t_x0), BW_MS(p->t_x0, p->t_xl), BW_MS(p->t_xl, p->t_x1),
            BW_MS(p->t_x1, p->t_c1), p->spv_new, (unsigned long long)p->code[0], (unsigned long long)p->code[1], p->gi != NULL,
            ai, au, p->vkey.depth_minus_one, p->vkey.flip_x, p->vkey.flip_y, p->vkey.xfb_n, rt, p->fkey.tex_uint, p->fkey.alpha_func, 10);
    } else {
        static const char *how[] = {"?", "cached", "linked", "whole"};
        int l = pp->libs_new;
        LOG("[build-wait] pipeline %.1f ms%s: queue %.1f build %.1f %s, new libs%s%s%s%s | prog vs %llx fs %llx%c",
            ms, pp->how & 0x80 ? " (prewarmed)" : "", BW_MS(pp->t_new, pp->t_b0), BW_MS(pp->t_b0, pp->t_b1), how[pp->how & 3],
            l & 1 ? " vi" : "", l & 2 ? " pre" : "", l & 4 ? " fs" : "", l & 8 ? " fo" : "",
            (unsigned long long)pp->k.prog->code[0], (unsigned long long)pp->k.prog->code[1], 10);
    }
    #undef BW_MS
}
static HTab pipe_tab = {.name = "pipeline"};  // [XCDE] T185 (grows: htab_slot)
static uint32_t pipe_count(void) { return pipe_tab.n; }

// [XCDE] T195: state Vulkan ignores is normalized before a key is hashed, so draws that differ only there share one
// pipeline (Mario Wonder: 18 % fewer). Same results by construction: a target with no channel written (or no fragment
// stage) has no blending; the advanced blend mode byte is never applied (audited); the alpha factors / equation of a
// target whose alpha isn't written, and the colour ones of a target whose RGB isn't written, give values that are
// discarded - set to the NVN defaults (ONE, ZERO, ADD). XCDE_CANON=0: off.
static void pipe_key_canon(PipeKey *k) {
    static int on = -1;
    if (on < 0) on = !(getenv("XCDE_CANON") && atoi(getenv("XCDE_CANON")) == 0);
    if (!on) return;
    for (int i = 0; i < 8; i++) {
        if (k->prog && k->prog->no_fs) k->write_mask[i] = 0;
        uint8_t *b = k->blend[i];
        if (!((k->blend_on >> i) & 1) || !k->write_mask[i]) {
            k->blend_on &= (uint8_t)~(1u << i);
            memset(b, 0, 8);
            continue;
        }
        b[7] = 0;
        if (!(k->write_mask[i] & 8)) { b[3] = 2; b[4] = 1; b[6] = 1; }
        if (!(k->write_mask[i] & 7)) { b[1] = 2; b[2] = 1; b[5] = 1; }
    }
}

// 8 bytes a step (PipeKey is ~700 bytes, hashed per draw); in-memory tables only - hmix values are compared with
// stored constants (fragment-shader hashes) and must not change.
// [XCDE] perf: four independent lanes over 32-byte steps (one serial multiply chain was ~2.6 % of the render thread
// in Colony 9 on the Steam Deck: the ~700-byte pipeline key every draw). Only in-memory tables use these values
// (variants.bin keeps the records and re-hashes them on load).
static uint64_t hash_words(uint64_t h, const void *p, size_t n) {
    const uint8_t *b = p;
    if (n >= 32) {
        uint64_t h1 = h ^ 0xC2B2AE3D27D4EB4Full, h2 = h ^ 0x165667B19E3779F9ull, h3 = h ^ 0x27D4EB2F165667C5ull;
        for (; n >= 32; n -= 32, b += 32) {
            uint64_t v[4];
            memcpy(v, b, 32);
            h = (h ^ v[0]) * 0x9E3779B97F4A7C15ull; h ^= h >> 29;
            h1 = (h1 ^ v[1]) * 0x9E3779B97F4A7C15ull; h1 ^= h1 >> 29;
            h2 = (h2 ^ v[2]) * 0x9E3779B97F4A7C15ull; h2 ^= h2 >> 29;
            h3 = (h3 ^ v[3]) * 0x9E3779B97F4A7C15ull; h3 ^= h3 >> 29;
        }
        h ^= h1 * 0xC2B2AE3D27D4EB4Full ^ h2 * 0x165667B19E3779F9ull ^ h3 * 0x27D4EB2F165667C5ull;
    }
    for (; n >= 8; n -= 8, b += 8) { uint64_t w; memcpy(&w, b, 8); h = (h ^ w) * 0x9E3779B97F4A7C15ull; h ^= h >> 29; }
    for (; n; n--, b++) { h = (h ^ *b) * 0x100000001B3ull; }
    return h ^ (h >> 32);
}

static uint64_t hmix(uint64_t h, const void *p, size_t n) {
    const uint8_t *b = p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001B3ull; }
    return h;
}

// [XCDE] perf: hmix of a fragment program's code (fs + 0x30, 0x200 bytes: the hash the skip / depth-of-field lists
// compare against), remembered per code address. hmix runs a byte at a time (~1,500 cycles); the program changes
// between most draws (~6 % of the render thread on the Steam Deck). An entry also keeps the first 16 code bytes, so a
// program loaded later at the same address gets hashed again. (Render thread only.)
static uint64_t fs_code_hash(uint64_t fs) {
    if (!fs) return 0;
    static struct { uint64_t fs, head[2], hash; } tab[4096];
    const uint64_t *code = (const uint64_t *)(uintptr_t)(fs + 0x30);
    typeof(tab[0]) *e = &tab[(fs >> 4) & 4095];
    if (e->fs == fs && e->head[0] == code[0] && e->head[1] == code[1]) return e->hash;
    e->fs = fs; e->head[0] = code[0]; e->head[1] = code[1];
    return e->hash = hmix(1469598103934665603ull, code, 0x200);
}

static VkPrimitiveTopology topo_class_base(uint8_t c) {
    return c == 0 ? VK_PRIMITIVE_TOPOLOGY_POINT_LIST : c == 1 ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
}

// ---- [XCDE] T188: graphics pipeline libraries (VK_EXT_graphics_pipeline_library, g_vk.gpl)
// A pipeline is four pieces: vertex input (attributes, streams, topology class), pre-rasterization (vertex / geometry
// shader, polygon mode), fragment shader (+ multisample) and fragment output (target formats, blend, write masks). Each
// piece is built once per its own key and kept; a new combination is a fast link of four pieces (well under a
// millisecond) instead of a whole compile (10-100 ms on a fast CPU). The pipeline with whole-program optimization is
// then compiled on the low-priority queue and replaces the linked one; next session the disk cache returns it directly
// (first try: FAIL_ON_PIPELINE_COMPILE_REQUIRED). Same SPIR-V, same fixed-function state: the same results.
// LARECOMP_VK_GPL=0 (device.c): whole pipelines as before. Deterministic runs: whole pipelines, unless XCDE_GPL_DET=1
// (linked pipelines only, built synchronously - to compare captures).
typedef struct { uint64_t key; VkPipeline lib; _Atomic int state; uint64_t last_used; _Atomic int refs; } GplLib;
// (refs: linked pipelines alive that were made from it - NVIDIA's driver needs the libraries while they exist: one
// destroyed under a fast-linked pipeline crashed in the next draw)  // state 0 building, 1 ready, 2 failed, 3 evicted
#define GPL_TAB 16384
static GplLib gpl_tab[GPL_TAB];
static pthread_mutex_t gpl_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t gpl_cond = PTHREAD_COND_INITIALIZER;
static pthread_rwlock_t gpl_rw = PTHREAD_RWLOCK_INITIALIZER;  // [XCDE] T242: links hold it shared, library eviction exclusive
static struct { VkPipeline p; uint64_t seq; void *libs[4]; } gpl_dead[1024];  // replaced linked pipelines (render thread frees them)
static int gpl_ndead;

static const VkDynamicState DYN_VI[] = {VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY, VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE};
static const VkDynamicState DYN_PRE[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS,
                                         VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE, VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE};
static const VkDynamicState DYN_FS[] = {VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
                                        VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
                                        VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE, VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
                                        VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE, VK_DYNAMIC_STATE_STENCIL_OP};
static const VkDynamicState DYN_FO[] = {VK_DYNAMIC_STATE_BLEND_CONSTANTS};

// The library for `key` (kind in its top bits), built by `make` if nobody has it yet; another thread building the same
// key is waited for. VK_NULL_HANDLE: failed.
static _Thread_local uint8_t tl_libs_new;  // (build-wait log: libraries this thread built, bit = kind)
static _Atomic uint64_t gpl_libs_total;  // [XCDE] T212: libraries in memory (T242: evicted over the budget, lib_evict)
// [XCDE] T212: libraries are prebuilt ahead only while fewer than a budget are in memory (~90 KB each, measured: one
// area's ~3,500 libraries ~ +320 MB): 12,000 with 24 GB of RAM or more, 6,000 from 12 GB, else 3,000.
// XCDE_LIB_BUDGET=<n>. Libraries a draw needs are still built on demand past it.
static uint64_t lib_budget(void) {
    static uint64_t budget;
    if (!budget) {
        const char *e = getenv("XCDE_LIB_BUDGET");
        int mb = SDL_GetSystemRAM();
        budget = e ? strtoull(e, NULL, 10) : mb >= 24000 ? 12000 : mb >= 12000 ? 6000 : 3000;
        if (!budget) budget = 1;
    }
    return budget;
}
static bool prebuild_room(void) { return atomic_load(&gpl_libs_total) < lib_budget(); }
static VkPipeline gpl_get_e(uint64_t key, VkResult (*make)(const PipeKey *, VkPipeline *), const PipeKey *k, GplLib **ent);
static VkPipeline gpl_get(uint64_t key, VkResult (*make)(const PipeKey *, VkPipeline *), const PipeKey *k) {
    return gpl_get_e(key, make, k, NULL);
}
static VkPipeline gpl_get_e(uint64_t key, VkResult (*make)(const PipeKey *, VkPipeline *), const PipeKey *k, GplLib **ent) {
    if (!key) key = 1;
    uint32_t i = (uint32_t)(key ^ (key >> 29)) & (GPL_TAB - 1);
    uint64_t now = renderer_present_count();
    GplLib *tomb = NULL;  // [XCDE] T242: the first evicted entry on the way (a new key may take it)
    pthread_mutex_lock(&gpl_lock);
    for (int n = 0; n < GPL_TAB; n++, i = (i + 1) & (GPL_TAB - 1)) {
        GplLib *e = &gpl_tab[i];
        if (e->key == key && atomic_load(&e->state) != 3) {
            while (atomic_load(&e->state) == 0) pthread_cond_wait(&gpl_cond, &gpl_lock);
            e->last_used = now;
            pthread_mutex_unlock(&gpl_lock);
            if (ent) *ent = e;
            return atomic_load(&e->state) == 1 ? e->lib : VK_NULL_HANDLE;
        }
        if (e->key && atomic_load(&e->state) == 3 && !tomb) tomb = e;
        if (!e->key || n == GPL_TAB - 1) {
            if (tomb) e = tomb;  // (the key is in no other slot: the probe reached an empty one)
            else if (e->key) break;  // (table full)
            e->key = key;
            e->last_used = now;
            atomic_store(&e->state, 0);
            pthread_mutex_unlock(&gpl_lock);
            VkPipeline lib = VK_NULL_HANDLE;
            tl_libs_new |= (uint8_t)(1u << (key >> 62));
            VkResult r = make(k, &lib);
            pthread_mutex_lock(&gpl_lock);
            e->lib = lib;
            atomic_store(&e->state, r == VK_SUCCESS ? 1 : 2);
            pthread_cond_broadcast(&gpl_cond);
            pthread_mutex_unlock(&gpl_lock);
            if (ent) *ent = e;
            if (r == VK_SUCCESS) { atomic_fetch_add(&stat_gpl_libs, 1); atomic_fetch_add(&gpl_libs_total, 1); }
            else { static int nl; if (nl++ < 10) LOG("[vk] pipeline library failed: %d (kind %llu)\n", r, (unsigned long long)(key >> 62)); }
            return r == VK_SUCCESS ? lib : VK_NULL_HANDLE;
        }
    }
    pthread_mutex_unlock(&gpl_lock);
    return VK_NULL_HANDLE;  // (table full)
}

static VkResult gpl_make(const VkGraphicsPipelineCreateInfo *base, VkGraphicsPipelineLibraryFlagsEXT what, VkPipeline *out) {
    VkGraphicsPipelineLibraryCreateInfoEXT gl = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT,
                                                 .pNext = base->pNext, .flags = what};
    VkGraphicsPipelineCreateInfo pi = *base;
    pi.pNext = &gl;
    // [XCDE] T212: no RETAIN_LINK_TIME_OPTIMIZATION_INFO: nothing links these with link-time optimization (the optimized
    // pipeline that replaces a linked one is compiled whole, gpl_optimize), and the retained info made each library
    // ~180 KB (one area's ~3,500 libraries: +630 MB of process memory, never freed). XCDE_GPL_RETAIN=1: keep it.
    static int retain = -1;
    if (retain < 0) retain = getenv("XCDE_GPL_RETAIN") && atoi(getenv("XCDE_GPL_RETAIN")) != 0;
    pi.flags |= VK_PIPELINE_CREATE_LIBRARY_BIT_KHR | (retain ? VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT : 0);
    return vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, &pi, NULL, out);
}

static VkResult gpl_vi(const PipeKey *k, VkPipeline *out) {
    VkVertexInputAttributeDescription va[32];
    VkVertexInputBindingDescription vb[17];
    VkVertexInputBindingDivisorDescriptionEXT vd[17];
    uint32_t na = 0, nb = 0, nd = 0;
    for (int i = 0; i < 32; i++)
        if (k->attr_mask & (1u << i)) va[na++] = (VkVertexInputAttributeDescription){(uint32_t)i, k->attr_bind[i], k->attr_fmt[i], k->attr_off[i]};
    for (uint32_t s = 0; s < 17; s++) {
        if (!(k->used_streams & (1u << s))) continue;
        bool inst = k->divisor[s] != 0;
        vb[nb++] = (VkVertexInputBindingDescription){s, k->stride[s], inst ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX};
        if (inst && k->divisor[s] > 1 && g_vk.divisor) vd[nd++] = (VkVertexInputBindingDivisorDescriptionEXT){s, k->divisor[s]};
    }
    VkPipelineVertexInputDivisorStateCreateInfoEXT dvi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT,
                                                          .vertexBindingDivisorCount = nd, .pVertexBindingDivisors = vd};
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, .pNext = nd ? &dvi : NULL,
                                               .vertexBindingDescriptionCount = nb, .pVertexBindingDescriptions = vb,
                                               .vertexAttributeDescriptionCount = na, .pVertexAttributeDescriptions = va};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = topo_class_base(k->topo_class)};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                           .dynamicStateCount = sizeof DYN_VI / sizeof DYN_VI[0], .pDynamicStates = DYN_VI};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pVertexInputState = &vi,
                                       .pInputAssemblyState = &ia, .pDynamicState = &dy};
    return gpl_make(&pi, VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT, out);
}

static VkResult gpl_pre(const PipeKey *k, VkPipeline *out) {
    VkPipelineShaderStageCreateInfo st[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = k->prog->vs, .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_GEOMETRY_BIT, .module = k->prog->gs, .pName = "main"}};
    VkPipelineViewportStateCreateInfo vs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .polygonMode = k->poly_fill ? VK_POLYGON_MODE_FILL : VK_POLYGON_MODE_LINE, .lineWidth = 1};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                           .dynamicStateCount = sizeof DYN_PRE / sizeof DYN_PRE[0], .pDynamicStates = DYN_PRE};
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri, .stageCount = k->prog->gs ? 2 : 1,
                                       .pStages = st, .pViewportState = &vs, .pRasterizationState = &rs, .pDynamicState = &dy,
                                       .layout = k->prog->layout};
    return gpl_make(&pi, VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT, out);
}

static VkResult gpl_fs(const PipeKey *k, VkPipeline *out) {
    VkPipelineShaderStageCreateInfo st = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                                          .module = k->prog->fs, .pName = "main"};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = (VkSampleCountFlagBits)k->samples, .alphaToCoverageEnable = k->a2c};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                           .dynamicStateCount = sizeof DYN_FS / sizeof DYN_FS[0], .pDynamicStates = DYN_FS};
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri, .stageCount = 1, .pStages = &st,
                                       .pMultisampleState = &ms, .pDepthStencilState = &ds, .pDynamicState = &dy, .layout = k->prog->layout};
    return gpl_make(&pi, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT, out);
}

static VkResult gpl_fo(const PipeKey *k, VkPipeline *out) {
    VkPipelineColorBlendAttachmentState cba[8] = {0};
    for (uint32_t i = 0; i < k->ncolor; i++) {
        cba[i].colorWriteMask = k->prog->no_fs ? 0 : k->write_mask[i];
        if ((k->blend_on >> i) & 1) {
            const uint8_t *b = k->blend[i];
            cba[i].blendEnable = VK_TRUE;
            cba[i].srcColorBlendFactor = blend_factor(b[1]);
            cba[i].dstColorBlendFactor = blend_factor(b[2]);
            cba[i].srcAlphaBlendFactor = blend_factor(b[3]);
            cba[i].dstAlphaBlendFactor = blend_factor(b[4]);
            cba[i].colorBlendOp = blend_op(b[5]);
            cba[i].alphaBlendOp = blend_op(b[6]);
        }
    }
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                              .attachmentCount = k->ncolor, .pAttachments = cba,
                                              .logicOpEnable = k->logic_op != 0, .logicOp = k->logic_op ? (VkLogicOp)(k->logic_op - 1) : VK_LOGIC_OP_COPY};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = (VkSampleCountFlagBits)k->samples, .alphaToCoverageEnable = k->a2c};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                           .dynamicStateCount = sizeof DYN_FO / sizeof DYN_FO[0], .pDynamicStates = DYN_FO};
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = k->ncolor,
                                        .pColorAttachmentFormats = k->color, .depthAttachmentFormat = k->depth,
                                        .stencilAttachmentFormat = k->stencil};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri, .pColorBlendState = &cb,
                                       .pMultisampleState = &ms, .pDynamicState = &dy};
    return gpl_make(&pi, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT, out);
}

// The four pieces of k, linked (no link-time optimization: the fast path).
// [XCDE] T211: the pre-rasterization / fragment-shader library keys (shared by gpl_link and the prebuild below)
static uint64_t gpl_key_pre(const PipeKey *k) {
    uint64_t w[4] = {(uint64_t)(uintptr_t)k->prog->vs, (uint64_t)(uintptr_t)k->prog->gs, (uint64_t)(uintptr_t)k->prog->layout, k->poly_fill};
    return (hash_words(0xC2B2AE3D27D4EB4Full, w, sizeof w) & ~(3ull << 62)) | (1ull << 62);
}
static uint64_t gpl_key_fs(const PipeKey *k) {
    uint64_t w[4] = {(uint64_t)(uintptr_t)k->prog->fs, (uint64_t)(uintptr_t)k->prog->layout, k->samples, k->a2c};
    return (hash_words(0x165667B19E3779F9ull, w, sizeof w) & ~(3ull << 62)) | (2ull << 62);
}
static bool gpl_ready(uint64_t key) {  // (the library is built and in memory; no waiting)
    if (!key) key = 1;
    uint32_t i = (uint32_t)(key ^ (key >> 29)) & (GPL_TAB - 1);
    bool r = false;
    pthread_mutex_lock(&gpl_lock);
    for (int n = 0; n < GPL_TAB; n++, i = (i + 1) & (GPL_TAB - 1)) {
        GplLib *e = &gpl_tab[i];
        if (e->key == key) { r = atomic_load(&e->state) == 1; break; }
        if (!e->key) break;
    }
    pthread_mutex_unlock(&gpl_lock);
    return r;
}
// [XCDE] T211: a program the game registered (loading screen) gets its shader libraries built ahead on the workers, for
// the fill mode / sample count almost every draw uses: its first draws then only link (vertex input and fragment output
// pieces have no shaders: cheap) instead of compiling or loading a whole pipeline from the driver cache (2-60 ms each,
// the "load burst" and new-area hitches of the [hitch] log). Discrete GPUs only (the libraries stay in memory).
static void gpl_prebuild(void *arg) {
    Prog *pr = arg;
    if (atomic_load(&pr->state) != 1 || !pr->vs || !pr->layout || !prebuild_room()) return;
    PipeKey k;
    memset(&k, 0, sizeof k);
    k.prog = pr; k.poly_fill = 1; k.samples = 1;
    gpl_get(gpl_key_pre(&k), gpl_pre, &k);
    if (pr->fs) gpl_get(gpl_key_fs(&k), gpl_fs, &k);
}
static bool prebuild_libs_on(void);
// [XCDE] T212: pipelines are linked from the shader libraries (built if missing, waited for if being built) rather
// than loaded whole from the driver cache, and the recorded variants' pipelines aren't built ahead (only their
// libraries): a program's libraries serve all its pipelines. Measured: an area's first frame waited 0.8 - 1.3 s for
// ~600 pipelines loaded whole one by one on the render thread (2-5 ms each). With libraries prebuilt (on a discrete
// GPU) or XCDE_PREWARM_WHOLE=1: whole pipelines first as before.
static bool link_first(void) {
    static int on = -1;
    if (on < 0) on = prebuild_libs_on() && !(getenv("XCDE_PREWARM_WHOLE") && atoi(getenv("XCDE_PREWARM_WHOLE")) != 0);
    return on;
}
// [XCDE] T218: whole pipelines built ahead (no pipeline libraries / integrated GPU) stay in memory: ~230 KB each
// (NVIDIA, measured: 2,771 -> 7,185 of them on the coldroot route once no registration was dropped, +975 MB). Below
// 24 GB of RAM they are built for the first programs of each registration batch only, as before T218 (when the queue
// kept 256): 256 below 12 GB, 768 below 24 GB. XCDE_WHOLE_BATCH=<n>.
static int whole_batch_max(void) {
    static int v;
    if (!v) {
        const char *e = getenv("XCDE_WHOLE_BATCH");
        int mb = SDL_GetSystemRAM();
        v = e ? atoi(e) : mb >= 24000 ? 1 << 30 : mb >= 12000 ? 768 : 256;
        if (v <= 0) v = 1;
    }
    return v;
}
// [XCDE] T218 (prewarm_drain): with 24 GB of RAM or more (the linked pipelines: ~+260 MB for one area, measured;
// the area's first frame 231 -> 195 ms). XCDE_PRELINK=0/1.
static bool prelink_on(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("XCDE_PRELINK"); on = e ? atoi(e) != 0 : SDL_GetSystemRAM() >= 24000; }
    return on;
}
// [XCDE] T212: the same for a program with recorded variants (cache/variants.bin, the seed), once per fill mode /
// sample count / alpha-to-coverage its variants use. Before, those programs only got their recorded pipelines queued
// whole: their libraries were never in memory, so every new state combination met in play went the slow route (a
// whole pipeline from the driver cache, 2-3 ms each; ~280 at once when an enemy / effect first appeared: 110-226 ms
// frames in the user's log). With the libraries ready it is a link (~0.3 ms).
typedef struct { Prog *pr; uint8_t poly_fill, samples, a2c; } LibJob;
static void gpl_prebuild_variant(void *arg) {
    LibJob *j = arg;
    Prog *pr = j->pr;
    if (atomic_load(&pr->state) == 1 && pr->vs && pr->layout && prebuild_room()) {
        PipeKey k;
        memset(&k, 0, sizeof k);
        k.prog = pr; k.poly_fill = j->poly_fill; k.samples = j->samples; k.a2c = j->a2c;
        gpl_get(gpl_key_pre(&k), gpl_pre, &k);
        if (pr->fs) gpl_get(gpl_key_fs(&k), gpl_fs, &k);
    }
    free(j);
}

static void libs_release(void *libs[4]) {
    for (int i = 0; i < 4; i++) if (libs[i]) { atomic_fetch_sub(&((GplLib *)libs[i])->refs, 1); libs[i] = NULL; }
}
static VkResult gpl_link(const PipeKey *k, VkPipeline *out, void *ents[4]) {
    uint64_t kv = 0x9E3779B97F4A7C15ull, kp, kf, ko;
    {   // vertex input: attributes, streams, topology class
        uint32_t w[32 * 3 + 17 * 2 + 3], n = 0;
        w[n++] = k->attr_mask; w[n++] = k->used_streams; w[n++] = k->topo_class;
        for (int i = 0; i < 32; i++) if (k->attr_mask & (1u << i)) { w[n++] = k->attr_fmt[i]; w[n++] = k->attr_off[i]; w[n++] = k->attr_bind[i]; }
        for (int s = 0; s < 17; s++) if (k->used_streams & (1u << s)) { w[n++] = k->stride[s]; w[n++] = k->divisor[s]; }
        kv = hash_words(kv, w, n * 4);
    }
    kp = gpl_key_pre(k);
    kf = gpl_key_fs(k);
    {   uint32_t w[8 + 3 + 3 + 16 + 8], n = 0;
        w[n++] = k->ncolor; w[n++] = k->depth; w[n++] = k->stencil; w[n++] = k->samples; w[n++] = k->a2c; w[n++] = k->prog->no_fs;
        w[n++] = k->blend_on; w[n++] = k->logic_op;
        for (uint32_t i = 0; i < k->ncolor && i < 8; i++) { w[n++] = k->color[i]; w[n++] = k->write_mask[i]; }
        ko = hash_words(0x27D4EB2F165667C5ull, w, n * 4);
        for (uint32_t i = 0; i < k->ncolor && i < 8; i++) if ((k->blend_on >> i) & 1) ko = hash_words(ko, k->blend[i], 8);
    }
    // (kind in the top two bits: the four tables share one hash table)
    pthread_rwlock_rdlock(&gpl_rw);  // [XCDE] T242: no library eviction while their handles are in use
    GplLib *e[4] = {0};
    VkPipeline libs[4] = {gpl_get_e((kv & ~(3ull << 62)) | (0ull << 62), gpl_vi, k, &e[0]), gpl_get_e(kp, gpl_pre, k, &e[1]),
                          gpl_get_e(kf, gpl_fs, k, &e[2]), gpl_get_e((ko & ~(3ull << 62)) | (3ull << 62), gpl_fo, k, &e[3])};
    VkResult r = VK_ERROR_UNKNOWN;
    if (libs[0] && libs[1] && libs[2] && libs[3]) {
        VkPipelineLibraryCreateInfoKHR li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR, .libraryCount = 4, .pLibraries = libs};
        VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &li, .layout = k->prog->layout};
        r = vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, &pi, NULL, out);
        if (r == VK_SUCCESS)  // (counted before the shared lock goes: eviction only takes unreferenced libraries)
            for (int i = 0; i < 4; i++) { atomic_fetch_add(&e[i]->refs, 1); ents[i] = e[i]; }
    }
    pthread_rwlock_unlock(&gpl_rw);
    return r;
}

static VkResult pipeline_whole(const PipeKey *k, VkPipelineCreateFlags flags, VkPipeline *out);

// low-priority job: the optimized whole pipeline replaces the linked one (the render thread frees the old one once
// every submission that may use it has completed: gpl_reap)
static bool gpl_optimize_on(void) {  // [XCDE] T242 measurement: XCDE_GPL_OPTIMIZE=0 keeps the fast-linked pipelines
    static int on = -1;
    if (on < 0) on = !(getenv("XCDE_GPL_OPTIMIZE") && atoi(getenv("XCDE_GPL_OPTIMIZE")) == 0);
    return on;
}
static void gpl_optimize_(void *arg);
static void gpl_optimize(void *arg) { gpl_optimize_(arg); atomic_fetch_sub(&((Pipe *)arg)->jobs, 1); }
static void gpl_optimize_(void *arg) {
    Pipe *pp = arg;
    VkPipeline p = VK_NULL_HANDLE;
    if (pipeline_whole(&pp->k, 0, &p) != VK_SUCCESS) return;  // (the linked one stays)
    pthread_mutex_lock(&gpl_lock);
    if (gpl_ndead == 1024) { pthread_mutex_unlock(&gpl_lock); vkDestroyPipeline(g_vk.dev, p, NULL); return; }  // (keep the linked one)
    VkPipeline old = pp->pipe;
    __atomic_store_n(&pp->pipe, p, __ATOMIC_RELEASE);
    gpl_dead[gpl_ndead].p = old;
    memcpy(gpl_dead[gpl_ndead].libs, pp->libs, sizeof pp->libs);
    memset(pp->libs, 0, sizeof pp->libs);
    gpl_dead[gpl_ndead++].seq = 0;  // (stamped by the render thread)
    pthread_mutex_unlock(&gpl_lock);
    atomic_fetch_add(&stat_gpl_opt, 1);
}

// [XCDE] Pipeline eviction. Pipelines were never destroyed: a 2 h 19 min soak across every map went from 5,680 to
// 21,081 and the process from 6.1 to 11.4 GB (~0.32 MB of driver memory each) - past what an 8 GB PC has. Over
// XCDE_PIPE_BUDGET live pipelines (0 = never evict; default by RAM, T241: 8000 below 12 GB, 16000 below 24 GB, else
// 24000 - every PC gets a cap, pipelines pile up with 32 GB too), the ones not drawn for XCDE_PIPE_MIN_AGE frames
// (default 3600, ~1-2 min) go, least recently drawn first, down to 90 % of the budget. Only ready ones with no build /
// optimize job queued; the Pipe stays in the table (jobs and keys point at it) with state 3, and its VkPipeline is
// destroyed once the GPU is past every list recorded so far. A later draw builds it again (whole from the driver's
// cache, or linked: ~0.3 ms) - and an area's loading screen prewarms its pipelines anyway.
static uint64_t stat_pipe_evicted, stat_pipe_rebuilt;
static uint32_t pipe_live;  // (ready or building: not evicted)
static struct { VkPipeline p; uint64_t seq; void *libs[4]; } *pe_dead;
static size_t pe_ndead, pe_cap;
static int pe_cmp(const void *a, const void *b) {
    uint64_t x = (*(Pipe *const *)a)->last_used, y = (*(Pipe *const *)b)->last_used;
    return x < y ? -1 : x > y;
}
// [XCDE] T242: the GPL libraries are what grew in long sessions (35 min over the battle tour: +465 MB with them, +63 MB
// with LARECOMP_VK_GPL=0; pipelines themselves freed ~0.025 MB each when evicted). Over the library budget (lib_room's,
// by RAM: 12,000 / 6,000 / 3,000) the ones no link asked for in min_age frames are destroyed, least recently used first,
// to 90 %: a linked pipeline does not need its libraries any more, and a later link builds them again. Only when no
// link is using library handles (the shared gpl_rw; otherwise next round) - the render thread never waits.
static uint64_t lib_budget(void);
static uint64_t stat_lib_evicted;
static int le_cmp(const void *a, const void *b) {
    uint64_t x = (*(GplLib *const *)a)->last_used, y = (*(GplLib *const *)b)->last_used;
    return x < y ? -1 : x > y;
}
static void lib_evict(uint64_t now, uint64_t min_age) {
    uint64_t total = atomic_load(&gpl_libs_total), budget = lib_budget();
    if (total <= budget || pthread_rwlock_trywrlock(&gpl_rw)) return;
    pthread_mutex_lock(&gpl_lock);
    static GplLib *c[GPL_TAB];
    uint32_t nc = 0;
    for (uint32_t i = 0; i < GPL_TAB; i++)
        if (gpl_tab[i].key && atomic_load(&gpl_tab[i].state) == 1 && gpl_tab[i].lib && !atomic_load(&gpl_tab[i].refs) &&
            gpl_tab[i].last_used + min_age < now)
            c[nc++] = &gpl_tab[i];
    qsort(c, nc, sizeof *c, le_cmp);
    uint64_t want = total - budget * 9 / 10, n = want < nc ? want : nc;
    for (uint64_t i = 0; i < n; i++) {
        vkDestroyPipeline(g_vk.dev, c[i]->lib, NULL);  // (libraries are never bound: no GPU work uses them)
        c[i]->lib = VK_NULL_HANDLE;
        atomic_store(&c[i]->state, 3);
    }
    atomic_fetch_sub(&gpl_libs_total, n);
    pthread_mutex_unlock(&gpl_lock);
    pthread_rwlock_unlock(&gpl_rw);
    if (n) {
        stat_lib_evicted += n;
        static int nlog;
        if (nlog++ < 20 || nlog % 100 == 0)
            LOG("[vk] library eviction: %llu not used for %llu+ frames destroyed (%llu left, budget %llu; %llu so far)\n",
                (unsigned long long)n, (unsigned long long)min_age, (unsigned long long)(total - n), (unsigned long long)budget,
                (unsigned long long)stat_lib_evicted);
    }
}

static void pipe_evict(void) {  // (render thread)
    static uint64_t reaped_at;  // [XCDE] perf: the dead list once per submission, not every draw (a timeline query each)
    if (pe_ndead && vk_pending_seq() != reaped_at) {
        reaped_at = vk_pending_seq();
        uint64_t done = vk_completed_seq();
        size_t k = 0;
        for (size_t i = 0; i < pe_ndead; i++)
            if (pe_dead[i].seq <= done) { vkDestroyPipeline(g_vk.dev, pe_dead[i].p, NULL); libs_release(pe_dead[i].libs); }
            else pe_dead[k++] = pe_dead[i];
        pe_ndead = k;
    }
    static int budget = -1, min_age;
    static uint64_t next;
    if (budget < 0) {
        const char *e = getenv("XCDE_PIPE_BUDGET"), *a = getenv("XCDE_PIPE_MIN_AGE");
        int ram = SDL_GetSystemRAM();  // (MB)
        budget = e ? atoi(e) : ram < 12000 ? 8000 : ram < 24000 ? 16000 : 24000;
        LOG("[vk] pipeline budget %d (%d MB of RAM)%c", budget, ram, 10);
        min_age = a ? atoi(a) : 3600;
    }
    uint64_t now = renderer_present_count();
    if (now < next) return;
    next = now + 30;  // (at most twice a second at 60 fps)
    lib_evict(now, (uint64_t)min_age);  // (its own budget: XCDE_LIB_BUDGET)
    if (budget <= 0) return;
    pipe_live = 0;
    for (uint32_t i = 0; i < pipe_tab.cap; i++) {
        Pipe *pp = pipe_tab.e[i].val;
        if (pp && atomic_load(&pp->state) != 3) pipe_live++;
    }
    int effective_budget = budget;
    VmaBudget heaps[VK_MAX_MEMORY_HEAPS];
    VkPhysicalDeviceMemoryProperties memory;
    vmaGetHeapBudgets(g_vk.vma, heaps);
    memory = g_vk.memory_props;
    for (uint32_t i = 0; i < memory.memoryHeapCount && g_vk.props.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU; ++i)  // [PERF-V5] not on UMA iGPUs
        if (heaps[i].budget && heaps[i].usage > heaps[i].budget - heaps[i].budget / 10) {
            effective_budget = budget * 3 / 4; break;
        }
    if (effective_budget < 1) effective_budget = 1;
    if (pipe_live <= (uint32_t)effective_budget) return;
    Pipe **c = malloc(pipe_live * sizeof *c);
    if (!c) return;
    uint32_t nc = 0;
    for (uint32_t i = 0; i < pipe_tab.cap; i++) {
        Pipe *pp = pipe_tab.e[i].val;
        if (pp && atomic_load(&pp->state) == 1 && !atomic_load(&pp->jobs) && pp->pipe && pp->last_used + (uint64_t)min_age < now)
            c[nc++] = pp;
    }
    qsort(c, nc, sizeof *c, pe_cmp);
    uint32_t want = pipe_live - (uint32_t)effective_budget * 9 / 10, n = want < nc ? want : nc;
    uint64_t seq = vk_pending_seq();  // (lists recorded until now may use them)
    for (uint32_t i = 0; i < n; i++) {
        if (pe_ndead == pe_cap) {
            size_t next_cap = pe_cap ? pe_cap * 2 : 1024;
            void *next_dead = realloc(pe_dead, next_cap * sizeof *pe_dead);
            if (!next_dead) { n = i; break; } // keep existing handles alive if retirement bookkeeping cannot grow
            pe_dead = next_dead; pe_cap = next_cap;
        }
        pe_dead[pe_ndead].p = c[i]->pipe;
        memcpy(pe_dead[pe_ndead].libs, c[i]->libs, sizeof c[i]->libs);
        memset(c[i]->libs, 0, sizeof c[i]->libs);
        pe_dead[pe_ndead++].seq = seq;
        c[i]->pipe = VK_NULL_HANDLE;
        atomic_store(&c[i]->state, 3);
    }
    free(c);
    if (n) {
        ss.known &= ~SS_PIPE;  // (a new pipeline may get a destroyed one's handle: bind again)
        pipe_live -= n;
        stat_pipe_evicted += n;
        static int nlog;
        if (nlog++ < 20 || nlog % 100 == 0)
            LOG("[vk] pipeline eviction: %u not drawn for %d+ frames destroyed (%u live, budget %d; %llu evicted, %llu rebuilt so far)\n",
                n, min_age, pipe_live, budget, (unsigned long long)stat_pipe_evicted, (unsigned long long)stat_pipe_rebuilt);
    }
}

static void gpl_reap(void) {  // (render thread)
    if (!__atomic_load_n(&gpl_ndead, __ATOMIC_RELAXED)) return;
    static uint64_t reaped_at;  // [XCDE] perf: once per submission, not every draw (a timeline query and a lock each)
    if (vk_pending_seq() == reaped_at) return;
    reaped_at = vk_pending_seq();
    uint64_t done = vk_completed_seq(), pend = vk_pending_seq();
    pthread_mutex_lock(&gpl_lock);
    int k = 0;
    for (int i = 0; i < gpl_ndead; i++) {
        if (!gpl_dead[i].seq) gpl_dead[i].seq = pend;  // draws recorded until now may still use it
        if (gpl_dead[i].seq <= done) { vkDestroyPipeline(g_vk.dev, gpl_dead[i].p, NULL); libs_release(gpl_dead[i].libs); }
        else gpl_dead[k++] = gpl_dead[i];
    }
    gpl_ndead = k;
    pthread_mutex_unlock(&gpl_lock);
}

void vk_hitch_counters(uint64_t c[9]) {
    c[0] = hc_wait_ns; c[1] = hc_wait_n; c[2] = atomic_load(&hc_cached_ns); c[3] = atomic_load(&hc_cached_n);
    c[4] = atomic_load(&hc_link_ns); c[5] = atomic_load(&hc_link_n); c[6] = atomic_load(&hc_whole_ns); c[7] = atomic_load(&hc_whole_n);
    c[8] = hc_upload;
}

static void build_pipeline_(void *arg);
static void build_pipeline(void *arg) { build_pipeline_(arg); atomic_fetch_sub(&((Pipe *)arg)->jobs, 1); }
static void build_pipeline_(void *arg) {
    Pipe *pp = arg;
    static int det_gpl = -1;
    if (det_gpl < 0) det_gpl = getenv("XCDE_GPL_DET") != NULL;
    VkPipeline p = VK_NULL_HANDLE;
    VkResult r;
    {   // [XCDE] T240 test: XCDE_TEST_SLOW_BUILD=<ms> - every pipeline build takes that much longer, so draws wait for
        // theirs as on a cold first viewing (Shulk's head missing for 2 frames after 9 build waits, lab scene)
        static int slow = -1;
        if (slow < 0) slow = getenv("XCDE_TEST_SLOW_BUILD") ? atoi(getenv("XCDE_TEST_SLOW_BUILD")) : 0;
        if (slow > 0) SDL_Delay((uint32_t)slow);
    }
    pp->t_b0 = SDL_GetTicksNS();
    tl_libs_new = 0;
    if (g_vk.gpl && (!sync_shaders() || det_gpl)) {
        // a whole pipeline the disk cache already holds: no compile
        // [XCDE] T211: link first when the shader libraries are already in memory (~0.3 ms; a whole pipeline from the
        // driver cache took 2-17 ms): the optimized one replaces it from the low-priority queue as for any link
        bool libs_ready = !det_gpl && (link_first() || (gpl_ready(gpl_key_pre(&pp->k)) && (!pp->k.prog->fs || gpl_ready(gpl_key_fs(&pp->k)))));
        // [XCDE] T240: a prewarmed pipeline (an area's / a scene's recorded variant, built on its loading screen) is
        // taken whole from the driver's cache when the precompilation put it there (precompile.c builds every recorded
        // variant's optimized pipeline): the scene starts with its final pipelines instead of fast links that a
        // low-priority job replaces after their first draw. A miss (not precompiled, another driver) links as before.
        static int precompiled = -1;
        if (precompiled < 0) { FILE *pf = fopen("cache/precompile_done.txt", "r"); precompiled = pf != NULL; if (pf) fclose(pf); }
        if ((pp->how & 0x80) && precompiled && !det_gpl) libs_ready = false;
        r = det_gpl || libs_ready ? VK_PIPELINE_COMPILE_REQUIRED
                                  : pipeline_whole(&pp->k, g_vk.pipeline_cache_control ? VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT : 0, &p);
        if (r == VK_SUCCESS) {
            atomic_fetch_add(&stat_gpl_cached, 1);
            atomic_fetch_add(&hc_cached_ns, SDL_GetTicksNS() - pp->t_b0); atomic_fetch_add(&hc_cached_n, 1);
            pp->how = (pp->how & 0x80) | 1;
        } else {
            uint64_t t_l0 = SDL_GetTicksNS();
            void *ents[4] = {0};
            r = gpl_link(&pp->k, &p, ents);
            if (r == VK_SUCCESS) {
                atomic_fetch_add(&stat_gpl_linked, 1);
                // [XCDE] T218: a prewarmed link (nothing drew it yet) gets its optimized pipeline at its first draw
                bool later = (pp->how & 0x80) != 0;
                pp->how = (pp->how & 0x80) | (later ? 0x40 : 0) | 2; pp->libs_new = tl_libs_new; pp->t_b1 = SDL_GetTicksNS();
                atomic_fetch_add(&hc_link_ns, pp->t_b1 - t_l0); atomic_fetch_add(&hc_link_n, 1);
                pp->pipe = p;
                memcpy(pp->libs, ents, sizeof pp->libs);
                atomic_fetch_add(&stat_compiles, 1);
                atomic_store(&pp->state, 1);
                if (!sync_shaders() && !later && gpl_optimize_on()) { atomic_fetch_add(&pp->jobs, 1); run_job_low(gpl_optimize, pp); }
                return;
            }
            r = pipeline_whole(&pp->k, 0, &p);  // (a piece failed: the whole pipeline as before)
            pp->how = (pp->how & 0x80) | 3;
        }
    } else {
        r = pipeline_whole(&pp->k, 0, &p);
        pp->how = (pp->how & 0x80) | 3;
    }
    pp->libs_new = tl_libs_new; pp->t_b1 = SDL_GetTicksNS();
    if ((pp->how & 3) == 3) { atomic_fetch_add(&hc_whole_ns, pp->t_b1 - pp->t_b0); atomic_fetch_add(&hc_whole_n, 1); }
    if (r != VK_SUCCESS) {
        static int n;
        if (n++ < 10) LOG("[vk] pipeline creation failed: %d\n", r);
    }
    pp->pipe = p;
    atomic_fetch_add(&stat_compiles, 1);
    atomic_store(&pp->state, r == VK_SUCCESS ? 1 : 2);
}

typedef struct { const VkGraphicsPipelineCreateInfo *pi; VkPipeline *out; VkResult r; } PwCall;
static void pw_call(void *a) {
    PwCall *c = a;
    c->r = vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, c->pi, NULL, c->out);
}
// [VEGA-FIX] what a pipeline the driver crashed on was made of (build/last_game.log)
static void pw_log_key(const PipeKey *k, int n) {
    char b[1500]; int o = 0;
    o += snprintf(b + o, sizeof b - o, "[vk] DRIVER CRASH #%d (divide by zero) in a pipeline: vs %llx fs %llx gs %d | ncolor %u color", n,
                  (unsigned long long)k->prog->code[0], (unsigned long long)k->prog->code[1], k->prog->gi != NULL, k->ncolor);
    for (uint32_t i = 0; i < k->ncolor && i < 8; i++) o += snprintf(b + o, sizeof b - o, " %d(mask %x blend %d)", (int)k->color[i], k->write_mask[i], (k->blend_on >> i) & 1);
    o += snprintf(b + o, sizeof b - o, " | depth %d stencil %d | samples %u a2c %d | topo %d fill %d logic %d | no_fs %d | attrs %x streams %x |",
                  (int)k->depth, (int)k->stencil, k->samples, k->a2c, k->topo_class, k->poly_fill, k->logic_op, k->prog->no_fs, k->attr_mask, k->used_streams);
    for (int s = 0; s < 17; s++) if (k->used_streams & (1u << s)) o += snprintf(b + o, sizeof b - o, " s%d(stride %u div %u)", s, k->stride[s], k->divisor[s]);
    o += snprintf(b + o, sizeof b - o, " | fmt");
    for (int i = 0; i < 32; i++) if (k->attr_mask & (1u << i)) o += snprintf(b + o, sizeof b - o, " a%d:%u@%u/b%u", i, k->attr_fmt[i], k->attr_off[i], k->attr_bind[i]);
    LOG("%s%c", b, 10);
}

// [FIX-TEX] XCDE_DUMP_FS_HASH=<hash>: every new pipeline drawn with that fragment shader logs its vertex layout (streams, formats, offsets)
static void attr_layout_log(const PipeKey *k, uint64_t fsh) {
    static int n;
    if (n++ >= 24) return;
    char b[1500]; int o = 0;
    o += snprintf(b + o, sizeof b - o, "[tex-fix] pipeline of fragment shader %016llx: vs %llx fs %llx | ncolor %u | attrs %x streams %x |", (unsigned long long)fsh,
                  (unsigned long long)k->prog->code[0], (unsigned long long)k->prog->code[1], k->ncolor, k->attr_mask, k->used_streams);
    for (int s = 0; s < 17; s++) if (k->used_streams & (1u << s)) o += snprintf(b + o, sizeof b - o, " s%d(stride %u div %u)", s, k->stride[s], k->divisor[s]);
    o += snprintf(b + o, sizeof b - o, " | VkFormat@offset/stream:");
    for (int i = 0; i < 32; i++) if (k->attr_mask & (1u << i)) o += snprintf(b + o, sizeof b - o, " a%d:%u@%u/b%u", i, k->attr_fmt[i], k->attr_off[i], k->attr_bind[i]);
    LOG("%s%c", b, 10);
}
static int g_mat_blue;  // [FIX-TEX] the last cb4 of the watched fragment shader had colour (0,0,1,1)
// [FIX-TEX] XCDE_BLUE_FIX=1 (or "r,g,b", 0..1): the toldo/tent material shader 69dc9c67ee198e5d draws untextured surfaces with colour = vertex colour
// x cb4[2]. Thousands of its draws read a placeholder material cb4 = (0.2,0.5,0.5,1 | 0,0,0,0 | 0,0,1,1) whose colour is pure blue; this
// replaces that cb4[2] by a neutral colour (default 0.9,0.8,0.65) in the constant-buffer copy. Other materials of the shader are untouched.
static const float *blue_fix_color(void) {
    static int init; static float c[3]; static int on;
    if (!init) {
        init = 1;
        const char *e = getenv("XCDE_BLUE_FIX");
        if (e && *e && strcmp(e, "0")) {
            c[0] = 0.9f; c[1] = 0.8f; c[2] = 0.65f;
            if (strchr(e, ',')) sscanf(e, "%f,%f,%f", &c[0], &c[1], &c[2]);
            on = 1;
        }
    }
    return on ? c : NULL;
}
static uint64_t dump_fs_want(void) {
    static int init; static uint64_t w;
    if (!init) { init = 1; const char *e = getenv("XCDE_DUMP_FS_HASH"); w = e && *e ? strtoull(e, NULL, 16) : 0; }
    return w;
}
static VkResult pipeline_whole(const PipeKey *k, VkPipelineCreateFlags flags, VkPipeline *out) {
    VkPipelineShaderStageCreateInfo stages[3] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = k->prog->vs, .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = k->prog->fs, .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_GEOMETRY_BIT, .module = k->prog->gs, .pName = "main"}};
    VkVertexInputAttributeDescription va[32];
    VkVertexInputBindingDescription vb[17];
    VkVertexInputBindingDivisorDescriptionEXT vd[17];
    uint32_t na = 0, nb = 0, nd = 0;
    for (int i = 0; i < 32; i++)
        if (k->attr_mask & (1u << i)) va[na++] = (VkVertexInputAttributeDescription){(uint32_t)i, k->attr_bind[i], k->attr_fmt[i], k->attr_off[i]};
    for (uint32_t s = 0; s < 17; s++) {
        if (!(k->used_streams & (1u << s))) continue;
        bool inst = k->divisor[s] != 0;
        vb[nb++] = (VkVertexInputBindingDescription){s, k->stride[s], inst ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX};
        if (inst && k->divisor[s] > 1 && g_vk.divisor) vd[nd++] = (VkVertexInputBindingDivisorDescriptionEXT){s, k->divisor[s]};
    }
    VkPipelineVertexInputDivisorStateCreateInfoEXT dvi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT,
                                                          .vertexBindingDivisorCount = nd, .pVertexBindingDivisors = vd};
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, .pNext = nd ? &dvi : NULL,
                                               .vertexBindingDescriptionCount = nb, .pVertexBindingDescriptions = vb,
                                               .vertexAttributeDescriptionCount = na, .pVertexAttributeDescriptions = va};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = topo_class_base(k->topo_class)};
    VkPipelineViewportStateCreateInfo vs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .polygonMode = k->poly_fill ? VK_POLYGON_MODE_FILL : VK_POLYGON_MODE_LINE, .lineWidth = 1};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = (VkSampleCountFlagBits)k->samples, .alphaToCoverageEnable = k->a2c};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState cba[8] = {0};
    for (uint32_t i = 0; i < k->ncolor; i++) {
        cba[i].colorWriteMask = k->prog->no_fs ? 0 : k->write_mask[i];  // [XCDE] no fragment stage: nothing to write
        if ((k->blend_on >> i) & 1) {
            const uint8_t *b = k->blend[i];
            cba[i].blendEnable = VK_TRUE;
            cba[i].srcColorBlendFactor = blend_factor(b[1]);
            cba[i].dstColorBlendFactor = blend_factor(b[2]);
            cba[i].srcAlphaBlendFactor = blend_factor(b[3]);
            cba[i].dstAlphaBlendFactor = blend_factor(b[4]);
            cba[i].colorBlendOp = blend_op(b[5]);
            cba[i].alphaBlendOp = blend_op(b[6]);
        }
    }
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                              .attachmentCount = k->ncolor, .pAttachments = cba,
                                              .logicOpEnable = k->logic_op != 0, .logicOp = k->logic_op ? (VkLogicOp)(k->logic_op - 1) : VK_LOGIC_OP_COPY};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                           .dynamicStateCount = NDYN, .pDynamicStates = DYN};
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = k->ncolor,
                                        .pColorAttachmentFormats = k->color, .depthAttachmentFormat = k->depth,
                                        .stencilAttachmentFormat = k->stencil};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri, .stageCount = k->prog->gs ? 3 : 2,
                                       .pStages = stages, .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vs,
                                       .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &ds,
                                       .pColorBlendState = &cb, .pDynamicState = &dy, .layout = k->prog->layout, .flags = flags};
    // [VEGA-FIX] amdvlk64.dll (Vega iGPU, Windows) died here with an integer divide by zero (c0000094) in its shader
    // compiler and the whole game closed. The call is guarded: that pipeline just fails (its draws are skipped like
    // any program that failed to build) and its key is logged, to find out which state the driver chokes on.
    PwCall pc = {&pi, out, VK_ERROR_INITIALIZATION_FAILED};
    if (!plat_try_call(pw_call, &pc)) {
        static int nc;
        if (nc++ < 40) pw_log_key(k, nc);
        *out = VK_NULL_HANDLE;
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    return pc.r;
}

// ================================================================== [XCDE] T163: shader variant cache (precompilation)
// Every pipeline this executor builds is recorded in cache/variants.bin by the content of its shaders (code hashes),
// its translation keys and its render state. When the game registers a program (nvnProgramSetShaders - typically on a
// loading screen), all the variants seen with those shaders in earlier sessions are translated, compiled and turned
// into pipelines on the workers right away, so the draws that need them later find them ready instead of waiting (a
// first-time pipeline build was the main source of in-play hitches once T174 stopped dropping draws). The SPIR-V and
// Vulkan pipeline caches make each rebuild cheap; this makes it happen ahead of time. XCDE_PREWARM=0: off.
typedef struct {
    uint64_t code_hash[3];
    uint8_t has_gs, gs_in, pad[6];
    ShaderKey vkey, fkey;
    PipeKey k;  // (k.prog is NULL in the file)
} Variant;
static Variant *var_db;
static size_t var_n, var_cap;
static uint64_t var_set[1 << 18];  // hashes of the recorded variants (open addressing; [XCDE] T240: room for the whole game's tours)
static FILE *var_out;
static bool var_loaded;
static bool prewarm_on(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("XCDE_PREWARM"); on = (!e || atoi(e) != 0) && !sync_shaders(); }
    return on;
}
static bool var_seen(uint64_t h) {
    if (!h) h = 1;
    for (uint32_t i = (uint32_t)(h ^ (h >> 32)) & 0x3FFFF, n = 0; n < 0x40000; i = (i + 1) & 0x3FFFF, n++) {
        if (var_set[i] == h) return true;
        if (!var_set[i]) { var_set[i] = h; return false; }
    }
    return true;  // (full: stop recording)
}
static void var_add(const Variant *v) {
    if (var_n == var_cap) { var_cap = var_cap ? var_cap * 2 : 4096; var_db = realloc(var_db, var_cap * sizeof *var_db); }
    var_db[var_n++] = *v;
}
static void var_write_header(FILE *f) { uint32_t hdr[2] = {0x56435843u, (uint32_t)sizeof(Variant)}; fwrite(hdr, 4, 2, f); }
static void var_load(void) {
    if (var_loaded) return;
    var_loaded = true;
    bool ok = false;
    FILE *f = fopen("cache/variants.bin", "rb");
    if (f) {
        uint32_t hdr[2];
        if (fread(hdr, 4, 2, f) == 2 && hdr[0] == 0x56435843u && hdr[1] == (uint32_t)sizeof(Variant)) {
            ok = true;
            Variant v;
            while (fread(&v, sizeof v, 1, f) == 1)
                if (!var_seen(hash_words(0xcbf29ce484222325ull, &v, sizeof v))) var_add(&v);
        }
        fclose(f);
        LOG("[prewarm] %zu shader variants known (cache/variants.bin)\n", var_n);
    }
    {   // [XCDE] T211: the shipped seed (data/variants_seed.bin: variants recorded in real play, tools/merge_variants.py):
        // a new player's first visit to an area finds them known as if seen before (prewarmed at its loading screen)
        size_t n0 = var_n;
        FILE *sf = fopen("data/variants_seed.bin", "rb");
        uint32_t hdr[2];
        if (sf && fread(hdr, 4, 2, sf) == 2 && hdr[0] == 0x56435843u && hdr[1] == (uint32_t)sizeof(Variant)) {
            Variant v;
            while (fread(&v, sizeof v, 1, sf) == 1)
                if (!var_seen(hash_words(0xcbf29ce484222325ull, &v, sizeof v))) var_add(&v);
            if (!ok && var_n > n0) ok = true;  // (then written into the player's own file below)
        }
        if (sf) fclose(sf);
        if (var_n > n0) LOG("[prewarm] %zu more from the seed (data/variants_seed.bin)\n", var_n - n0);
    }
    mkdir("cache", 0755);
    if (ok) {
        // rewrite the deduplicated set (also drops a torn last record)
        var_out = fopen("cache/variants.bin", "wb");
        if (var_out) { var_write_header(var_out); fwrite(var_db, sizeof(Variant), var_n, var_out); fflush(var_out); }
    } else {
        var_out = fopen("cache/variants.bin", "wb");
        if (var_out) var_write_header(var_out);
    }
}
// a pipeline built for a draw: remember its variant
static void var_record(const NvnProgram *np, const ShaderKey *vkey, const ShaderKey *fkey, const PipeKey *k) {
    if (!prewarm_on()) return;
    var_load();
    Variant v;
    memset(&v, 0, sizeof v);
    for (int i = 0; i < 3; i++) v.code_hash[i] = np->stages[i].code_hash;
    v.has_gs = (np->stage_mask & 4) && np->stages[2].code_addr;
    v.gs_in = g_draw_gs_in;
    v.vkey = *vkey; v.fkey = *fkey;
    v.k = *k; v.k.prog = NULL;
    if (var_seen(hash_words(0xcbf29ce484222325ull, &v, sizeof v))) return;
    var_add(&v);
    if (var_out) { fwrite(&v, sizeof v, 1, var_out); fflush(var_out); }
}
// programs the game registered (game thread -> render thread)
// [XCDE] T218: the queue grows - an area registers ~1,000 programs between two draws of the black transition and
// only the first 256 were kept: the rest got no prewarm (no libraries, no recorded pipelines) and their first draws
// translated / built on the render thread (~350 programs, ~300 pipelines in the area's first frame, coldroot bench).
static NvnProgram **pw_queue;
static int pw_n, pw_cap, pw_dropped;
static pthread_mutex_t pw_lock = PTHREAD_MUTEX_INITIALIZER;
static _Atomic int g_reg_burst;  // [XCDE] T240: programs registered since the last present (vk_prewarm_hold)
void vk_program_registered(const NvnProgram *np) {  // (objects.c set_shaders, game thread)
    if (!prewarm_on() || !(np->stage_mask & 1)) return;
    atomic_fetch_add(&g_reg_burst, 1);
    NvnProgram *c = malloc(sizeof *c);
    *c = *np;
    c->backend = NULL;
    pthread_mutex_lock(&pw_lock);
    if (pw_n == pw_cap) {
        int cap = pw_cap ? pw_cap * 2 : 1024;
        NvnProgram **q = realloc(pw_queue, cap * sizeof *q);
        if (q) { pw_queue = q; pw_cap = cap; }
    }
    if (pw_n < pw_cap) pw_queue[pw_n++] = c; else { free(c); pw_dropped++; }
    pthread_mutex_unlock(&pw_lock);
}
#define PW_MAX 8192  // [XCDE] T218: (was 1024; ~2,600 recorded pipelines for one area are linked ahead)
static Pipe *pw_pending[PW_MAX];
static void gpl_prebuild(void *arg);
static LibJob *lv_pending[8192];  // [XCDE] T212: variant library jobs whose program isn't translated yet
static bool prebuild_libs_on(void) {  // XCDE_PREBUILD_LIBS=0/1 (default: on with pipeline libraries on a discrete GPU)
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("XCDE_PREBUILD_LIBS");
        on = g_vk.gpl && (e ? atoi(e) != 0 : g_vk.props.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);
    }
    return on;
}
// pipelines whose program is ready go to the workers (render thread; also called while a draw waits)
static void prewarm_pump(void) {
    {
        int k = 0;
        for (int i = 0; i < pl_npending; i++) {
            Prog *pr = (Prog *)pl_pending[i];
            int st_ = atomic_load(&pr->state);
            if (st_ == 1) run_job(gpl_prebuild, pr);
            else if (st_ == 0) pl_pending[k++] = pr;
        }
        pl_npending = k;
        k = 0;
        for (int i = 0; i < lv_npending; i++) {  // [XCDE] T212 (before the whole pipelines: they then link)
            LibJob *j = lv_pending[i];
            int st_ = atomic_load(&j->pr->state);
            if (st_ == 1) run_job(gpl_prebuild_variant, j);
            else if (st_ == 0) lv_pending[k++] = j;
            else free(j);
        }
        lv_npending = k;
    }
    int k = 0;
    for (int i = 0; i < pw_npending; i++) {
        Pipe *pp = pw_pending[i];
        int s = atomic_load(&pp->k.prog->state);
        if (s == 1) { atomic_fetch_add(&pp->jobs, 1); run_job(build_pipeline, pp); }
        else if (s == 2) atomic_store(&pp->state, 2);
        else pw_pending[k++] = pp;
    }
    pw_npending = k;
}
static void gpl_reap(void);
static void pipe_evict(void);
static void prewarm_drain(void) {  // (render thread, start of each draw)
    gpl_reap();  // [XCDE] T188
    pipe_evict();
    if (pw_npending || pl_npending || lv_npending) prewarm_pump();
    if (!__atomic_load_n(&pw_n, __ATOMIC_RELAXED)) return;
    pthread_mutex_lock(&pw_lock);
    NvnProgram **q = pw_queue;  // (taken whole: registrations meanwhile start a new array)
    int n = pw_n;
    pw_queue = NULL; pw_n = pw_cap = 0;
    int dropped = pw_dropped; pw_dropped = 0;
    pthread_mutex_unlock(&pw_lock);
    if (dropped) LOG("[prewarm] %d registered programs dropped (out of memory)\n", dropped);
    var_load();
    uint8_t saved_gs_in = g_draw_gs_in;
    g_prewarming = true;
    for (int i = 0; i < n; i++) {
        NvnProgram *np = q[i];  // (kept: the Prog objects copy what they need from it; one small copy per registration)
        bool has_gs = (np->stage_mask & 4) && np->stages[2].code_addr;
        bool any = false;
        uint32_t lib_seen[16];  // [XCDE] T212: (program, fill mode, samples, a2c) of its variants already queued
        int nlib_seen = 0;
        for (size_t v = 0; v < var_n; v++) {
            Variant *vr = &var_db[v];
            if (vr->code_hash[0] != np->stages[0].code_hash || vr->code_hash[1] != np->stages[1].code_hash ||
                vr->has_gs != has_gs || (has_gs && vr->code_hash[2] != np->stages[2].code_hash)) continue;
            g_draw_gs_in = vr->gs_in;
            Prog *pr = get_program(np, &vr->vkey, &vr->fkey);
            if (!pr) continue;
            any = true;
            PipeKey k = vr->k;
            k.prog = pr;
            pipe_key_canon(&k);  // [XCDE] T195 (variants recorded before it)
            if (prebuild_libs_on()) {  // [XCDE] T212: its shader libraries too, once per fill mode / samples / a2c
                uint32_t lk = (uint32_t)(uintptr_t)pr ^ ((uint32_t)k.poly_fill << 24 | (uint32_t)k.samples << 8 | k.a2c);
                bool seen = false;
                for (int t = 0; t < nlib_seen; t++) seen |= lib_seen[t] == lk;
                if (!seen && lv_npending < 8192) {
                    if (nlib_seen < 16) lib_seen[nlib_seen++] = lk;
                    LibJob *j = malloc(sizeof *j);
                    if (j) {
                        j->pr = pr; j->poly_fill = k.poly_fill; j->samples = (uint8_t)k.samples; j->a2c = k.a2c;
                        lv_pending[lv_npending++] = j;
                        stat_prewarm_libs++;
                    }
                }
            }
            // [XCDE] T212: with its libraries prebuilt the recorded pipelines aren't built ahead whole: building them
            // whole (2-5 ms each, ~2,600 when an area registers ~1,000 programs right before its first frame) saturated
            // the workers and the first frame waited for them (0.8 - 1.3 s). XCDE_PREWARM_WHOLE=1: build them whole.
            // [XCDE] T218: they are linked ahead instead (queued after their libraries; ~0.3 ms each on the workers):
            // the area's first frame linked ~730 of them one by one on the render thread (1.2 s, coldroot fallen_arm).
            // Their optimized pipelines wait for a first draw. prelink_on: when.
            if (link_first() ? !prelink_on() || !prebuild_room() : i >= whole_batch_max()) continue;
            uint64_t ph = hash_words(0xcbf29ce484222325ull, &k, sizeof k);
            if (!ph) ph = 1;
            HEnt *pslot = htab_slot(&pipe_tab, ph);
            if (pslot->val || pw_npending == PW_MAX) continue;
            Pipe *pp = calloc(1, sizeof *pp);
            pp->k = k;
            pp->t_new = SDL_GetTicksNS();
            pp->how = 0x80;  // (prewarmed)
            pp->last_used = renderer_present_count();  // (eviction: as young as a drawn one)
            if (!pslot->key) pipe_tab.n++;
            pslot->key = ph;
            pslot->val = pp;
            pw_pending[pw_npending++] = pp;
            stat_prewarm_pipes++;
        }
        if (any) stat_prewarm_progs++;
        else if (prebuild_libs_on() && pl_npending < PL_MAX) {
            // [XCDE] T211: no variant seen yet (a new area / a new player): the variant precompilation guesses
            ShaderKey vk_ = {0}, fk_ = {0};
            vk_.glsl = fk_.glsl = 1;
            vk_.depth_minus_one = true;
            Prog *pr = get_program(np, &vk_, &fk_);
            if (pr) { pl_pending[pl_npending++] = pr; stat_prewarm_libs++; }
        }
    }
    g_draw_gs_in = saved_gs_in;
    free(q);
    g_prewarming = false;
    prewarm_pump();
}

// [XCDE] T240: a loading frame waits for the build work. The game registers an area's / a scene's programs on its
// loading screen or black transition, and their prewarm (libraries, prelinked pipelines, T163 / T212 / T218) ran on
// while the scene started: its first draws then waited for pipelines one by one on the render thread - and Shulk's
// head went missing for 2 frames at the start of the lab scene on a cold first viewing (9 such waits in that frame;
// none and no glitch in the runs where the prewarm had finished first). A presented frame with few draws (a loading
// screen, a black or white transition: XCDE_LOAD_HOLD_DRAWS, default 150) is held, the render thread driving the
// prewarm, until the build work is done (at most XCDE_LOAD_HOLD_MS, default 8000, per frame): the transition lasts a
// little longer and the scene starts with its pipelines ready. Gameplay frames are never held. XCDE_LOAD_HOLD_MS=0: off.
void vk_prewarm_hold(uint64_t frame_draws) {
    static int max_draws = -1, max_ms;
    if (max_draws < 0) {
        const char *e = getenv("XCDE_LOAD_HOLD_DRAWS");
        max_draws = e ? atoi(e) : 150;
        e = getenv("XCDE_LOAD_HOLD_MS");
        max_ms = e ? atoi(e) : 8000;
    }
    // a scene that starts without a loading screen (a cutscene, a new character or model set appearing) registers a
    // burst of programs between two frames: the next frame waits too (XCDE_REG_HOLD=<programs>, default 16; at most
    // XCDE_REG_HOLD_MS, default 3000) - between frames, never inside one (a draw waiting for its pipeline in the middle
    // of a frame is when hair / faces / a weapon came out wrong, T240)
    static int reg_min = -1, reg_ms;
    if (reg_min < 0) {
        const char *e = getenv("XCDE_REG_HOLD");
        reg_min = e ? atoi(e) : 16;
        e = getenv("XCDE_REG_HOLD_MS");
        reg_ms = e ? atoi(e) : 3000;
    }
    int burst = atomic_exchange(&g_reg_burst, 0);
    bool reg_hold = reg_min > 0 && burst >= reg_min && reg_ms > 0;
    if (!max_ms || sync_shaders() || (frame_draws > (uint64_t)max_draws && !reg_hold)) return;
    int lim = frame_draws > (uint64_t)max_draws && reg_ms < max_ms ? reg_ms : max_ms;
    uint64_t t0 = SDL_GetTicksNS(), dl = t0 + (uint64_t)lim * 1000000ull;
    int n0 = -1;
    for (;;) {
        prewarm_drain();  // (programs registered meanwhile, ready translations -> their library / pipeline jobs)
        int left = __atomic_load_n(&pw_n, __ATOMIC_RELAXED) + pw_npending + pl_npending + lv_npending + atomic_load(&g_jobs_queued) +
                   atomic_load(&g_jobs_busy) + atomic_load(&g_xlat_queued);
        if (n0 < 0) n0 = left;
        if (!left || SDL_GetTicksNS() >= dl) {
            uint64_t ms = (SDL_GetTicksNS() - t0) / 1000000;
            static int nlog;
            if (n0 > 0 && nlog++ < 200)
                LOG("[prewarm] %s (%llu draws, %d programs registered) held %llu ms for %d builds%s\n",
                    frame_draws > (uint64_t)max_draws ? "scene start" : "loading frame", (unsigned long long)frame_draws, burst,
                    (unsigned long long)ms, n0, left ? " (time limit: some left)" : "");
            return;
        }
        SDL_Delay(1);
    }
}

// [XCDE] T187 (precompile.c, `xcde.exe --precompile`): one program built the way its first draw builds it -
// translate_program + compile_program (SPIR-V into cache/spirv) and its vertex / fragment pipeline libraries (gpl_pre,
// gpl_fs: they depend only on the shaders, the layout, fill mode, sample count and alpha-to-coverage) - then freed. The
// SPIR-V cache, our pipeline cache (cache/vk_pipelines.bin) and the driver's own shader cache keep the results, so the
// game's first draw with the program finds its pieces compiled: a fast link instead of native compiles. Synchronous
// (the calling thread translates and compiles). Returns 0 if not built; *vs_attr = the vertex inputs it reads.
int vk_precompile_program(const NvnProgram *np, const ShaderKey *vkey, const ShaderKey *fkey, uint32_t *vs_attr) {
    Prog *p = calloc(1, sizeof *p);
    for (int i = 0; i < 2; i++) {
        p->code[i] = np->stages[i].code_addr;
        p->src[i] = nvn_stage_code(&np->stages[i]);
        p->cmax[i] = np->stages[i].const_offset ? np->stages[i].const_offset : 0x10000;
    }
    p->vkey = *vkey;
    p->fkey = *fkey;
    p->no_fs = !(np->stage_mask & 2) || !np->stages[1].code_addr;
    tl_inline_jobs = true;
    translate_program(p);  // (and compile_program, run here)
    int ok = atomic_load(&p->state) == 1;
    if (vs_attr) *vs_attr = p->vi ? p->vi->attr_mask : 0;
    if (ok && g_vk.gpl) {
        PipeKey k;
        memset(&k, 0, sizeof k);
        k.prog = p;
        k.poly_fill = 1;
        k.samples = 1;
        VkPipeline a = VK_NULL_HANDLE, b = VK_NULL_HANDLE;
        if (gpl_pre(&k, &a) != VK_SUCCESS) ok = 0;
        if (gpl_fs(&k, &b) != VK_SUCCESS) ok = 0;
        if (a) vkDestroyPipeline(g_vk.dev, a, NULL);
        if (b) vkDestroyPipeline(g_vk.dev, b, NULL);
    }
    if (p->vs) vkDestroyShaderModule(g_vk.dev, p->vs, NULL);
    if (p->fs) vkDestroyShaderModule(g_vk.dev, p->fs, NULL);
    if (p->layout) vkDestroyPipelineLayout(g_vk.dev, p->layout, NULL);
    for (int i = 0; i < 3; i++) if (p->set_layout[i]) vkDestroyDescriptorSetLayout(g_vk.dev, p->set_layout[i], NULL);
    maxwell_free(p->vi);
    maxwell_free(p->fi);
    free(p);
    return ok;
}

// [XCDE] T240 (precompile.c): one recorded variant (cache/variants.bin / data/variants_seed.bin record, rec_size bytes)
// built the way the game ends up drawing it: its program (as vk_precompile_program) and then its whole optimized
// pipeline with the recorded render state (vertex layout, targets, blend, ...). Precompiling runs without our pipeline
// cache (g_vk.pcache null): the driver's own disk cache keeps the result, so the game's optimized build of that
// pipeline - the one that replaces its fast link a few frames after the first draw - is a cache hit. Returns 1 if built.
int vk_precompile_variant(const NvnProgram *np, const void *rec, size_t rec_size) {
    if (rec_size != sizeof(Variant)) return 0;
    Variant v;
    memcpy(&v, rec, sizeof v);
    if (v.has_gs) return 0;
    Prog *p = calloc(1, sizeof *p);
    for (int i = 0; i < 2; i++) {
        p->code[i] = np->stages[i].code_addr;
        p->src[i] = nvn_stage_code(&np->stages[i]);
        p->cmax[i] = np->stages[i].const_offset ? np->stages[i].const_offset : 0x10000;
    }
    p->vkey = v.vkey;
    p->fkey = v.fkey;
    p->no_fs = !(np->stage_mask & 2) || !np->stages[1].code_addr;
    tl_inline_jobs = true;
    g_draw_gs_in = v.gs_in;
    translate_program(p);  // (and compile_program, run here)
    int ok = atomic_load(&p->state) == 1;
    if (ok) {
        // the recorded depth / stencil format is this machine's choice for NVN's D24S8 (textures.c depth24_stencil8):
        // D24_UNORM_S8 where the GPU has it with every feature used (NVIDIA), else D32_SFLOAT_S8 (AMD, Intel). A list
        // recorded on one maps to the other here; a D32_SFLOAT_S8 record may also be NVN's own D32S8: built both ways.
        extern VkFormat vk_depth24_stencil8(void);
        VkFormat d24 = vk_depth24_stencil8();
        PipeKey k = v.k;
        k.prog = p;
        bool both = false;
        for (int i = 0; i < 2; i++) {
            VkFormat *f = i ? &k.stencil : &k.depth;
            if (*f == VK_FORMAT_D24_UNORM_S8_UINT && d24 != VK_FORMAT_D24_UNORM_S8_UINT) *f = d24;
            else if (*f == VK_FORMAT_D32_SFLOAT_S8_UINT && d24 == VK_FORMAT_D24_UNORM_S8_UINT) both = true;
        }
        for (int pass = 0; pass < (both ? 2 : 1) && ok; pass++) {
            if (pass) {
                if (k.depth == VK_FORMAT_D32_SFLOAT_S8_UINT) k.depth = d24;
                if (k.stencil == VK_FORMAT_D32_SFLOAT_S8_UINT) k.stencil = d24;
            }
            PipeKey kc = k;
            pipe_key_canon(&kc);
            VkPipeline pipe = VK_NULL_HANDLE;
            ok = pipeline_whole(&kc, 0, &pipe) == VK_SUCCESS;
            if (pipe) vkDestroyPipeline(g_vk.dev, pipe, NULL);
        }
    }
    if (p->vs) vkDestroyShaderModule(g_vk.dev, p->vs, NULL);
    if (p->fs) vkDestroyShaderModule(g_vk.dev, p->fs, NULL);
    if (p->layout) vkDestroyPipelineLayout(g_vk.dev, p->layout, NULL);
    for (int i = 0; i < 3; i++) if (p->set_layout[i]) vkDestroyDescriptorSetLayout(g_vk.dev, p->set_layout[i], NULL);
    maxwell_free(p->vi);
    maxwell_free(p->fi);
    free(p);
    return ok;
}

// ================================================================== feedback snapshots (renderer.m resolve_feedback)
static NvnTexture *feedback_tex[4];
static VkImageView feedback_view[4];
static int nfeedback;
static uint64_t g_cur_fs_hash;
static NvnTexture *snap_tex[8];
static bool snap_ok[8];
static const char *snap_why[8];
static const char *snap_reason = "?";
static void snapshots_invalidate(NvnTexture *t) {
    for (int i = 0; i < 8; i++) if ((!t || snap_tex[i] == t) && snap_ok[i]) { snap_ok[i] = false; snap_why[i] = snap_reason; }
}

typedef struct { NvnTexture *key; TexBackend copy; } Snap;
static Snap snaps[16];
static int nsnaps;

static NvnTexture *shader_texture_object(int stage, ShaderTexture *st_) {
    int slot = ((int)st_->handle_offset - 0x20) / 8;
    uint64_t h = (slot >= 0 && slot < 32) ? st.tex[stage][slot] : 0;
    if (st_->is_image) { int k = ((int)st_->handle_offset - 0x120) / 8; h = (k >= 0 && k < 8) ? st.img[stage][k] : 0; }  // [XCDE]
    uint32_t tic = h & 0xFFFFF;
    if (!h || !st.tpool || tic >= (uint32_t)st.tpool->count) return NULL;
    return ((NvnTexEntry *)st.tpool->entries)[tic].tex;
}

static bool is_attachment(NvnTexture *t) {
    if (!t) return false;
    if (t == st.rt.depth) return true;
    for (int i = 0; i < st.rt.ncolor; i++) if (t == st.rt.color[i]) return true;
    return false;
}

static TexBackend *snapshot_image(NvnTexture *t, TexBackend *src) {
    Snap *s = NULL;
    for (int i = 0; i < nsnaps; i++) if (snaps[i].key == t) s = &snaps[i];
    if (!s) {
        if (nsnaps == 16) { static int rr; s = &snaps[rr++ & 15]; } else s = &snaps[nsnaps++];
        s->key = t;
    }
    TexBackend *d = &s->copy;
    if (d->image && (d->width != src->width || d->height != src->height || d->format != src->format)) {
        vk_views_forget(d);  // (resolution change)
        vk_defer_image(d->image, d->mem);
        d->image = VK_NULL_HANDLE;
        d->mem = VK_NULL_HANDLE;
    }
    if (!d->image) {
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = src->format,
                                .extent = {src->width, src->height, 1}, .mipLevels = 1, .arrayLayers = 1,
                                .samples = (VkSampleCountFlagBits)src->samples, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &d->image, &d->mem, NULL));
        d->format = src->format; d->width = src->width; d->height = src->height; d->depth = 1; d->layers = 1; d->levels = 1;
        d->samples = src->samples; d->aspect = src->aspect; d->base_type = VK_IMAGE_VIEW_TYPE_2D;
        vk_image_init_layout(vk_cmd(), d->image, d->aspect);
    }
    return d;
}

// (debug frames) a depth image's depth values as a greyscale PGM, stretched over its own min..max
static void debug_save_depth(TexBackend *b, const char *path) {
    if (!b || !b->image || !(b->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) || b->format != VK_FORMAT_D32_SFLOAT_S8_UINT) return;
    uint64_t n = (uint64_t)b->width * b->height;
    VkSpan out = vk_readback(n * 4);
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    VkBufferImageCopy c = {.bufferOffset = out.off, .imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1},
                           .imageExtent = {b->width, b->height, 1}};
    vkCmdCopyImageToBuffer(cb, b->image, VK_IMAGE_LAYOUT_GENERAL, out.buf, 1, &c);
    renderer_flush();
    const float *d = out.ptr;
    float lo = 1e30f, hi = -1e30f;
    for (uint64_t i = 0; i < n; i++) { if (d[i] < lo) lo = d[i]; if (d[i] > hi) hi = d[i]; }
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "P5 %u %u 255\n", b->width, b->height);
    for (uint64_t i = 0; i < n; i++) fputc(hi > lo ? (int)((d[i] - lo) / (hi - lo) * 255.0f) : 0, f);
    fclose(f);
    LOG("[vk] saved %s (%ux%u depth %.6f..%.6f)\n", path, b->width, b->height, lo, hi);
}

// [XCDE] (debug) XCDE_WATCH_FS=<fs hash>: the depth image of that shader's draw is watched; every later Vulkan write to it
// (copies, blits, clears, render passes using it) is logged with its call site.
void vk_watch(VkImage dst, const char *site) {
    if (g_watch_img && dst == g_watch_img && g_debug_frame) LOG("[watch] write to watched image at %s (pass %d)\n", site, debug_pass_no);
}

// [XCDE] (debug) histogram of a D16 / D32 depth image: the distinct values and how many pixels hold each
static void debug_depth_hist(TexBackend *b, const char *what) {
    vk_end_pass();  // (no copies inside a render pass)
    if (!b || !b->image || !(b->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) return;
    bool d16 = b->format == VK_FORMAT_D16_UNORM;
    if (!d16 && b->format != VK_FORMAT_D32_SFLOAT) { LOG("[depth] %s: format %d not handled\n", what, (int)b->format); return; }
    uint64_t n = (uint64_t)b->width * b->height;
    VkSpan out = vk_readback(n * (d16 ? 2 : 4));
    memset(out.ptr, 0xAB, n * (d16 ? 2 : 4));  // sentinel: a region the copy does not write stays 0xABAB
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    VkBufferImageCopy c = {.bufferOffset = out.off, .imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1}, .imageExtent = {b->width, b->height, 1}};
    vkCmdCopyImageToBuffer(cb, b->image, VK_IMAGE_LAYOUT_GENERAL, out.buf, 1, &c);
    renderer_flush();
    static uint32_t val[4096]; static uint64_t cnt[4096];
    int nv = 0;
    for (uint64_t i = 0; i < n; i++) {
        uint32_t v = d16 ? ((const uint16_t *)out.ptr)[i] : ((const uint32_t *)out.ptr)[i];
        int k = 0;
        while (k < nv && val[k] != v) k++;
        if (k == nv) { if (nv == 4096) continue; val[nv++] = v; cnt[k] = 0; }
        cnt[k]++;
    }
    {   // the image itself as a P5 map (values mod 256 x 4, so that neighbouring classes differ visibly)
        char fn[128]; snprintf(fn, sizeof fn, "build/depth_%s.pgm", getenv("XCDE_DEPTH_HIST"));
        FILE *pf = fopen(fn, "wb");
        if (pf) {
            fprintf(pf, "P5 %u %u 255\n", b->width, b->height);
            for (uint64_t i = 0; i < n; i++) { uint32_t v = d16 ? ((const uint16_t *)out.ptr)[i] : ((const uint32_t *)out.ptr)[i]; fputc(v == 0 ? 0 : (int)(64 + (v * 4) % 192), pf); }
            fclose(pf);
        }
    }
    LOG("[depth] %s img %p %ux%u %s: %d distinct values\n", what, (void *)b->image, b->width, b->height, d16 ? "D16" : "D32", nv);
    for (int rank = 0; rank < 12 && rank < nv; rank++) {
        int best = -1;
        for (int k = 0; k < nv; k++) if (cnt[k] && (best < 0 || cnt[k] > cnt[best])) best = k;
        if (best < 0) break;
        if (d16) LOG("[depth]   %5u (%.7f)  x%llu\n", val[best], val[best] / 65535.0, (unsigned long long)cnt[best]);
        else { float f; memcpy(&f, &val[best], 4); LOG("[depth]   %.7f  x%llu\n", f, (unsigned long long)cnt[best]); }
        cnt[best] = 0;
    }
}

// [XCDE] (debug) histogram of the stencil plane of a depth-stencil image
static void debug_stencil_hist(TexBackend *b, const char *what) {
    vk_end_pass();  // (no copies inside a render pass)
    if (!b || !b->image || !(b->aspect & VK_IMAGE_ASPECT_STENCIL_BIT)) return;
    uint64_t n = (uint64_t)b->width * b->height;
    VkSpan out = vk_readback(n);
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    VkBufferImageCopy c = {.bufferOffset = out.off, .imageSubresource = {VK_IMAGE_ASPECT_STENCIL_BIT, 0, 0, 1}, .imageExtent = {b->width, b->height, 1}};
    vkCmdCopyImageToBuffer(cb, b->image, VK_IMAGE_LAYOUT_GENERAL, out.buf, 1, &c);
    renderer_flush();
    uint64_t cnt[256] = {0};
    for (uint64_t i = 0; i < n; i++) cnt[((const uint8_t *)out.ptr)[i]]++;
    {
        FILE *pf = fopen("build/stencil.pgm", "wb");
        if (pf) { fprintf(pf, "P5 %u %u 255\n", b->width, b->height); fwrite(out.ptr, 1, n, pf); fclose(pf); }
    }
    LOG("[stencil] %s %ux%u:", what, b->width, b->height);
    for (int v = 0; v < 256; v++) if (cnt[v]) LOG(" %#x:x%llu", v, (unsigned long long)cnt[v]);
    LOG("\n");
}

// (debug) LARECOMP_COPY_TRACE=<frame>: draws' attachments / sampled textures, copies and snapshots of that frame ("[ct]")
static bool ct_on(void) {
    static const char *e = (const char *)1;
    if (e == (const char *)1) e = getenv("LARECOMP_COPY_TRACE");
    if (!e) return false;
    static uint64_t memo_f = ~0ull;
    static bool memo;
    uint64_t f = renderer_present_count();
    if (f == memo_f) return memo;
    memo_f = f;
    memo = false;
    for (const char *p = e; *p;) {
        if ((uint64_t)strtoll(p, (char **)&p, 10) == f) { memo = true; break; }
        if (*p == ',') p++; else break;
    }
    if (memo) LOG("[ct] ===== frame %llu\n", (unsigned long long)f);
    return memo;
}
static void ct_draw(Prog *pr, const char *what) {
    LOG("[ct] %s fs %016llx rt", what, (unsigned long long)g_cur_fs_hash);
    for (int i = 0; i < st.rt.ncolor; i++) if (st.rt.color[i]) LOG(" c%d=%p/%#x", i, (void *)st.rt.color[i], st.rt.color[i]->format);
    if (st.rt.depth) LOG(" d=%p/%#x%s", (void *)st.rt.depth, st.rt.depth->format, st.ds[0] && st.ds[1] ? "W" : "");
    LOG(" cm %08x vp %.0f %.0f %.0f %.0f sc %d %d %d %d | tex", st.channel_mask, st.viewport[0][0], st.viewport[0][1], st.viewport[0][2],
        st.viewport[0][3], st.scissor[0][0], st.scissor[0][1], st.scissor[0][2], st.scissor[0][3]);
    ShaderInfo *infos[2] = {pr->vi, pr->fi};
    for (int sg = 0; sg < 2; sg++)
        for (int i = 0; i < (infos[sg] ? infos[sg]->ntex : 0); i++) {
            NvnTexture *t = shader_texture_object(sg, &infos[sg]->tex[i]);
            if (t) LOG(" %s%p/%#x", sg ? "" : "v:", (void *)t, t->format);
        }
    LOG("\n");
}

// ================================================================== copy elision (depth copies)
// The game copies its main depth buffer whole, every frame, into textures that are then only sampled while the
// source keeps its depth (the water / transparency depth input), or that get their depth cleared right away
// (stencil kept). At 3x internal resolution each such copy is ~0.15 ms of pure bandwidth. Instead of copying, the
// destination becomes an alias: sampling it (and copying from it) reads the source's image. The alias ends before
// anything changes either side:
//   - the source is about to be overwritten whole (both aspects cleared): the two images swap - the destination
//     keeps the contents without a copy, the source is cleared in the destination's old image;
//   - the destination is about to be overwritten in some aspects (its depth cleared): only the other aspects are
//     copied (stencil, a quarter of the bytes);
//   - anything else (a draw writing depth / stencil, a partial clear, a copy or upload into either, the
//     destination bound as an attachment, a texture freed or reallocated): the real copy is made first.
// Only same-size, same-format, single-level, single-sample depth/stencil images, whole-image copies.
// LARECOMP_VK_COPY_OPT=0: always copy (the old behaviour).
typedef struct { NvnTexture *src, *dst; VkImage simg, dimg; } CopyAlias;
extern uint32_t vk_view_gen;
static CopyAlias aliases[8];
static int naliases;
static uint64_t stat_alias_made, stat_alias_renamed, stat_alias_partial, stat_alias_copied;

static int copy_opt(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("LARECOMP_VK_COPY_OPT"); on = e && *e ? atoi(e) : 1; }
    return on;
}

// still valid: both textures keep the images the alias was made with (a resolution change reallocates them)
static bool alias_valid(const CopyAlias *a) {
    const TexBackend *s = a->src->backend, *d = a->dst->backend;
    return s && d && s->image == a->simg && d->image == a->dimg;
}

static void alias_drop(int i) {
    aliases[i] = aliases[--naliases];
    vk_view_gen++;  // (exec.c's per-frame view memo: the destination's views change)
}

// the real copy of the aspects asked for, into the destination's own image
static void alias_copy(CopyAlias *a, VkImageAspectFlags aspects, const char *why) {
    TexBackend *s = a->src->backend, *d = a->dst->backend;
    aspects &= d->aspect;
    if (!aspects) return;
    vk_end_pass();
    RT_DIRTY();
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    gp_mark(cb, "copy texture (deferred, %s%s) %ux%u", why, aspects == d->aspect ? "" : ", stencil only", a->dst->width, a->dst->height);
    VkImageCopy c = {.srcSubresource = {aspects, 0, 0, 1}, .dstSubresource = {aspects, 0, 0, 1}, .extent = {d->width, d->height, 1}};
    vk_watch(d->image, "exec.c:1814 vkCmdCopyImage"); vkCmdCopyImage(cb, s->image, VK_IMAGE_LAYOUT_GENERAL, d->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    if (aspects == d->aspect) stat_alias_copied++; else stat_alias_partial++;
    if (ct_on()) LOG("[ct] alias copy %p -> %p aspects %#x (%s)\n", (void *)a->src, (void *)a->dst, aspects, why);
}

// t is about to be written; full: the aspects of it the write overwrites whole (0: unknown / partial)
static void alias_before_write(NvnTexture *t, VkImageAspectFlags full, const char *why) {
    if (!naliases || !t) return;
    for (int i = naliases - 1; i >= 0; i--) {
        CopyAlias *a = &aliases[i];
        if (!alias_valid(a)) { alias_drop(i); continue; }
        if (a->dst != t) continue;
        // (LARECOMP_VK_COPY_OPT=2: all aspects even then - for comparing the stencil-only copy's cost)
        alias_copy(a, ((TexBackend *)t->backend)->aspect & (copy_opt() == 2 ? ~0u : ~full), why);
        alias_drop(i);
    }
    NvnTexture *heir = NULL;  // t overwritten whole: one destination can take its image instead of a copy
    if (t->backend && (((TexBackend *)t->backend)->aspect & ~full) == 0)
        for (int i = 0; i < naliases && !heir; i++) if (aliases[i].src == t) heir = aliases[i].dst;
    for (int i = naliases - 1; i >= 0; i--) {
        CopyAlias *a = &aliases[i];
        if (a->src != t || a->dst == heir) continue;
        alias_copy(a, ((TexBackend *)a->dst->backend)->aspect, why);
        alias_drop(i);
    }
    int rename = -1;
    for (int i = 0; i < naliases; i++) if (heir && aliases[i].src == t && aliases[i].dst == heir) rename = i;
    if (rename >= 0) {
        CopyAlias *a = &aliases[rename];
        TexBackend *s = a->src->backend, *d = a->dst->backend;
        vk_end_pass();
        RT_DIRTY();
        VkImage im = s->image; s->image = d->image; d->image = im;
        VmaAllocation m = s->mem; s->mem = d->mem; d->mem = m;
        vk_views_forget(s);
        vk_views_forget(d);
        stat_alias_renamed++;
        if (ct_on()) LOG("[ct] alias rename %p -> %p (%s)\n", (void *)a->src, (void *)a->dst, why);
        alias_drop(rename);
    }
}

// the texture whose image holds t's contents (t itself unless it is an alias)
static NvnTexture *alias_read(NvnTexture *t) {
    for (int i = 0; i < naliases; i++)
        if (aliases[i].dst == t) {
            if (alias_valid(&aliases[i])) return aliases[i].src;
            alias_drop(i);
            return t;
        }
    return t;
}

// copy_texture: a whole-image copy that can be an alias instead (true: done)
static bool alias_make(NvnTexture *src, NvnTexture *dst, TexBackend *s, TexBackend *d) {
    if (!copy_opt() || src == dst) return false;
    if (!(s->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT))) return false;
    if (s->format != d->format || s->width != d->width || s->height != d->height || s->aspect != d->aspect || s->levels != 1 ||
        d->levels != 1 || s->layers != 1 || d->layers != 1 || s->depth != 1 || d->depth != 1 || s->samples != 1 || d->samples != 1 ||
        s->base_type != d->base_type || s->presented || d->presented || !s->image || !d->image || !s->gpu_owned || !d->gpu_owned)
        return false;
    // the destination's previous contents (or alias) are replaced whole; a destination that is itself the source of
    // aliases gets those resolved first
    alias_before_write(dst, d->aspect, "copy dst");
    src = alias_read(src);  // (a copy of an alias: of its source)
    s = src->backend;
    if (src == dst) return true;  // (copied back onto itself)
    if (naliases == (int)(sizeof aliases / sizeof aliases[0])) return false;
    aliases[naliases++] = (CopyAlias){src, dst, s->image, d->image};
    vk_view_gen++;
    stat_alias_made++;
    return true;
}

// textures.c: t is being freed - an alias source hands its contents over first
void vk_alias_texture_destroyed(NvnTexture *t) {
    if (naliases) alias_before_write(t, 0, "freed");
}
void vk_alias_reset(void) { naliases = 0; }  // (reboot: every texture goes)

// (renderer_stats, once per stats interval) copies made aliases / renamed / partly or fully copied later
void vk_alias_stats(void) {
    static int n;
    if (!stat_alias_made || (n++ > 5 && n % 30)) return;
    LOG("[vk] copy alias: %llu made, %llu renamed, %llu stencil-only, %llu copied later (live %d)\n",
        (unsigned long long)stat_alias_made, (unsigned long long)stat_alias_renamed, (unsigned long long)stat_alias_partial,
        (unsigned long long)stat_alias_copied, naliases);
    stat_alias_made = stat_alias_renamed = stat_alias_partial = stat_alias_copied = 0;
}

// draws: attachments written (or a destination bound at all) end the aliases involved
static void alias_draw_check(void) {
    if (!naliases) return;
    for (int i = 0; i < st.rt.ncolor; i++)
        if (st.rt.color[i]) alias_before_write(st.rt.color[i], 0, "color target");
    NvnTexture *d = st.rt.depth;
    if (!d) return;
    bool writes = (st.ds[0] && st.ds[1]) || (st.ds[3] && ((st.stencil_mask[0] | st.stencil_mask[1]) & 0xFF));
    bool is_dst = false;
    for (int i = 0; i < naliases; i++) is_dst |= aliases[i].dst == d;
    if (writes || is_dst) alias_before_write(d, 0, writes ? "depth write" : "dst bound");
}

// [XCDE] T161 (XCDE_FB_STATS): feedback snapshot kinds, counted per stats interval
static struct { char key[96]; uint64_t n; } fb_stats[64];
static int n_fb_stats;
static void fb_stat_add(const char *key) {
    for (int i = 0; i < n_fb_stats; i++) if (!strcmp(fb_stats[i].key, key)) { fb_stats[i].n++; return; }
    if (n_fb_stats < 64) { snprintf(fb_stats[n_fb_stats].key, sizeof fb_stats[0].key, "%s", key); fb_stats[n_fb_stats++].n = 1; }
}
static void fb_stats_flush(void) {
    if (!n_fb_stats) return;
    LOG("[fbstats]");
    for (int i = 0; i < n_fb_stats; i++) LOG(" | %s: %llu", fb_stats[i].key, (unsigned long long)fb_stats[i].n);
    LOG("\n");
    n_fb_stats = 0;
}

// [XCDE] T161: a texture bound both as an attachment and for sampling is only a feedback loop when the sampled levels /
// layers overlap the attachment's. Mip chains (bloom, depth of field downsamples) render level N while sampling level
// N-1 of the same texture: those were snapshotted on every draw (99.8 % of 8.2 M copies on the prologue route), and the
// snapshot holds only level 0, so they also sampled the wrong level.
static void view_range(const NvnTextureView *v, bool has, const NvnTexture *t, uint32_t *l0, uint32_t *l1, uint32_t *y0, uint32_t *y1) {
    uint32_t nl = t->levels ? t->levels : 1, ny = t->depth > 1 ? 1 : (t->layout.layers ? t->layout.layers : 1);
    *l0 = 0, *l1 = nl, *y0 = 0, *y1 = ny;
    if (has && (v->setmask & 1)) { *l0 = v->base_level; *l1 = v->num_levels ? v->base_level + v->num_levels : nl; }
    if (has && (v->setmask & 2)) { *y0 = v->min_layer; *y1 = v->num_layers ? v->min_layer + v->num_layers : ny; }
}
static bool feedback_overlaps(int stage, ShaderTexture *st_, NvnTexture *t) {
    static int all = -1;  // LARECOMP_FEEDBACK_ANY_LEVEL=1: the old rule (any shared texture object)
    if (all < 0) all = getenv("LARECOMP_FEEDBACK_ANY_LEVEL") != NULL;
    if (all) return true;
    int slot = ((int)st_->handle_offset - 0x20) / 8;
    uint64_t h = (slot >= 0 && slot < 32) ? st.tex[stage][slot] : 0;
    if (st_->is_image) { int k = ((int)st_->handle_offset - 0x120) / 8; h = (k >= 0 && k < 8) ? st.img[stage][k] : 0; }
    NvnTexEntry *e = &((NvnTexEntry *)st.tpool->entries)[h & 0xFFFFF];
    uint32_t sl0, sl1, sy0, sy1;
    view_range(&e->view, e->has_view, t, &sl0, &sl1, &sy0, &sy1);
    const NvnTextureView *av = NULL;
    bool ahas = false;
    if (t == st.rt.depth) av = &st.rt.dview, ahas = st.rt.has_dview;
    for (int i = 0; !av && i < st.rt.ncolor; i++) if (st.rt.color[i] == t) av = &st.rt.cview[i], ahas = st.rt.has_cview[i];
    if (!av) return false;
    uint32_t al0, al1, ay0, ay1;
    view_range(av, ahas, t, &al0, &al1, &ay0, &ay1);
    al1 = al0 + 1;  // (an attachment is one level)
    return sl0 < al1 && al0 < sl1 && sy0 < ay1 && ay0 < sy1;
}

static void resolve_feedback(Prog *pr) {
    nfeedback = 0;
    static int off = -1;
    if (off < 0) off = getenv("LARECOMP_NO_FEEDBACK") != NULL;
    if (off) return;
    ShaderInfo *infos[2] = {pr->vi, pr->fi};
    for (int sg = 0; sg < 2; sg++)
        for (int i = 0; i < infos[sg]->ntex; i++) {
            NvnTexture *t = shader_texture_object(sg, &infos[sg]->tex[i]);
            if (!is_attachment(t) || !feedback_overlaps(sg, &infos[sg]->tex[i], t)) continue;
            // [XCDE] T161: colour feedback samples the attachment itself (all images are in GENERAL layout), as the
            // Maxwell texture unit reads render-target memory. The snapshot path (end the pass, full barrier, copy the
            // whole image) ran ~1,700 times a frame in battle - the game's UI uber-shaders (13 texture slots) have the
            // scene target bound in slots they read through uniform-selected branches. Measured: 5.57 M copies -> 0 on
            // the prologue route, captures identical (max 2/255 on 0.2 % of one frame), battle 23.8 -> 27.7 fps
            // (deterministic), validation layers quiet. LARECOMP_FEEDBACK_DIRECT=0: snapshots again. Depth: unchanged.
            static int direct = -1;
            // T252 made snapshots the default on every vendor. [PERF-V5] the direct path (V4 behaviour, +16 % in battle) is the
            // default again on ALL vendors. LARECOMP_FEEDBACK_DIRECT=0 brings the snapshots back (if blue textures / flicker appear).
            if (direct < 0) { const char *e = getenv("LARECOMP_FEEDBACK_DIRECT"); direct = !e || atoi(e) != 0; }
            if (direct && t != st.rt.depth) {
                // [XCDE] the attachment is read in the draw that writes it: no colour compression (textures.c want_no_dcc)
                if (t->backend && ((TexBackend *)t->backend)->nlean) { vk_end_pass(); RT_DIRTY(); vk_tex_drop_dcc(t, vk_cmd()); }
                continue;
            }
            bool have = false;
            for (int k = 0; k < nfeedback; k++) have |= feedback_tex[k] == t;
            if (have || nfeedback == 4) continue;
            static int nosnap = -1;
            if (nosnap < 0) nosnap = getenv("LARECOMP_NO_SNAPSHOT_REUSE") != NULL;
            int si = -1;
            for (int k = 0; k < 8; k++) if (snap_tex[k] == t) si = k;
            bool self_write = t == st.rt.depth && st.ds[0] && st.ds[1];
            TexBackend *src = vk_tex(t, true);
            Snap *existing = NULL;
            for (int k = 0; k < nsnaps; k++) if (snaps[k].key == t && snaps[k].copy.image) existing = &snaps[k];
            VkImageAspectFlags sa = src->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : src->aspect;
            static const VkComponentMapping id = {0};
            if (!nosnap && si >= 0 && (snap_ok[si] || self_write) && existing) {
                feedback_tex[nfeedback] = t;
                feedback_view[nfeedback++] = vk_image_view(&existing->copy, VK_IMAGE_VIEW_TYPE_2D, src->format, 0, 1, 0, 1, id, sa);
                continue;
            }
            stat_feedback_copies++;
            {   // [XCDE] T161: XCDE_FB_STATS=1 - what the feedback snapshots are (depth/colour, written by this draw?, why
                // the last copy was invalidated); logged with the per-2 s stats
                static int fbs = -1;
                if (fbs < 0) fbs = getenv("XCDE_FB_STATS") != NULL;
                if (fbs) {
                    bool depth = t == st.rt.depth;
                    bool wr = depth ? (st.ds[0] && st.ds[1]) : false;
                    for (int c2 = 0; !depth && c2 < st.rt.ncolor; c2++) if (st.rt.color[c2] == t) wr = ((st.channel_mask >> (c2 * 4)) & 0xF) != 0;
                    char key[96];
                    snprintf(key, sizeof key, "%s %s was=%s %ux%u f%#x fs %llx vs %llx", depth ? "depth" : "color", wr ? "rw" : "ro",
                             si >= 0 && snap_why[si] ? snap_why[si] : "new", src->width, src->height, t->format,
                             (unsigned long long)pr->code[1], (unsigned long long)pr->code[0]);
                    fb_stat_add(key);
                }
            }
            if (ct_on()) LOG("[ct] snapshot %p %ux%u (was %s, si %d)\n", (void *)t, src->width, src->height, si >= 0 ? snap_why[si] : "new", si);
            vk_end_pass();
            TexBackend *dst = snapshot_image(t, src);
            VkCommandBuffer cb = vk_cmd();
            vk_full_barrier(cb);
            gp_mark(cb, "feedback snapshot %ux%u fmt %d", src->width, src->height, src->format);
            VkImageCopy rg = {.srcSubresource = {src->aspect, 0, 0, 1}, .dstSubresource = {src->aspect, 0, 0, 1},
                              .extent = {src->width, src->height, 1}};
            vk_watch(dst->image, "exec.c:1954 vkCmdCopyImage"); vkCmdCopyImage(cb, src->image, VK_IMAGE_LAYOUT_GENERAL, dst->image, VK_IMAGE_LAYOUT_GENERAL, 1, &rg);
            if (g_debug_frame && (src->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) {
                static int nd;
                char path[128];
                snprintf(path, sizeof path, "build/fbdepth_%03d_src.pgm", nd);
                debug_save_depth(src, path);
                snprintf(path, sizeof path, "build/fbdepth_%03d_snap.pgm", nd++);
                debug_save_depth(dst, path);
                cb = vk_cmd();
            }
            feedback_tex[nfeedback] = t;
            feedback_view[nfeedback++] = vk_image_view(dst, VK_IMAGE_VIEW_TYPE_2D, src->format, 0, 1, 0, 1, id, sa);
            if (si < 0) { static int rr; si = rr++ & 7; snap_tex[si] = t; }
            snap_ok[si] = true;
            RT_DIRTY();
        }
}

// ================================================================== resource binding
static VkDescriptorBufferInfo ubo_span(VkSpan s, uint64_t size) {
    uint64_t max = g_vk.props.limits.maxUniformBufferRange;
    return (VkDescriptorBufferInfo){s.buf, s.off, size > max ? max : size};
}

// Constant buffer k of a stage (renderer.m bind_cbufs), copied into transient memory.
static VkDescriptorBufferInfo cbuf_info(int stage, ShaderInfo *si, int k, uint32_t gx, uint32_t gy, uint32_t gz) {
    static const uint8_t zeros[0x10000];
    // [XCDE] perf: a graphics stage's driver constant buffer is built from its bound storage buffers only: the same
    // submission, size and storage bindings give the same span (was a 2 KB clear + fill + compare per stage per draw).
    static struct { uint64_t seq, size; uint32_t gen; VkDescriptorBufferInfo bi; bool valid; } drv_memo[3];
    static int drv_memo_on = -1;
    if (drv_memo_on < 0) { const char *e = getenv("XCDE_CBUF_MEMO"); drv_memo_on = !e || atoi(e) != 0; }
    if (k == 0 && stage < 3 && drv_memo_on && !g_debug_frame) {
        uint64_t size = si->cbuf_size[0] > 0x800 ? 0x800 : (si->cbuf_size[0] ? si->cbuf_size[0] : 256);
        typeof(drv_memo[0]) *m = &drv_memo[stage];
        if (m->valid && m->seq == vk_pending_seq() && m->size == size && m->gen == g_sbuf_bind_gen[stage]) return m->bi;
    }
    if (k == 0) {
        uint32_t drv[0x800 / 4] = {0};
        if (stage == 5) {
            drv[0x200 / 4] = gx; drv[0x204 / 4] = gy; drv[0x208 / 4] = gz;
            for (int i = 0; i < 16; i++) {
                uint32_t *u = &drv[(0x210 + 16 * i) / 4];
                u[0] = (uint32_t)st.ubuf[5][i].addr; u[1] = (uint32_t)(st.ubuf[5][i].addr >> 32); u[2] = (uint32_t)st.ubuf[5][i].size;
                uint32_t *e = &drv[(0x310 + 16 * i) / 4];
                e[0] = (uint32_t)st.sbuf[5][i].addr; e[1] = (uint32_t)(st.sbuf[5][i].addr >> 32); e[2] = (uint32_t)st.sbuf[5][i].size;
            }
        } else {
            uint32_t base = 0x110 + 0x100 * (stage == 1 ? 4 : 0);
            for (int i = 0; i < 16; i++) {
                uint32_t *e = &drv[(base + 16 * i) / 4];
                e[0] = (uint32_t)st.sbuf[stage][i].addr;
                e[1] = (uint32_t)(st.sbuf[stage][i].addr >> 32);
                e[2] = (uint32_t)st.sbuf[stage][i].size;
            }
        }
        if (g_debug_frame && stage <= 1 && getenv("XCDE_TRACE_CB") && getenv("XCDE_TRACE_CB")[0] == '2') {  // [XCDE] bound storage buffers
            for (int i = 0; i < 16; i++) {
                if (!st.sbuf[stage][i].addr) continue;
                const uint32_t *w = (const uint32_t *)(uintptr_t)st.sbuf[stage][i].addr;
                LOG("[sbuf] pass %d stage %d s%d addr %llx size %llu :", debug_pass_no, stage, i, (unsigned long long)st.sbuf[stage][i].addr, (unsigned long long)st.sbuf[stage][i].size);
                for (int q = 0; q < 12; q++) LOG(" %08x", w[q]);
                LOG("\n");
            }
        }
        uint64_t size = si->cbuf_size[0] > sizeof drv ? sizeof drv : (si->cbuf_size[0] ? si->cbuf_size[0] : 256);
        if (size < 0x500 && stage == 5) size = 0x500;
        VkDescriptorBufferInfo bi = ubo_span(copy_dedupe(stage == 5 ? DEDUPE_DRV_C : stage == 1 ? DEDUPE_DRV_F : DEDUPE_DRV_V, drv, size, 256), size);
        if (stage < 3) {
            drv_memo[stage].seq = vk_pending_seq(); drv_memo[stage].size = size; drv_memo[stage].gen = g_sbuf_bind_gen[stage];
            drv_memo[stage].bi = bi; drv_memo[stage].valid = true;
        }
        return bi;
    }
    if (k == 1) {  // compiler constants stored after the shader code
        NvnShaderStage *ss = &st.program->stages[stage];
        uint64_t size = ss->const_size ? ss->const_size : 16;
        if (size < si->cbuf_size[1]) size = si->cbuf_size[1];
        return ubo_span(guest_copy(ss->code_addr + ss->const_offset, size), size);  // (constant: shared per submission)
    }
    int idx = k - 3;  // cbuf_base
    Range r = (idx >= 0 && idx < 18) ? st.ubuf[stage][idx] : (Range){0, 0};
    uint64_t size = si->cbuf_size[k] ? si->cbuf_size[k] : 256;
    if (!r.addr) return ubo_span(vk_transient_copy(zeros, size, 256), size);
    if ((si->cbuf_dynamic & (1u << k)) && r.size) size = r.size > 0x10000 ? 0x10000 : r.size;
    if (g_debug_frame && getenv("XCDE_TRACE_CB") && getenv("XCDE_TRACE_CB")[0] == '2' && stage == 1 && size >= 16) {  // [XCDE] whole buffer
        const float *f = (const float *)(uintptr_t)r.addr;
        LOG("[cbfull] k %d size %llu (pass %d)\n", k, (unsigned long long)size, debug_pass_no);
        for (uint64_t q = 0; q < (r.size < size ? r.size : size) / 16 && q < 64; q++) LOG("[cbfull]  %2llu: %10.5g %10.5g %10.5g %10.5g\n", (unsigned long long)q, f[q*4], f[q*4+1], f[q*4+2], f[q*4+3]);
    }
    if (stage == 1 && k == 4 && dump_fs_want() && dump_fs_want() == g_cur_fs_hash) {  // [FIX-TEX] the material constants (cb4) of the draws with that fragment shader
        static unsigned cn, shown;
        if ((cn++ < 6 || cn % 200 == 0) && shown < 100) {
            shown++;
            const float *f = (const float *)(uintptr_t)r.addr;
            unsigned nf = (unsigned)((r.size ? (r.size < 64 ? r.size : 64) : 64) / 4);
            LOG("[tex-fix] cb4 draw#%u addr %llx bound %llu used %llu, first %u floats:", cn, (unsigned long long)r.addr, (unsigned long long)r.size, (unsigned long long)size, nf);
            for (unsigned q = 0; q < nf; q++) LOG(" %g", f[q]);
            LOG("%c", 10);
        }
    }
    if (stage == 1 && k == 4) g_mat_blue = 0;
    if (stage == 1 && k == 4 && g_cur_fs_hash && (g_cur_fs_hash == dump_fs_want() || (blue_fix_color() && g_cur_fs_hash == 0x69dc9c67ee198e5dull))) {
        const float *f = (const float *)(uintptr_t)r.addr;
        g_mat_blue = (!r.size || r.size >= 48) && f[8] == 0.0f && f[9] == 0.0f && f[10] == 1.0f;
    }
    if (size > r.size && r.size) {
        // [XCDE] T126: the shader may read past the range the game bound. Maxwell returns 0 there; a shorter Vulkan
        // range only promises "some value of the buffer" (robustBufferAccess): leftovers of the transient ring
        // (another port saw garbage loop counts and GPU hangs from this). Copy the bound part, zero the rest.
        VkSpan s = vk_transient(size, 256);
        memcpy(s.ptr, (const void *)(uintptr_t)r.addr, r.size);
        memset((uint8_t *)s.ptr + r.size, 0, size - r.size);
        if (g_mat_blue && blue_fix_color() && r.size >= 48) {  // [FIX-TEX] placeholder blue -> neutral colour (see blue_fix_color)
            float *pf = (float *)s.ptr;
            pf[8] = blue_fix_color()[0]; pf[9] = blue_fix_color()[1]; pf[10] = blue_fix_color()[2];
        }
        return ubo_span(s, size);
    }
    if (size < 16) size = 16;
    // snapshot: later CPU writes must not leak in. Shared by the draws of one submission binding the same range:
    // within a command list nothing writes guest memory (no inline UpdateUniformBuffer: that writes through below
    // and drops the shared copies).
    if (g_debug_frame && getenv("XCDE_TRACE_CB")) { const float *f = (const float *)(uintptr_t)r.addr; LOG("[cb] stage %d k %d addr %llx size %llu : %g %g %g %g\n", stage, k, (unsigned long long)r.addr, (unsigned long long)size, f[0], f[1], f[2], f[3]); }
    if (g_mat_blue && blue_fix_color() && size >= 48) {  // [FIX-TEX] same as above for a range bound at its full size
        VkSpan s2 = vk_transient(size, 256);
        memcpy(s2.ptr, (const void *)(uintptr_t)r.addr, size);
        float *pf = (float *)s2.ptr;
        pf[8] = blue_fix_color()[0]; pf[9] = blue_fix_color()[1]; pf[10] = blue_fix_color()[2];
        return ubo_span(s2, size);
    }
    return ubo_span(guest_copy(r.addr, size), size);
}

// Global memory (shader LDG/STG, maxwell_gmem_base = HEAP_BASE): the guest heap imported as a GPU buffer
// (VK_EXT_external_memory_host), zero-copy like metal.m's g_guest_buffer, so compute results land in guest
// memory. The import pins its pages, so it covers only the heap the game set up (SetHeapSize), made on first use.
// LARECOMP_VK_NO_GMEM=1: no import, reads return zero.
static VkBuffer gmem_buf;
static VkDeviceMemory gmem_mem;
static uint64_t gmem_size;

// [XCDE] XCDE_GMEM_CHECK="<frame>:<hex addr>:<hex size>": at that presented frame, what the GPU reads through the
// heap import vs what the CPU sees there (pages decommitted / recommitted after the import would keep their old copy
// on the GPU side)
void vk_gmem_check(uint64_t frame) {
    static unsigned long long f, a, n;
    static int init;
    if (!init) { init = 1; const char *e = getenv("XCDE_GMEM_CHECK"); if (e) sscanf(e, "%llu:%llx:%llx", &f, &a, &n); }
    if (!f || frame != f || !gmem_buf || a < HEAP_BASE || a + n > HEAP_BASE + gmem_size) return;
    VkSpan out = vk_readback(n);
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    VkBufferCopy c = {a - HEAP_BASE, out.off, n};
    vkCmdCopyBuffer(cb, gmem_buf, out.buf, 1, &c);
    renderer_flush();
    const uint8_t *g = out.ptr, *m = (const uint8_t *)(uintptr_t)a;
    uint64_t diff = 0, first = ~0ull, pages = 0, last_page = ~0ull;
    for (uint64_t i = 0; i < n; i++)
        if (g[i] != m[i]) { diff++; if (first == ~0ull) first = i; if (i / 4096 != last_page) { pages++; last_page = i / 4096; } }
    LOG("[gmem] frame %llu %llx+%llx: %llu bytes differ in %llu pages (first at +%llx)\n", f, a, n, (unsigned long long)diff,
        (unsigned long long)pages, (unsigned long long)first);
}

// [XCDE] Linux: the heap was placed in GPU memory at boot (heap_placed.c): import that buffer object (all of HEAP_MAX,
// so it never grows) instead of the host pointer.
static bool gmem_import_placed(void) {
    extern int vk_heap_fd(VkPhysicalDevice phys, uint32_t *type);
    uint32_t type;
    int fd = g_vk.heap_fd_import ? vk_heap_fd(g_vk.phys, &type) : -1;
    if (fd < 0) return false;
    VkExternalMemoryBufferCreateInfo eb = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
                                           .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &eb, .size = HEAP_MAX,
                             .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT};
    VkBuffer buf;
    if (vkCreateBuffer(g_vk.dev, &bi, NULL, &buf) != VK_SUCCESS) { close(fd); return false; }
    VkMemoryRequirements rq;
    vkGetBufferMemoryRequirements(g_vk.dev, buf, &rq);
    VkImportMemoryFdInfoKHR imp = {.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
                                   .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT, .fd = fd};
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &imp, .allocationSize = HEAP_MAX,
                               .memoryTypeIndex = type};
    VkDeviceMemory mem;
    if (!(rq.memoryTypeBits & (1u << type)) || rq.size > HEAP_MAX || vkAllocateMemory(g_vk.dev, &ai, NULL, &mem) != VK_SUCCESS) {
        close(fd);  // (a successful import owns the fd; a failed one does not)
        vkDestroyBuffer(g_vk.dev, buf, NULL);
        LOG("[vk] placed guest heap: import failed, using the host-memory import\n");
        return false;
    }
    if (vkBindBufferMemory(g_vk.dev, buf, mem, 0) != VK_SUCCESS) { vkDestroyBuffer(g_vk.dev, buf, NULL); vkFreeMemory(g_vk.dev, mem, NULL); return false; }
    gmem_buf = buf;
    gmem_mem = mem;
    gmem_size = HEAP_MAX;
    LOG("[vk] guest heap shared with the GPU (placed, %llu MB): no host-memory import\n", (unsigned long long)(HEAP_MAX >> 20));
    return true;
}

static bool gmem_import(uint64_t size) {
    if (!gmem_buf && gmem_import_placed()) return true;
    PFN_vkGetMemoryHostPointerPropertiesEXT props =
        (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(g_vk.dev, "vkGetMemoryHostPointerPropertiesEXT");
    VkMemoryHostPointerPropertiesEXT mp = {.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    if (!props || props(g_vk.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, (void *)(uintptr_t)HEAP_BASE, &mp) != VK_SUCCESS ||
        !mp.memoryTypeBits)
        return false;
    VkImportMemoryHostPointerInfoEXT imp = {.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
                                            .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                            .pHostPointer = (void *)(uintptr_t)HEAP_BASE};
    VkExternalMemoryBufferCreateInfo eb = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
                                           .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &eb, .size = size,
                             .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT};
    VkBuffer buf;
    if (vkCreateBuffer(g_vk.dev, &bi, NULL, &buf) != VK_SUCCESS) return false;
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(g_vk.dev, buf, &requirements);
    uint32_t compatible = mp.memoryTypeBits & requirements.memoryTypeBits;
    VkPhysicalDeviceMemoryProperties memory;
    memory = g_vk.memory_props;
    // CPU changes to imported memory are observed directly: prefer coherent types. [PERF-V5] but do not give up the import when
    // the driver exposes none (V4 took the first compatible type); failing here sends every gmem shader to the whole-heap staging copy.
    uint32_t coherent = compatible;
    for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
        if (!(memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) coherent &= ~(1u << i);
    if (coherent) compatible = coherent;
    else LOG("[vk] guest heap import: no host-coherent memory type, using the first compatible one\n");
    if (!compatible || requirements.size > size) { vkDestroyBuffer(g_vk.dev, buf, NULL); return false; }
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &imp, .allocationSize = requirements.size,
                               .memoryTypeIndex = (uint32_t)__builtin_ctz(compatible)};
    VkDeviceMemory mem;
    if (vkAllocateMemory(g_vk.dev, &ai, NULL, &mem) != VK_SUCCESS) { vkDestroyBuffer(g_vk.dev, buf, NULL); return false; }
    if (vkBindBufferMemory(g_vk.dev, buf, mem, 0) != VK_SUCCESS) {
        vkDestroyBuffer(g_vk.dev, buf, NULL); vkFreeMemory(g_vk.dev, mem, NULL); return false;
    }
    if (gmem_buf) {  // the heap grew: the old import may still be in use by submitted work
        vk_defer_import(gmem_buf, gmem_mem); // T254: current unsubmitted draws may still refer to it
    }
    gmem_buf = buf;
    gmem_mem = mem;
    gmem_size = size;
    LOG("[vk] guest heap imported for shader global memory: %llu MB (maxStorageBufferRange %u MB)\n", (unsigned long long)(size >> 20),
        g_vk.props.limits.maxStorageBufferRange >> 20);
    return true;
}

static uint64_t gmem_tried;  // heap size the import was last attempted for (gmem_info imports / re-imports)

static VkDescriptorBufferInfo gmem_info(void) {
    static int off = -1;
    if (off < 0) off = getenv("LARECOMP_VK_NO_GMEM") != NULL || !g_vk.host_import;
    uint64_t want = (kernel_heap_size() + 0xFFFF) & ~0xFFFFull;
    if (want > HEAP_MAX) want = HEAP_MAX;
    if (!off && want > gmem_size && !gmem_import(want)) {
        LOG("[vk] guest heap import failed (%llu MB): using staging for shader global memory\n", (unsigned long long)(want >> 20));
        off = 1;
    }
    // [PERF-V5] a descriptor range above maxStorageBufferRange is invalid (Intel HD: 1 GiB): bind the first window only.
    if (gmem_buf) return (VkDescriptorBufferInfo){gmem_buf, 0, gmem_size > g_vk.props.limits.maxStorageBufferRange ? g_vk.props.limits.maxStorageBufferRange : gmem_size};
    static uint32_t zero[1024];
    VkSpan s = vk_transient_copy(zero, sizeof zero, 256);
    return (VkDescriptorBufferInfo){s.buf, s.off, sizeof zero};
}

// [XCDE] T254: correctness-first fallback when host import is absent/fails. Vertex/index
// uploads already use bounded transient copies. Global-memory shaders need a shared heap
// snapshot with write-back, not the old all-zero dummy. Windowed shader addressing for heaps
// larger than maxStorageBufferRange is still required on some Intel/older drivers.
static struct { VkBuffer buf; VmaAllocation mem; uint64_t size, seq; uint32_t generation; VkSpan before; bool used; } gmem_stage;
static VkDescriptorBufferInfo gmem_shader_info(void) {
    VkDescriptorBufferInfo imported = gmem_info();
    uint64_t size = (kernel_heap_size() + 0xFFFF) & ~0xFFFFull;
    if (size > HEAP_MAX) size = HEAP_MAX;
    if (!size || getenv("LARECOMP_VK_NO_GMEM")) return imported; // explicit diagnostic switch
    if (size > g_vk.props.limits.maxStorageBufferRange) {
        // [PERF-V5] was FATAL (the game closed on Intel HD, 1 GiB limit). Windowed addressing is not implemented, so
        // shaders only see the first maxStorageBufferRange bytes of the heap: possible glitches in global-memory effects, but it runs.
        static int warned;
        if (!warned++) LOG("[gmem] heap %llu MiB exceeds maxStorageBufferRange %u MiB on %s: using the first window only\n",
            (unsigned long long)(size >> 20), g_vk.props.limits.maxStorageBufferRange >> 20, g_vk.props.deviceName);
        size = g_vk.props.limits.maxStorageBufferRange & ~0xFFFFull;
    }
    if (gmem_buf) return imported;
    uint64_t seq = vk_pending_seq();
    if (gmem_stage.seq != seq || gmem_stage.generation != gcopy_gen || gmem_stage.size != size) {
        // A fresh buffer avoids changing the snapshot of earlier, still-pending draws.
        if (gmem_stage.buf) vk_defer_buffer(gmem_stage.buf, gmem_stage.mem);
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateBuffer(g_vk.vma, &bi, &ai, &gmem_stage.buf, &gmem_stage.mem, NULL));
        gmem_stage.before = vk_transient_copy((const void *)(uintptr_t)HEAP_BASE, size, 256);
        metric_gmem_upload+=size;
        VkBufferCopy copy = {gmem_stage.before.off, 0, size};
        vkCmdCopyBuffer(vk_upload_cmd(), gmem_stage.before.buf, gmem_stage.buf, 1, &copy);
        gmem_stage.size = size; gmem_stage.seq = seq; gmem_stage.generation = gcopy_gen;
        static int logged;
        if (!logged++) LOG("[gmem] staging fallback: %llu MiB per snapshot; slower than host import\n", (unsigned long long)(size >> 20));
    }
    gmem_stage.used = true;
    return (VkDescriptorBufferInfo){gmem_stage.buf, 0, size};
}
static void gmem_stage_finish(bool stores) {
    if (!gmem_stage.used) return;
    gmem_stage.used = false;
    if (!stores) return;
    vk_end_pass();
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    VkSpan result = vk_readback(gmem_stage.size);
    metric_gmem_readback+=gmem_stage.size;
    VkBufferCopy copy = {0, result.off, gmem_stage.size};
    vkCmdCopyBuffer(cb, gmem_stage.buf, result.buf, 1, &copy);
    renderer_flush(); // copy and host-visibility barrier finish before CPU merge
    xcde_merge_gpu_words((uint32_t *)(uintptr_t)HEAP_BASE, gmem_stage.before.ptr, result.ptr, (size_t)(gmem_stage.size / 4));
    gmem_stage.seq = 0; guest_copy_flush();
}

// Compute dispatches that use global memory: the imported heap is host memory across PCIe on a discrete GPU, and a
// shader storing to it in a loop crawls (the little chain chomps in Madam MeowMeow's house: ~8 ms a frame on a GTX 1660
// Super and on an RTX 3080 Ti alike, which the game then waits for on its fence - 49-53 fps there on the 1660). Before
// such a dispatch its bound storage buffers (sorted, merged) are copied into a device-local buffer, the address table
// (binding GMAP_BUFFER_INDEX) tells the shader (translate.c, GLSL compute) where each range sits, loads and stores
// inside them use that copy, and a shader that stores gets the ranges copied back to the heap right after it (the
// game alternates its buffer sets per frame: the CPU isn't writing them meanwhile). Accesses outside the table use
// the heap as before. LARECOMP_VK_GMEM_FAST=0: no copy (empty table: everything goes to the heap, the old behaviour).
#define GFAST_SIZE (32ull << 20)
static VkBuffer gfast_buf;
static VmaAllocation gfast_mem;
static uint32_t gmap_tab[(GMAP_MAX + 1) * 4];  // this dispatch's table (stage_set copies it into binding 20)
static VkBufferCopy gfast_cp[GMAP_MAX];       // its copies heap -> gfast_buf (reversed after a storing dispatch)
static int gfast_n;

static VkDescriptorBufferInfo gfast_info(void) {
    if (!gfast_buf) {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = GFAST_SIZE,
                                 .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateBuffer(g_vk.vma, &bi, &ai, &gfast_buf, &gfast_mem, NULL));
    }
    return (VkDescriptorBufferInfo){gfast_buf, 0, GFAST_SIZE};
}

static void gfast_prepare(VkCommandBuffer cb) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("LARECOMP_VK_GMEM_FAST"); on = !e || atoi(e) != 0; }
    memset(gmap_tab, 0, sizeof gmap_tab);
    gfast_n = 0;
    gmem_info();  // (the heap import, made on first use)
    gfast_info();
    if (!on || !gmem_buf) return;
    struct { uint64_t lo, hi; } r[16];
    int n = 0;
    for (int i = 0; i < 16; i++) {  // heap offsets of the bound storage buffers, 256-aligned
        uint64_t a = st.sbuf[5][i].addr, sz = st.sbuf[5][i].size;
        if (!a || !sz || a < HEAP_BASE || a - HEAP_BASE + sz > gmem_size) continue;
        uint64_t lo = (a - HEAP_BASE) & ~255ull, hi = (a - HEAP_BASE + sz + 255) & ~255ull;
        if (hi > gmem_size) hi = gmem_size;
        int k = n++;
        while (k > 0 && r[k - 1].lo > lo) { r[k] = r[k - 1]; k--; }  // (insertion sort by start)
        r[k].lo = lo; r[k].hi = hi;
    }
    int m = 0;
    for (int i = 0; i < n; i++) {  // merge overlapping / nearby ranges (a small gap costs less than another table entry)
        if (m && r[i].lo <= r[m - 1].hi + 4096) { if (r[i].hi > r[m - 1].hi) r[m - 1].hi = r[i].hi; }
        else r[m++] = r[i];
    }
    VkBufferCopy cp[16];
    uint64_t off = 0;
    for (int i = 0; i < m; i++) {
        uint64_t len = r[i].hi - r[i].lo;
        if (m > GMAP_MAX || off + len > GFAST_SIZE) return;  // (too much: the heap as before)
        cp[i] = (VkBufferCopy){r[i].lo, off, len};
        gmap_tab[i * 4 + 0] = (uint32_t)r[i].lo;
        gmap_tab[i * 4 + 1] = (uint32_t)len;
        gmap_tab[i * 4 + 2] = (uint32_t)(off >> 2);
        off += (len + 255) & ~255ull;
    }
    if (!m) return;
    if (getenv("LARECOMP_LOG_GFAST")) {  // (debug) =<shader code address hex>: that shader's first tables
        static int nlog;
        if (st.program->stages[5].code_addr == strtoull(getenv("LARECOMP_LOG_GFAST"), NULL, 16) && nlog++ < 40) {
            LOG("[gfast] cs %llx: %d ranges, %llu KB:", (unsigned long long)st.program->stages[5].code_addr, m, (unsigned long long)(off >> 10));
            for (int i = 0; i < m; i++) LOG(" %llx+%llx", (unsigned long long)(r[i].lo + HEAP_BASE), (unsigned long long)(r[i].hi - r[i].lo));
            LOG("\n");
        }
    }
    vkCmdCopyBuffer(cb, gmem_buf, gfast_buf, (uint32_t)m, cp);
    vk_full_barrier(cb);
    gmap_tab[GMAP_MAX * 4] = (uint32_t)m;
    memcpy(gfast_cp, cp, (size_t)m * sizeof cp[0]);
    gfast_n = m;
}

// After a storing dispatch: its ranges back to the heap (before anything else reads them: draws, the CPU after the fence).
static void gfast_finish(VkCommandBuffer cb) {
    if (!gfast_n) return;
    VkBufferCopy back[GMAP_MAX];
    for (int i = 0; i < gfast_n; i++) back[i] = (VkBufferCopy){gfast_cp[i].dstOffset, gfast_cp[i].srcOffset, gfast_cp[i].size};
    vk_full_barrier(cb);
    vkCmdCopyBuffer(cb, gfast_buf, gmem_buf, (uint32_t)gfast_n, back);
    vk_full_barrier(cb);
    gfast_n = 0;
}

// ---- vertex / index cache
// Reading vertex / index data in place from the imported heap crosses PCIe for every vertex fetched on a discrete GPU
// (outdoors at 3x internal resolution on an RTX 3080 Ti: ~0.7-1.9 ms of GPU a frame; more on PCIe 3.0), and copying it
// per draw instead moves about as much onto the CPU. So each range bound for drawing gets a device-local copy, refreshed
// only where the CPU wrote since (the OS's page write tracking, plat_written_pages, asked at submission: vc_flush): one
// sequential GPU copy of the written pages in the submission's upload buffer (which completes before its draws) instead
// of fetches across the bus. Overlapping ranges are separate entries; the per-page generation keeps them all current.
// Pages the GPU writes itself (a compute dispatch's copy-back, gfast_finish) are never cached: draws over them read the
// heap in place as before, and so does everything when the OS gives no write tracking. LARECOMP_VK_VCACHE=0: off.
#define VC_PAGE 4096ull
#define VC_TAB 8192                   // entries (open addressing)
#define VC_MAX_RANGE (64ull << 20)    // larger bound ranges: read in place
#define VC_BUDGET (768ull << 20)      // device-local bytes kept; the least recently used go first
typedef struct { uint64_t lo, hi, used_seq; uint32_t gen, epoch; VkBuffer buf; VmaAllocation mem; bool gpu, logged; } VcEnt;
static VcEnt vc_tab[VC_TAB];
static uint32_t vc_pend[VC_TAB];  // entries the pending submission draws from (vc_flush brings them up to date)
static int vc_npend;
static uint32_t *vc_page_gen;  // per heap page: the generation of its last CPU write seen (vc_gen when observed)
static uint8_t *vc_page_gpu;   // per heap page: written by the GPU
static uint32_t vc_gen = 1, vc_epoch = 1;  // (epoch: bumped when pages turn GPU-written: entries check again)
static uint64_t vc_bytes;
static double vc_bytes_mb(void) { return vc_bytes / 1048576.0; }
#define VC_DEAD 1024
static struct { VkBuffer b; VmaAllocation m; uint64_t seq; } vc_dead[VC_DEAD];
static int vc_ndead;
static int vc_on = -1;
int vk_vcache_state(void) { return vc_on; }  // -1 not decided yet, 0 off, 1 on (objects.c frame lag)
// (profiler: draws served / read in place, bytes refreshed, write tracking queries, written pages seen, query time)
uint64_t prof_vc_hits, prof_vc_bypass, prof_vc_copied, prof_vc_queries, prof_vc_scan_pages, prof_vc_scan_ns;

static void vc_free_dead(bool all) {
    uint64_t done = vk_completed_seq();
    int k = 0;
    for (int i = 0; i < vc_ndead; i++) {
        if (!all && vc_dead[i].seq > done) { vc_dead[k++] = vc_dead[i]; continue; }
        vmaDestroyBuffer(g_vk.vma, vc_dead[i].b, vc_dead[i].m);
    }
    vc_ndead = k;
}

static bool vc_drop(VcEnt *e) {  // (freed once the submissions that may read it are done)
    if (vc_ndead == VC_DEAD) { vk_wait_seq(vk_pending_seq() - 1); vc_free_dead(false); }
    if (vc_ndead == VC_DEAD) return false;  // (all read by the pending submission: keep it)
    vc_dead[vc_ndead].b = e->buf; vc_dead[vc_ndead].m = e->mem; vc_dead[vc_ndead++].seq = e->used_seq;
    vc_bytes -= e->hi - e->lo;
    memset(e, 0, sizeof *e);
    return true;
}

// The page tables (vc_page_gen / vc_page_gpu) exist whenever the OS gives write tracking: the vertex cache and the
// texture upload checks (vk_pages_changed) share them. Each write tracking query resets the OS bits, so every query
// stamps the pages it saw written with a new generation, and each user compares against the generation it last saw.
static int pt_state = -1;  // 1: tracking, 0: none
static bool pt_init(void) {
    if (pt_state >= 0) return pt_state;
    uint64_t pg = VC_PAGE, one;
    int n = plat_written_pages(HEAP_BASE, VC_PAGE, &one, 1, &pg);  // (is there write tracking?)
    pt_state = n >= 0 && pg == VC_PAGE;
    if (!pt_state) { LOG("[vk] no page write tracking (%d, page %llu)\n", n, (unsigned long long)pg); return false; }
    vc_page_gen = calloc(HEAP_MAX / VC_PAGE, sizeof *vc_page_gen);
    vc_page_gpu = calloc(HEAP_MAX / VC_PAGE, 1);
    vc_page_gen[0] = ++vc_gen;  // (the probe took page 0's tracking)
    return true;
}

// Texture upload checks (textures.c): may guest memory [addr, addr + size) have changed since generation `since`?
// 0: no CPU write since, 1: written, -1: unknown (no tracking, outside the heap, or pages the GPU writes).
// *now: the generation to pass next time.
int vk_pages_changed(uint64_t addr, uint64_t size, uint32_t since, uint32_t *now) {
    if (!size || addr < HEAP_BASE || addr - HEAP_BASE + size > HEAP_MAX || !pt_init()) return -1;
    uint64_t lo = (addr - HEAP_BASE) & ~(VC_PAGE - 1), hi = (addr - HEAP_BASE + size + VC_PAGE - 1) & ~(VC_PAGE - 1);
    static uint64_t pages[4096];
    uint64_t pg = VC_PAGE;
    int n = plat_written_pages(HEAP_BASE + lo, hi - lo, pages, 4096, &pg);
    if (n < 0) return -1;
    uint32_t g = ++vc_gen;
    if (n >= 4096) for (uint64_t p = lo / VC_PAGE; p < hi / VC_PAGE; p++) vc_page_gen[p] = g;
    else for (int k = 0; k < n; k++) vc_page_gen[(pages[k] - HEAP_BASE) / VC_PAGE] = g;
    *now = g;
    int changed = 0;
    for (uint64_t p = lo / VC_PAGE; p < hi / VC_PAGE; p++) {
        if (vc_page_gpu[p]) return -1;
        if (vc_page_gen[p] > since) changed = 1;
    }
    return changed;
}

// ---- [XCDE] XCDE_OVERLAP_STATS=1 (Linux): what one frame in flight would collide with. Every guest-heap range the
// GPU reads in place (vertex / index data, the storage buffers of gmem shaders) or writes (transform feedback, storing
// shaders) is noted while a frame is recorded; at the game's frame boundary (objects.c, nvnSyncWait from the limiter)
// those pages are protected (platform/linux.c plat_ov_*) and the CPU's touches of them during the next frame counted.
static int ov_on = -1;
static pthread_mutex_t ov_lock = PTHREAD_MUTEX_INITIALIZER;
static struct { uint64_t lo, hi; uint8_t w; } *ov_r;
static int ov_rn, ov_rcap;
static void ov_note(uint64_t addr, uint64_t size, bool gpu_writes) {
    extern bool vk_ov_wait_mode(void);
    if (ov_on < 0) ov_on = getenv("XCDE_OVERLAP_STATS") ? atoi(getenv("XCDE_OVERLAP_STATS")) : vk_ov_wait_mode() ? 1 : 0;  // 1 all, 2 GPU reads, 3 GPU writes
    if (!ov_on || !size || addr < HEAP_BASE || addr - HEAP_BASE >= HEAP_MAX || (ov_on == 2 && gpu_writes) || (ov_on == 3 && !gpu_writes)) return;
    pthread_mutex_lock(&ov_lock);
    if (ov_rn == ov_rcap) { ov_rcap = ov_rcap ? ov_rcap * 2 : 4096; ov_r = realloc(ov_r, ov_rcap * sizeof *ov_r); }
    ov_r[ov_rn].lo = addr & ~0xFFFull; ov_r[ov_rn].hi = (addr + size + 0xFFF) & ~0xFFFull; ov_r[ov_rn].w = gpu_writes;
    ov_rn++;
    pthread_mutex_unlock(&ov_lock);
}
static void ov_note_sbufs(int stage, bool gpu_writes) {
    for (int i = 0; i < 16; i++) ov_note(st.sbuf[stage][i].addr, st.sbuf[stage][i].size, gpu_writes);
}
static int ov_cmp(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}
// (game thread, at the frame boundary: the worker has run every job of the frame) protect the frame's ranges.
// seq: the frame's last submission (wait mode, XCDE_OVERLAP=1, one frame in flight: CPU accesses wait for it); 0: stats.
// XCDE_OVERLAP: one frame in flight made safe by page protection (wait mode). On by default on Linux (Steam Deck: the
// same field view 41 -> 55-57 fps; battle and pause menu clean, unlike the unprotected frame lag), XCDE_OVERLAP=0: off.
bool vk_ov_wait_mode(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("XCDE_OVERLAP");
#ifdef __linux__
        on = e ? atoi(e) != 0 : !getenv("XCDE_OVERLAP_STATS");
#else
        on = e && atoi(e) != 0;
#endif
    }
    return on;
}
void vk_ov_frame(uint64_t seq) {
    if (ov_on < 0) ov_note(0, 0, false);  // (reads the switches)
    if (ov_on <= 0) return;
    if (seq) {
        // (the frame is already on the GPU's timeline: the worker submits after every fence in wait mode, CMD_FENCE_SYNC;
        // an access must never wait for unsubmitted work)
        plat_ov_set_gpu_done(vk_completed_seq);
        plat_ov_set_gpu_wait(vk_block_seq);
        if (seq > submit_seq) seq = submit_seq;
    }
    pthread_mutex_lock(&ov_lock);
    qsort(ov_r, (size_t)ov_rn, sizeof *ov_r, ov_cmp);  // (by lo: the struct starts with it)
    static uint64_t *lo, *hi; static uint8_t *w; static int cap;
    if (cap < ov_rn) { cap = ov_rn; lo = realloc(lo, cap * 8); hi = realloc(hi, cap * 8); w = realloc(w, cap); }
    int n = 0;
    for (int i = 0; i < ov_rn; i++) {  // merge overlapping / touching ranges; a run with any GPU write is GPU-written
        if (n && ov_r[i].lo <= hi[n - 1]) { if (ov_r[i].hi > hi[n - 1]) hi[n - 1] = ov_r[i].hi; w[n - 1] |= ov_r[i].w; }
        else { lo[n] = ov_r[i].lo; hi[n] = ov_r[i].hi; w[n] = ov_r[i].w; n++; }
    }
    ov_rn = 0;
    pthread_mutex_unlock(&ov_lock);
    plat_ov_protect(lo, hi, w, n, seq);
    static int frames;
    static int every = -1;  // (stats: every 30 frames; wait mode: every 1800, a minute at 30 fps)
    if (every < 0) every = getenv("XCDE_OVERLAP_STATS") ? 30 : 1800;
    if (++frames % every == 0) plat_ov_report();
}

// Heap pages [lo, hi) (offsets) written by the GPU: never cached from now on.
static void vc_mark_gpu(uint64_t lo, uint64_t hi) {
    ov_note(lo + HEAP_BASE, hi - lo, true);  // [XCDE] (XCDE_OVERLAP_STATS)
    if (!pt_init()) return;
    { static int nl; if (getenv("XCDE_TRACE_VC") && nl < 40 && !vc_page_gpu[lo / VC_PAGE]) { nl++; LOG("[vc] GPU writes %llx..%llx\n", (unsigned long long)(lo + HEAP_BASE), (unsigned long long)(hi + HEAP_BASE)); } }
    bool changed = false;
    for (uint64_t p = lo / VC_PAGE; p < (hi + VC_PAGE - 1) / VC_PAGE && p < HEAP_MAX / VC_PAGE; p++)
        if (!vc_page_gpu[p]) { vc_page_gpu[p] = 1; changed = true; }
    // [XCDE] perf: only when a page newly turned GPU-written (every grass / skinning writer draw re-marks the same
    // scratch pages: each bump made every cache entry rescan its pages, up to 16,384 of them, on its next lookup)
    if (changed) vc_epoch++;
}

// A storing shader (stage 0 / 1 / 5) may write any of its bound storage buffers.
static void vc_mark_sbufs(int stage) {
    for (int i = 0; i < 16; i++) {
        uint64_t a = st.sbuf[stage][i].addr, sz = st.sbuf[stage][i].size;
        if (a >= HEAP_BASE && sz && a - HEAP_BASE < HEAP_MAX) vc_mark_gpu(a - HEAP_BASE, a - HEAP_BASE + sz);
    }
}

static bool vcache_lookup(uint64_t addr, uint64_t size, VkBuffer *buf, VkDeviceSize *off) {
    if (vc_on < 0) {
        const char *e = getenv("LARECOMP_VK_VCACHE");
        vc_on = !e || atoi(e) != 0;
        // [XCDE] T169: off by default on integrated GPUs. The cache exists to keep vertex fetches off the PCIe bus; an
        // integrated GPU reads the imported heap from the same RAM, so the device-local copies would only take up to
        // 768 MB of that RAM (measured on the dev PC: process private 5.6 GB without the cache, 6.2 GB and still growing
        // with it) plus a copy per written page. On the dGPU it matters: GPU busy 4.0 ms a frame with it, 9.8 without.
        if (!e && g_vk.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
            vc_on = 0;
            LOG("[vk] vertex cache off: integrated GPU (LARECOMP_VK_VCACHE=1 turns it on)\n");
        }
        // [XCDE] T119: off in the deterministic mode (T029): with it, two runs differ in ~7 % of the intro movie's
        // frames (shifted by a frame); without it 99.9 % match. Why its refresh varies between runs: T121.
        if (diag_deterministic() && !(e && atoi(e) != 0)) vc_on = 0;
        if (vc_on && !pt_init()) {
            LOG("[vk] vertex cache off: no page write tracking\n");
            vc_on = 0;
        }
    }
    if (!vc_on || !gmem_buf || size > VC_MAX_RANGE) return false;
    uint64_t lo = (addr - HEAP_BASE) & ~(VC_PAGE - 1), hi = (addr - HEAP_BASE + size + VC_PAGE - 1) & ~(VC_PAGE - 1);
    if (hi > gmem_size) return false;
    uint64_t pend = vk_pending_seq();
    uint64_t h = (lo * 0x9E3779B97F4A7C15ull) ^ (hi * 0xC2B2AE3D27D4EB4Full);
    uint32_t slot = (uint32_t)(h ^ (h >> 32)) & (VC_TAB - 1), free_slot = ~0u, old_slot = ~0u;
    VcEnt *e = NULL;
    for (int probe = 0; probe < 16; probe++, slot = (slot + 1) & (VC_TAB - 1)) {
        VcEnt *c = &vc_tab[slot];
        if (c->buf && c->lo == lo && c->hi == hi) { e = c; break; }
        if (!c->buf && free_slot == ~0u) free_slot = slot;
        if (c->buf && c->used_seq < pend && (old_slot == ~0u || c->used_seq < vc_tab[old_slot].used_seq)) old_slot = slot;
    }
    if (!e) {  // a new range: its own device-local buffer (gen 0: copied whole at the submission)
        if (vc_npend == VC_TAB) return false;
        if (vc_ndead) vc_free_dead(false);
        if (free_slot == ~0u) {
            if (old_slot == ~0u || !vc_drop(&vc_tab[old_slot])) return false;
            free_slot = old_slot;
        }
        while (vc_bytes + (hi - lo) > VC_BUDGET) {  // over budget: the least recently used entry goes
            int lru = -1;
            for (int i = 0; i < VC_TAB; i++)
                if (vc_tab[i].buf && vc_tab[i].used_seq < pend && (lru < 0 || vc_tab[i].used_seq < vc_tab[lru].used_seq)) lru = i;
            if (lru < 0 || !vc_drop(&vc_tab[lru])) return false;
        }
        e = &vc_tab[free_slot];
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = hi - lo,
                                 .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateBuffer(g_vk.vma, &bi, &ai, &e->buf, &e->mem, NULL) != VK_SUCCESS) { memset(e, 0, sizeof *e); return false; }
        e->lo = lo; e->hi = hi;
        vc_bytes += hi - lo;
    }
    if (e->epoch != vc_epoch) {  // (pages turned GPU-written since: is this range one of them?)
        e->gpu = false;
        for (uint64_t p = lo / VC_PAGE; p < hi / VC_PAGE && !e->gpu; p++) e->gpu = vc_page_gpu[p];
        e->epoch = vc_epoch;
    }
    if (e->gpu) {
        prof_vc_bypass++;
        static int nl;  // XCDE_TRACE_VC=1: ranges the cache refuses because the GPU writes (part of) them
        static int tvc = -1;  // ([XCDE] perf: bypasses happen on every grass draw; getenv is slow)
        if (tvc < 0) tvc = getenv("XCDE_TRACE_VC") != NULL;
        if (tvc && nl < 40 && !e->logged) { nl++; e->logged = true; LOG("[vc] bypass %llx..%llx (GPU-written pages)\n", (unsigned long long)(lo + HEAP_BASE), (unsigned long long)(hi + HEAP_BASE)); }
        return false;
    }
    if (e->used_seq != pend) {  // first use in this submission: brought up to date by vc_flush before it runs
        e->used_seq = pend;
        vc_pend[vc_npend++] = (uint32_t)(e - vc_tab);
    }
    prof_vc_hits++;
    *buf = e->buf;
    *off = addr - HEAP_BASE - lo;
    return true;
}

static int vc_pend_cmp(const void *a, const void *b) {
    uint64_t x = vc_tab[*(const uint32_t *)a].lo, y = vc_tab[*(const uint32_t *)b].lo;
    return x < y ? -1 : x > y;
}

// At submission (vk_submit_extra, before the upload buffer closes): the pages the CPU wrote in the ranges this
// submission draws from - one write tracking query per run of nearby ranges (a query costs about as much as scanning
// ~64 pages: ranges closer than that share one) - and their copies into the cache, ahead of the draws.
void vc_flush(void) {
    if (!vc_npend) return;
    qsort(vc_pend, (size_t)vc_npend, sizeof vc_pend[0], vc_pend_cmp);
    uint32_t g = ++vc_gen;
    static uint64_t pages[4096];
    uint64_t t0 = SDL_GetTicksNS();
    for (int i = 0; i < vc_npend;) {
        uint64_t lo = vc_tab[vc_pend[i]].lo, hi = vc_tab[vc_pend[i]].hi;
        int j = i + 1;
        for (; j < vc_npend && vc_tab[vc_pend[j]].lo <= hi + 64 * VC_PAGE; j++)
            if (vc_tab[vc_pend[j]].hi > hi) hi = vc_tab[vc_pend[j]].hi;
        uint64_t pg = VC_PAGE;
        int n = plat_written_pages(HEAP_BASE + lo, hi - lo, pages, 4096, &pg);
        prof_vc_queries++;
        if (n < 0 || n >= 4096) for (uint64_t p = lo / VC_PAGE; p < hi / VC_PAGE; p++) vc_page_gen[p] = g;  // (all)
        else for (int k = 0; k < n; k++) vc_page_gen[(pages[k] - HEAP_BASE) / VC_PAGE] = g;
        if (n > 0) prof_vc_scan_pages += (uint64_t)n;
        i = j;
    }
    prof_vc_scan_ns += SDL_GetTicksNS() - t0;
    VkCommandBuffer cb = vk_upload_cmd();
    for (int i = 0; i < vc_npend; i++) {
        VcEnt *e = &vc_tab[vc_pend[i]];
        VkBufferCopy cp[64];
        int nc = 0;
        for (uint64_t p = e->lo / VC_PAGE; p < e->hi / VC_PAGE; p++) {
            if (e->gen && vc_page_gen[p] <= e->gen) continue;
            uint64_t o = p * VC_PAGE;
            if (nc && cp[nc - 1].srcOffset + cp[nc - 1].size == o) { cp[nc - 1].size += VC_PAGE; continue; }
            if (nc == 64) { nc = 1; cp[0] = (VkBufferCopy){e->lo, 0, e->hi - e->lo}; break; }  // (scattered: the whole range)
            cp[nc++] = (VkBufferCopy){o, o - e->lo, VC_PAGE};
        }
        if (nc) {
            vkCmdCopyBuffer(cb, gmem_buf, e->buf, (uint32_t)nc, cp);
            for (int k = 0; k < nc; k++) prof_vc_copied += cp[k].size;
        }
        e->gen = g;
    }
    vc_npend = 0;
}

// Vertex / index data from the imported guest heap (renderer.m bind_guest reads g_guest_buffer the same way): the
// device-local cache above, else read in place - no per-draw copy, no index scan for the copy size. The game fences its
// CPU writes as on the Switch. Falls back to copies outside the heap or without the import; LARECOMP_VK_COPY_VERTICES=1
// forces copies.
static void audit(const char *fmt, ...);
static uint64_t audit_vs(void);
static uint64_t audit_fs(void);
// [XCDE] T175 redo: device-local mirrors of the transform-feedback buffers. XCDE's grass is drawn from instance data
// that a "writer" point draw captures with transform feedback into a scratch buffer in the guest heap; the heap is host
// memory imported into Vulkan, so every grass vertex fetched its instance data over PCIe (the depth prepass and colour
// pass of the grass: 6.8 of 13.4 ms a frame at 4K in the heavy meadow view on the RTX GPU). Now after each writer
// draw the bytes it captured are copied once into a device-local mirror of that buffer, and vertex / index fetches in
// the buffer's range read the mirror. The heap keeps the data too (shader loads, the CPU see it as before); a mirror is
// filled whole when created. Discrete GPUs only (an iGPU reads the heap from its own memory). XCDE_XFB_SHADOW=0: off.
// [XCDE] T197: ranges written by storing shaders (the vertex-stage skinning / cloth writers, compute) that draws then
// read as vertex data get the same mirrors: an in-place read of a GPU-written range is noted, and from the next storing
// draw / dispatch that binds it on, each such store refreshes the mirror whole after the draw. XCDE_STORE_SHADOW=0: off.
typedef struct { uint64_t base, size; VkBuffer buf; VmaAllocation mem; bool sb; } XfbShadow;
#define NXSH 64
static XfbShadow xsh[NXSH];
static int nxsh;
static bool xsh_on(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("XCDE_XFB_SHADOW");
        on = !(e && *e == '0') && (e ? true : g_vk.props.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);
    }
    return on;
}
static XfbShadow *xsh_find(uint64_t a, uint64_t n) {
    (void)n;  // (streams are bound with the buffer's whole size from an offset: past its end - the binding is clamped)
    for (int i = 0; i < nxsh; i++) if (a >= xsh[i].base && a < xsh[i].base + xsh[i].size) return &xsh[i];
    return NULL;
}
static void xsh_copy(XfbShadow *s, VkCommandBuffer cb, uint64_t at, uint64_t n) {  // (outside a render pass)
    vk_full_barrier(cb);  // (the writer's stores done; earlier readers of the mirror done)
    VkBufferCopy c = {at - HEAP_BASE, at - s->base, n};
    vkCmdCopyBuffer(cb, gmem_buf, s->buf, 1, &c);
    vk_full_barrier(cb);  // (the next readers see it)
}
static void xsh_retire_overlaps(uint64_t a, uint64_t sz);
// after a transform-feedback writer draw (its pass ended): [at, at + n) of the buffer [base, base + size) was written
static void xsh_written(uint64_t base, uint64_t size, uint64_t at, uint64_t n) {
    if (!xsh_on() || !gmem_buf || !n || base < HEAP_BASE || base + size > HEAP_BASE + gmem_size) return;
    XfbShadow *s = NULL;
    for (int i = 0; i < nxsh; i++) if (xsh[i].base == base && xsh[i].size == size) s = &xsh[i];
    VkCommandBuffer cb = vk_cmd();
    if (!s) {
        xsh_retire_overlaps(base, size);  // [XCDE] T205 (redo): a transform-feedback mirror too (see xsh_retire_overlaps)
        if (nxsh == NXSH) return;
        s = &xsh[nxsh];
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
                                 .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateBuffer(g_vk.vma, &bi, &ai, &s->buf, &s->mem, NULL) != VK_SUCCESS) return;
        s->base = base; s->size = size;
        nxsh++;
        xsh_copy(s, cb, base, size);  // (whole: parts not written this frame hold what the heap holds)
        LOG("[vk] transform feedback mirror %llx +%llu in device memory\n", (unsigned long long)base, (unsigned long long)size);
    }
    if (at + n > base + size) n = base + size - at;
    xsh_copy(s, cb, at, n);
}

static uint64_t xsh_want[32];  // (T197) addresses of in-place vertex reads of GPU-written pages
static int nwant;
static bool xsh_store_on(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("XCDE_STORE_SHADOW"); on = !(e && *e == '0') && xsh_on(); }
    return on;
}
// after a storing draw (its pass ended) or dispatch: refresh / make the mirrors of the stage's storage buffers
// [XCDE] T205: the writers rebind the same scratch memory with other sizes / offsets as scenes change (the user's
// log: 7265f2ef00 +40224, then 7265f2ef00 +81632). The older mirror over it was no longer refreshed but still
// matched lookups first: draws read stale data, and the binding clamped to its end left the vertices past it reading
// out of bounds - characters' faces drawn as vertical strips in cutscenes. A new store mirror now retires every
// store mirror it overlaps (its buffer freed once the submissions that may read it are done).
// [XCDE] T205 redo (2026-10-21, the user's Fallen Arm shot: Shulk's face as strips again): any new mirror - store or
// transform feedback - retires every mirror of either kind it overlaps. The user's log had a new mirror 7265f79c00
// +61248 over two older ones (7265f7c000 +288, 7265f7c300 +61472) that only store mirrors were checked against, and
// lookups (xsh_find) return the first match: the stale one.
static void xsh_retire_overlaps(uint64_t a, uint64_t sz) {
    for (int k = 0; k < nxsh;) {
        XfbShadow *o = &xsh[k];
        if (o->base >= a + sz || a >= o->base + o->size) { k++; continue; }
        static int nlog;
        if (nlog++ < 40) LOG("[vk] mirror %llx +%llu retired (overlapped by %llx +%llu)\n", (unsigned long long)o->base,
                             (unsigned long long)o->size, (unsigned long long)a, (unsigned long long)sz);
        if (vc_ndead == VC_DEAD) { vk_wait_seq(vk_pending_seq() - 1); vc_free_dead(false); }
        if (vc_ndead < VC_DEAD) { vc_dead[vc_ndead].b = o->buf; vc_dead[vc_ndead].m = o->mem; vc_dead[vc_ndead++].seq = vk_pending_seq(); }
        xsh[k] = xsh[--nxsh];
    }
}

static void xsh_stored(int stage) {
    if (!xsh_store_on()) return;
    for (int i = 0; i < 16; i++) {
        uint64_t a = st.sbuf[stage][i].addr, sz = st.sbuf[stage][i].size;
        if (a < HEAP_BASE || !sz || sz > (1ull << 20)) continue;
        XfbShadow *s = NULL;
        for (int k = 0; k < nxsh; k++) if (xsh[k].base == a && xsh[k].size == sz) s = &xsh[k];
        if (s && !s->sb) continue;  // (a transform-feedback mirror: refreshed by its captures)
        if (!s) {
            // [XCDE] T205 (third look, the user's shot: Shulk's face as strips after dying in a battle, log without any
            // overlapping new mirror): a store into a range that only overlaps a mirror (another base / size than the
            // one it was made for) refreshed nothing - and since draws read the mirror, no in-place read was ever
            // noted for the new range either, so the stale copy stayed. The overlapping part is copied now.
            // XCDE_MIRROR_OVERLAP=0: as before.
            static int ov = -1;
            if (ov < 0) ov = !(getenv("XCDE_MIRROR_OVERLAP") && atoi(getenv("XCDE_MIRROR_OVERLAP")) == 0);
            // The user's log after a defeat: mirror 7265eeda00 +72160, stores 7265ef5900 +72160 every frame - the same
            // buffer moved up 32 KB. Draws reading the new range found the old mirror and ran off its end (the strips).
            // A store reaching past an overlapping mirror retires it: reads go to the heap, and the next store makes
            // a mirror of the new range. A store inside a mirror refreshes that part.
            bool outside = false;
            for (int k = 0; ov && k < nxsh; k++) {
                XfbShadow *o = &xsh[k];
                if (o->base < a + sz && a < o->base + o->size && (a < o->base || a + sz > o->base + o->size)) outside = true;
            }
            if (outside) xsh_retire_overlaps(a, sz);
            for (int k = 0; ov && k < nxsh; k++) {
                XfbShadow *o = &xsh[k];
                uint64_t lo = a > o->base ? a : o->base, hi = a + sz < o->base + o->size ? a + sz : o->base + o->size;
                if (lo >= hi) continue;
                static int nlog;
                if (nlog++ < 20) LOG("[vk] mirror %llx +%llu refreshed by an overlapping store %llx +%llu\n", (unsigned long long)o->base,
                                     (unsigned long long)o->size, (unsigned long long)a, (unsigned long long)sz);
                xsh_copy(o, vk_cmd(), lo, hi - lo);
            }
            bool want = false;
            for (int k = 0; k < nwant; k++) want |= xsh_want[k] >= a && xsh_want[k] < a + sz;
            if (!want) continue;
            xsh_retire_overlaps(a, sz);
            int n0 = nxsh;
            xsh_written(a, sz, a, sz);
            if (nxsh > n0) xsh[n0].sb = true;
            continue;
        }
        xsh_copy(s, vk_cmd(), a, sz);
    }
}

static uint64_t g_direct_limit;  // (T175) bytes the buffer guest_direct returned holds from its offset (~0: the rest of it)
static bool guest_direct(uint64_t addr, uint64_t size, VkBuffer *buf, VkDeviceSize *off) {
    static int copy = -1;
    if (copy < 0) copy = getenv("LARECOMP_VK_COPY_VERTICES") != NULL;
    if (copy || addr < HEAP_BASE) return false;
    if (addr + size > HEAP_BASE + gmem_size && kernel_heap_size() != gmem_tried) { gmem_tried = kernel_heap_size(); gmem_info(); }
    if (!gmem_buf || addr + size > HEAP_BASE + gmem_size) return false;
    g_direct_limit = ~0ull;
    if (nxsh) { XfbShadow *s = xsh_find(addr, size); if (s) { *buf = s->buf; *off = addr - s->base; g_direct_limit = s->base + s->size - addr; return true; } }  // (T175)
    if (vcache_lookup(addr, size, buf, off)) return true;
    if (pt_state == 1 && nwant < 32 && vc_page_gpu[(addr - HEAP_BASE) / VC_PAGE] && xsh_store_on()) {  // (T197)
        bool k = false;
        for (int i = 0; i < nwant; i++) k |= xsh_want[i] == addr;
        if (!k) xsh_want[nwant++] = addr;
    }
    *buf = gmem_buf;
    *off = addr - HEAP_BASE;
    ov_note(addr, size, false);  // [XCDE] (XCDE_OVERLAP_STATS) read in place by the GPU
    {   // (debugging) XCDE_TRACE_DIRECT=1: vertex / index ranges read in place from host memory, once per program and range
        static int tr = -1;
        if (tr < 0) tr = getenv("XCDE_TRACE_DIRECT") != NULL;
        if (tr) audit("direct heap read: vs %llx fs %llx range %llx +%llu", (unsigned long long)audit_vs(), (unsigned long long)audit_fs(),
                      (unsigned long long)(addr & ~0xFFFull), (unsigned long long)size);
    }
    return true;
}

static int g_cs_n;  // (compute program table below)

// In-place reboot (device.c backend_reset; GPU idle): drop what is keyed by guest addresses - programs by code
// address (the old Prog / Pipe objects are leaked, so pipeline keys holding their pointers can't collide), compute
// programs, shared copies - and the heap import: its pages are replaced (re-imported on first use).
void vk_exec_reset(void) {
    g_pipe_state_gen++; g_pass_gen++;
    for (int i = 0; i < 6; i++) g_tex_bind_gen[i]++, g_sbuf_bind_gen[i]++;
    memset(descriptor_cache,0,sizeof descriptor_cache);
    memset(gpu_frames,0,sizeof gpu_frames); gpu_last_tag=0; gpu_last_ms=-1;
    for (int i=0;i<SUBS;i++) subs[i].ts=subs[i].upload_ts=false;
    if (prog_tab.cap) memset(prog_tab.e, 0, prog_tab.cap * sizeof *prog_tab.e);
    if (pipe_tab.cap) memset(pipe_tab.e, 0, pipe_tab.cap * sizeof *pipe_tab.e);
    prog_tab.n = pipe_tab.n = 0;
    prog_tab_gen++;
    g_cs_n = 0;
    gcopy_gen++;
    nwant = 0;
    for (int i = 0; i < nxsh; i++) vmaDestroyBuffer(g_vk.vma, xsh[i].buf, xsh[i].mem);  // (GPU idle; refilled on their next write)
    nxsh = 0;
    if (gmem_buf) {
        vkDestroyBuffer(g_vk.dev, gmem_buf, NULL);
        vkFreeMemory(g_vk.dev, gmem_mem, NULL);
        gmem_buf = VK_NULL_HANDLE;
        gmem_mem = VK_NULL_HANDLE;
    }
    if (gmem_stage.buf) vmaDestroyBuffer(g_vk.vma, gmem_stage.buf, gmem_stage.mem);
    memset(&gmem_stage, 0, sizeof gmem_stage);
    gmem_size = gmem_tried = 0;
    for (int i = 0; i < VC_TAB; i++)  // (the vertex cache: copies of the old pages)
        if (vc_tab[i].buf) vmaDestroyBuffer(g_vk.vma, vc_tab[i].buf, vc_tab[i].mem);
    memset(vc_tab, 0, sizeof vc_tab);
    vc_npend = 0;
    vc_free_dead(true);
    vc_bytes = 0;
    if (vc_page_gpu) memset(vc_page_gpu, 0, HEAP_MAX / VC_PAGE);
    vc_epoch++;
    st_defaults();
}

// [XCDE] audit: state the game sets that this renderer does not apply (or applies approximately), logged once per
// distinct message as "[audit] ...". The class behind most graphics bugs so far (ignored logic op, late stencil with
// image stores, dummy storage views, unscaled image coordinates): this makes them visible before a player finds them.
static void audit(const char *fmt, ...) {
    static uint64_t seen[1024];
    static int nseen;
    char msg[400];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    uint64_t h = 1469598103934665603ull;
    for (const char *q = msg; *q; q++) h = (h ^ (uint8_t)*q) * 1099511628211ull;
    for (int i = 0; i < nseen; i++) if (seen[i] == h) return;
    if (nseen < 1024) seen[nseen++] = h;
    LOG("[audit] %s\n", msg);
}
static uint64_t audit_vs(void) { return st.program ? st.program->stages[0].code_addr : 0; }
static uint64_t audit_fs(void) { return st.program ? st.program->stages[1].code_addr : 0; }

static void texture_info(int stage, ShaderInfo *si, int i, VkDescriptorImageInfo *out, float sc[2], uint32_t wh[2]) {
    ShaderTexture *st_ = &si->tex[i];
    int slot = ((int)st_->handle_offset - 0x20) / 8;  // 64-bit handles from offset 0x20
    uint64_t h = (slot >= 0 && slot < 32) ? st.tex[stage][slot] : 0;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler smp = VK_NULL_HANDLE;
    uint32_t tic = h & 0xFFFFF, tsc = (uint32_t)(h >> 20) & 0xFFF;
    sc[0] = sc[1] = 1;
    wh[0] = wh[1] = 0;
    ti_d16 = false;
    if (h && st.tpool && tic < (uint32_t)st.tpool->count) {
        NvnTexEntry *e = &((NvnTexEntry *)st.tpool->entries)[tic];
        if (e->tex) {
            // Per-frame memo: ~2400 lookups per overworld frame recomputed the view (format / range math, cache
            // probe). The first lookup of a frame still runs vk_view (upload check); a view that may have been
            // destroyed (eviction, image re-created) bumps vk_view_gen. LARECOMP_VK_NO_TEX_MEMO=1 disables.
            extern uint32_t vk_view_gen;
            extern uint64_t vk_frame_counter(void);
            static int nomemo = -1;
            if (nomemo < 0) nomemo = getenv("LARECOMP_VK_NO_TEX_MEMO") != NULL;
            static struct { NvnTexture *tex; uint64_t vh, frame; uint32_t gen; uint8_t type, has_view; VkImageView view; } memo[1024];
            uint64_t vh = e->has_view ? hash_words(0, &e->view, sizeof e->view) : 0, frame = vk_frame_counter();
            uint32_t ms = (uint32_t)((((uintptr_t)e->tex >> 4) ^ vh ^ (uint64_t)st_->type * 0x9E3779B97F4A7C15ull) & 1023);
            if (!nomemo && memo[ms].tex == e->tex && memo[ms].vh == vh && memo[ms].frame == frame && memo[ms].gen == vk_view_gen &&
                memo[ms].type == st_->type && memo[ms].has_view == e->has_view) {
                view = memo[ms].view;
            } else {
                extern bool vk_view_mismatch;
                vk_view_mismatch = false;
                view = vk_view(naliases ? alias_read(e->tex) : e->tex, &e->view, e->has_view, false, st_->type);
                if (vk_view_mismatch && getenv("XCDE_GLITCH_WATCH")) {  // [XCDE] T240: the shader samples a type this texture can't be (black dummy)
                    static int nl;
                    extern uint64_t renderer_present_count(void);
                    if (nl++ < 2000)
                        LOG("[glitch] frame %llu present %llu: stage %d unit %d pool slot %u: shader type %d, texture %llx target %u fmt %#x %ux%ux%u"
                            " (vs %llx fs %llx) - black; slot registered in frame %u, held target %u before%c", (unsigned long long)frame,
                            (unsigned long long)renderer_present_count(), stage, slot, tic, st_->type, (unsigned long long)e->tex->addr, e->tex->target,
                            e->tex->format, e->tex->width, e->tex->height, e->tex->depth, (unsigned long long)audit_vs(), (unsigned long long)audit_fs(),
                            e->reg_frame, e->prev_target, 10);
                }
                if (vk_view_mismatch) dummy_report("tipo incompatible (textura dummy)", stage, slot, tic, st_->type, audit_fs(), e->tex->target, e->tex->format, e->tex->width, e->tex->height);
                if (st_->shadow && e->tex->format == 0x32) {  // [XCDE] a depth compare on D16 (Maxwell rounds the reference first)
                    static int nd16;
                    if (nd16++ < 10) LOG("[vk] depth compare on a D16 texture %ux%u (fs %llx)\n", e->tex->width, e->tex->height,
                                         (unsigned long long)audit_fs());
                }
                memo[ms].tex = e->tex; memo[ms].vh = vh; memo[ms].frame = frame; memo[ms].gen = vk_view_gen;
                memo[ms].type = (uint8_t)st_->type; memo[ms].has_view = e->has_view; memo[ms].view = view;
            }
            renderer_texture_scale(e->tex, &sc[0], &sc[1]);
            wh[0] = e->tex->width; wh[1] = e->tex->height;
            ti_d16 = e->tex->format == 0x32;
            for (int k = 0; k < nfeedback; k++)
                if (e->tex == feedback_tex[k] && (st_->type == TEXTYPE_2D || st_->type == TEXTYPE_2D_ARRAY)) {
                    view = feedback_view[k];
                    if (st_->type == TEXTYPE_2D_ARRAY) view = vk_dummy_view(TEXTYPE_2D_ARRAY);  // (2D snapshot as array: rare)
                }
        }
    }
    if (h && st.spool && tsc < (uint32_t)st.spool->count) smp = vk_sampler(&((NvnSamplerDesc *)st.spool->entries)[tsc]);
    if (!view && h && st.tpool && tic < (uint32_t)st.tpool->count && ((NvnTexEntry *)st.tpool->entries)[tic].tex) {
        NvnTexture *t = ((NvnTexEntry *)st.tpool->entries)[tic].tex;
        audit("dummy texture view: stage %d unit %d shader type %d / texture target %u fmt %#x %ux%ux%u levels %u (vs %llx fs %llx)",
              stage, slot, st_->type, t->target, t->format, t->width, t->height, t->depth, t->levels, (unsigned long long)audit_vs(),
              (unsigned long long)audit_fs());
    }
    {   // [XCDE] T240 glitch watch (XCDE_GLITCH_WATCH=1): a texture the shader samples whose pool slot is empty - never
        // registered, or finalized (objects.c nvn_pools_forget_texture) while the game still draws with it. The Switch
        // would read the old descriptor and its memory; here it is the black dummy, for as long as the slot stays so.
        static int gw = -1;
        if (gw < 0) gw = getenv("XCDE_GLITCH_WATCH") ? atoi(getenv("XCDE_GLITCH_WATCH")) : 0;
        if (gw && !view && h && st.tpool && tic < (uint32_t)st.tpool->count && !((NvnTexEntry *)st.tpool->entries)[tic].tex) {
            static int nl;
            if (nl++ < 400)
                LOG("[glitch] frame %llu: stage %d unit %d samples empty pool slot %u (vs %llx fs %llx)\n", (unsigned long long)vk_frame_counter(),
                    stage, slot, tic, (unsigned long long)audit_vs(), (unsigned long long)audit_fs());
        }
    }
    if (!view && h && st.tpool && tic < (uint32_t)st.tpool->count && !((NvnTexEntry *)st.tpool->entries)[tic].tex)
        dummy_report("ranura vacia (textura dummy)", stage, slot, tic, st_->type, audit_fs(), 0, 0, 0, 0);
    if (!view) view = vk_dummy_view(st_->type);
    if (!smp) smp = vk_default_sampler(true);
    *out = (VkDescriptorImageInfo){smp, view, VK_IMAGE_LAYOUT_GENERAL};
}

// [XCDE] The unsigned-integer storage view of shader texture slot i (SULD/SUST), or the dummy one.
static VkImageView storage_view_for(int stage, ShaderInfo *si, int i, float sc[2]) {
    ShaderTexture *st_ = &si->tex[i];
    // [XCDE] images live in their own table (BindImage): handle word 72 + 2k of the driver constant buffer = image k
    int slot = ((int)st_->handle_offset - 0x120) / 8;
    uint64_t h = (slot >= 0 && slot < 8) ? st.img[stage][slot] : 0;
    uint32_t tic = h & 0xFFFFF;
    VkImageView v = VK_NULL_HANDLE;
    if (h && st.tpool && tic < (uint32_t)st.tpool->count) {
        NvnTexEntry *e = &((NvnTexEntry *)st.tpool->entries)[tic];
        if (e->tex) {
            NvnTexture *t = naliases ? alias_read(e->tex) : e->tex;
            v = vk_storage_view(t);
            if (v && sc) renderer_texture_scale(t, &sc[0], &sc[1]);  // [XCDE] T178: its allocation scale (rscale table)
            if (!v) audit("dummy storage view: stage %d image %d fmt %#x %ux%ux%u target %u (vs %llx fs %llx)", stage, slot, t->format,
                          t->width, t->height, t->depth, t->target, (unsigned long long)audit_vs(), (unsigned long long)audit_fs());
            VkFormat vf = vk_format(t->format);
            bool intf = vf == VK_FORMAT_R32_UINT || vf == VK_FORMAT_R32_SINT || vf == VK_FORMAT_R32G32_UINT || vf == VK_FORMAT_R32G32B32A32_UINT ||
                        vf == VK_FORMAT_R8_UINT || vf == VK_FORMAT_R16_UINT || vf == VK_FORMAT_R8G8B8A8_UINT || vf == VK_FORMAT_R16G16_UINT;
            if (st_->image_p && !intf)
                audit("SULD/SUST formatted (P) mode on non-integer image fmt %#x (vk %d): raw bits used (stage %d image %d, vs %llx fs %llx)",
                      t->format, (int)vf, stage, slot, (unsigned long long)audit_vs(), (unsigned long long)audit_fs());
        }
    }
    return v ? v : vk_dummy_storage_view();
}

// Writes the descriptor set of one stage. gx..gz: compute group counts (driver constants).
// pre_ii: texture descriptors already resolved by the caller (draw: fragment textures, needed earlier for the
// viewport decision), or NULL.
static float g_mirror_w;  // mirror mode's composite: its render target width (rscale[65].x; 0 otherwise)

static VkDescriptorSet stage_set(Prog *p, VkDescriptorSetLayout layout, uint32_t set, int stage, ShaderInfo *si, const float *rscale_row0,
                                 const float tint[8], uint32_t gx, uint32_t gy, uint32_t gz, const VkDescriptorImageInfo *pre_ii) {
    VkDescriptorSet ds = VK_NULL_HANDLE; // T260: allocate only after an exact-value cache miss
    VkWriteDescriptorSet w[MAX_BIND];
    VkDescriptorBufferInfo bi[MAX_BIND];
    VkDescriptorImageInfo ii[MAX_BIND];
    int n = 0;
    int sg = stage == 1 ? 1 : 0;
    float rs[66][4];  // ([65]: mirror mode, renderer.m) (filled only for a binding 18: [XCDE] perf, 1 KB a set)
    for (int i = 0; i < p->nbinds; i++) {
        VkBindingInfo *b = &p->binds[i];
        if (b->set != set) continue;
        w[n] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = b->binding,
                                      .descriptorCount = 1, .descriptorType = b->type};
        if (b->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER && b->binding >= 32) {
            int ti = (int)b->binding - 32;
            float sc[2]; uint32_t wh[2];
            if (ti < si->ntex && pre_ii && ti < 64) { ii[n] = pre_ii[ti]; sc[0] = tex_sc[sg][ti][0]; sc[1] = tex_sc[sg][ti][1]; wh[0] = tex_wh[sg][ti][0]; wh[1] = tex_wh[sg][ti][1]; }
            else if (ti < si->ntex) { texture_info(stage, si, ti, &ii[n], sc, wh); if (ti < 64) tex_d16[sg][ti] = ti_d16; }
            else { ii[n] = (VkDescriptorImageInfo){vk_default_sampler(true), vk_dummy_view(TEXTYPE_2D), VK_IMAGE_LAYOUT_GENERAL}; sc[0] = sc[1] = 1; wh[0] = wh[1] = 0; if (ti < 64) tex_d16[sg][ti] = 0; }
            if (ti < 64) {
                tex_sc[sg][ti][0] = sc[0]; tex_sc[sg][ti][1] = sc[1];
                tex_wh[sg][ti][0] = wh[0]; tex_wh[sg][ti][1] = wh[1];
            }
            w[n].pImageInfo = &ii[n];
        } else if (b->type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE && b->binding >= 100) {  // [XCDE] SULD/SUST
            int ti = (int)b->binding - 100;
            float sc[2] = {1, 1};
            ii[n] = (VkDescriptorImageInfo){VK_NULL_HANDLE, ti < si->ntex ? storage_view_for(stage, si, ti, sc) : vk_dummy_storage_view(),
                                            VK_IMAGE_LAYOUT_GENERAL};
            if (ti < 64) { tex_sc[sg][ti][0] = sc[0]; tex_sc[sg][ti][1] = sc[1]; tex_d16[sg][ti] = 0; }  // [XCDE] T178
            w[n].pImageInfo = &ii[n];
        } else if (b->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER && b->binding < 17) {
            bi[n] = cbuf_info(stage, si, (int)b->binding, gx, gy, gz);
            w[n].pBufferInfo = &bi[n];
        } else if (b->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
            bi[n] = b->binding == GFAST_BUFFER_INDEX ? gfast_info() : gmem_shader_info();
            w[n].pBufferInfo = &bi[n];
        } else {
            continue;  // rscale / tint: written below, once the textures (and their scales) are known
        }
        n++;
    }
    for (int i = 0; i < p->nbinds; i++) {
        VkBindingInfo *b = &p->binds[i];
        if (b->set != set || b->type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
            (b->binding != 18 && b->binding != 19 && b->binding != GMAP_BUFFER_INDEX && b->binding != XFB_PARAM_INDEX)) continue;
        w[n] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = b->binding,
                                      .descriptorCount = 1, .descriptorType = b->type};
        if (b->binding == XFB_PARAM_INDEX) {  // [XCDE] T134: transform feedback placement (draw_)
            bi[n] = ubo_span(vk_transient_copy(g_xfbp, sizeof g_xfbp, 256), sizeof g_xfbp);
        } else if (b->binding == GMAP_BUFFER_INDEX) {  // (compute) this dispatch's fast-copy table: gfast_prepare
            bi[n] = ubo_span(vk_transient_copy(gmap_tab, sizeof gmap_tab, 256), sizeof gmap_tab);
        } else if (b->binding == 18) {
            memset(rs, 0, sizeof rs);
            rs[0][0] = rs[0][1] = 1;
            if (rscale_row0) memcpy(rs[0], rscale_row0, 16);
            rs[65][0] = g_mirror_w;
            for (int t = 0; t < si->ntex && t < 64; t++) { rs[1 + t][0] = tex_sc[sg][t][0]; rs[1 + t][1] = tex_sc[sg][t][1]; rs[1 + t][2] = tex_d16[sg][t]; }
            bi[n] = ubo_span(copy_dedupe(sg ? DEDUPE_RSCALE_F : DEDUPE_RSCALE_V, rs, sizeof rs, 256), sizeof rs);
        } else {
            float zero[8] = {0};
            bi[n] = ubo_span(copy_dedupe(DEDUPE_TINT, tint ? tint : zero, 32, 256), 32);
        }
        w[n].pBufferInfo = &bi[n];
        n++;
    }
    XcdeDescriptorKey keys[MAX_BIND];
    for (int i=0;i<n;i++) {
        keys[i]=(XcdeDescriptorKey){{((uint64_t)w[i].dstBinding<<32)|(uint32_t)w[i].descriptorType,0,0,0}};
        if (w[i].descriptorType==VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER || w[i].descriptorType==VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
            const VkDescriptorImageInfo *v=w[i].pImageInfo;
            keys[i].word[1]=(uint64_t)(uintptr_t)v->sampler;
            keys[i].word[2]=(uint64_t)(uintptr_t)v->imageView; keys[i].word[3]=v->imageLayout;
        } else {
            const VkDescriptorBufferInfo *v=w[i].pBufferInfo;
            keys[i].word[1]=(uint64_t)(uintptr_t)v->buffer; keys[i].word[2]=v->offset; keys[i].word[3]=v->range;
        }
    }
    uint64_t seq=vk_pending_seq(), lid=(uint64_t)(uintptr_t)layout;
    XcdeDescriptorEntry *entry=&descriptor_cache[xcde_descriptor_slot(lid,keys,(unsigned)n)];
    static int enabled = -1;  // (was a getenv per call: a scan of Steam's large environment per stage per draw)
    if (enabled < 0) { const char *e = getenv("XCDE_DESCRIPTOR_CACHE"); enabled = !e || atoi(e) != 0; }
    if (enabled && xcde_descriptor_match(entry,seq,lid,keys,(unsigned)n)) {
        metric_desc_hits++; return (VkDescriptorSet)(uintptr_t)entry->handle;
    }
    uint64_t pt0 = prof ? SDL_GetTicksNS() : 0;
    ds=vk_alloc_set(layout);
    uint64_t pt1 = prof ? SDL_GetTicksNS() : 0;
    for (int i=0;i<n;i++) w[i].dstSet=ds;
    if (n) { vkUpdateDescriptorSets(g_vk.dev, (uint32_t)n, w, 0, NULL); metric_desc_updates++; stat_ds_writes += (uint64_t)n; }
    if (prof) { uint64_t pt2 = SDL_GetTicksNS(); prof_ds_alloc += pt1 - pt0; prof_ds_upd += pt2 - pt1; }
    if (enabled) {
        entry->sequence=seq; entry->layout=lid; entry->handle=(uint64_t)(uintptr_t)ds; entry->count=(unsigned)n;
        memcpy(entry->key,keys,(size_t)n*sizeof *keys);
    }
    return ds;
}

// ================================================================== compute
typedef struct { Prog p; ShaderInfo *si; VkPipeline pipe; uint32_t local[3]; } CsProg;

static void build_compute(void *arg) {
    CsProg *cp = arg;
    size_t n = 0;
    uint32_t *spv = vk_glsl_to_spirv(cp->si->msl, STAGE_COMPUTE, "cs", &n);
    if (!spv) { atomic_store(&cp->p.state, 2); return; }
    cp->p.nbinds = vk_reflect(spv, n, cp->p.binds, MAX_BIND);
    VkShaderModule m = vk_module(spv, n);
    free(spv);
    cp->p.layout = make_layout(&cp->p, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0);
    VkSpecializationMapEntry me[3] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}};
    VkSpecializationInfo spec = {3, me, 12, cp->local};
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main",
                                                .pSpecializationInfo = &spec},
                                      .layout = cp->p.layout};
    VkResult r = vkCreateComputePipelines(g_vk.dev, g_vk.pcache, 1, &ci, NULL, &cp->pipe);
    atomic_fetch_add(&stat_compiles, 1);
    atomic_store(&cp->p.state, r == VK_SUCCESS ? 1 : 2);
    if (r != VK_SUCCESS) LOG("[compute] pipeline failed: %d\n", r);
}

static struct CsEnt { uint64_t addr; CsProg *cp; } g_cs_tab[256];  // compute programs by code address

static CsProg *compute_program(NvnProgram *np) {
    NvnShaderStage *ss = &np->stages[5];
    if (!ss->code_addr) return NULL;
    struct CsEnt *tab = g_cs_tab;
    int n = g_cs_n;
    for (int i = 0; i < n; i++) if (tab[i].addr == ss->code_addr) return tab[i].cp;
    CsProg *cp = calloc(1, sizeof *cp);
    if (n < 256) { tab[n].addr = ss->code_addr; tab[n].cp = cp; g_cs_n = n + 1; }
    maxwell_gmem_base = HEAP_BASE;
    maxwell_gmem_size = HEAP_MAX;
    ShaderKey k = {0};
    k.glsl = 1;
    xlat_lock();
    cp->si = maxwell_translate((const uint8_t *)(uintptr_t)nvn_stage_code(ss), ss->const_offset ? ss->const_offset : 0x10000, STAGE_COMPUTE, &k);
    pthread_mutex_unlock(&translate_lock);
    cp->p.vi = cp->si;
    if (!cp->si->msl) {
        LOG("[compute] translate failed: %s (code %llx)\n", cp->si->error, (unsigned long long)ss->code_addr);
        atomic_store(&cp->p.state, 2);
        return cp;
    }
    for (int i = 0; i < 3; i++) {
        cp->local[i] = ss->control ? *(const uint32_t *)(ss->control + 0x748 + 4 * i) : 1;
        if (!cp->local[i]) cp->local[i] = 1;
    }
    save_glsl(cp->si->msl, ss->code_addr, "cs");
    run_job(build_compute, cp);
    return cp;
}

static void dispatch_compute(uint32_t gx, uint32_t gy, uint32_t gz) {
    snap_reason = "compute"; snapshots_invalidate(NULL);
    if (getenv("XCDE_TRACE_CS")) {  // [XCDE] every compute dispatch the game issues, once per shader: is it skipped, and why
        static uint64_t seen[64]; static int ns;
        uint64_t a = st.program ? st.program->stages[5].code_addr : 0;
        bool dup = false;
        for (int i = 0; i < ns; i++) dup |= seen[i] == a;
        if (!dup && ns < 64) { seen[ns++] = a; LOG("[cs] game dispatch cs %llx %ux%ux%u program %p\n", (unsigned long long)a, gx, gy, gz, (void *)st.program); }
    }
    if (!st.program || !gx || !gy || !gz) return;
    CsProg *cp = compute_program(st.program);
    if (getenv("XCDE_TRACE_CS") && (!cp || atomic_load(&cp->p.state) != 1)) {
        static int n; if (n++ < 20) LOG("[cs] dispatch skipped: cp %p state %d\n", (void *)cp, cp ? (int)atomic_load(&cp->p.state) : -9);
    }
    if (!cp || atomic_load(&cp->p.state) != 1) { SKIPPED(); return; }
    {   // LARECOMP_SKIP_CS=<code address hex> (debug): skip that compute shader's dispatches, to see what it does
        static uint64_t skip = ~0ull;
        if (skip == ~0ull) { const char *e = getenv("LARECOMP_SKIP_CS"); skip = e ? strtoull(e, NULL, 16) : 0; }
        if (skip && st.program->stages[5].code_addr == skip) { SKIPPED(); return; }
    }
    if (cp->si->uses_gmem && getenv("LARECOMP_LOG_CS_SBUF")) {  // (debug) the storage buffers bound to gmem dispatches
        static uint64_t seen_cs[16];
        static int seen_n[16];
        int k = 0;
        while (k < 15 && seen_cs[k] && seen_cs[k] != st.program->stages[5].code_addr) k++;
        seen_cs[k] = st.program->stages[5].code_addr;
        if (seen_n[k]++ < 24) {
            LOG("[cs] %llx %ux%ux%u:", (unsigned long long)st.program->stages[5].code_addr, gx, gy, gz);
            for (int i = 0; i < 16; i++)
                if (st.sbuf[5][i].addr) LOG(" sb%d %llx+%llx", i, (unsigned long long)st.sbuf[5][i].addr, (unsigned long long)st.sbuf[5][i].size);
            LOG("\n");
        }
    }
    vk_end_pass();
    for (int i = 0; i < cp->si->ntex; i++) {  // [XCDE] T192 (as in draw_)
        if (!cp->si->tex[i].is_image) continue;
        NvnTexture *t = shader_texture_object(5, &cp->si->tex[i]);
        if (t && naliases) t = alias_read(t);
        if (t) vk_tex_add_storage(t, vk_cmd());
    }
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    gp_mark(cb, "dispatch %ux%ux%u (local %ux%ux%u) cs %llx gmem %d", gx, gy, gz, cp->local[0], cp->local[1], cp->local[2],
            (unsigned long long)st.program->stages[5].code_addr, cp->si->uses_gmem);
    dl_note("dispatch %ux%ux%u cs %llx gmem %d", gx, gy, gz, (unsigned long long)st.program->stages[5].code_addr, cp->si->uses_gmem);
    nfeedback = 0;
    if (ct_on()) {
        LOG("[ct] dispatch cs %llx %ux%ux%u | tex", (unsigned long long)st.program->stages[5].code_addr, gx, gy, gz);
        for (int i = 0; i < cp->si->ntex; i++) {
            NvnTexture *t = shader_texture_object(5, &cp->si->tex[i]);
            if (t) LOG(" %s%p/%#x", cp->si->tex[i].is_image ? "I:" : "", (void *)t, t->format);
        }
        LOG("\n");
    }
    if (cp->si->uses_gmem_map) gfast_prepare(cb);
    VkDescriptorSet ds = stage_set(&cp->p, cp->p.set_layout[0], 0, 5, cp->si, NULL, NULL, gx, gy, gz, NULL);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, cp->pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, cp->p.layout, 0, 1, &ds, 0, NULL);
    vkCmdDispatch(cb, gx, gy, gz);
    if (cp->si->uses_gmem_map && cp->si->uses_gmem_store) gfast_finish(cb);
    if (cp->si->uses_gmem_store) { vc_mark_sbufs(5); if (nwant) xsh_stored(5); atomic_store(&host_read_pending, true); }
    else if (cp->si->uses_gmem) ov_note_sbufs(5, false);  // [XCDE] (XCDE_OVERLAP_STATS)
    gmem_stage_finish(cp->si->uses_gmem_store);
    stat_dispatches++;
}

// ================================================================== draws
static void debug_draw(Prog *pr, CmdRec *r, bool indexed);

// Quads / fans / polygons as a 32-bit triangle index list.
static uint32_t *expand_indices(uint32_t mode, uint32_t count, const void *indices, int itype, uint32_t first, uint32_t *out_count) {
    uint32_t tris = mode == 7 ? count / 4 * 2 : count >= 3 ? count - 2 : 0;
    // T260: render-thread scratch; the caller copies it before the next expansion. Never read back WC GPU memory.
    static uint32_t *scratch;
    static size_t capacity;
    size_t needed = ((size_t)tris * 3 + 1) * sizeof(uint32_t);
    if (needed > capacity) {
        uint32_t *next = realloc(scratch, needed);
        if (!next) FATAL("[vk] cannot expand %u primitive indices\n", count);
        scratch=next; capacity=needed;
    }
    uint32_t *out=scratch;
    uint32_t n = 0;
#define IDX(i) (indices ? (itype == 0 ? ((const uint8_t *)indices)[i] : itype == 1 ? ((const uint16_t *)indices)[i] : ((const uint32_t *)indices)[i]) : first + (i))
    if (mode == 7) {
        for (uint32_t q = 0; q + 3 < count; q += 4) {
            out[n++] = IDX(q); out[n++] = IDX(q + 1); out[n++] = IDX(q + 2);
            out[n++] = IDX(q); out[n++] = IDX(q + 2); out[n++] = IDX(q + 3);
        }
    } else {
        for (uint32_t i = 1; i + 1 < count; i++) { out[n++] = IDX(0); out[n++] = IDX(i); out[n++] = IDX(i + 1); }
    }
#undef IDX
    *out_count = n;
    out[n]=0; // copied padding is initialized too
    return out;
}

static bool topology(uint32_t mode, VkPrimitiveTopology *t, uint8_t *cls) {
    switch (mode) {
    case 0: *t = VK_PRIMITIVE_TOPOLOGY_POINT_LIST; *cls = 0; return true;
    case 1: *t = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; *cls = 1; return true;
    case 3: *t = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; *cls = 1; return true;
    case 4: *t = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; *cls = 2; return true;
    case 5: *t = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; *cls = 2; return true;
    default: *t = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; *cls = 2; return false;
    }
}

// [XCDE] T134: transform feedback layout of the bound vertex shader, from its NVN control section (layout read off the
// game's capture shaders: [0x6e8] captured varyings, [0x6dc] stride of buffer 0, [0xdc + i] slot (0x20 + k = output v_k),
// [0x4dc + i] its component count; the varyings are packed in order). false: nothing to capture or a layout we don't know.
static bool xfb_key(ShaderKey *k) {
    const uint8_t *ctl = st.program->stages[0].control;
    if (!ctl) return false;
    uint32_t n = ctl[0x6e8], stride = *(const uint32_t *)(ctl + 0x6dc), sum = 0;
    for (uint32_t i = 0; i < n && i < 16; i++) sum += ctl[0x4dc + i];
    if (!n || n > 16 || !stride || stride % 4 || sum * 4 > stride) {
        static int warned;
        if (!warned++) LOG("[xfb] unsupported capture layout: %u varyings, stride %u (vs %llx)\n", n, stride,
                           (unsigned long long)st.program->stages[0].code_addr);
        return false;
    }
    k->xfb_n = (uint8_t)n;
    k->xfb_stride = (uint16_t)stride;
    for (uint32_t i = 0; i < n; i++) { k->xfb_slot[i] = ctl[0xdc + i]; k->xfb_comps[i] = ctl[0x4dc + i]; }
    return true;
}

static void draw_(CmdRec *r, bool indexed);
static void writeback_texture(NvnTexture *t);
// [XCDE] T181: is this the save-thumbnail draw (see writeback_texture)? One 320x180 RGBA8 colour target, no depth,
// drawn by the thumbnail fragment program (content hash; other programs draw the title screen's 320x180 slots)
static NvnTexture *thumbnail_target(void) {
    if (st.rt.ncolor != 1 || st.rt.depth || !st.rt.color[0] || !st.program) return NULL;
    NvnTexture *t = st.rt.color[0];
    if (t->width != 320 || t->height != 180 || t->format != 0x25) return NULL;
    uint64_t fs = st.program->stages[1].code_addr;
    uint64_t h = fs_code_hash(fs);
    static uint64_t want = 1;
    if (want == 1) want = getenv("XCDE_THUMB_FS") ? strtoull(getenv("XCDE_THUMB_FS"), NULL, 16) : 0xb515bb7fd9974090ull;
    static int logged;
    if (getenv("XCDE_THUMB_LOG") && renderer_present_count() >= strtoull(getenv("XCDE_THUMB_LOG"), NULL, 10) && logged++ < 2000) LOG("[t181] 320x180 draw frame %llu fs hash %016llx\n", (unsigned long long)renderer_present_count(), (unsigned long long)h);
    return h == want ? t : NULL;
}
// ================================================================== temporal upscaling (FSR 3.1 / DLSS: temporal.c)
// The game's scene camera, found each frame in the vertex constant buffers by its shape (measured in the meadow: a
// 512-byte buffer, vec4 rows, clip = dot(row, pos): rows 0-2 view (3x4), 3-6 projection (row 6 = 0, 0, -1, 0:
// perspective), 7-10 view-projection = projection x view). The depth attachment of the draws that use it is the scene
// depth. Draws into that depth with the depth test on get the frame's sub-pixel jitter (viewport offset, render pixels).
static TmpCamera tmp_cam;            // this frame's (ok = found)
static uint64_t tmp_frame = ~0ull;
static NvnTexture *tmp_depth_tex;    // the scene depth (kept across frames: the prepass draws before the camera is seen)
static float tmp_jit[2];
static bool tmp_cam_check(const float *f, TmpCamera *c) {
    if (fabsf(f[24]) > 1e-4f || fabsf(f[25]) > 1e-4f || fabsf(f[26] + 1.0f) > 1e-4f || fabsf(f[27]) > 1e-4f) return false;
    if (fabsf(f[12]) < 0.1f || fabsf(f[17]) < 0.1f) return false;  // (focal lengths)
    for (int i = 0; i < 4; i++)      // rows 7-10 = P (rows 3-6) x V (rows 0-2, then 0 0 0 1)
        for (int j = 0; j < 4; j++) {
            const float *p = f + 12 + i * 4;
            float s = p[0] * f[j] + p[1] * f[4 + j] + p[2] * f[8 + j] + (j == 3 ? p[3] : 0);
            if (fabsf(s - f[28 + i * 4 + j]) > 1e-3f * (1.0f + fabsf(s))) return false;
        }
    memcpy(c->view, f, 12 * 4);
    memcpy(c->proj, f + 12, 16 * 4);
    memcpy(c->viewproj, f + 28, 16 * 4);
    c->ok = true;
    return true;
}
static bool tmp_dm1 = true;          // the scene draws' NVN depth mode (clip z in [-1, 1])
static void tmp_draw(const Prog *pr, float *jx, float *jy) {
    uint64_t fr = renderer_present_count();
    if (fr != tmp_frame) {
        tmp_frame = fr;
        tmp_cam.ok = false;
        tmp_jitter(tmp_jit);
    }
    if (!st.rt.depth) return;
    if (!tmp_cam.ok)
        for (int k = 0; k < 14; k++) {
            const Range *u = &st.ubuf[0][k];
            if (u->addr < HEAP_BASE || u->size < 176) continue;
            if (tmp_cam_check((const float *)(uintptr_t)u->addr, &tmp_cam)) { tmp_depth_tex = st.rt.depth; tmp_dm1 = pr->vkey.depth_minus_one; break; }
        }
    // (the same image: the game draws its prepass and colour passes through different texture objects over one depth)
    TexBackend *db = st.rt.depth->backend, *sb = tmp_depth_tex ? tmp_depth_tex->backend : NULL;
    static int nojit = -1;
    if (nojit < 0) nojit = getenv("XCDE_TMP_NOJITTER") != NULL;  // (tests)
    static int bysize = -1;
    if (bysize < 0) bysize = !getenv("XCDE_TMP_JIT_SAME_IMAGE");
    bool scene = sb && db && (db->image == sb->image || (bysize && st.rt.depth->width == tmp_depth_tex->width &&
                                                          st.rt.depth->height == tmp_depth_tex->height));
    // [XCDE] T213: nothing after the frame's scene composite (the HUD draws with a depth buffer of the scene's size:
    // it was jittered too - the HUD shook once the HUD mask showed it unblended)
    // [XCDE] T260: world geometry drawn WITHOUT the depth test (sky, water, particles, some effects) was never jittered,
    // but the upscaler removes the jitter from the whole image: those pixels shook / flickered. A draw that reads the scene
    // camera's constant buffer is world geometry whatever its depth state (a fullscreen post pass does not read it).
    // XCDE_TMP_JIT_NODEPTH=0 restores the old rule (depth-tested draws only).
    bool world = st.ds[0] != 0;
    if (scene && !world) {
        static int nodepth = -1;
        if (nodepth < 0) { const char *e = getenv("XCDE_TMP_JIT_NODEPTH"); nodepth = !(e && *e == '0'); }
        if (nodepth) {
            TmpCamera scratch;
            for (int k = 0; k < 14 && !world; k++) {
                const Range *u = &st.ubuf[0][k];
                if (u->addr < HEAP_BASE || u->size < 176) continue;
                world = tmp_cam_check((const float *)(uintptr_t)u->addr, &scratch);
            }
        }
    }
    if (scene && world && !nojit && !tmp_hud_started()) { *jx = tmp_jit[0]; *jy = tmp_jit[1]; tmp_scene_frame = fr; }
    if (getenv("XCDE_TMP_LOG")) {  // (debugging) per frame: draws jittered / scene draws not jittered (after the HUD began) / others
        static uint64_t fr; static int nj, nh, no, nl;
        if (fr != renderer_present_count()) {
            if (fr && nl < 12 && (fr % 60) == 0) { nl++; LOG("[tmp] frame %llu: %d draws jittered, %d scene draws after the HUD began (not jittered), %d others\n", (unsigned long long)fr, nj, nh, no); }
            fr = renderer_present_count(); nj = nh = no = 0;
        }
        if (scene && world) { if (tmp_hud_started()) nh++; else nj++; } else no++;
    }
}

// [XCDE] T181: save / load list thumbnails. Each slot's image is drawn into its 320x180 target by fragment program
// 93ee0aecfbae2ac9, which reads the picture (320x180 RGB) from storage buffer 0 in guest memory; the vertex stage
// offsets the quad by two floats from its storage buffer 2 - (100000, 100000) while the game holds the picture
// back. Both alternate every frame between two buffers per slot; the game loads a .tmb into only one of the two
// (the other keeps a marker: 16 bytes of 0xf3, the rest 0) and never releases the offset, so the list showed an empty
// box on Switch-correct saves (measured: real time, every frame of the load list). Per slot (the n-th such draw of a
// frame) the two picture buffers are paired: the loaded picture is copied into its twin, and a slot whose picture is
// loaded gets the offset (0, 0). XCDE_SLOT_THUMB=0: off.
static bool slot_pic_empty(const uint8_t *p) {  // (the marker - 16 bytes of 0xf3, then zeros - or nothing loaded yet)
    return (p[0] == 0xf3 || p[0] == 0) && p[16] == 0 && p[17] == 0 && p[86400] == 0 && p[86401] == 0 && p[172799] == 0;
}
// [XCDE] T222: pictures the game itself loaded (games/xcde/patches.py hook on its memcpy, game thread): buffer and
// content hash. A buffer holds the game's picture while its contents still hash to what was loaded.
static struct { _Atomic uint64_t addr; uint64_t hash; } thumb_loaded[32];
static _Atomic unsigned thumb_nloaded;
void xcde_thumb_loaded(uint64_t d, uint64_t n) {
    if (n != 320 * 180 * 3 || d < HEAP_BASE) return;
    unsigned i = atomic_fetch_add(&thumb_nloaded, 1) % 32;
    atomic_store(&thumb_loaded[i].addr, 0);
    thumb_loaded[i].hash = hmix(1469598103934665603ull, (const void *)(uintptr_t)d, n);
    atomic_store(&thumb_loaded[i].addr, d);
}
static bool thumb_is_loaded(uint64_t a, uint64_t h) {  // (h: the buffer's current content hash)
    for (int i = 0; i < 32; i++) if (atomic_load(&thumb_loaded[i].addr) == a && thumb_loaded[i].hash == h) return true;
    return false;
}
static bool thumb_any_loaded(void) { return atomic_load(&thumb_nloaded) != 0; }
static void slot_thumb_fix(void) {
    static int on = -1;
    if (on < 0) on = !(getenv("XCDE_SLOT_THUMB") && *getenv("XCDE_SLOT_THUMB") == '0');
    if (!on || st.rt.ncolor != 1 || st.rt.depth || !st.rt.color[0] || !st.program) return;
    NvnTexture *t = st.rt.color[0];
    if (t->width != 320 || t->height != 180 || t->format != 0x25) return;
    uint64_t fs = st.program->stages[1].code_addr;
    if (!fs || fs_code_hash(fs) != 0x93ee0aecfbae2ac9ull) return;
    uint64_t a = st.sbuf[1][0].addr, off = st.sbuf[0][2].addr;
    const uint64_t sz = 320 * 180 * 3;
    if (a < HEAP_BASE || st.sbuf[1][0].size < sz || off < HEAP_BASE || st.sbuf[0][2].size < 8) return;
    static uint64_t frame = ~0ull;
    static int ord;
    if (renderer_present_count() != frame) { frame = renderer_present_count(); ord = 0; }
    // per slot: its two picture buffers, the frame each was last drawn, its content hash then, and the hash of what
    // this function copied into it (0: the game's own content)
    static uint64_t pair[16][2], seen[16][2], seen_f[16][2], wrote[16][2];
    if (ord >= 16) return;
    int o = ord++;
    int me = pair[o][0] == a ? 0 : pair[o][1] == a ? 1 : -1;
    if (me < 0) {  // (a buffer not seen for this slot: its second one, or a new pair - the list changed)
        if (!pair[o][0] || frame - seen_f[o][0] > 2) me = 0;
        else if (!pair[o][1] || frame - seen_f[o][1] > 2) me = 1;
        else { me = 0; pair[o][1] = 0; }
        pair[o][me] = a; seen[o][me] = 0; wrote[o][me] = 0;
    }
    uint8_t *pa = (uint8_t *)(uintptr_t)a;
    uint64_t b = pair[o][me ^ 1];
    uint64_t ha = hmix(1469598103934665603ull, pa, sz);
    // [XCDE] T222: the twin is used only while it is this slot's live twin (drawn for it within the last 2 frames)
    // and its picture is complete (unchanged since its own last draw). Before, a pair kept from an earlier list
    // (another menu visit) pointed into memory the game had reused: its contents were copied in - pink / green
    // garbage or black thumbnails until saving rebuilt the list (user shots).
    bool known = thumb_any_loaded();  // (the load hook works: which buffer holds the picture is known, not guessed)
    bool good = known ? thumb_is_loaded(a, ha) || (wrote[o][me] && ha == wrote[o][me]) : !slot_pic_empty(pa);
    if (b && frame - seen_f[o][me ^ 1] <= 2) {
        const uint8_t *pb = (const uint8_t *)(uintptr_t)b;
        uint64_t hb = hmix(1469598103934665603ull, pb, sz);
        bool stable = hb == seen[o][me ^ 1];
        bool ours = wrote[o][me] && ha == wrote[o][me];  // (what is in this buffer is a copy made here)
        bool b_good = known ? thumb_is_loaded(b, hb) : stable && !slot_pic_empty(pb);
        bool a_mine = known ? !thumb_is_loaded(a, ha) : slot_pic_empty(pa) || ours;
        if (b_good && ha != hb && a_mine) { memcpy(pa, pb, sz); ha = hb; wrote[o][me] = hb; good = true; }
    }
    seen[o][me] = ha; seen_f[o][me] = frame;
    float *xy = (float *)(uintptr_t)off;
    if (good && xy[0] == 100000.0f && xy[1] == 100000.0f) { xy[0] = 0; xy[1] = 0; }
}

static void draw(CmdRec *r, bool indexed) {
    PROF_T0();
    slot_thumb_fix();
    NvnTexture *thumb = thumbnail_target();
    draw_(r, indexed);
    if (thumb) writeback_texture(thumb);
    PROF_ADD(prof_draw);
}
static uint64_t prof_tp_;
static void draw_(CmdRec *r, bool indexed) {
    if (prof) prof_tp_ = prof_pp_ = SDL_GetTicksNS();
    prewarm_drain();  // [XCDE] T163
    if (!st.program) { SKIPPED(); return; }
    {   // [XCDE] debugging: XCDE_SKIP_PROG=<fragment code addr hex>: drop the draws of that program
        static uint64_t skip = ~0ull;
        if (skip == ~0ull) skip = getenv("XCDE_SKIP_PROG") ? strtoull(getenv("XCDE_SKIP_PROG"), NULL, 16) : 0;
        if (skip && st.program->stages[1].code_addr == skip) return;
        // XCDE_ONLY_VS=<vs code addr hex>[,...]: in the debug frame, G-buffer draws (4+ colour targets) of other
        // programs are dropped - the pass image shows that geometry alone
        static uint64_t only[16];
        static int nonly = -1;
        if (nonly < 0) {
            nonly = 0;
            for (const char *p = getenv("XCDE_ONLY_VS"); p && *p && nonly < 16;) { only[nonly++] = strtoull(p, (char **)&p, 16); if (*p == ',') p++; else break; }
        }
        if (nonly && g_debug_frame && st.rt.ncolor >= 4) {
            bool keep = false;
            for (int i = 0; i < nonly; i++) keep |= st.program->stages[0].code_addr == only[i];
            if (!keep) return;
        }
    }
    { static int no_ss = -1; if (no_ss < 0) no_ss = getenv("XCDE_NO_SS") != NULL; if (no_ss) ss_forget(); }  // [XCDE] debugging: no dynamic-state caching
    ShaderKey vkey = {0}, fkey = {0};
    vkey.glsl = fkey.glsl = 1;
    vkey.flip_y = fkey.flip_y = nvn_window_origin() == 0;
    vkey.depth_minus_one = nvn_depth_mode() == 0;
    for (int i = 0; i < st.nattrib; i++) vkey.attr_kind[i] = (uint8_t)attr_kind(st.attrib[i] & 0xFF);
    for (int i = 0; i < st.rt.ncolor; i++) if (st.rt.color[i]) fkey.rt_kind[i] = (uint8_t)rt_kind(st.rt.color[i]->format);
    vkey.tex_uint = stencil_unit_mask(0);  // [XCDE]
    fkey.tex_uint = stencil_unit_mask(1);
    if (st.color_state[2] && st.color_state[2] < 8 && alpha_test_on()) {  // [XCDE] T191: alpha test (a shader variant)
        fkey.alpha_func = st.color_state[2];
        fkey.alpha_ref = f32_to_f16(st.alpha_ref);
    }
    // [XCDE] T134: transform feedback (the game's grass and cloth "writer" draws): a vertex shader variant that stores
    // its captured outputs to the buffer. Capture order is exact for non-indexed lists (the game's point draws).
    bool xfb = false;
    if (st.xfb_on && st.xfb_buf[0].addr >= HEAP_BASE && !(st.program->stage_mask & 4)) {
        uint32_t m = (uint32_t)r->x[0];
        if (indexed || (m != 0 && m != 1 && m != 4)) {
            static int warned;
            if (!warned++) LOG("[xfb] capture of an %s draw, mode %u: not emulated\n", indexed ? "indexed" : "array", m);
        } else {
            xfb = xfb_key(&vkey);
        }
    }
    { uint32_t m = (uint32_t)r->x[0]; g_draw_gs_in = m == 0 ? 1 : (m <= 3 || m == 0xa || m == 0xb) ? 2 : 3; }  // [XCDE] NVN draw mode -> GS input
    Prog *pr = get_program(st.program, &vkey, &fkey);
    bool pr_ok = pr && wait_ready(&pr->state, pr);
    PRE_MARK(0);
    if (pr && g_last_wait_ns > build_wait_min()) build_wait_log(pr, NULL, g_last_wait_ns);
    if (!pr_ok) {
        static int nlog_;
        if ((nlog_++ & 255) == 0) LOG("[vk] program not ready: %s state %d fs %llx vs %llx""%c", pr ? "present" : "NULL", pr ? (int)atomic_load(&pr->state) : -1,
                              (unsigned long long)st.program->stages[1].code_addr, (unsigned long long)st.program->stages[0].code_addr, 10);
        SKIPPED(); return;
    }
    // [XCDE] T186: the Link's Awakening recomp's per-draw hooks (mirror mode, pot / bottle / heart tints, the pets
    // item cursor, widescreen backdrop detection) ran here on every draw; XCDE has none of them. The fragment
    // program hash is computed only for the debug switches that use it.
    static int fsh = -1;
    if (fsh < 0) fsh = getenv("LARECOMP_SKIP_FSHASH") || getenv("LARECOMP_LOG_FSHASH") || getenv("XCDE_WATCH_FS") || ct_on() || amd_igpu_heavy_skip() || hunt_on() || getenv("XCDE_DUMP_FS_HASH") || getenv("XCDE_BLUE_FIX") || getenv("XCDE_GPUPROF_DRAWS");
    if (fsh || g_settings.dof_off) {  // (depth of field: the F1 menu turns it off and on while playing)
        static uint64_t last_fs, last_hash;
        uint64_t fs = st.program->stages[1].code_addr;
        if (fs != last_fs) {
            last_fs = fs;
            last_hash = fs_code_hash(fs);
        }
        if (g_nskipfs < 0) {
            g_nskipfs = 0;
            for (const char *p = getenv("LARECOMP_SKIP_FSHASH"); p && *p && g_nskipfs < 64;) {
                g_skipfs[g_nskipfs++] = strtoull(p, (char **)&p, 16);
                while (*p == ',') p++;
            }
            // [AMD-IGPU-FIX] heavy volume effect: see amd_igpu_heavy_skip()
            if (amd_igpu_heavy_skip()) {
                static const uint64_t heavy[] = { 0x53329ef60c533b43ull, 0x36580ad1389195c3ull, 0x071492ad778cf10cull };
                for (size_t h = 0; h < sizeof heavy / sizeof heavy[0] && g_nskipfs < 64; h++) g_skipfs[g_nskipfs++] = heavy[h];
                LOG("[vk] AMD integrated GPU: %d heavy volume-effect fragment shaders skipped (XCDE_AMD_KEEP_HEAVY=1 keeps them)%c", (int)(sizeof heavy / sizeof heavy[0]), 10);
            }
            hunt_load_file();
        }
        for (int i = 0; i < g_nskipfs; i++) if (last_hash == g_skipfs[i]) { SKIPPED(); return; }
        if (hunt_on() && hunt_draw(last_hash)) { SKIPPED(); return; }
        // [XCDE] T183: launcher "depth of field" off (xcde.ini depth_of_field = 0). The game's DOF is a bokeh chain after
        // the lighting: CoC + half-res colour (fs 69b277b005fbc48a), 64x36 tiles, one point sprite per tile splatted into
        // a 640x360 buffer (f818597443f5fa73), then the sprites composited over the scene (8cb02bca8ec7e152). Without
        // the splat and the composite the scene stays as lit; the setup / tile passes feed nothing else.
        if (g_settings.dof_off && (last_hash == 0xf818597443f5fa73ull || last_hash == 0x8cb02bca8ec7e152ull)) { SKIPPED(); return; }
        if (g_debug_frame && getenv("LARECOMP_LOG_FSHASH")) LOG("[fshash] fs=%llx hash %016llx\n", (unsigned long long)fs, (unsigned long long)last_hash);
        g_cur_fs_hash = last_hash;
    }
    static const char *watch_fs = (const char *)1;  // ([XCDE] perf: was a getenv per draw)
    if (watch_fs == (const char *)1) watch_fs = getenv("XCDE_WATCH_FS");
    if (watch_fs && g_cur_fs_hash == strtoull(watch_fs, NULL, 16) && st.rt.depth && st.rt.depth->backend)
        g_watch_img = ((TexBackend *)st.rt.depth->backend)->image;
    if (g_debug_frame) debug_draw(pr, r, indexed);
    if (ct_on()) ct_draw(pr, "draw");
    PRE_MARK(1);
    alias_draw_check();
    {   // [XCDE] T192: a texture used through SULD / SUST without storage usage yet gets it (re-created, contents copied)
        ShaderInfo *sis[3] = {pr->vi, pr->fi, pr->gi};
        for (int s = 0; s < 3; s++)
            for (int i = 0; sis[s] && i < sis[s]->ntex; i++) {
                if (!sis[s]->tex[i].is_image) continue;
                NvnTexture *t = shader_texture_object(s == 2 ? 2 : s, &sis[s]->tex[i]);
                if (t && naliases) t = alias_read(t);
                if (t && sis[s]->tex[i].image_atomic) {  // [XCDE] T226: SUATOM images are declared r32ui
                    static int told;
                    if (told < 4) { told++; LOG("[vk] SUATOM image: nvn fmt %#x %ux%u (stage %d)\n", t->format, t->width, t->height, s); }
                }
                if (!t || (t->backend && (((TexBackend *)t->backend)->has_storage || ((TexBackend *)t->backend)->storage_failed))) continue;
                vk_end_pass(); RT_DIRTY();
                vk_tex_add_storage(t, vk_cmd());
            }
    }
    resolve_feedback(pr);
    snap_reason = "draw-depthwrite";
    if (st.rt.depth && st.ds[0] && st.ds[1]) snapshots_invalidate(st.rt.depth);
    snap_reason = "draw-colorwrite";
    for (int i = 0; i < st.rt.ncolor; i++)
        if (st.rt.color[i] && ((st.channel_mask >> (i * 4)) & 0xF)) snapshots_invalidate(st.rt.color[i]);
    // [XCDE] T136: a draw that writes memory (transform feedback capture, shader stores) starts a new pass, i.e. a full
    // barrier *before* it too (one after it was already there). The game refills one grass scratch buffer from offset
    // 0 per batch, right after the previous batch's draws read it as instance data: Maxwell runs draws in order, Vulkan
    // overlaps them inside a pass, so the new capture overwrote data still being read - a grass tuft drawn with another
    // batch's positions missed its depth prepass and showed black for a frame (user report; timing dependent).
    PRE_MARK(2);
    VkPipelineStageFlags2 wst = 0;  // [XCDE] T162 part 2: the stages of this draw that write guest memory
    if (xfb || pr->vi->uses_gmem_store) wst |= VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    if (pr->fi->uses_gmem_store) wst |= VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    if (pr->gi && pr->gi->uses_gmem_store) wst |= VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT;
    if (wst) {
        if (in_pass) RT_DIRTY();
        pre_wr |= wst;  // (the barrier before it: earlier reads of guest memory first, T136)
    }
    static int simg_full = -1;  // XCDE_SIMG_FULL=1: storage images untracked (full barriers around their passes, as before)
    if (simg_full < 0) simg_full = getenv("XCDE_SIMG_FULL") != NULL;
    bool simg = false;
    if (simg_full) {
        ShaderInfo *sis[3] = {pr->vi, pr->fi, pr->gi};
        for (int s = 0; s < 3; s++)
            for (int i = 0; sis[s] && i < sis[s]->ntex; i++) simg |= sis[s]->tex[i].is_image;
        if (simg) pass_dirty = true;
    }
    {   // [XCDE] T162: a draw sampling an image written since the last full barrier needs one (unless the image is an
        // attachment of the pass it continues: the feedback case, read in place as before)
        bool cont = in_pass && !st.rt_dirty;
        ShaderInfo *sis[3] = {pr->vi, pr->fi, pr->gi};
        for (int s = 0; s < 3 && !pass_hazard; s++) {
            ShaderInfo *si = sis[s];
            if (!si) continue;
            for (int i = 0; i < si->ntex && !pass_hazard; i++) {
                NvnTexture *t = shader_texture_object(s == 2 ? 2 : s, &si->tex[i]);
                TexBackend *b = t ? t->backend : NULL;
                if (!b || b->w_epoch != pass_epoch) continue;
                if (cont && is_attachment(t)) continue;
                pass_hazard = true;
                if (cont) RT_DIRTY();  // (restart the pass behind a full barrier)
            }
        }
    }
    {   // [XCDE] T220: the pass that composites the scene into the presented image can go on with the HUD in the same
        // pass (the title screen: composite, then "Press any button" / the copyright as instanced glyphs). The image
        // before the HUD (tmp_hud_snapshot, at the pass end) then held the text, FSR 3 / DLSS took it as scene and
        // FSR 3 frayed the letters with the grass behind them. The pass ends after its first draw instead, so the
        // snapshot is the scene alone and the rest is the HUD (composited back by tmp_hud.frag).
        static int pass_draws;
        static uint64_t last_pass_seq;
        if (in_pass && tmp_enabled() && pass_dbg_rt && pass_dbg_rt->presented && tmp_scene_frame == renderer_present_count() &&
            !tmp_hud_started() && pass_draws >= 1 && last_pass_seq == pass_seq_no)
            RT_DIRTY();
        if (last_pass_seq != pass_seq_no || st.rt_dirty || !in_pass) pass_draws = 0;
        uint64_t pb0 = prof ? SDL_GetTicksNS() : 0;
        begin_pass();
        if (prof) prof_bp += SDL_GetTicksNS() - pb0;
        last_pass_seq = pass_seq_no;
        pass_draws++;
    }
    // [XCDE] T162 part 2: guest memory this draw reads (vertex / index fetch, heap loads) and writes
    mem_rd |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT | VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT |
              (pr->vi->uses_gmem ? VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT : 0) | (pr->fi->uses_gmem ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : 0) |
              (pr->gi && pr->gi->uses_gmem ? VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT : 0);
    mem_wr |= wst;
    if (simg) pass_dirty = true;
    {   // [XCDE] T162: what this draw samples, as of this epoch (a later pass writing it must wait for the reads)
        ShaderInfo *sis[3] = {pr->vi, pr->fi, pr->gi};
        for (int s = 0; s < 3; s++) {
            ShaderInfo *si = sis[s];
            if (!si) continue;
            for (int i = 0; i < si->ntex; i++) {
                NvnTexture *t = shader_texture_object(s == 2 ? 2 : s, &si->tex[i]);
                if (!t || !t->backend) continue;
                ((TexBackend *)t->backend)->r_epoch = pass_epoch;
                // [XCDE] T162: a storage image (SULD / SUST) counts as written too: whatever reads or writes it next
                // (sampling, another image draw, an attachment) gets a graphics barrier first
                if (si->tex[i].is_image) ((TexBackend *)t->backend)->w_epoch = pass_epoch;
            }
        }
    }
    PRE_MARK(3);
    VkCommandBuffer cb = vk_cmd();
    if (g_gpuprof_frame) {  // [XCDE] XCDE_GPUPROF_VS=<hex vs>[,...]: with LARECOMP_GPUPROF_AT, a timestamp before each draw of those programs
        static uint64_t pvs[8]; static int npvs = -1;
        if (npvs < 0) { npvs = 0; for (const char *p = getenv("XCDE_GPUPROF_VS"); p && *p && npvs < 8;) { pvs[npvs++] = strtoull(p, (char **)&p, 16); if (*p == ',') p++; else break; } }
        // [XCDE] XCDE_GPUPROF_DRAWS=1: a timestamp before every draw, labelled with its fragment program hash
        // (scripts/linux/gpuprof_sum.py adds them up per shader)
        static int pdraws = -1;
        if (pdraws < 0) pdraws = getenv("XCDE_GPUPROF_DRAWS") != NULL;
        if (pdraws) gp_mark(cb, "  draw fs %016llx", (unsigned long long)g_cur_fs_hash);
        for (int i = 0; i < npvs; i++)
            if (st.program->stages[0].code_addr == pvs[i])
                gp_mark(cb, "  draw vs %llx inst %llu count %llu", (unsigned long long)pvs[i], (unsigned long long)(indexed ? r->x[6] : r->x[4]),
                        (unsigned long long)r->x[2]);
    }

    uint32_t mode = (uint32_t)r->x[0];
    VkPrimitiveTopology topo;
    uint8_t cls;
    bool native = topology(mode, &topo, &cls);
    if (!native && mode != 6 && mode != 7 && mode != 9) { SKIPPED(); return; }

    PRE_MARK(4);
    uint64_t prof_tk_ = prof ? SDL_GetTicksNS() : 0;
    if (prof) prof_pre += prof_tk_ - prof_tp_;
    // ---- pipeline
    // [XCDE] perf: the previous draw's pipeline when nothing the key reads changed (g_pipe_state_gen / g_pass_gen).
    // XCDE_PIPEKEY_MEMO=0: off. XCDE_MEMO_CHECK=1: the key is built anyway and a different pipeline logged.
    // (the pass part of the key: its formats, compared as values - passes are restarted with the same ones constantly)
    struct PkPass { uint32_t ncolor, samples; VkFormat color[8], depth, stencil; } pkp = {pass_ncolor, pass_samples, {0}, pass_depth, pass_stencil};
    memcpy(pkp.color, pass_color, sizeof pkp.color);
    static struct { Prog *pr; Pipe *pp; uint32_t gen, pass_gen, tab_gen, desc_gen; uint8_t cls; bool valid; struct PkPass ps; } pkm;
    uint32_t desc_gen = atomic_load_explicit(&g_tex_desc_gen, memory_order_relaxed);
    static int pkm_on = -1, memo_check = -1;
    if (pkm_on < 0) { const char *e = getenv("XCDE_PIPEKEY_MEMO"); pkm_on = !e || atoi(e) != 0; memo_check = getenv("XCDE_MEMO_CHECK") != NULL; }
    bool pk_hit = pkm_on && pkm.valid && pkm.pr == pr && pkm.cls == cls && pkm.gen == g_pipe_state_gen &&
                  pkm.pass_gen == g_pass_gen && pkm.tab_gen == prog_tab_gen && pkm.desc_gen == desc_gen &&
                  !memcmp(&pkm.ps, &pkp, sizeof pkp);
    PipeKey k;
    Pipe *pp = NULL;
    if (pk_hit) { pp = pkm.pp; memcpy(&k, &pp->k, sizeof k); stat_pipekey_memo++; }
    if (!pk_hit || memo_check) {
    memset(&k, 0, sizeof k);
    k.prog = pr;
    k.ncolor = pass_ncolor;
    memcpy(k.color, pass_color, sizeof k.color);
    k.depth = pass_depth;
    k.stencil = pass_stencil;
    k.samples = pass_samples;
    for (int i = 0; i < 32; i++) {
        if (!(pr->vi->attr_mask & (1u << i))) continue;
        k.attr_mask |= 1u << i;
        uint32_t a = i < st.nattrib ? st.attrib[i] : 0x2E;
        uint32_t fmt = a & 0xFF, stream = (a >> 8) & 0x1F, off = a >> 13;
        if (i >= st.nattrib || fmt == 0 || stream >= 16) {  // unbound attribute: (0, 0, 0, 1) from a constant stream
            k.attr_fmt[i] = VK_FORMAT_R32G32B32A32_SFLOAT;
            k.attr_bind[i] = ZERO_BINDING;
            k.used_streams |= 1u << ZERO_BINDING;
            continue;
        }
        k.attr_fmt[i] = vertex_format_supported(vertex_format(fmt));
        k.attr_off[i] = off;
        k.attr_bind[i] = stream;
        k.used_streams |= 1u << stream;
    }
    for (int s = 0; s < 16; s++) {
        if (!(k.used_streams & (1u << s))) continue;
        k.stride[s] = s < st.nstream ? st.stream[s][0] : 0;
        k.divisor[s] = s < st.nstream && k.stride[s] ? st.stream[s][1] : 0;
        if (k.divisor[s] > 1 && !g_vk.divisor) k.divisor[s] = 1;
    }
    for (uint32_t i = 0; i < k.ncolor && i < 8; i++) {
        if (!st.rt.color[i] || k.color[i] == VK_FORMAT_UNDEFINED) continue;
        // Targets the fragment shader doesn't output keep their contents on Metal, but are undefined in Vulkan
        // (NVIDIA writes black): mask them. Foliage depth pre-passes left black silhouettes (user 2026-10-08).
        k.write_mask[i] = ((pr->fi->out_mask >> (i * 4)) & 0xF) ? (st.channel_mask >> (i * 4)) & 0xF : 0;
        if (((st.color_state[0] >> i) & 1) && !rt_kind(st.rt.color[i]->format)) {
            k.blend_on |= (uint8_t)(1u << i);
            memcpy(k.blend[i], st.blend[i], 8);
        }
    }
    // [XCDE] T176: NVN ColorState logic op (GL order, as VkLogicOp; COPY = off). Integer / unorm targets only, and
    // blending is off while it is on (GL and Vulkan alike). The probe-lighting index map (640x360 RG32UI) ORs each
    // probe volume's index and weight into a byte slot; without it the last full-screen draw replaced them all.
    if (st.color_state[1] != 3 && st.color_state[1] < 16 && g_vk.logic_op) k.logic_op = (uint8_t)(st.color_state[1] + 1);
    k.a2c = st.ms[0xC] != 0;
    {   // [XCDE] audit: draw state not applied
        for (uint32_t i = 0; i < k.ncolor && i < 8; i++)
            if (((k.blend_on >> i) & 1) && st.blend[i][7]) audit("advanced blend mode %u on target %u ignored (fs %llx)", st.blend[i][7], i, (unsigned long long)audit_fs());
        uint32_t rsamp; memcpy(&rsamp, &st.ms[8], 4);
        if (rsamp > 1) audit("multisample raster samples %u ignored (fs %llx)", rsamp, (unsigned long long)audit_fs());
        if (st.ms[0xE]) audit("coverage to color ignored (fs %llx)", (unsigned long long)audit_fs());
        if (st.ms[0x10]) audit("coverage modulation %u ignored (fs %llx)", st.ms[0x10], (unsigned long long)audit_fs());
        if (st.poly[2] == 0) audit("polygon mode POINT drawn as fill (fs %llx)", (unsigned long long)audit_fs());
        if (st.poly[3] & 3) audit("polygon offset for point/line modes %#x ignored", st.poly[3]);
    }
    k.topo_class = cls;
    k.poly_fill = st.poly[2] != 1;
    pipe_key_canon(&k);  // [XCDE] T195
    uint64_t ph = hash_words(0xcbf29ce484222325ull, &k, sizeof k);
    if (!ph) ph = 1;
    HEnt *pslot = htab_slot(&pipe_tab, ph);
    if (pk_hit) {  // (XCDE_MEMO_CHECK)
        if (pslot->val != pp) {
            static int nmis;
            if (nmis++ < 50) LOG("[memo-check] pipeline key memo: another pipeline (fs %llx)\n", (unsigned long long)audit_fs());
            pp = pslot->val;
            if (!pp) { SKIPPED(); return; }
        }
    } else {
    pp = pslot->val;
    if (!pp) {
        pp = calloc(1, sizeof *pp);
        pp->k = k;
        pp->t_new = SDL_GetTicksNS();
        if (!pslot->key) pipe_tab.n++;
        pslot->key = ph;
        pslot->val = pp;
        atomic_fetch_add(&pp->jobs, 1);
        run_job(build_pipeline, pp);
        var_record(st.program, &vkey, &fkey, &k);  // [XCDE] T163
        if (dump_fs_want() && dump_fs_want() == g_cur_fs_hash) attr_layout_log(&k, dump_fs_want());
    }
    else if (atomic_load(&pp->state) == 3) {  // [XCDE] evicted (pipe_evict): built again, from the warm caches
        pp->how &= 0x80;
        atomic_store(&pp->state, 0);
        atomic_fetch_add(&pp->jobs, 1);
        run_job(build_pipeline, pp);
        stat_pipe_rebuilt++;
    }
    }  // (not a memo hit)
    }  // (key built)
    {   // [XCDE] XCDE_MEMO_WHY=1: the same pipeline as the previous draw without a memo hit - which input moved
        static int why = -1;
        static uint32_t hist[96], npass, nprog, n;
        if (why < 0) why = getenv("XCDE_MEMO_WHY") != NULL;
        if (why && !pk_hit && pkm.valid && pkm.pp == pp) {
            if (pkm.pass_gen != g_pass_gen || memcmp(&pkm.ps, &pkp, sizeof pkp)) npass++;
            else if (pkm.pr != pr || pkm.cls != cls) nprog++;
            else if (pkm.gen != g_pipe_state_gen) hist[g_pipe_bump_op < 96 ? g_pipe_bump_op : 95]++;
            if (++n % 20000 == 0) {
                LOG("[memo-why] %u same-pipeline misses: pass %u, program/class %u, state:", n, npass, nprog);
                for (int i = 0; i < 96; i++) if (hist[i]) LOG(" op%d=%u", i, hist[i]);
                LOG("\n");
            }
        }
    }
    if (!pk_hit) { pkm.desc_gen = desc_gen; pkm.ps = pkp; pkm.pr = pr; pkm.pp = pp; pkm.gen = g_pipe_state_gen; pkm.pass_gen = g_pass_gen; pkm.tab_gen = prog_tab_gen; pkm.cls = cls; pkm.valid = true; }
    if (atomic_load(&pp->state) == 3) {  // [XCDE] evicted since the memo (pipe_evict): built again
        pp->how &= 0x80;
        atomic_store(&pp->state, 0);
        atomic_fetch_add(&pp->jobs, 1);
        run_job(build_pipeline, pp);
        stat_pipe_rebuilt++;
    }
    pp->last_used = renderer_present_count();
    bool pp_ok;
    if (atomic_load(&pp->state) == 0 && pipe_wait_over_budget()) { pp_ok = false; g_last_wait_ns = 0; }  // [PIPE-BUDGET] (still building: skipped this frame)
    else { pp_ok = wait_ready(&pp->state, pp); pb_used_ns += g_last_wait_ns; }
    if (g_last_wait_ns > build_wait_min()) build_wait_log(NULL, pp, g_last_wait_ns);
    if ((pp->how & 0x40) && gpl_optimize_on()) { pp->how &= ~0x40; atomic_fetch_add(&pp->jobs, 1); run_job_low(gpl_optimize, pp); }  // [XCDE] T218: a prelinked pipeline's first draw
    if (!pp_ok) {
        static int nlog2_;
        if (nlog2_++ < 8) LOG("[vk] pipeline not ready: state %d""%c", (int)atomic_load(&pp->state), 10);
        SKIPPED(); return;
    }
    ss_enabled();
    {   // (profiler) how often a draw repeats the previous draw's program / pipeline
        static Pipe *last_pp; static Prog *last_pr;
        prof_same_pipe += pp == last_pp; prof_same_prog += pr == last_pr;
        last_pp = pp; last_pr = pr;
    }
    if (ss_need(SS_PIPE, ss.pipe == pp->pipe)) { ss.pipe = pp->pipe; vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pp->pipe); }
    uint64_t prof_td_ = prof ? SDL_GetTicksNS() : 0;
    if (prof) prof_key += prof_td_ - prof_tk_;

    // ---- depth / stencil (dynamic)
    static int nodt = -1;
    if (nodt < 0) nodt = getenv("LARECOMP_NO_DEPTH_TEST") != NULL;
    bool dt = st.ds[0] && !nodt && pass_depth;
    if (ss_need(SS_DT, ss.dt == (VkBool32)dt)) { ss.dt = dt; vkCmdSetDepthTestEnable(cb, dt); }
    VkBool32 dw = dt && st.ds[1];
    if (ss_need(SS_DW, ss.dw == dw)) { ss.dw = dw; vkCmdSetDepthWriteEnable(cb, dw); }
    VkCompareOp dop = dt ? compare_fn(st.ds[2]) : VK_COMPARE_OP_ALWAYS;
    // EQUAL against a depth pre-pass written by a different vertex shader (foliage, trees) needs both shaders to land
    // on identical depths. They do since the translator stopped using unpackHalf2x16 (NVIDIA ran the half math after
    // it in fp16, differently per shader: black slashes in foliage). LARECOMP_VK_EQUAL_TOL=1 brings back the old
    // workaround: LESS_OR_EQUAL with a few ulps of bias towards the camera, which also let walls paint over the
    // plants stuck flat on them (beach) - so it is off.
    static int tol = -1;
    if (tol < 0) tol = getenv("LARECOMP_VK_EQUAL_TOL") != NULL;
    bool equal_tol = dop == VK_COMPARE_OP_EQUAL && tol;
    if (equal_tol) dop = VK_COMPARE_OP_LESS_OR_EQUAL;
    if (ss_need(SS_DOP, ss.dop == dop)) { ss.dop = dop; vkCmdSetDepthCompareOp(cb, dop); }
    bool sten = st.ds[3] && pass_stencil;
    {   // (debugging) XCDE_NO_STENCIL_FS=<fs code addr hex>: stencil test off for that program's draws
        static uint64_t ns = ~0ull;
        if (ns == ~0ull) ns = getenv("XCDE_NO_STENCIL_FS") ? strtoull(getenv("XCDE_NO_STENCIL_FS"), NULL, 16) : 0;
        if (ns && st.program->stages[1].code_addr == ns) sten = false;
    }
    if (ss_need(SS_STEN, ss.sten == (VkBool32)sten)) { ss.sten = sten; vkCmdSetStencilTestEnable(cb, sten); }
    if (sten) {
        uint32_t ops = st.ds[5] | st.ds[6] << 8 | st.ds[7] << 16;
        for (int f = 0; f < 2; f++) {
            uint32_t o = f ? ops >> 12 : ops;
            VkStencilFaceFlags face = f ? VK_STENCIL_FACE_BACK_BIT : VK_STENCIL_FACE_FRONT_BIT;
            uint32_t v[7] = {stencil_op(o & 0xF), stencil_op((o >> 8) & 0xF), stencil_op((o >> 4) & 0xF),
                             compare_fn(f ? st.ds[4] >> 4 : st.ds[4] & 0xF), st.stencil_vmask[f] & 0xFF, st.stencil_mask[f] & 0xFF,
                             st.stencil_ref[f]};
            if (!ss_need(f ? SS_SFACE1 : SS_SFACE0, !memcmp(v, ss.sface[f], sizeof v))) continue;
            memcpy(ss.sface[f], v, sizeof v);
            vkCmdSetStencilOp(cb, face, (VkStencilOp)v[0], (VkStencilOp)v[1], (VkStencilOp)v[2], (VkCompareOp)v[3]);
            vkCmdSetStencilCompareMask(cb, face, v[4]);
            vkCmdSetStencilWriteMask(cb, face, v[5]);
            vkCmdSetStencilReference(cb, face, v[6]);
        }
    }
    if (ss_need(SS_BC, !memcmp(ss.bc, st.blend_color, sizeof ss.bc))) { memcpy(ss.bc, st.blend_color, sizeof ss.bc); vkCmdSetBlendConstants(cb, st.blend_color); }

    // ---- rasterizer
    uint8_t cull = st.poly[0];
    if (vkey.flip_x) cull = 0;  // (mirror mode's composite: one full-screen quad, either winding)
    // [FIX-TEX] NVN cull face 3 = FRONT_AND_BACK: every triangle is culled (the game draws its placeholder / hidden helper objects with it:
    // the blue tents, shader 69dc9c67ee198e5d, cb4 = 0,0,1,1). It was mapped to NONE, so they were drawn. LARECOMP_CULL3_NONE=1 brings that back.
    static int cull3_none = -1;
    if (cull3_none < 0) cull3_none = getenv("LARECOMP_CULL3_NONE") != NULL;
    VkCullModeFlags cm = cull == 1 ? VK_CULL_MODE_FRONT_BIT : cull == 2 ? VK_CULL_MODE_BACK_BIT : (cull == 3 && !cull3_none) ? VK_CULL_MODE_FRONT_AND_BACK : VK_CULL_MODE_NONE;
    if (ss_need(SS_CULL, ss.cull == cm)) { ss.cull = cm; vkCmdSetCullMode(cb, cm); }
    bool ccw = st.poly[1] == 1;
    static int flipw = -1;  // LARECOMP_VK_FLIP_WINDING: invert the front-face mapping (A/B against Metal)
    if (flipw < 0) flipw = getenv("LARECOMP_VK_FLIP_WINDING") != NULL;
    bool front_ccw = (ccw != vkey.flip_y) != (flipw != 0);
    VkFrontFace ff = front_ccw ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
    if (ss_need(SS_FF, ss.ff == ff)) { ss.ff = ff; vkCmdSetFrontFace(cb, ff); }
    VkPrimitiveTopology tp = native ? topo : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    if (ss_need(SS_TOPO, ss.topo == tp)) { ss.topo = tp; vkCmdSetPrimitiveTopology(cb, tp); }
    VkBool32 pre = st.prim_restart && native && (mode == 3 || mode == 5);
    if (ss_need(SS_PRE, ss.pre == pre)) { ss.pre = pre; vkCmdSetPrimitiveRestartEnable(cb, pre); }
    {
        float f[3] = {0, 0, 0};
        static int off = -1;
        if (off < 0) off = getenv("LARECOMP_NO_DEPTH_BIAS") != NULL;
        bool on = (st.poly[3] & 4) && !off;
        if (on) memcpy(f, st.poly_offset, sizeof f);
        if (equal_tol) { f[1] -= 8; on = true; }  // (see the EQUAL comment above: a few depth ulps nearer)
        if (ss_need(SS_DBE, ss.dbe == (VkBool32)on)) { ss.dbe = on; vkCmdSetDepthBiasEnable(cb, on); }
        if (ss_need(SS_DB, !memcmp(ss.db, f, sizeof f))) { memcpy(ss.db, f, sizeof f); vkCmdSetDepthBias(cb, f[1], f[2], f[0]); }
    }

    // ---- textures first (their scales feed the widescreen decision and the rscale table), then the viewport
    float *vp = st.viewport[0];
    float zn = st.depth_range[0][0], zf = st.depth_range[0][1];
    vp_sx = pass_sx; vp_sy = pass_sy; vp_ox = 0; vp_oy = 0;
    float tint[8] = {0};
    // fragment textures are bound before deciding the viewport (tex_sc / tex_wh), rscale depends on the decision:
    // build the vertex set now and the fragment set after the decision (stage_set fills tex_sc for stage 1).
    if (prof) prof_dyn += SDL_GetTicksNS() - prof_td_;
    uint64_t xfb_at = 0, xfb_bytes = 0;  // (T175: what this draw captures)
    if (xfb) {  // [XCDE] T134: where this draw's vertices go (appended after what the capture already holds)
        uint32_t xfirst = (uint32_t)r->x[1], xcount = (uint32_t)r->x[2], xbase = (uint32_t)r->x[3], xinst = r->x[4] ? (uint32_t)r->x[4] : 1;
        uint64_t at = st.xfb_buf[0].addr + st.xfb_written, end = st.xfb_buf[0].addr + st.xfb_buf[0].size;
        memset(g_xfbp, 0, sizeof g_xfbp);
        g_xfbp[0] = (uint32_t)((at - HEAP_BASE) / 4); g_xfbp[1] = xbase; g_xfbp[2] = xcount; g_xfbp[3] = xfirst;
        g_xfbp[4] = (uint32_t)((end - HEAP_BASE) / 4);
        uint64_t bytes = (uint64_t)xcount * xinst * vkey.xfb_stride;
        if (at < end) vc_mark_gpu(at - HEAP_BASE, (at + bytes < end ? at + bytes : end) - HEAP_BASE);
        if (at < end) { xfb_at = at; xfb_bytes = bytes; }
        st.xfb_written += bytes;
        if (g_debug_frame) LOG("[xfb] vs %llx -> %llx (+%llu bytes, %u x %u, stride %u)\n", (unsigned long long)st.program->stages[0].code_addr,
                               (unsigned long long)at, (unsigned long long)bytes, xcount, xinst, vkey.xfb_stride);
    }
    PROF_T0();
    VkDescriptorSet vset = stage_set(pr, pr->set_layout[0], 0, 0, pr->vi, NULL, NULL, 0, 0, 0, NULL);
    PROF_ADD(prof_sets);
    VkDescriptorImageInfo fii[64];
    uint64_t prof_tx_ = prof ? SDL_GetTicksNS() : 0;
    for (int i = 0; i < pr->fi->ntex && i < 64; i++) {
        float sc[2]; uint32_t wh[2];
        texture_info(1, pr->fi, i, &fii[i], sc, wh);
        tex_sc[1][i][0] = sc[0]; tex_sc[1][i][1] = sc[1]; tex_wh[1][i][0] = wh[0]; tex_wh[1][i][1] = wh[1]; tex_d16[1][i] = ti_d16;
    }
    if (prof) prof_tex += SDL_GetTicksNS() - prof_tx_;
    if (fabsf(pass_sx - pass_sy) > 0.01f * pass_sy && st.rt.ncolor && renderer_texture_presented(st.rt.color[0])) {
        // Draws into the swapchain image (see renderer.m): composites fill the window, UI goes in a centred 16:9 box.
        bool reads_scene = false;
        for (int i = 0; i < pr->fi->ntex && i < 64; i++)
            if ((tex_sc[1][i][0] != 1 || tex_sc[1][i][1] != 1) && tex_wh[1][i][0] == st.rt.color[0]->width &&
                tex_wh[1][i][1] == st.rt.color[0]->height)
                reads_scene = true;
        bool fullscreen_quad = r->x[2] <= 6 && st.viewport[0][0] == 0 && st.viewport[0][1] == 0 &&
                               st.viewport[0][2] >= st.rt.color[0]->width && st.viewport[0][3] >= st.rt.color[0]->height;
        bool fullscreen_fill = pr->fi->ntex == 0 && fullscreen_quad;
        if (fullscreen_quad && !reads_scene && pr->fi->ntex == 1 && (tex_sc[1][0][0] != 1 || tex_sc[1][0][1] != 1)) reads_scene = true;
        // [XCDE] V9: a full-target pass into the window that samples ANY scene-scaled texture is a composite of the scene
        // even when none of its textures has exactly the window texture's size (post effects that read a smaller or
        // differently sized copy of the scene). Before, such a pass was shrunk into the centred 16:9 box: black bars while
        // the effect ran (e.g. while the camera moves).
        bool v9_relaxed = false;
        if (fullscreen_quad && !reads_scene && pr->fi->ntex > 1) {
            for (int i = 0; i < pr->fi->ntex && i < 64; i++)
                if (tex_sc[1][i][0] != 1 || tex_sc[1][i][1] != 1) { reads_scene = true; v9_relaxed = true; break; }
        }
        static uint64_t wide_frame = ~0ull;
        if (reads_scene) wide_frame = renderer_present_count();
        bool fade_fill = false;
        if (fullscreen_fill && st.ubuf[1][0].addr && st.ubuf[1][0].size >= 12) {
            const float *c = (const float *)(uintptr_t)st.ubuf[1][0].addr;
            bool black = c[0] < 0.01f && c[1] < 0.01f && c[2] < 0.01f, white = c[0] > 0.99f && c[1] > 0.99f && c[2] > 0.99f;
            fade_fill = black || white;
        }
        bool composite = reads_scene || fade_fill || (fullscreen_fill && wide_frame == renderer_present_count());
        if (!composite) wide_frame = ~0ull;
        if ((v9_relaxed || (!composite && fullscreen_quad)) && !getenv("XCDE_NO_WIDE_LOG")) {
            // [XCDE] V9 diagnostic ("[wide-box]" in last_game.log): one line per distinct fragment shader, at most 80
            static uint64_t seen_fs[96];
            static int nseen;
            uint64_t fsh = st.program->stages[1].code_hash;
            bool known = false;
            for (int i = 0; i < nseen; i++) if (seen_fs[i] == fsh) { known = true; break; }
            if (!known && nseen < 80) {
                seen_fs[nseen++] = fsh;
                LOG("[wide-box] %s: fs %016llx ntex %d rt %ux%u vp %.0f %.0f %.0f %.0f pass %.3f %.3f tex0 sc %.3f,%.3f wh %ux%u\n",
                    v9_relaxed ? "now a composite (fills the window)" : "fullscreen quad kept in the 16:9 box",
                    (unsigned long long)fsh, (int)pr->fi->ntex, (unsigned)st.rt.color[0]->width, (unsigned)st.rt.color[0]->height,
                    (double)st.viewport[0][0], (double)st.viewport[0][1], (double)st.viewport[0][2], (double)st.viewport[0][3],
                    (double)pass_sx, (double)pass_sy, (double)tex_sc[1][0][0], (double)tex_sc[1][0][1],
                    (unsigned)tex_wh[1][0][0], (unsigned)tex_wh[1][0][1]);
            }
        }
        if (!composite) {
            int pw, ph;
            pass_size(&pw, &ph);
            float u = fminf(pass_sx, pass_sy);
            vp_sx = vp_sy = u;
            vp_ox = (pw - st.rt.color[0]->width * u) * 0.5f;
            vp_oy = (ph - st.rt.color[0]->height * u) * 0.5f;
        }
    }
    float jx = 0, jy = 0;
    if (tmp_enabled()) tmp_draw(pr, &jx, &jy);  // [XCDE] temporal upscaling: the camera, the scene depth, jitter
    if (tmp_enabled()) {  // [XCDE] T215: the game's velocity buffer (gTVelocity of its temporal AA / AO / shadow passes)
        // read at the gTVelocity unit of the game's own readers, found by their code hashes (container names,
        // tools/spch_attrs.py + the SPCH layout). Written in the G-buffer pass, also with the game's AA off.
        static const struct { uint64_t h; int unit; } vr[] = {
            {0xbb44ef3b36ad9cebull, 0}, {0x7e44df8503cc1e3dull, 4}, {0x920e4c73a20a21beull, 3}, {0x99e2b215439c052dull, 2},
            {0xfc143ddc53c1c59eull, 0}, {0x1abac2378f5ba855ull, 0}, {0x21ec426dc97769c6ull, 2}, {0xec955aaaac94168aull, 4},
            {0x9afc15fdfd34bf79ull, 5}, {0x565bcdfbb1b87719ull, 5}, {0x7053600a7399c690ull, 4}, {0x1626d8d25e7c6aeeull, 4},
            {0x71721e8cf70be93bull, 2}, {0xe14a094d7bb38c80ull, 2}, {0x5587ed44ae49cb8full, 2}, {0xaf42f7bfcaac8f78ull, 2},
            {0x6c9d5853f7c5ee06ull, 0}};
        uint64_t fh = st.program->stages[1].code_hash;
        for (size_t i = 0; i < sizeof vr / sizeof vr[0]; i++) {
            if (vr[i].h != fh) continue;
            uint64_t h = st.tex[1][vr[i].unit];
            uint32_t tic = h & 0xFFFFF;
            NvnTexture *t = h && st.tpool && tic < (uint32_t)st.tpool->count ? ((NvnTexEntry *)st.tpool->entries)[tic].tex : NULL;
            if (t && t->backend) tmp_set_velocity((TexBackend *)t->backend);
            static int nl;
            if (getenv("XCDE_VEL_FIND") && nl++ < 8)
                LOG("[vel] fs %016llx (code %llx) unit %d: %ux%u fmt %#x addr %llx\n", (unsigned long long)fh,
                    (unsigned long long)st.program->stages[1].code_addr, vr[i].unit, t ? t->width : 0, t ? t->height : 0,
                    t ? t->format : 0, t ? (unsigned long long)t->addr : 0);
            break;
        }
    }
    {   // camera-attached effects sized for 16:9 are widened with the view (runtime/hle/widescreen.c)
        float k = 1.0f;  // [XCDE] T186: (LAS widened camera-attached effects here; XCDE has none)
        float vx = vp[0] * vp_sx + vp_ox + jx, vw = vp[2] * vp_sx;
        set_viewport(vx, vp[1] * vp_sy + vp_oy + jy, vw, vp[3] * vp_sy, zn, zf);
        if (g_debug_frame) {
            VkRect2D sr = scissor_rect();
            LOG("[draw]   vp %.2f %.2f %.2f %.2f (stretch %.3f, fs hash %016llx) scissor %d %d %u %u | guest sc %d %d %d %d | pass %ux%u s %.4f %.4f\n",
                vx, vp[1] * vp_sy + vp_oy, vw, vp[3] * vp_sy, k, (unsigned long long)g_cur_fs_hash, sr.offset.x, sr.offset.y,
                sr.extent.width, sr.extent.height, st.scissor[0][0], st.scissor[0][1], st.scissor[0][2], st.scissor[0][3], pass_w, pass_h,
                vp_sx, vp_sy);
        }
    }
    apply_scissor();
    if (st.raster_discard) {  // [XCDE] T134: nvnCommandBufferSetRasterizerDiscard: the vertex stage runs, nothing is drawn
        VkRect2D z = {{0, 0}, {0, 0}};
        if (ss_need(SS_SC, !memcmp(&z, &ss.sc, sizeof z))) { ss.sc = z; vkCmdSetScissor(cb, 0, 1, &z); }
    }
    float rs0[4] = {vp_sx, vp_sy, vp_ox, vp_oy};
    uint64_t prof_t1_ = prof ? SDL_GetTicksNS() : 0;
    VkDescriptorSet fset = stage_set(pr, pr->set_layout[1], 1, 1, pr->fi, rs0, tint, 0, 0, 0, fii);
    if (prof) prof_sets += SDL_GetTicksNS() - prof_t1_;
    VkDescriptorSet sets[3] = {vset, fset, VK_NULL_HANDLE};
    if (pr->gi) sets[2] = stage_set(pr, pr->set_layout[2], 2, 2, pr->gi, NULL, NULL, 0, 0, 0, NULL);  // [XCDE] geometry
    vk_bind_graphics_sets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pr->layout, 0, pr->gi ? 3 : 2, sets, 0, NULL);  // (always new sets)
    if (pr->vi->uses_gmem_store) vc_mark_sbufs(0);
    if (pr->fi->uses_gmem_store) vc_mark_sbufs(1);
    if (pr->vi->uses_gmem && !pr->vi->uses_gmem_store) ov_note_sbufs(0, false);  // [XCDE] (XCDE_OVERLAP_STATS) gmem loads
    if (pr->fi->uses_gmem && !pr->fi->uses_gmem_store) ov_note_sbufs(1, false);

    uint64_t prof_t2_ = prof ? SDL_GetTicksNS() : 0;
    // ---- indices (copied; the largest index bounds the vertex data copied below)
    uint32_t first = 0, count = (uint32_t)r->x[2], inst, base_inst, max_index = 0;
    int32_t base_vertex = 0;
    VkSpan ib = {0};
    VkIndexType itype_vk = VK_INDEX_TYPE_UINT32;
    uint32_t icount = 0;
    bool use_index = indexed || !native;
    const void *scan_src = NULL;  // deferred index scan (direct index buffer)
    uint32_t scan_isz = 0, scan_restart = 0;
    if (!indexed) {
        first = (uint32_t)r->x[1]; base_inst = (uint32_t)r->x[3]; inst = (uint32_t)r->x[4] ? (uint32_t)r->x[4] : 1;
        max_index = first + count;
        if (!native) {
            uint32_t *e = expand_indices(mode, count, NULL, 0, first, &icount);
            ib = vk_transient_copy(e, (uint64_t)icount * 4 + 4, 4);
            // T260: expansion scratch is retained by the render thread.
        }
    } else {
        uint32_t itype = (uint32_t)r->x[1];
        uint64_t iaddr = r->x[3];
        base_vertex = (int32_t)r->x[4];
        base_inst = (uint32_t)r->x[5];
        inst = (uint32_t)r->x[6] ? (uint32_t)r->x[6] : 1;
        const void *src = (const void *)(uintptr_t)iaddr;
        uint32_t restart = st.prim_restart ? st.restart_index : 0xFFFFFFFFu;
        if (native && itype != 0) {
            uint32_t isz = itype == 1 ? 2 : 4;
            itype_vk = itype == 1 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
            icount = count;
            if (iaddr % isz == 0 && guest_direct(iaddr, (uint64_t)count * isz, &ib.buf, &ib.off)) {
                scan_src = src; scan_isz = isz; scan_restart = restart;  // max_index only if a stream gets copied
            } else {
                ib = vk_transient_copy(src, (uint64_t)count * isz + 4, 4);
                if (isz == 2) { const uint16_t *s = src; for (uint32_t i = 0; i < count; i++) if (s[i] > max_index && s[i] != (uint16_t)restart) max_index = s[i]; }
                else { const uint32_t *s = src; for (uint32_t i = 0; i < count; i++) if (s[i] > max_index && s[i] != restart) max_index = s[i]; }
            }
        } else if (native) {  // 8-bit indices: widen to 16
            ib = vk_transient((uint64_t)count * 2 + 4, 4);
            uint16_t *w = ib.ptr;
            const uint8_t *s = src;
            for (uint32_t i = 0; i < count; i++) { w[i] = s[i] == 0xFF && st.prim_restart ? 0xFFFF : s[i]; if (s[i] > max_index && w[i] != 0xFFFF) max_index = s[i]; }
            itype_vk = VK_INDEX_TYPE_UINT16;
            icount = count;
        } else {
            uint32_t *e = expand_indices(mode, count, src, (int)itype, 0, &icount);
            for (uint32_t i = 0; i < icount; i++) if (e[i] > max_index) max_index = e[i];
            ib = vk_transient_copy(e, (uint64_t)icount * 4 + 4, 4);
            // T260: expansion scratch is retained by the render thread.
        }
        if (!scan_src) {
            max_index += 1;
            if (base_vertex > 0) max_index += (uint32_t)base_vertex;
        }
    }
    if (g_debug_frame && indexed && getenv("XCDE_TRACE_IDX")) {  // [XCDE] the index data this draw reads
        const uint8_t *ip = (const uint8_t *)(uintptr_t)r->x[3]; uint32_t itype = (uint32_t)r->x[1];
        LOG("[idx] itype %d count %u base_vertex %d native %d :", (int)itype, count, base_vertex, (int)native);
        for (uint32_t i = 0; i < 6 && ip; i++) LOG(" %u", itype == 1 ? ((const uint16_t *)ip)[i] : itype == 2 ? ((const uint32_t *)ip)[i] : ip[i]);
        LOG("\n");
    }
    if (use_index && ss_need(SS_IB, ss.ib == ib.buf && ss.io == ib.off && ss.it == itype_vk)) {
        ss.ib = ib.buf; ss.io = ib.off; ss.it = itype_vk;
        vkCmdBindIndexBuffer(cb, ib.buf, ib.off, itype_vk);
    }

    // ---- vertex streams: the range this draw can read, copied once per submission
    uint64_t dbg_vb_addr[17] = {0}, dbg_vb_len[17] = {0};  // [XCDE] XCDE_DUMP_VB
    for (int s = 0; s < 17; s++) {
        if (!(k.used_streams & (1u << s))) continue;
        VkSpan sp;
        VkDeviceSize span_size = VK_WHOLE_SIZE;  // (T173: bytes this binding may read)
        if (s == ZERO_BINDING) {
            static const float zero[4] = {0, 0, 0, 1};
            sp = vk_transient_copy(zero, 16, 16);
        } else {
            Range v = st.vbuf[s];
            dbg_vb_addr[s] = v.addr; dbg_vb_len[s] = v.size;
            VkBuffer dbuf;
            VkDeviceSize doff;
            if (v.addr && v.size && guest_direct(v.addr, v.size, &dbuf, &doff)) {
                if (ss_on && (ss.vb_known >> s & 1) && ss.vb[s] == dbuf && ss.vo[s] == doff && ss.vsz[s] == v.size) continue;
                ss.vb[s] = dbuf; ss.vo[s] = doff; ss.vsz[s] = v.size; ss.vb_known |= 1u << s;
                // [XCDE] T173: bound with the stream's own size - fetches past it read 0 (robustBufferAccess2), as
                // on Maxwell. Unbounded, a 4-byte per-object opacity stream read the guest memory after it.
                VkDeviceSize vsz = g_vk.robust2 ? v.size : VK_WHOLE_SIZE;
                if (g_direct_limit != ~0ull && (vsz == VK_WHOLE_SIZE || vsz > g_direct_limit)) vsz = g_direct_limit;  // (T175 mirror end)
                vkCmdBindVertexBuffers2(cb, (uint32_t)s, 1, &dbuf, &doff, &vsz, NULL);
                continue;
            }
            if (scan_src) {  // copying after all: the index scan bounds the copy
                if (scan_isz == 2) { const uint16_t *q = scan_src; for (uint32_t i = 0; i < icount; i++) if (q[i] > max_index && q[i] != (uint16_t)scan_restart) max_index = q[i]; }
                else { const uint32_t *q = scan_src; for (uint32_t i = 0; i < icount; i++) if (q[i] > max_index && q[i] != scan_restart) max_index = q[i]; }
                max_index += 1;
                if (base_vertex > 0) max_index += (uint32_t)base_vertex;
                scan_src = NULL;
            }
            uint64_t stride = k.stride[s], need;
            if (!stride) need = 256;
            else if (k.divisor[s]) need = ((uint64_t)base_inst + inst + k.divisor[s] - 1) / k.divisor[s] * stride + 256;
            else need = (uint64_t)max_index * stride + 256;
            if (v.size && need > v.size) need = v.size;
            if (!v.addr) { static const uint8_t zero[256]; sp = vk_transient_copy(zero, 256, 16); }
            else { sp = guest_copy(v.addr, need ? need : 16); span_size = need ? need : 16; }
        }
        VkDeviceSize off = sp.off;
        if (!g_vk.robust2) span_size = VK_WHOLE_SIZE;
        if (ss_on && (ss.vb_known >> s & 1) && ss.vb[s] == sp.buf && ss.vo[s] == off && ss.vsz[s] == span_size) continue;
        ss.vb[s] = sp.buf; ss.vo[s] = off; ss.vsz[s] = span_size; ss.vb_known |= 1u << s;
        vkCmdBindVertexBuffers2(cb, (uint32_t)s, 1, &sp.buf, &off, &span_size, NULL);
    }

    if (prof) prof_vtx += SDL_GetTicksNS() - prof_t2_;
    // ---- draw
    // Instanced draws with item spots among their instances (bushsanity: bushes are drawn in batches, each instance's
    // model matrix in vertex storage buffer 0, indexed by instance id): drawn as runs of instances, gold runs tinted
    // (as renderer.m; gl_InstanceIndex includes the base instance, so a run [b + k, b + k + n) draws exactly those).
    enum { MAX_RUNS = 64 };
    uint32_t run_at[MAX_RUNS], run_n[MAX_RUNS];
    bool run_gold[MAX_RUNS];
    int nruns = 0;
    if (pr->fi->uses_tint && inst > 1 && st.sbuf[0][0].addr) {
        extern bool pot_glow_active(void);
        extern bool pot_instance_gold(const float *m);
        uint64_t recs = st.sbuf[0][0].size / 64;
        if (pot_glow_active() && (uint64_t)base_inst + inst <= recs) {
            const float *m = (const float *)(uintptr_t)st.sbuf[0][0].addr;
            for (uint32_t i = 0; i < inst; i++) {
                bool g = pot_instance_gold(m + (size_t)(base_inst + i) * 16);
                if (nruns && run_gold[nruns - 1] == g) { run_n[nruns - 1]++; continue; }
                if (nruns == MAX_RUNS) { nruns = 0; break; }  // (too fragmented: draw as is)
                run_at[nruns] = i; run_n[nruns] = 1; run_gold[nruns] = g; nruns++;
            }
            if (nruns == 1 && !run_gold[0]) nruns = 0;
        }
    }
    if (!nruns) { run_at[0] = 0; run_n[0] = inst; nruns = 1; run_gold[0] = false; }
    int ck_q = -1;  // [XCDE] T143: EQUAL-depth draw check (XCDE_CHECK_EQ)
    if (ck_on() && st.ds[0] && st.ds[2] == 3 && pass_depth && oq_open < 0 && !st.raster_discard) {
        CheckQ *q = &ckq[cur_sub];
        int pi = 0;
        uint64_t vs = st.program->stages[0].code_addr, fs = st.program->stages[1].code_addr;
        while (pi < ck_nprog && (ck_prog[pi].vs != vs || ck_prog[pi].fs != fs)) pi++;
        if (pi == ck_nprog && ck_nprog < CK_PROGS) { ck_prog[pi].vs = vs; ck_prog[pi].fs = fs; ck_nprog++; }
        if (pi < CK_PROGS && q->pool && q->n < CK_MAX) {
            ck_q = (int)q->n;
            q->prog[q->n++] = (uint16_t)pi;
            vkCmdBeginQuery(cb, q->pool, (uint32_t)ck_q, 0);
        }
    }
    for (int ri = 0; ri < nruns; ri++) {
        if (nruns > 1 || run_gold[0]) {  // this run's tint: a fragment set of its own
            float rtint[8] = {0};
            extern void pot_gold_tint(float tint[8]);
            if (run_gold[ri]) pot_gold_tint(rtint);
            VkDescriptorSet rset = stage_set(pr, pr->set_layout[1], 1, 1, pr->fi, rs0, rtint, 0, 0, 0, fii);
            vk_bind_graphics_sets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pr->layout, 1, 1, &rset, 0, NULL);
        }
        if (!use_index) vkCmdDraw(cb, count, run_n[ri], first, base_inst + run_at[ri]);
        else vkCmdDrawIndexed(cb, icount, run_n[ri], 0, base_vertex, base_inst + run_at[ri]);
    }
    if (ck_q >= 0) vkCmdEndQuery(cb, ckq[cur_sub].pool, (uint32_t)ck_q);
    if (g_debug_frame) {  // [XCDE] debugging: XCDE_SPLIT_DRAWS=<pass>: from that debug pass on, one image per draw (build/pass_NNN.png)
        static int split = -2, split_to = 1 << 30;  // XCDE_SPLIT_DRAWS_TO=<pass>: stop splitting after that debug pass
        if (split == -2) {
            split = getenv("XCDE_SPLIT_DRAWS") ? atoi(getenv("XCDE_SPLIT_DRAWS")) : -1;
            if (getenv("XCDE_SPLIT_DRAWS_TO")) split_to = atoi(getenv("XCDE_SPLIT_DRAWS_TO"));
        }
        if (split >= 0 && debug_pass_no >= split && debug_pass_no <= split_to) { vk_end_pass(); RT_DIRTY(); }
    }
    stat_draws++, g_draws_total++;
    cum_draws++;
    // [XCDE] T134: a draw whose shaders store to guest memory (XCDE's grass and skinning "writer" point draws) is read by
    // the next draws' vertex fetch; the game issues no barrier in between (Maxwell runs them in order), and Vulkan gives
    // no ordering inside a render pass. Without it the depth prepass drew grass from a half-written vertex buffer, the
    // G-buffer grass then failed its EQUAL depth test: black grass. The next draw starts a new pass behind a full barrier.
    if (pr->vi->uses_gmem_store || pr->fi->uses_gmem_store || (pr->gi && pr->gi->uses_gmem_store)) {
        vk_end_pass(); RT_DIRTY(); pass_untracked = !(mem_track_mode() & 1);
        if (nwant) { if (pr->vi->uses_gmem_store) xsh_stored(0); if (pr->fi->uses_gmem_store) xsh_stored(1); }  // (T197)
    }
    if (xfb_bytes || pr->vi->uses_gmem_store || pr->fi->uses_gmem_store || (pr->gi && pr->gi->uses_gmem_store)) atomic_store(&host_read_pending, true);
    gmem_stage_finish(xfb_bytes || pr->vi->uses_gmem_store || pr->fi->uses_gmem_store || (pr->gi && pr->gi->uses_gmem_store));
    if (xfb_bytes) { vk_end_pass(); RT_DIRTY(); xsh_written(st.xfb_buf[0].addr, st.xfb_buf[0].size, xfb_at, xfb_bytes); }  // [XCDE] T175
    if (g_debug_frame && getenv("XCDE_DUMP_VB")) {  // [XCDE] debugging: what each stream holds when the GPU has run this draw
        char code[32]; snprintf(code, sizeof code, "%llx", (unsigned long long)st.program->stages[0].code_addr);
        if (strstr(getenv("XCDE_DUMP_VB"), code)) {
            vk_end_pass();
            renderer_flush();
            LOG("[vb-hash] vs %s inst %u base %u:", code, r->x[4] ? (unsigned)r->x[4] : 1u, (unsigned)r->x[3]);
            for (int s = 0; s < 17; s++) {
                if (!(k.used_streams & (1u << s)) || s == ZERO_BINDING || !dbg_vb_addr[s]) continue;
                uint64_t n = k.divisor[s] ? ((uint64_t)base_inst + inst + k.divisor[s] - 1) / k.divisor[s] * k.stride[s] : (uint64_t)(max_index ? max_index : count) * k.stride[s];
                if (!n || n > (1u << 20)) n = n ? (1u << 20) : 16;
                if (dbg_vb_len[s] && n > dbg_vb_len[s]) n = dbg_vb_len[s];
                const uint8_t *m = (const uint8_t *)(uintptr_t)dbg_vb_addr[s];
                uint64_t h = 1469598103934665603ull;
                for (uint64_t i = 0; i < n; i++) h = (h ^ m[i]) * 1099511628211ull;
                LOG(" s%d %llx+%llu=%08x", s, (unsigned long long)dbg_vb_addr[s], (unsigned long long)n, (unsigned)(h ^ (h >> 32)));
            }
            LOG("\n");
            RT_DIRTY();
        }
    }
    if (dump_fs_want() && dump_fs_want() == g_cur_fs_hash) {  // [FIX-TEX] the first bytes of every vertex stream of the first draws with that fragment shader
        static unsigned nd;
        unsigned qd = nd++;
        {   // render state of the draws with the (0,0,1,1) material vs the rest: 8 of each class, then one in 400
            static unsigned sc[2];
            unsigned cn2 = sc[g_mat_blue ? 1 : 0]++;
            if (cn2 < 8 || cn2 % 400 == 0) {
                LOG("[tex-fix] state %s#%u: ncolor %u", g_mat_blue ? "BLUE" : "other", cn2, k.ncolor);
                for (uint32_t i = 0; i < k.ncolor && i < 8; i++) LOG(" rt%u(fmt %d mask %x blend %d)", i, (int)k.color[i], k.write_mask[i], (k.blend_on >> i) & 1);
                LOG(" | depth fmt %d stencil fmt %d | ds test %u write %u fn %u stencil %u ref %u | cull %u front %u fill %d | a2c %d samples %u logic %d | vp %g %g %g %g%c",
                    (int)k.depth, (int)k.stencil, st.ds[0], st.ds[1], st.ds[2], st.ds[3], (unsigned)st.stencil_ref[0], st.poly[0], st.poly[1], k.poly_fill, k.a2c, k.samples,
                    k.logic_op, st.viewport[0][0], st.viewport[0][1], st.viewport[0][2], st.viewport[0][3], 10);
            }
        }
        if (qd < 6 || (qd % 200 == 0 && qd < 20000)) {
            LOG("[tex-fix] draw#%u layout:", qd);
            for (int a = 0; a < 32; a++) if (k.attr_mask & (1u << a)) LOG(" a%d:%u@%u/b%u", a, k.attr_fmt[a], k.attr_off[a], k.attr_bind[a]);
            for (int s2 = 0; s2 < 17; s2++) if (k.used_streams & (1u << s2)) LOG(" s%d(stride %u div %u)", s2, k.stride[s2], k.divisor[s2]);
            LOG("%c", 10);
            LOG("[tex-fix] vertex data, fs %016llx, vs %llx, count %u:", (unsigned long long)g_cur_fs_hash, (unsigned long long)st.program->stages[0].code_addr, (unsigned)count);
            for (int s = 0; s < 17; s++) {
                if (!(k.used_streams & (1u << s)) || s == ZERO_BINDING || !dbg_vb_addr[s]) continue;
                uint64_t n = (uint64_t)k.stride[s] * 3;
                if (n > 192) n = 192;
                if (dbg_vb_len[s] && n > dbg_vb_len[s]) n = dbg_vb_len[s];
                const uint8_t *m = (const uint8_t *)(uintptr_t)dbg_vb_addr[s];
                LOG(" s%d(stride %u):", s, k.stride[s]);
                for (uint64_t i = 0; i < n; i++) LOG("%s%02x", i && i % k.stride[s] == 0 ? " | " : (i ? " " : " "), m[i]);
            }
            LOG("%c", 10);
        }
    }
    if (g_debug_frame && xfb && getenv("XCDE_DUMP_XFB")) {  // [XCDE] debugging: what the capture holds once this draw ran
        vk_end_pass();
        renderer_flush();
        uint64_t at = (uint64_t)g_xfbp[0] * 4 + HEAP_BASE, bytes = (uint64_t)((uint32_t)r->x[2]) * (r->x[4] ? (uint32_t)r->x[4] : 1) * vkey.xfb_stride;
        const uint8_t *m = (const uint8_t *)(uintptr_t)at;
        uint64_t h = 1469598103934665603ull;
        for (uint64_t i = 0; i < bytes; i++) h = (h ^ m[i]) * 1099511628211ull;
        LOG("[xfb-data] %llx +%llu hash %016llx :", (unsigned long long)at, (unsigned long long)bytes, (unsigned long long)h);
        for (int i = 0; i < 24 && (uint64_t)i < bytes; i++) LOG(" %02x", m[i]);
        LOG("\n");
        RT_DIRTY();
    }
    if (g_gpuprof_frame && gp_n) {
        if (!gp_draws[gp_n - 1]) {  // first draw of the row: its fragment shader hash
            size_t l = strlen(gp_label[gp_n - 1]);
            snprintf(gp_label[gp_n - 1] + l, sizeof gp_label[0] - l, ", first fs %016llx vs %016llx (x2 %u x4 %u)", (unsigned long long)g_cur_fs_hash,
                     (unsigned long long)(pr ? pr->code[0] : 0), (unsigned)r->x[2], (unsigned)r->x[4]);  // [GPUPROF-SLOW] + the vertex shader and counts
        }
        gp_draws[gp_n - 1]++;
    }
}

static void debug_draw(Prog *pr, CmdRec *r, bool indexed) {
    if ((st.program->stage_mask & 4) && getenv("XCDE_DUMP_GS")) {  // [XCDE] the geometry shader binary (header + code)
        char fn[128]; snprintf(fn, sizeof fn, "build/gs_%llx.bin", (unsigned long long)st.program->stages[2].code_addr);
        FILE *f = fopen(fn, "wb");
        if (f) { fwrite((const void *)(uintptr_t)(st.program->stages[2].code_addr - 0x80), 1, 0x1080, f); fclose(f); }
    }
    if (st.program->stage_mask & ~3u) LOG("[draw] NOTE program stage mask %#x (stages beyond VS/FS: gs %llx)\n", st.program->stage_mask, (unsigned long long)st.program->stages[2].code_addr);
    LOG("[draw] (pass %d) %s inst %llu mode %llu count %llu prog vs=%llx fs=%llx | vs cb %#x fs cb %#x | rt %d", debug_pass_no, indexed ? "e" : "a",
        (unsigned long long)(indexed ? r->x[6] : r->x[4]), (unsigned long long)r->x[0], (unsigned long long)r->x[2],
        (unsigned long long)st.program->stages[0].code_addr, (unsigned long long)st.program->stages[1].code_addr,
        pr->vi->cbuf_mask, pr->fi->cbuf_mask, st.rt.ncolor);
    for (int i = 0; i < st.rt.ncolor; i++)
        if (st.rt.color[i]) LOG(" %ux%u/f%#x@%llx[%p/%p]", st.rt.color[i]->width, st.rt.color[i]->height, st.rt.color[i]->format,
                                (unsigned long long)st.rt.color[i]->addr, (void *)st.rt.color[i], st.rt.color[i]->backend),
                                st.rt.has_cview[i] ? LOG("{tex tgt %u d %u layers %u levels %u; vk layers %u | view m%#x lv %u+%u ly %u+%u tgt %u}", st.rt.color[i]->target, st.rt.color[i]->depth, st.rt.color[i]->layout.layers, st.rt.color[i]->levels, st.rt.color[i]->backend ? ((TexBackend *)st.rt.color[i]->backend)->layers : 0, st.rt.cview[i].setmask, st.rt.cview[i].base_level,
                                                         st.rt.cview[i].num_levels, st.rt.cview[i].min_layer, st.rt.cview[i].num_layers,
                                                         st.rt.cview[i].target) : 0;
    if (st.rt.depth) {
        LOG(" d%#x %p", st.rt.depth->format, (void *)st.rt.depth);
        if (getenv("XCDE_TRACE_RT"))  // [XCDE] the depth target's storage and view (layer / level), for aliasing questions
            LOG(" img %p @%llx %ux%ux%u t%u lv%u view %d/%#x L%u+%u M%u",
                st.rt.depth->backend ? (void *)((TexBackend *)st.rt.depth->backend)->image : NULL, (unsigned long long)st.rt.depth->addr, st.rt.depth->width,
                st.rt.depth->height, st.rt.depth->depth, st.rt.depth->target, st.rt.depth->levels, st.rt.has_dview,
                st.rt.dview.setmask, st.rt.dview.min_layer, st.rt.dview.num_layers, st.rt.dview.base_level);
    }
    if (st.color_state[1] != 3) LOG(" | logicop %u", st.color_state[1]);
    if (st.color_state[2] != 8) LOG(" | atest %u ref %g", st.color_state[2], st.alpha_ref);
    LOG(" | vp %.0f %.0f %.0f %.0f | attrs %#x blend %#x ds %02x%02x%02x poly %02x%02x chmask %08x\n", st.viewport[0][0], st.viewport[0][1],
        st.viewport[0][2], st.viewport[0][3], pr->vi->attr_mask, st.color_state[0], st.ds[0], st.ds[1], st.ds[2], st.poly[0], st.poly[1],
        st.channel_mask);
    if (st.ds[3])  // [XCDE] stencil state (debug frame)
        LOG("[draw]   stencil ops %02x %02x %02x funcs %02x vmask %02x/%02x wmask %02x/%02x ref %02x/%02x\n", st.ds[5], st.ds[6],
            st.ds[7], st.ds[4], st.stencil_vmask[0] & 0xFF, st.stencil_vmask[1] & 0xFF, st.stencil_mask[0] & 0xFF,
            st.stencil_mask[1] & 0xFF, st.stencil_ref[0] & 0xFF, st.stencil_ref[1] & 0xFF);
    if (getenv("XCDE_TRACE_ATTR")) {  // [XCDE] vertex attributes: format, stream, offset, stride and the first vertex's bytes
        for (int i = 0; i < 32; i++) {
            if (!(pr->vi->attr_mask & (1u << i)) || i >= st.nattrib) continue;
            uint32_t a = st.attrib[i], fmt = a & 0xFF, stream = (a >> 8) & 0x1F, off = a >> 13;
            uint32_t stride = stream < (uint32_t)st.nstream ? st.stream[stream][0] : 0;
            LOG("[attr]   a%d fmt %#x stream %u off %u stride %u vbuf %llx+%llx :", i, fmt, stream, off, stride,
                stream < 16 ? (unsigned long long)st.vbuf[stream].addr : 0ull, stream < 16 ? (unsigned long long)st.vbuf[stream].size : 0ull);
            if (stream < 16 && st.vbuf[stream].addr) {
                const uint8_t *p = (const uint8_t *)(uintptr_t)st.vbuf[stream].addr + off;
                for (int k = 0; k < 16; k++) LOG(" %02x", p[k]);
            }
            LOG("\n");
        }
    }
    if (getenv("LARECOMP_DEBUG_DATA")) {  // vertex streams 0-1 and fragment cbufs: the first floats
        for (int s = 0; s < 2; s++) {
            if (!st.vbuf[s].addr) continue;
            const float *v = (const float *)(uintptr_t)st.vbuf[s].addr;
            LOG("[draw]   vbuf%d %llx size %llu:", s, (unsigned long long)st.vbuf[s].addr, (unsigned long long)st.vbuf[s].size);
            for (int k = 0; k < 16; k++) LOG(" %.4g", v[k]);
            LOG("\n");
        }
        for (int c = 3; c < 18; c++) {  // [XCDE] the vertex stage's too
            if (!((pr->vi->cbuf_mask >> c) & 1) || !st.ubuf[0][c - 3].addr) continue;
            const float *u = (const float *)(uintptr_t)st.ubuf[0][c - 3].addr;
            LOG("[draw]   vcb%d %llx size %u:", c, (unsigned long long)st.ubuf[0][c - 3].addr, (unsigned)st.ubuf[0][c - 3].size);
            for (int k = 0; k < 24 && k * 4 < (int)st.ubuf[0][c - 3].size; k++) LOG(" %.6g", u[k]);
            LOG("\n");
        }
        for (int c = 3; c < 18; c++) {
            if (!((pr->fi->cbuf_mask >> c) & 1) || !st.ubuf[1][c - 3].addr) continue;
            const float *u = (const float *)(uintptr_t)st.ubuf[1][c - 3].addr;
            LOG("[draw]   fcb%d %llx size %u:", c, (unsigned long long)st.ubuf[1][c - 3].addr, (unsigned)st.ubuf[1][c - 3].size);
            for (int k = 0; k < 96 && k * 4 < (int)st.ubuf[1][c - 3].size; k++) LOG(" %.6g", u[k]);
            LOG("\n");
        }
    }
    const int dpass = debug_pass_no;  // (saving a texture ends the pass and advances the counter)
    if (getenv("XCDE_STENCIL_HIST") && atoi(getenv("XCDE_STENCIL_HIST")) == dpass) {
        for (int i = 0; i < pr->fi->ntex && i < 64; i++) {
            NvnTexture *t = shader_texture_object(1, &pr->fi->tex[i]);
            if (t && t->backend && renderer_format_has_stencil(t->format)) { debug_stencil_hist((TexBackend *)t->backend, "sampled"); }
        }
    }
    if (getenv("XCDE_DEPTH_HIST") && atoi(getenv("XCDE_DEPTH_HIST")) == dpass && st.rt.depth && st.rt.depth->backend) {
        static int done; if (!done++) debug_depth_hist((TexBackend *)st.rt.depth->backend, "light pass");
    }
    for (int i = 0; i < pr->fi->ntex && i < 64; i++) {  // fragment textures: guest size / format, allocation, scale
        NvnTexture *t = shader_texture_object(1, &pr->fi->tex[i]);
        const TexBackend *b = t ? (const TexBackend *)t->backend : NULL;
        if (!t) { LOG("[draw]   tex%d: none\n", i); continue; }
        LOG("[draw]   tex%d: %p %ux%u f%#x lv %u -> %ux%u scale %.4f %.4f%s addr %llx stride %llu flags %#x '%s'\n", i, (void *)t, t->width, t->height, t->format,
            b ? b->levels : 0, b ? b->width : 0, b ? b->height : 0, b ? b->sx : 0, b ? b->sy : 0, b && b->scalable ? " scalable" : "",
            (unsigned long long)t->addr, (unsigned long long)t->stride, t->flags, t->label);
        if (getenv("XCDE_TRACE_RT"))
            LOG("[draw]     tex%d target %u depth %u img %p\n", i, t->target, t->depth, b ? (void *)b->image : NULL);
        // [XCDE] each distinct sampled texture once, as seen by the shader (decoded): XCDE_DUMP_TEX=<pass>, or
        // XCDE_DUMP_TEX_ADDR=<guest address hex> for one texture wherever the debug frame samples it
        bool by_addr = getenv("XCDE_DUMP_TEX_ADDR") && strtoull(getenv("XCDE_DUMP_TEX_ADDR"), NULL, 16) == t->addr;
        if (b && (by_addr || (getenv("XCDE_DUMP_TEX") && atoi(getenv("XCDE_DUMP_TEX")) == dpass))) {
            static void *seen[96]; static int nseen;
            bool dup = false;
            for (int k = 0; k < nseen; k++) dup |= seen[k] == (void *)t;
            if (!dup && nseen < 96) {
                char path[256];
                seen[nseen++] = (void *)t;
                snprintf(path, sizeof path, "build/tex_p%03d_u%02d_%ux%u_f%x_%llx.png", dpass, i, t->width, t->height, t->format, (unsigned long long)t->addr);
                renderer_save_texture_tex((TexBackend *)b, path);
            }
        }
    }
}

// ================================================================== texture copies
typedef struct { uint64_t tex; NvnTextureView view; uint8_t has_view; int32_t region[6]; } TexRefRec;

static void copy_region_dims(NvnTexture *t, const TexRefRec *ref, const FormatInfo **fi, int32_t rg[6], int *level,
                             uint64_t *row, uint64_t *img) {
    *fi = nvn_format_info(t->format);
    memcpy(rg, ref->region, sizeof ref->region);
    if (!rg[3]) rg[3] = (int32_t)t->width;
    if (!rg[4]) rg[4] = (int32_t)t->height;
    if (!rg[5]) rg[5] = 1;
    *level = ref->has_view && (ref->view.setmask & 1) ? ref->view.base_level : 0;
    uint32_t wb = ((uint32_t)rg[3] + (*fi)->bw - 1) / (*fi)->bw, hb = ((uint32_t)rg[4] + (*fi)->bh - 1) / (*fi)->bh;
    *row = st.copy_row_stride ? st.copy_row_stride : (uint64_t)wb * (*fi)->bpb;
    *img = st.copy_image_stride ? st.copy_image_stride : *row * hb;
}

static void copy_buffer_to_texture(uint64_t src, const TexRefRec *dst) {
    snap_reason = "buf2tex"; snapshots_invalidate(NULL);
    NvnTexture *t = (NvnTexture *)(uintptr_t)dst->tex;
    if (!t || !src) return;
    alias_before_write(t, 0, "upload");
    vk_end_pass();
    const FormatInfo *fi;
    int32_t rg[6];
    int level;
    uint64_t row, img;
    copy_region_dims(t, dst, &fi, rg, &level, &row, &img);
    if (g_debug_frame) LOG("[copy] buf->tex %p fmt %#x region %d %d %d %dx%dx%d src %llx row %llu\n", (void *)t, t->format, rg[0], rg[1], rg[2], rg[3], rg[4], rg[5], (unsigned long long)src, (unsigned long long)row);
    { static int tr = -1, n; if (tr < 0) tr = getenv("XCDE_TRACE_MOVIECOPY") != NULL;
      if (tr && src >= HLE_HEAP && src < HLE_HEAP + HLE_HEAP_SIZE && n++ < 40)
          LOG("[copy] movie buf->tex %p fmt %#x %ux%u region %d %d %dx%d src %llx row %llu bpb %u\n", (void *)t, t->format, t->width, t->height, rg[0], rg[1], rg[3], rg[4], (unsigned long long)src, (unsigned long long)row, fi ? fi->bpb : 0); }
    if (!fi || !fi->bpb) return;
    TexBackend *b = vk_tex(t, false);  // prior CPU contents uploaded first
    renderer_mark_gpu_owned(t);
    if (b->aspect == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) return;
    if (b->astc_bw) {  // (blocks would need the CPU decode: not seen in the game yet)
        static int nlog;
        if (nlog++ < 5) LOG("[vk] buffer -> ASTC texture copy skipped (%ux%u)\n", t->width, t->height);
        return;
    }
    bool array = b->layers > 1 && b->depth == 1;
    uint32_t hb = ((uint32_t)rg[4] + fi->bh - 1) / fi->bh, wb = ((uint32_t)rg[3] + fi->bw - 1) / fi->bw;
    uint32_t out_bpb = fi->bpb == 3 ? 4 : fi->bpb;
    uint32_t nz = (uint32_t)rg[5];
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    uint64_t out_row = (uint64_t)wb * out_bpb;
    VkSpan s = vk_transient(out_row * hb * nz, 16);
    for (uint32_t z = 0; z < nz; z++)
        for (uint32_t y = 0; y < hb; y++) {
            const uint8_t *sr = (const uint8_t *)(uintptr_t)(src + (uint64_t)z * img + (uint64_t)y * row);
            uint8_t *dr = (uint8_t *)s.ptr + ((uint64_t)z * hb + y) * out_row;
            if (out_bpb == fi->bpb) memcpy(dr, sr, out_row);
            else for (uint32_t x = 0; x < wb; x++) { memcpy(dr + x * 4, sr + x * 3, 3); dr[x * 4 + 3] = 255; }
        }
    VkBufferImageCopy c = {.bufferOffset = s.off, .bufferRowLength = wb * fi->bw, .bufferImageHeight = hb * fi->bh,
                           .imageSubresource = {b->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : b->aspect,
                                                (uint32_t)level, array ? (uint32_t)rg[2] : 0, array ? nz : 1},
                           .imageOffset = {rg[0], rg[1], array ? 0 : rg[2]},
                           .imageExtent = {(uint32_t)rg[3], (uint32_t)rg[4], array ? 1 : nz}};
    vk_watch(b->image, "exec.c:3324 vkCmdCopyBufferToImage"); vkCmdCopyBufferToImage(cb, s.buf, b->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
}

static void copy_texture_to_buffer(const TexRefRec *src, uint64_t dst) {
    NvnTexture *t = (NvnTexture *)(uintptr_t)src->tex;
    if (!t || !dst || dst < HEAP_BASE) return;
    t = alias_read(t);
    vk_end_pass();
    const FormatInfo *fi;
    int32_t rg[6];
    int level;
    uint64_t row, img;
    copy_region_dims(t, src, &fi, rg, &level, &row, &img);
    if (!fi || !fi->bpb || fi->bpb == 3 || row % fi->bpb) return;
    TexBackend *b = vk_tex(t, false);
    float tsx, tsy;
    renderer_texture_scale(t, &tsx, &tsy);
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    VkImage srcimg = b->image;
    VmaAllocation small_mem = NULL;
    if ((tsx != 1 || tsy != 1) && level == 0 && !renderer_format_is_depth(t->format)) {
        // the CPU expects guest-sized texels: filter the scaled allocation down first
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = b->format,
                                .extent = {t->width, t->height, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                                .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &srcimg, &small_mem, NULL));
        vk_image_init_layout(cb, srcimg, VK_IMAGE_ASPECT_COLOR_BIT);
        VkImageBlit bl = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .srcOffsets = {{0, 0, 0}, {(int32_t)b->width, (int32_t)b->height, 1}},
                          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{0, 0, 0}, {(int32_t)t->width, (int32_t)t->height, 1}}};
        vk_watch(srcimg, "exec.c:3355 vkCmdBlitImage"); vkCmdBlitImage(cb, b->image, VK_IMAGE_LAYOUT_GENERAL, srcimg, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, VK_FILTER_LINEAR);
        vk_full_barrier(cb);
    }
    bool array = b->layers > 1 && b->depth == 1;
    uint32_t nz = (uint32_t)rg[5];
    uint64_t total = img * nz;
    VkSpan out = vk_readback(total);
    VkBufferImageCopy c = {.bufferOffset = out.off, .bufferRowLength = (uint32_t)(row / fi->bpb) * fi->bw,
                           .bufferImageHeight = (uint32_t)(img / row) * fi->bh,
                           .imageSubresource = {b->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : b->aspect,
                                                (uint32_t)level, array ? (uint32_t)rg[2] : 0, array ? nz : 1},
                           .imageOffset = {rg[0], rg[1], array ? 0 : rg[2]},
                           .imageExtent = {(uint32_t)rg[3], (uint32_t)rg[4], array ? 1 : nz}};
    vkCmdCopyImageToBuffer(cb, srcimg, VK_IMAGE_LAYOUT_GENERAL, out.buf, 1, &c);
    wait_why = "texture -> buffer readback";
    renderer_flush();  // the CPU may read the result right after the fence
    memcpy((void *)(uintptr_t)dst, out.ptr, total);
    if (small_mem) vmaDestroyImage(g_vk.vma, srcimg, small_mem);
}

// [XCDE] T181: the save thumbnail. The game draws a 320x180 RGBA8 target once (one quad) and then reads its guest
// storage on the CPU (block-linear, as the GPU left it) to build the .tmb file; our draws only reach the VkImage, so the
// thumbnail was all zeros. After that draw: wait for the GPU, read level 0 back at guest size and swizzle it into the
// texture's storage. Only for the thumbnail program (fs content hash), once per save: one short stall.
#include "../tegra.h"
static void writeback_texture(NvnTexture *t) {
    t = alias_read(t);
    const FormatInfo *fi = t->layout.fmt;
    if (!t->addr || !fi || fi->bw != 1 || fi->bh != 1 || fi->bpb != 4 || renderer_format_is_depth(t->format)) return;
    vk_end_pass();
    TexBackend *b = vk_tex(t, false);
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    VkImage srcimg = b->image;
    VmaAllocation small_mem = NULL;
    if (b->width != t->width || b->height != t->height) {  // scaled allocation: filter it down to the guest size
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = b->format,
                                .extent = {t->width, t->height, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                                .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &srcimg, &small_mem, NULL));
        vk_image_init_layout(cb, srcimg, VK_IMAGE_ASPECT_COLOR_BIT);
        VkImageBlit bl = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .srcOffsets = {{0, 0, 0}, {(int32_t)b->width, (int32_t)b->height, 1}},
                          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{0, 0, 0}, {(int32_t)t->width, (int32_t)t->height, 1}}};
        vkCmdBlitImage(cb, b->image, VK_IMAGE_LAYOUT_GENERAL, srcimg, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, VK_FILTER_LINEAR);
        vk_full_barrier(cb);
    }
    uint64_t total = (uint64_t)t->width * t->height * 4;
    VkSpan out = vk_readback(total);
    VkBufferImageCopy c = {.bufferOffset = out.off, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {t->width, t->height, 1}};
    vkCmdCopyImageToBuffer(cb, srcimg, VK_IMAGE_LAYOUT_GENERAL, out.buf, 1, &c);
    wait_why = "thumbnail readback";
    renderer_flush();
    LevelLayout l = tegra_level(&t->layout, 0);
    tegra_swizzle(out.ptr, (uint8_t *)(uintptr_t)t->addr + l.offset, l.wb, l.hb, 1, 4, l.block_height_log2, l.block_depth_log2);
    if (small_mem) vmaDestroyImage(g_vk.vma, srcimg, small_mem);
    LOG("[t181] thumbnail %ux%u written back to guest storage %llx\n", t->width, t->height, (unsigned long long)t->addr);
}

static void copy_texture(const TexRefRec *src, const TexRefRec *dst, uint64_t flags) {
    if (g_debug_frame) LOG("[copy] tex->tex %p (fmt %#x) -> %p (fmt %#x) flags %llx\n", (void *)(uintptr_t)src->tex, src->tex ? ((NvnTexture *)(uintptr_t)src->tex)->format : 0, (void *)(uintptr_t)dst->tex, dst->tex ? ((NvnTexture *)(uintptr_t)dst->tex)->format : 0, (unsigned long long)flags);
    snap_reason = "tex2tex"; snapshots_invalidate(NULL);
    NvnTexture *st_ = (NvnTexture *)(uintptr_t)src->tex, *dt = (NvnTexture *)(uintptr_t)dst->tex;
    if (!st_ || !dt) return;
    vk_end_pass();
    if (dl_on == 2) renderer_flush();  // (LARECOMP_VK_DEVLOST_TRACE=2: each copy its own submission, waited on)
    gp_mark(vk_cmd(), "copy texture %ux%u -> %ux%u", st_->width, st_->height, dt->width, dt->height);
    dl_note("copy texture %ux%u f%#x -> %ux%u f%#x (%p -> %p)", st_->width, st_->height, st_->format, dt->width, dt->height,
            dt->format, (void *)st_, (void *)dt);
    TexBackend *s = vk_tex(st_, false), *d = vk_tex(dt, true);
    renderer_mark_gpu_owned(dt);
    if (renderer_texture_presented(dt) && st_->width == dt->width && st_->height == dt->height) renderer_texture_mark_presented(st_);
    renderer_texture_first_write(dt);
    renderer_texture_level_fresh(dt);
    const int32_t *sr = src->region, *dr = dst->region;
    float ssx, ssy, dsx, dsy;
    renderer_texture_scale(st_, &ssx, &ssy);
    renderer_texture_scale(dt, &dsx, &dsy);
    int sx0 = (int)lroundf(sr[0] * ssx), sy0 = (int)lroundf(sr[1] * ssy);
    int dx0 = (int)lroundf(dr[0] * dsx), dy0 = (int)lroundf(dr[1] * dsy);
    int sw = sr[3] ? (int)lroundf(sr[3] * ssx) : (int)s->width, sh = sr[4] ? (int)lroundf(sr[4] * ssy) : (int)s->height;
    int dw = dr[3] ? (int)lroundf(dr[3] * dsx) : (int)d->width, dh = dr[4] ? (int)lroundf(dr[4] * dsy) : (int)d->height;
    uint32_t slv = src->has_view && (src->view.setmask & 1) ? src->view.base_level : 0;
    uint32_t dlv = dst->has_view && (dst->view.setmask & 1) ? dst->view.base_level : 0;
    uint32_t sly = src->has_view && (src->view.setmask & 2) ? src->view.min_layer : 0;
    uint32_t dly = dst->has_view && (dst->view.setmask & 2) ? dst->view.min_layer : 0;
    if (slv >= s->levels || dlv >= d->levels || sly >= s->layers || dly >= d->layers) return;
    {   // whole-image regions are the level's extent, and every region stays inside its level: scaled allocations
        // (internal resolution below 1x) round guest regions past the edge, and out-of-bounds copies / blits are
        // undefined in Vulkan (garbage that changed every frame on NVIDIA: flicker in the depth pyramid's users)
        int slw = (int)level_extent(s->width, slv), slh = (int)level_extent(s->height, slv);
        int dlw = (int)level_extent(d->width, dlv), dlh = (int)level_extent(d->height, dlv);
        if (!sr[3]) sw = slw - sx0;
        if (!sr[4]) sh = slh - sy0;
        if (!dr[3]) dw = dlw - dx0;
        if (!dr[4]) dh = dlh - dy0;
        if (sx0 + sw > slw) sw = slw - sx0;
        if (sy0 + sh > slh) sh = slh - sy0;
        if (dx0 + dw > dlw) dw = dlw - dx0;
        if (dy0 + dh > dlh) dh = dlh - dy0;
        if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) return;
    }
    if (g_debug_frame || ct_on())
        LOG("[copytex] %p %ux%u f%#x lv%u (%d,%d %dx%d) [%ux%u] -> %p %ux%u f%#x lv%u (%d,%d %dx%d) [%ux%u] phys (%d,%d %dx%d) -> (%d,%d %dx%d) flags %llx\n",
            (void *)st_, st_->width, st_->height, st_->format, slv, sr[0], sr[1], sr[3], sr[4], s->width, s->height, (void *)dt, dt->width,
            dt->height, dt->format, dlv, dr[0], dr[1], dr[3], dr[4], d->width, d->height, sx0, sy0, sw, sh, dx0, dy0, dw, dh,
            (unsigned long long)flags);
    if (copy_opt() && sx0 == 0 && sy0 == 0 && dx0 == 0 && dy0 == 0 && sw == (int)s->width && sh == (int)s->height && dw == (int)d->width &&
        dh == (int)d->height && slv == 0 && dlv == 0 && sly == 0 && dly == 0 && alias_make(st_, dt, s, d)) {
        if (ct_on()) LOG("[ct] alias made %p -> %p\n", (void *)st_, (void *)dt);
        return;
    }
    alias_before_write(dt, 0, "copy dst");
    if (naliases) s = alias_read(st_)->backend;
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    VkImageAspectFlags sa = s->aspect, da = d->aspect;
    if (s->samples > 1 && d->samples == 1 && sw == dw && sh == dh && s->format == d->format && !(sa & VK_IMAGE_ASPECT_DEPTH_BIT)) {
        VkImageResolve rv = {.srcSubresource = {sa, slv, sly, 1}, .srcOffset = {sx0, sy0, 0}, .dstSubresource = {da, dlv, dly, 1},
                             .dstOffset = {dx0, dy0, 0}, .extent = {(uint32_t)sw, (uint32_t)sh, 1}};
        vk_watch(d->image, "exec.c:3436 vkCmdResolveImage"); vkCmdResolveImage(cb, s->image, VK_IMAGE_LAYOUT_GENERAL, d->image, VK_IMAGE_LAYOUT_GENERAL, 1, &rv);
        return;
    }
    if (s->samples != d->samples) return;
    if (sw == dw && sh == dh && s->format == d->format) {
        VkImageCopy c = {.srcSubresource = {sa, slv, sly, 1}, .srcOffset = {sx0, sy0, 0}, .dstSubresource = {da, dlv, dly, 1},
                         .dstOffset = {dx0, dy0, 0}, .extent = {(uint32_t)sw, (uint32_t)sh, 1}};
        vk_watch(d->image, "exec.c:3443 vkCmdCopyImage"); vkCmdCopyImage(cb, s->image, VK_IMAGE_LAYOUT_GENERAL, d->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
        return;
    }
    // scaled / converting copy: the game builds its depth pyramid this way (nearest), colour copies filter if asked
    bool depth = (sa | da) & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
    if (depth && s->format != d->format) return;
    VkImageBlit bl = {.srcSubresource = {depth ? (sa & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : sa) : sa, slv, sly, 1},
                      .srcOffsets = {{sx0, sy0, 0}, {sx0 + sw, sy0 + sh, 1}},
                      .dstSubresource = {depth ? (da & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : da) : da, dlv, dly, 1},
                      .dstOffsets = {{dx0, dy0, 0}, {dx0 + dw, dy0 + dh, 1}}};
    bool linear = (flags & 1) && !depth && !rt_kind(st_->format);
    {   // [XCDE] T240: a blit needs BLIT_SRC / BLIT_DST on both formats (optimal tiling). NVIDIA has them for the depth
        // formats; other drivers needn't (PokeZ-ARecomp 71ab3ff: RADV's D32_SFLOAT_S8_UINT, which non-NVIDIA GPUs get
        // instead of D24S8) - an unsupported blit is undefined, a GPU fault at worst. Logged once per format and skipped.
        static uint8_t known[256];  // per format: 0 unknown, bit 0 known, bit 1 BLIT_SRC, bit 2 BLIT_DST
        uint8_t fs = 7, fd = 7;
        for (int k = 0; k < 2; k++) {
            VkFormat f = k ? d->format : s->format;
            uint8_t *slot = (unsigned)f < 256 ? &known[f] : NULL, v = slot ? *slot : 0;
            if (!v) {
                VkFormatProperties fp;
                vkGetPhysicalDeviceFormatProperties(g_vk.phys, f, &fp);
                v = 1 | (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT ? 2 : 0) | (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT ? 4 : 0);
                if (slot) *slot = v;
            }
            if (k) fd = v; else fs = v;
        }
        bool ok = (fs & 2) && (fd & 4);
        if (!ok) {
            static int nl;
            if (nl++ < 8) LOG("[vk] blit %d -> %d not supported by the GPU (depth %d): skipped\n", (int)s->format, (int)d->format, (int)depth);
            return;
        }
    }
    dl_note("  blit %p lv%u/%u %ux%u (%d,%d %dx%d) -> %p lv%u/%u %ux%u (%d,%d %dx%d) %s", (void *)s->image, slv, s->levels, s->width,
            s->height, sx0, sy0, sw, sh, (void *)d->image, dlv, d->levels, d->width, d->height, dx0, dy0, dw, dh, linear ? "linear" : "nearest");
    vk_watch(d->image, "exec.c:3456 vkCmdBlitImage"); vkCmdBlitImage(cb, s->image, VK_IMAGE_LAYOUT_GENERAL, d->image, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
}

// ================================================================== present helpers (device.c)
static VkPipelineLayout present_layout;
static VkDescriptorSetLayout present_set_layout;

static VkPipeline present_pipeline(VkFormat fmt) {
    static struct { VkFormat f; VkPipeline p; } cache[4];
    for (int i = 0; i < 4; i++) if (cache[i].p && cache[i].f == fmt) return cache[i].p;
    if (!present_layout) {
        VkDescriptorSetLayoutBinding b = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL};
        VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1, .pBindings = &b};
        VK_CHECK(vkCreateDescriptorSetLayout(g_vk.dev, &li, NULL, &present_set_layout));
        VkPushConstantRange r = {VK_SHADER_STAGE_VERTEX_BIT, 0, 32};
        VkPipelineLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                         .pSetLayouts = &present_set_layout, .pushConstantRangeCount = 1, .pPushConstantRanges = &r};
        VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &present_layout));
    }
    VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vk_builtin_shader("present.vert"), .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = vk_builtin_shader("present.frag"), .pName = "main"}};
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineViewportStateCreateInfo vs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState cba = {.colorWriteMask = 0xF};
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1,
                                              .pAttachments = &cba};
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 2, .pDynamicStates = dyn};
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = 1,
                                        .pColorAttachmentFormats = &fmt};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri, .stageCount = 2,
                                       .pStages = stages, .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vs,
                                       .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &ds,
                                       .pColorBlendState = &cb, .pDynamicState = &dy, .layout = present_layout};
    VkPipeline p;
    VK_CHECK(vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, &pi, NULL, &p));
    for (int i = 0; i < 4; i++) if (!cache[i].p) { cache[i].f = fmt; cache[i].p = p; break; }
    return p;
}

// Scanout shows the bytes the game wrote: sRGB images are read through a UNORM view (no decode), like metal.m.
static VkImageView raw_view(TexBackend *b) {
    VkFormat f = b->format == VK_FORMAT_R8G8B8A8_SRGB ? VK_FORMAT_R8G8B8A8_UNORM
               : b->format == VK_FORMAT_B8G8R8A8_SRGB ? VK_FORMAT_B8G8R8A8_UNORM : b->format;
    static const VkComponentMapping id = {0};
    return vk_image_view(b, VK_IMAGE_VIEW_TYPE_2D, f, 0, 1, 0, 1, id, VK_IMAGE_ASPECT_COLOR_BIT);
}

// ---- post passes (Video tab "Anti-aliasing" / "Upscaler"): SMAA 1x at the game image's size, then AMD FSR 1 (EASU
// upscale to the image's size in the window, RCAS sharpen) when it is smaller than its place in the window, as
// metal.m does with MetalFX. Each pass: a fullscreen triangle into an RGBA8 image (gamma-encoded, like the input);
// up to three inputs (bindings 0-2), sampled linear / clamp.
static VkPipelineLayout post_layout;
static VkDescriptorSetLayout post_set_layout;

static VkPipeline post_pipeline(const char *fs) {
    static struct { const char *fs; VkPipeline p; } cache[8];
    for (int i = 0; i < 8; i++) if (cache[i].p && cache[i].fs == fs) return cache[i].p;
    if (!post_layout) {
        VkDescriptorSetLayoutBinding b[3];
        for (uint32_t i = 0; i < 3; i++)
            b[i] = (VkDescriptorSetLayoutBinding){i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL};
        VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 3, .pBindings = b};
        VK_CHECK(vkCreateDescriptorSetLayout(g_vk.dev, &li, NULL, &post_set_layout));
        VkPushConstantRange r = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32};
        VkPipelineLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                         .pSetLayouts = &post_set_layout, .pushConstantRangeCount = 1, .pPushConstantRanges = &r};
        VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &post_layout));
    }
    VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vk_builtin_shader("post.vert"), .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = vk_builtin_shader(fs), .pName = "main"}};
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineViewportStateCreateInfo vs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState cba = {.colorWriteMask = 0xF};
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1,
                                              .pAttachments = &cba};
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 2, .pDynamicStates = dyn};
    VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = 1,
                                        .pColorAttachmentFormats = &fmt};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri, .stageCount = 2,
                                       .pStages = stages, .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vs,
                                       .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &ds,
                                       .pColorBlendState = &cb, .pDynamicState = &dy, .layout = post_layout};
    VkPipeline p;
    VK_CHECK(vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, &pi, NULL, &p));
    for (int i = 0; i < 8; i++) if (!cache[i].p) { cache[i].fs = fs; cache[i].p = p; break; }
    return p;
}

// Post image `i` at w x h (RGBA8, kept until the size changes). 0, 1: SMAA edges / weights, 2: SMAA output,
// 3, 4: FSR EASU / RCAS.
static TexBackend *post_image(int i, uint32_t w, uint32_t h) {
    static TexBackend img[6];  // (5: T213 the upscaled scene with the HUD)
    TexBackend *d = &img[i];
    if (d->image && (d->width != w || d->height != h)) {
        vk_views_forget(d);
        vk_defer_image(d->image, d->mem);
        d->image = VK_NULL_HANDLE;
        d->mem = VK_NULL_HANDLE;
    }
    if (!d->image) {
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                                .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1,
                                .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &d->image, &d->mem, NULL));
        d->format = VK_FORMAT_R8G8B8A8_UNORM; d->width = w; d->height = h; d->depth = 1; d->layers = 1; d->levels = 1;
        d->samples = 1; d->aspect = VK_IMAGE_ASPECT_COLOR_BIT; d->base_type = VK_IMAGE_VIEW_TYPE_2D;
        vk_image_init_layout(vk_cmd(), d->image, d->aspect);
    }
    return d;
}

static VkImageView post_view(TexBackend *d) {
    static const VkComponentMapping id = {0};
    return vk_image_view(d, VK_IMAGE_VIEW_TYPE_2D, d->format, 0, 1, 0, 1, id, VK_IMAGE_ASPECT_COLOR_BIT);
}

// SMAA's lookup textures (runtime/nvn/smaa_data.c): 0 = area (160 x 560 R8G8), 1 = search (64 x 16 R8), uploaded once.
static VkImageView smaa_lut(int i) {
    static TexBackend t[2];
    TexBackend *d = &t[i];
    if (!d->image) {
        extern const unsigned char *smaa_area_tex(void);
        extern const unsigned char *smaa_search_tex(void);
        uint32_t w = i ? 64 : 160, h = i ? 16 : 560, bpp = i ? 1 : 2;
        VkFormat f = i ? VK_FORMAT_R8_UNORM : VK_FORMAT_R8G8_UNORM;
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = f,
                                .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                                .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &d->image, &d->mem, NULL));
        d->format = f; d->width = w; d->height = h; d->depth = 1; d->layers = 1; d->levels = 1;
        d->samples = 1; d->aspect = VK_IMAGE_ASPECT_COLOR_BIT; d->base_type = VK_IMAGE_VIEW_TYPE_2D;
        VkCommandBuffer cb = vk_cmd();
        vk_image_init_layout(cb, d->image, d->aspect);
        VkSpan s = vk_transient_copy(i ? smaa_search_tex() : smaa_area_tex(), (uint64_t)w * h * bpp, 16);
        VkBufferImageCopy c = {.bufferOffset = s.off, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {w, h, 1}};
        vk_watch(d->image, "exec.c:3617 vkCmdCopyBufferToImage"); vkCmdCopyBufferToImage(cb, s.buf, d->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    }
    return post_view(d);
}

// One post pass (outside the window image's rendering): srcs (bindings 0..n-1) -> dst, fragment shader `fs`.
// clear: start from zero (SMAA's edge pass only writes edge pixels; its weights / output read them).
static void post_pass(VkCommandBuffer cb, const char *fs, const VkImageView *srcs, int n, TexBackend *dst, const float pc[8],
                      bool clear) {
    VkPipeline pipe = post_pipeline(fs);  // (before the rendering begins: created inside it, the NVIDIA driver crashed)
    vk_full_barrier(cb);
    VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = post_view(dst),
                                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                    .loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                    .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {dst->width, dst->height}},
                          .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &ca};
    vkCmdBeginRendering(cb, &ri);
    VkDescriptorSet ds = vk_alloc_set(post_set_layout);
    VkDescriptorImageInfo ii[3];
    VkWriteDescriptorSet w[3];
    for (int i = 0; i < n; i++) {
        ii[i] = (VkDescriptorImageInfo){vk_default_sampler(true), srcs[i], VK_IMAGE_LAYOUT_GENERAL};
        w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = (uint32_t)i,
                                      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                      .pImageInfo = &ii[i]};
    }
    vkUpdateDescriptorSets(g_vk.dev, (uint32_t)n, w, 0, NULL);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vk_bind_graphics_sets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, post_layout, 0, 1, &ds, 0, NULL);
    VkViewport vp = {0, 0, (float)dst->width, (float)dst->height, 0, 1};
    VkRect2D sc = {{0, 0}, {dst->width, dst->height}};
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdSetScissor(cb, 0, 1, &sc);
    vkCmdPushConstants(cb, post_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32, pc);
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRendering(cb);
}

// The view the present samples: the game image, or its anti-aliased / upscaled version. ow x oh: the whole image's
// size in the window (its crop is drawn into the viewport).
static VkImageView fg_scene;  // [XCDE] this frame's upscaled scene before the HUD composite (frame generation's HUD-less copy)
float vk_frame_clear[4];       // (device.c: the window image's clear colour this frame)
static VkImageView post_process(VkCommandBuffer cb, TexBackend *b, double ow_d, double oh_d) {
    VkImageView v = raw_view(b);
    fg_scene = VK_NULL_HANDLE;
    bool smaa = g_settings.antialias == AA_SMAA;
    uint32_t ow = (uint32_t)lround(ow_d), oh = (uint32_t)lround(oh_d);
    if (ow > 16384) ow = 16384;
    if (oh > 16384) oh = 16384;
    bool fsr = g_settings.upscaler == UPSCALE_FSR1 && (ow > b->width + 1 || oh > b->height + 1);  // (an upscale only)
    extern void vk_frame_suspend(void);
    extern void vk_frame_resume(void);
    if (tmp_enabled()) {  // [XCDE] FSR 3 / DLSS (temporal.c); without this frame's camera: FSR 1 as the upscaler
        bool cam = tmp_cam.ok && tmp_frame == renderer_present_count() && tmp_depth_tex && tmp_depth_tex->backend;
        vk_frame_suspend();
        VkImageView t = tmp_upscale(cb, b, v, cam ? (TexBackend *)tmp_depth_tex->backend : NULL, cam ? &tmp_cam : NULL, tmp_jit,
                                    tmp_dm1, ow, oh);
        VkImageView scene = t ? tmp_hud_scene_view(b->width, b->height) : VK_NULL_HANDLE;
        if (scene) {  // [XCDE] T213: the upscaled scene + what the HUD passes added (render size, bilinear)
            fg_scene = t;
            TexBackend *hud = post_image(5, ow, oh);
            float ph[8] = {1.0f / ow, 1.0f / oh};
            VkImageView hi[3] = {t, v, scene};
            post_pass(cb, "tmp_hud.frag", hi, 3, hud, ph, false);
            vk_full_barrier(cb);
            t = post_view(hud);
        }
        vk_frame_resume();
        if (t) return t;
        smaa = false;
        fsr = ow > b->width + 1 || oh > b->height + 1;
    }
    if (!smaa && !fsr) return v;
    vk_frame_suspend();
    if (smaa) {  // SMAA 1x: luma edges -> blending weights -> neighbourhood blending
        float w = (float)b->width, h = (float)b->height;
        float m[8] = {1.0f / w, 1.0f / h, w, h};
        TexBackend *edges = post_image(0, b->width, b->height), *weights = post_image(1, b->width, b->height);
        TexBackend *out = post_image(2, b->width, b->height);
        VkImageView in[3] = {v};
        post_pass(cb, "smaa_edge.frag", in, 1, edges, m, true);
        VkImageView wi[3] = {post_view(edges), smaa_lut(0), smaa_lut(1)};
        post_pass(cb, "smaa_weight.frag", wi, 3, weights, m, true);
        VkImageView bi[3] = {v, post_view(weights)};
        post_pass(cb, "smaa_blend.frag", bi, 2, out, m, false);
        v = post_view(out);
    }
    if (fsr) {
        float iw = (float)b->width, ih = (float)b->height;
        TexBackend *e = post_image(3, ow, oh), *r = post_image(4, ow, oh);
        float pe[8] = {iw / ow, ih / oh, 0.5f * iw / ow - 0.5f, 0.5f * ih / oh - 0.5f, iw, ih};
        VkImageView ei[3] = {v};
        post_pass(cb, "easu.frag", ei, 1, e, pe, false);
        float pr[8] = {exp2f(-0.2f), 0, 0, 0, (float)ow, (float)oh};  // (sharpness 0.2 stops: AMD's default)
        VkImageView ri_[3] = {post_view(e)};
        post_pass(cb, "rcas.frag", ri_, 1, r, pr, false);
        v = post_view(r);
    }
    vk_full_barrier(cb);
    vk_frame_resume();
    {
        static uint32_t piw, pih, pow_, poh;
        static int psm = -1, pfs = -1;
        if (b->width != piw || b->height != pih || ow != pow_ || oh != poh || smaa != psm || fsr != pfs)
            LOG("[post] %ux%u%s%s -> %ux%u\n", b->width, b->height, smaa ? " smaa" : "", fsr ? " fsr1" : "", ow, oh);
        piw = b->width; pih = b->height; pow_ = ow; poh = oh; psm = smaa; pfs = fsr;
    }
    return v;
}

static VkImageView rp_src;  // (vk_present_redraw) the last final image and its placement
static VkViewport rp_vp;
static float rp_pv[8];
// Inside the window image's rendering (device.c vk_frame_begin): the game image into (vx, vy, vw, vh).
void vk_present_draw(VkCommandBuffer cb, TexBackend *b, float vx, float vy, float vw, float vh, const float pv[8]) {
    if (!b || !b->image || (b->aspect & ~VK_IMAGE_ASPECT_COLOR_BIT)) return;
    VkImageView src = post_process(cb, b, pv[6] > 0 ? vw / pv[6] : vw, pv[7] > 0 ? vh / pv[7] : vh);
    VkPipeline p = present_pipeline(g_vk.sc_format);
    {   // [XCDE] FSR 3 frame generation: the same frame without the HUD, drawn the same way (temporal_fg.inc). Without a
        // HUD-less scene this frame (menus, no snapshot) the full image: no HUD is told apart, as before.
        extern VkImageView tmp_fg_hudless_target(void);
        extern void vk_frame_suspend(void);
        extern void vk_frame_resume(void);
        VkImageView hl = tmp_fg_hudless_target();
        if (hl) {
            vk_frame_suspend();
            VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = hl, .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                            .clearValue.color.float32 = {vk_frame_clear[0], vk_frame_clear[1], vk_frame_clear[2], vk_frame_clear[3]}};
            VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, g_vk.sc_extent}, .layerCount = 1,
                                  .colorAttachmentCount = 1, .pColorAttachments = &ca};
            vkCmdBeginRendering(cb, &ri);
            VkDescriptorSet hs = vk_alloc_set(present_set_layout);
            VkDescriptorImageInfo hi = {vk_default_sampler(true), fg_scene ? fg_scene : src, VK_IMAGE_LAYOUT_GENERAL};
            VkWriteDescriptorSet hw = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = hs, .dstBinding = 0, .descriptorCount = 1,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &hi};
            vkUpdateDescriptorSets(g_vk.dev, 1, &hw, 0, NULL);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
            vk_bind_graphics_sets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, present_layout, 0, 1, &hs, 0, NULL);
            VkViewport hvp = {vx, vy, vw, vh, 0, 1};
            VkRect2D hsc = {{0, 0}, g_vk.sc_extent};
            vkCmdSetViewport(cb, 0, 1, &hvp);
            vkCmdSetScissor(cb, 0, 1, &hsc);
            vkCmdPushConstants(cb, present_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 32, pv);
            vkCmdDraw(cb, 3, 1, 0, 0);
            vkCmdEndRendering(cb);
            vk_full_barrier(cb);
            vk_frame_resume();
        }
    }
    VkDescriptorSet ds = vk_alloc_set(present_set_layout);
    VkDescriptorImageInfo ii = {vk_default_sampler(true), src, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0, .descriptorCount = 1,
                              .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii};
    vkUpdateDescriptorSets(g_vk.dev, 1, &w, 0, NULL);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    vk_bind_graphics_sets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, present_layout, 0, 1, &ds, 0, NULL);
    VkViewport vp = {vx, vy, vw, vh, 0, 1};
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdPushConstants(cb, present_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 32, pv);
    vkCmdDraw(cb, 3, 1, 0, 0);
    VkViewport full = {0, 0, (float)g_vk.sc_extent.width, (float)g_vk.sc_extent.height, 0, 1};
    vkCmdSetViewport(cb, 0, 1, &full);
    rp_src = src; rp_vp = vp; memcpy(rp_pv, pv, sizeof rp_pv);
}

// [XCDE] device.c repeat presents (FPS 30 with V-Sync on a 60 Hz monitor): the last frame's final image drawn again the
// same way - no post-processing / upscaling run again (the temporal upscalers' history would take it as a frame).
void vk_present_redraw(VkCommandBuffer cb) {
    if (!rp_src) return;
    VkPipeline p = present_pipeline(g_vk.sc_format);
    VkDescriptorSet ds = vk_alloc_set(present_set_layout);
    VkDescriptorImageInfo ii = {vk_default_sampler(true), rp_src, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0, .descriptorCount = 1,
                              .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii};
    vkUpdateDescriptorSets(g_vk.dev, 1, &w, 0, NULL);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    vk_bind_graphics_sets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, present_layout, 0, 1, &ds, 0, NULL);
    vkCmdSetViewport(cb, 0, 1, &rp_vp);
    vkCmdPushConstants(cb, present_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 32, rp_pv);
    vkCmdDraw(cb, 3, 1, 0, 0);
    VkViewport full = {0, 0, (float)g_vk.sc_extent.width, (float)g_vk.sc_extent.height, 0, 1};
    vkCmdSetViewport(cb, 0, 1, &full);
}

// Screenshot of a game texture (presented crop applied to build/frame_* shots), via the present shader into RGBA8.
void renderer_save_texture_tex(TexBackend *b, const char *path) {
    if (!b || !b->image || (b->aspect & ~VK_IMAGE_ASPECT_COLOR_BIT)) {  // [XCDE] say why nothing is saved
        LOG("[vk] screenshot %s skipped: %s\n", path, !b ? "no texture" : !b->image ? "no image" : "not a color image");
        return;
    }
    extern const float *vk_shot_crop(void);
    float crop[4] = {0, 0, 1, 1};
    if (strstr(path, "frame_")) memcpy(crop, vk_shot_crop(), sizeof crop);
    uint32_t w = (uint32_t)(b->width * crop[2]), h = (uint32_t)(b->height * crop[3]);
    if (!w || !h) return;
    vk_end_pass();
    VkImage img;
    VmaAllocation mem;
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
                            .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &img, &mem, NULL));
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    vk_image_init_layout(cb, img, VK_IMAGE_ASPECT_COLOR_BIT);
    VkImageViewCreateInfo vci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                 .format = VK_FORMAT_R8G8B8A8_UNORM, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkImageView tv;
    VK_CHECK(vkCreateImageView(g_vk.dev, &vci, NULL, &tv));
    VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = tv, .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {w, h}}, .layerCount = 1,
                          .colorAttachmentCount = 1, .pColorAttachments = &ca};
    vkCmdBeginRendering(cb, &ri);
    VkPipeline p = present_pipeline(VK_FORMAT_R8G8B8A8_UNORM);
    VkDescriptorSet ds = vk_alloc_set(present_set_layout);
    VkDescriptorImageInfo ii = {vk_default_sampler(true), raw_view(b), VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet wr = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0, .descriptorCount = 1,
                               .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii};
    vkUpdateDescriptorSets(g_vk.dev, 1, &wr, 0, NULL);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    vk_bind_graphics_sets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, present_layout, 0, 1, &ds, 0, NULL);
    VkViewport vp = {0, 0, (float)w, (float)h, 0, 1};
    VkRect2D sc = {{0, 0}, {w, h}};
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdSetScissor(cb, 0, 1, &sc);
    float pv[8] = {renderer_origin_lower_left() && !strstr(path, "window") ? 1.0f : 0.0f, 0, 0, 0, crop[0], crop[1], crop[2], crop[3]};
    vkCmdPushConstants(cb, present_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 32, pv);
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRendering(cb);
    vk_full_barrier(cb);
    VkSpan out = vk_readback((uint64_t)w * h * 4);
    VkBufferImageCopy c = {.bufferOffset = out.off, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {w, h, 1}};
    vkCmdCopyImageToBuffer(cb, img, VK_IMAGE_LAYOUT_GENERAL, out.buf, 1, &c);
    renderer_flush();
    if (path[0] == '#') {  // [XCDE] T119: "#<label>": log a hash of the image instead of saving it (XCDE_PASS_HASH)
        uint64_t hv = 0xcbf29ce484222325ull;
        for (uint64_t i = 0; i < (uint64_t)w * h * 4; i++) hv = (hv ^ ((const uint8_t *)out.ptr)[i]) * 0x100000001b3ull;
        LOG("[passhash] %s %016llx\n", path + 1, (unsigned long long)hv);
    } else
        vk_save_png(path, out.ptr, w, h);
    vkDestroyImageView(g_vk.dev, tv, NULL);
    vmaDestroyImage(g_vk.vma, img, mem);
}

// ================================================================== executor (renderer.m exec_list)
// [XCDE] perf: the game submits its whole frame as one or two command lists and waits for the previous frame's fence
// about a frame later. One Vulkan submission per list meant the GPU started only once the render thread had recorded
// all of it: frame time = recording + GPU time (8 + 12.8 ms in the heavy meadow view at 4K: 49 fps at 60). Now the
// recorded work is submitted every XCDE_SUBMIT_DRAWS draws (default 400, Linux 100; 0 = off) so the GPU runs alongside the
// recording. (An occlusion count split across two submissions reports "visible": safe, as before T139.)
// (split_since: declared with split_pending)
static void split_submit(void) {
    static int every = -1;
    // [XCDE] Linux: every 100 (Steam Deck: the GPU otherwise waits for most of the frame's recording; with the placed
    // heap a submission costs ~0.1 ms. 100 vs 400 in the field: ~50 -> ~60 fps, the CPU then boosts to 3.1 GHz).
#ifdef __linux__
    if (every < 0) { const char *e = getenv("XCDE_SUBMIT_DRAWS"); every = e ? atoi(e) : 100; }
#else
    if (every < 0) { const char *e = getenv("XCDE_SUBMIT_DRAWS"); every = e ? atoi(e) : 400; }
#endif
    if (!every || ++split_since < (uint32_t)every) return;
    // at the next pass boundary (no extra pass break: each break stores and reloads every attachment); a run of
    // draws with no boundary for 4x as long is split right here as before
    if (split_since < 4u * (uint32_t)every) { split_pending = true; return; }
    split_since = 0;
    split_pending = false;
    renderer_submit();
}

static void exec_list(uint64_t handle, int depth) {
    uint64_t p = handle;
    for (int guard = 0; guard < 10000000; guard++) {
        CmdRec *r = (CmdRec *)(uintptr_t)p;
        void *x = r->extra;
        if (r->op < CMD_COUNT) op_counts[r->op]++;
        {   // [XCDE] XCDE_SKIP_GPU_UNTIL=<frame>: test runs only - no GPU work (draws, clears, dispatches, texture copies)
            // before that presented frame; the guest, fences, counters and buffer copies behave as always. Gets a route to
            // a late scene in minutes instead of half an hour (the images of the first frames after the switch lack
            // history: exposure, bloom).
            static uint64_t until = ~0ull - 1;
            if (until == ~0ull - 1) until = getenv("XCDE_SKIP_GPU_UNTIL") ? strtoull(getenv("XCDE_SKIP_GPU_UNTIL"), NULL, 10) : 0;
            if (until && renderer_present_count() < until) {
                switch (r->op) {
                case CMD_CLEAR_COLOR: case CMD_CLEAR_TEXTURE: case CMD_CLEAR_DEPTH_STENCIL: case CMD_DISPATCH:
                case CMD_DRAW_ARRAYS: case CMD_DRAW_ELEMENTS: case CMD_COPY_TEXTURE_TO_TEXTURE: case CMD_DOWNSAMPLE:
                case CMD_COPY_BUFFER_TO_TEXTURE: case CMD_COPY_TEXTURE_TO_BUFFER:
                    p += r->size;
                    continue;
                default: break;
                }
            }
        }
        if (!cmd_keeps_pipe_key(r->op)) g_pipe_state_gen++, g_pipe_bump_op = r->op;  // [XCDE] (pipeline key memo)
        switch (r->op) {
        case CMD_END: return;
        case CMD_JUMP: p = r->x[0]; continue;
        case CMD_CALL: {
            const uint64_t *h = x;
            for (uint64_t i = 0; i < r->x[0] && depth < 8; i++) exec_list(h[i], depth + 1);
            break;
        }
        // [XCDE] (pipeline key memo) the game sends these before most draws, mostly unchanged: the key's generation only
        // moves when a value does (STATE_SET: compare, then copy)
        #define STATE_SET(dst, src, n) do { if (memcmp((dst), (src), (n))) { memcpy((dst), (src), (n)); g_pipe_state_gen++; } } while (0)
        case CMD_BIND_PROGRAM: st.program = (NvnProgram *)(uintptr_t)r->x[0]; break;
        case CMD_BIND_BLEND: { uint8_t *b = x; STATE_SET(st.blend[b[0] & 7], b, 8); break; }
        case CMD_BIND_CHANNEL_MASK: STATE_SET(&st.channel_mask, x, 4); break;
        case CMD_BIND_COLOR: STATE_SET(st.color_state, x, 4); break;
        case CMD_BIND_DEPTH_STENCIL: memcpy(st.ds, x, 8); break;
        case CMD_BIND_MULTISAMPLE: STATE_SET(st.ms, x, 0x18); break;
        case CMD_BIND_POLYGON: STATE_SET(st.poly, x, 4); break;
        case CMD_BIND_VERTEX_ATTRIB: {
            int n = (int)r->x[0] > 16 ? 16 : (int)r->x[0];
            if (n != st.nattrib) { st.nattrib = n; g_pipe_state_gen++; }
            STATE_SET(st.attrib, x, n * 4);
            break;
        }
        case CMD_BIND_VERTEX_STREAM: {
            int n = (int)r->x[0] > 16 ? 16 : (int)r->x[0];
            if (n != st.nstream) { st.nstream = n; g_pipe_state_gen++; }
            STATE_SET(st.stream, x, n * 8);
            break;
        }
        #undef STATE_SET
        case CMD_BIND_VERTEX_BUFFERS:
            for (uint64_t i = 0; i < r->x[1] && r->x[0] + i < 16; i++) memcpy(&st.vbuf[r->x[0] + i], (uint8_t *)x + i * 16, 16);
            break;
        case CMD_BIND_UNIFORM_BUFFERS:
            for (uint64_t i = 0; i < r->x[2] && r->x[1] + i < 18 && r->x[0] < 6; i++)
                memcpy(&st.ubuf[r->x[0]][r->x[1] + i], (uint8_t *)x + i * 16, 16);
            break;
        case CMD_BIND_STORAGE_BUFFERS:
            for (uint64_t i = 0; i < r->x[2] && r->x[1] + i < 16 && r->x[0] < 6; i++)
                if (memcmp(&st.sbuf[r->x[0]][r->x[1] + i], (uint8_t *)x + i * 16, 16)) {  // [XCDE] (cbuf_info memo: on a change)
                    memcpy(&st.sbuf[r->x[0]][r->x[1] + i], (uint8_t *)x + i * 16, 16);
                    g_sbuf_bind_gen[r->x[0]]++;
                }
            break;
        case CMD_BIND_TEXTURES:
            for (uint64_t i = 0; i < r->x[2] && r->x[1] + i < 32 && r->x[0] < 6; i++)
                if (memcmp(&st.tex[r->x[0]][r->x[1] + i], (uint8_t *)x + i * 8, 8)) {  // [XCDE] (stencil_unit_mask memo: on a change)
                    memcpy(&st.tex[r->x[0]][r->x[1] + i], (uint8_t *)x + i * 8, 8);
                    g_tex_bind_gen[r->x[0]]++;
                }
            break;
        case CMD_BIND_IMAGES:
            for (uint64_t i = 0; i < r->x[2] && r->x[1] + i < 8 && r->x[0] < 6; i++)
                memcpy(&st.img[r->x[0]][r->x[1] + i], (uint8_t *)x + i * 8, 8);
            break;
        case CMD_SET_TEXTURE_POOL: st.tpool = (NvnDescPool *)(uintptr_t)r->x[0]; break;
        case CMD_SET_SAMPLER_POOL: st.spool = (NvnDescPool *)(uintptr_t)r->x[0]; break;
        case CMD_SET_RENDER_TARGETS: {
            const struct { uint64_t tex[8]; NvnTextureView views[8]; uint8_t has_view[8]; uint64_t depth;
                           NvnTextureView dview; uint8_t has_dview; } *rt = x;
            RtState prev = st.rt;  // the same targets again: keep the pass (a restart = full barrier + reload)
            memset(&st.rt, 0, sizeof st.rt);
            st.rt.ncolor = (int)r->x[0] > 8 ? 8 : (int)r->x[0];
            for (int i = 0; i < st.rt.ncolor; i++) {
                st.rt.color[i] = (NvnTexture *)(uintptr_t)rt->tex[i];
                st.rt.cview[i] = rt->views[i];
                st.rt.has_cview[i] = rt->has_view[i];
            }
            st.rt.depth = (NvnTexture *)(uintptr_t)rt->depth;
            st.rt.dview = rt->dview;
            st.rt.has_dview = rt->has_dview;
            static int always = -1;
            if (always < 0) always = getenv("LARECOMP_VK_PASS_RESTART") != NULL;
            if (always || memcmp(&prev, &st.rt, sizeof prev)) RT_DIRTY(), g_pipe_state_gen++;  // [XCDE] (key memo: on a change)
            break;
        }
        case CMD_SET_VIEWPORTS:
            for (uint64_t i = 0; i < r->x[1] && r->x[0] + i < 16; i++) memcpy(st.viewport[r->x[0] + i], (uint8_t *)x + i * 16, 16);
            break;
        case CMD_SET_SCISSORS:
            for (uint64_t i = 0; i < r->x[1] && r->x[0] + i < 16; i++) memcpy(st.scissor[r->x[0] + i], (uint8_t *)x + i * 16, 16);
            break;
        case CMD_SET_DEPTH_RANGES:
            for (uint64_t i = 0; i < r->x[1] && r->x[0] + i < 16; i++) memcpy(st.depth_range[r->x[0] + i], (uint8_t *)x + i * 8, 8);
            break;
        case CMD_SET_BLEND_COLOR: memcpy(st.blend_color, x, 16); break;
        case CMD_SET_ALPHA_REF: st.alpha_ref = r->f[0]; break;
        case CMD_SET_STENCIL: {
            int face = (int)r->x[0];
            for (int f = 0; f < 2; f++) {
                if (!(face & (1 << f))) continue;
                if (r->x[6] == 0) st.stencil_ref[f] = (uint32_t)r->x[1];
                else if (r->x[6] == 1) st.stencil_mask[f] = (uint32_t)r->x[1];
                else st.stencil_vmask[f] = (uint32_t)r->x[1];
            }
            break;
        }
        case CMD_SET_POLYGON_OFFSET: memcpy(st.poly_offset, r->f, sizeof st.poly_offset); break;
        case CMD_SET_PRIMITIVE_RESTART: st.prim_restart = r->x[0] != 0; st.restart_index = (uint32_t)r->x[1]; break;
        case CMD_SET_COPY_STRIDES: st.copy_row_stride = r->x[0]; st.copy_image_stride = r->x[1]; break;
        case CMD_BIND_XFB_BUFFERS: {  // [XCDE] T134 (first, count, NVNbufferRange[count])
            const uint64_t *rg = (const uint64_t *)r->extra;
            for (uint64_t i = 0; i < r->x[1] && r->x[0] + i < 4; i++) { st.xfb_buf[r->x[0] + i].addr = rg[2 * i]; st.xfb_buf[r->x[0] + i].size = rg[2 * i + 1]; }
            break;
        }
        // [XCDE] T136: Begin/Pause/Resume/End(cmd, control buffer). The control buffer holds the capture's write
        // position: Pause stores it there, Resume continues from it, Begin starts at 0. One counter for all sessions
        // (the old code) resumed a paused capture where the *last* capture stopped.
        case CMD_BEGIN_XFB:
            st.xfb_on = true;
            if (!r->x[1]) st.xfb_written = 0;
            else {
                int k = xfb_ctrl_find(r->x[0]);
                if (k >= 0) st.xfb_written = xfb_ctrl[k].written;
            }
            if (g_debug_frame) LOG("[xfb] %s ctrl %llx -> offset %llu (buffer %llx)\n", r->x[1] ? "resume" : "begin", (unsigned long long)r->x[0],
                                   (unsigned long long)st.xfb_written, (unsigned long long)st.xfb_buf[0].addr);
            break;
        case CMD_END_XFB:
            st.xfb_on = false;
            if (r->x[1]) {
                int k = xfb_ctrl_find(r->x[0]);
                if (k < 0) { k = n_xfb_ctrl < 16 ? n_xfb_ctrl++ : (int)(xfb_ctrl_rr++ & 15); xfb_ctrl[k].ctrl = r->x[0]; }
                xfb_ctrl[k].written = st.xfb_written;
            }
            if (g_debug_frame) LOG("[xfb] %s ctrl %llx at offset %llu\n", r->x[1] ? "pause" : "end", (unsigned long long)r->x[0], (unsigned long long)st.xfb_written);
            break;
        case CMD_SET_RASTERIZER_DISCARD: st.raster_discard = (r->x[0] & 1) != 0; break;
        case CMD_RESET_COUNTER:  // [XCDE] T139 (counter type): a samples-passed count starts from zero here
            if ((uint32_t)r->x[0] == 1 && oq_enabled()) {
                VkCommandBuffer cb = vk_cmd();
                oq_end(cb);
                oq_counting = true;
                oq_slot = cur_sub;
                oq_from = oq[cur_sub].n;
                if (in_pass) oq_begin(cb);
            }
            break;
        case CMD_BARRIER:
            // [XCDE] T133: nvnCommandBufferBarrier(bits). XCDE computes skinning palettes on the GPU (vertex shaders of
            // point draws storing to memory) into one scratch range that each mesh's writer refills from index 0;
            // the barrier keeps a mesh's draw after its writer and before the next writer. Dropped, the draws
            // overlapped on the GPU: exploded hair and cloth, run-to-run differences (T124). The next draw starts a
            // new pass, which begins with a full barrier. (bits 0: the list-start marker of BeginRecording)
            if (g_debug_frame) LOG("[barrier] (pass %d) bits %#x in_pass %d\n", debug_pass_no, (unsigned)r->x[0], (int)in_pass);
            if ((uint32_t)r->x[0] && in_pass) { vk_end_pass(); RT_DIRTY(); pass_untracked = !(mem_track_mode() & 2); }  // (T162: tracked)
            break;
        case CMD_CLEAR_COLOR: {
            snap_reason = "clear-color"; snapshots_invalidate(NULL);
            int index = (int)r->x[0];
            if (getenv("LARECOMP_LOG_CLEARS") && index < st.rt.ncolor && st.rt.color[index])
                LOG("[cclear] frame %llu idx %d %ux%u f%#x mask %#x @%llx obj %p img %p sc %d %d %d %d\n", (unsigned long long)renderer_present_count(), index,
                    st.rt.color[index]->width, st.rt.color[index]->height, st.rt.color[index]->format, (uint32_t)r->x[2],
                    (unsigned long long)st.rt.color[index]->addr, (void *)st.rt.color[index],
                    st.rt.color[index]->backend ? (void *)((TexBackend *)st.rt.color[index]->backend)->image : NULL,
                    st.scissor[0][0], st.scissor[0][1], st.scissor[0][2], st.scissor[0][3]);
            if (ct_on() && index < st.rt.ncolor && st.rt.color[index])
                LOG("[ct] clear-color %p mask %#x sc %d %d %d %d\n", (void *)st.rt.color[index], (uint32_t)r->x[2], st.scissor[0][0], st.scissor[0][1],
                    st.scissor[0][2], st.scissor[0][3]);
            if (index < st.rt.ncolor && st.rt.color[index]) alias_before_write(st.rt.color[index], 0, "clear");
            if (index < st.rt.ncolor && st.rt.color[index]) clear(index, (int)r->x[5], x, (uint32_t)r->x[2], false, 0, false, 0, 0);
            break;
        }
        case CMD_CLEAR_TEXTURE: {
            // ClearTexture(cmd, texture, view, region, color, mask): a render-target clear of the texture alone (renderer.m)
            const struct { uint64_t tex; NvnTextureView view; uint8_t has_view; int32_t region[6]; float color[4]; } *ct = x;
            NvnTexture *t = (NvnTexture *)(uintptr_t)ct->tex;
            if (!t) break;
            if (g_debug_frame)
                LOG("[clear-tex] (pass %d) %llx %ux%u f%#x view %d region %d %d %d %d mask %#x color %08x %08x\n", debug_pass_no,
                    (unsigned long long)t->addr, t->width, t->height, t->format, ct->has_view, ct->region[0], ct->region[1], ct->region[3],
                    ct->region[4], (uint32_t)r->x[4], ((const uint32_t *)ct->color)[0], ((const uint32_t *)ct->color)[1]);
            snap_reason = "clear-texture"; snapshots_invalidate(NULL);
            alias_before_write(t, 0, "clear");
            RtState saved = st.rt;
            int32_t sc[4];
            memcpy(sc, st.scissor[0], sizeof sc);
            vk_end_pass();
            memset(&st.rt, 0, sizeof st.rt);
            st.rt.ncolor = 1;
            st.rt.color[0] = t;
            if (ct->has_view) { st.rt.cview[0] = ct->view; st.rt.has_cview[0] = 1; }
            RT_DIRTY();
            const int32_t *rg = ct->region;
            if (rg[3] > 0 && rg[4] > 0) { st.scissor[0][0] = rg[0]; st.scissor[0][1] = rg[1]; st.scissor[0][2] = rg[3]; st.scissor[0][3] = rg[4]; }
            else { st.scissor[0][0] = st.scissor[0][1] = 0; st.scissor[0][2] = st.scissor[0][3] = 16384; }
            clear(0, (int)r->x[6], ct->color, (uint32_t)r->x[4] ? (uint32_t)r->x[4] : 0xF, false, 0, false, 0, 0);
            vk_end_pass();
            st.rt = saved;
            memcpy(st.scissor[0], sc, sizeof sc);
            RT_DIRTY();
            break;
        }
        case CMD_CLEAR_DEPTH_STENCIL:
            snap_reason = "clear-depth"; snapshots_invalidate(NULL);
            if (ct_on() && st.rt.depth)
                LOG("[ct] clear-depth %p d%d s%d sc %d %d %d %d\n", (void *)st.rt.depth, r->x[0] != 0, r->x[2] != 0, st.scissor[0][0],
                    st.scissor[0][1], st.scissor[0][2], st.scissor[0][3]);
            if (st.rt.depth && naliases) {  // (copy aliases: which aspects of the depth target this clear replaces whole)
                NvnTexture *d = st.rt.depth;
                bool whole = st.scissor[0][0] <= 0 && st.scissor[0][1] <= 0 && (int64_t)st.scissor[0][0] + st.scissor[0][2] >= (int64_t)d->width &&
                             (int64_t)st.scissor[0][1] + st.scissor[0][3] >= (int64_t)d->height &&
                             !(st.rt.has_dview && (((st.rt.dview.setmask & 1) && st.rt.dview.base_level) ||
                                                   ((st.rt.dview.setmask & 2) && st.rt.dview.min_layer)));
                for (int i = 0; i < st.rt.ncolor; i++)  // (the clear covers the smallest attachment only)
                    if (st.rt.color[i] && (st.rt.color[i]->width < d->width || st.rt.color[i]->height < d->height)) whole = false;
                VkImageAspectFlags full = 0;
                if (whole && r->x[0] != 0) full |= VK_IMAGE_ASPECT_DEPTH_BIT;
                if (whole && renderer_format_has_stencil(d->format) && (r->x[2] & 0xFF) == 0xFF) full |= VK_IMAGE_ASPECT_STENCIL_BIT;
                if (!renderer_format_has_stencil(d->format)) full |= VK_IMAGE_ASPECT_STENCIL_BIT;
                if (ct_on())
                    LOG("[ct] alias clear %p full %#x (whole %d, x0 %llu x1 %#llx x2 %#llx, dview %d/%#x, ncolor %d)\n", (void *)d, full, whole,
                        (unsigned long long)r->x[0], (unsigned long long)r->x[1], (unsigned long long)r->x[2], st.rt.has_dview,
                        st.rt.dview.setmask, st.rt.ncolor);
                alias_before_write(d, full, "clear");
            }
            if (st.rt.depth) {
                bool has_stencil = renderer_format_has_stencil(st.rt.depth->format);
                clear(-1, 0, NULL, 0, r->x[0] != 0, r->f[0], has_stencil && r->x[2] != 0, (uint32_t)r->x[1], (uint32_t)r->x[2]);
            }
            break;
        case CMD_DISPATCH: dispatch_compute((uint32_t)r->x[0], (uint32_t)r->x[1], (uint32_t)r->x[2]); break;
        case CMD_DRAW_ARRAYS: draw(r, false); split_submit(); break;
        case CMD_DRAW_ELEMENTS: draw(r, true); split_submit(); break;
        case CMD_UPDATE_UNIFORM_BUFFER:
            watch_mem_note("inline uniform update", r->x[0] + r->x[2], r->x[3]);
            memcpy((void *)(uintptr_t)(r->x[0] + r->x[2]), x, r->x[3]);
            guest_copy_flush();  // (cbuf copies shared within the submission are stale now)
            break;
        case CMD_FENCE_SYNC: {
            static int fence_flush = -1;
            if (fence_flush < 0) fence_flush = getenv("LARECOMP_FENCE_FLUSH") ? 1 : 0;
            if (fence_flush) renderer_flush();
            extern void nvn_sync_set_seq(uint64_t guest, uint64_t seq);
            if (r->x[1]) nvn_sync_set_seq(r->x[1], renderer_fence_seq());
            extern bool vk_ov_wait_mode(void);
            if (r->x[1] && vk_ov_wait_mode()) renderer_submit();  // [XCDE] XCDE_OVERLAP: a waited-for fence is always submitted
            break;
        }
        case CMD_REPORT_COUNTER: {
            // The game times its GPU work with these (two per frame) and drops to 30 fps when that looks slow: the
            // host clock now, replaced by the GPU's time at this point once the submission is done (cq_resolve).
            uint64_t *out = (uint64_t *)(uintptr_t)r->x[1];
            uint32_t type = (uint32_t)r->x[0];
            static uint32_t seen;
            if (type < 32 && !(seen & (1u << type))) { seen |= 1u << type; LOG("[nvn] ReportCounter type %u\n", type); }
            int mode = vk_counter_mode();
            static uint64_t frame_seen = ~0ull, frame_ts;
            extern uint64_t vk_frame_counter(void);
            if (mode == 1 && vk_frame_counter() != frame_seen) { frame_seen = vk_frame_counter(); frame_ts = ns_to_gpu_ticks(diag_deterministic() ? kernel_vnow(0) : plat_now_ns()); }
            if (!out) break;
            out[0] = 0;
            if (type == 1) {  // [XCDE] T137 / T139: samples passed - "visible" until the queries are resolved (oq_resolve)
                out[0] = OQ_VISIBLE;
                out[1] = ns_to_gpu_ticks(diag_deterministic() ? kernel_vnow(0) : plat_now_ns());  // (not a frame-timing pair)
                if (oq_enabled() && oq_counting && oq_slot == cur_sub) {
                    VkCommandBuffer cb = vk_cmd();
                    bool reopen = oq_open >= 0;
                    oq_end(cb);  // (the segment so far counts; counting goes on in a new one)
                    OccQ *q = &oq[cur_sub];
                    pthread_mutex_lock(&cq_lock);
                    if (q->pool && q->nrep < OQ_REPORTS && q->n < OQ_MAX) q->rep[q->nrep++] = (OccRep){out, oq_from, q->n};
                    pthread_mutex_unlock(&cq_lock);
                    if (reopen) oq_begin(cb);
                }
                break;
            }
            out[1] = mode == 1 ? frame_ts : mode == 3 ? SDL_GetTicksNS() : ns_to_gpu_ticks(plat_now_ns());  // (3 legacy: ns, the old bug)
            static uint64_t *prev_out;  // (frame pacing: the pair's start)
            if (mode == 0 || mode == 2) out[1] = pace_end_counter(prev_out, out, out[1]);
            prev_out = out;
            {   // (test) LARECOMP_TEST_COUNTER_DELAY_MS=<ms>: stall the host after a frame's first counter - a slow CPU
                static double delay = -1;
                static uint64_t delayed_frame = ~0ull;
                if (delay < 0) delay = getenv("LARECOMP_TEST_COUNTER_DELAY_MS") ? atof(getenv("LARECOMP_TEST_COUNTER_DELAY_MS")) : 0;
                if (delay > 0 && vk_frame_counter() != delayed_frame) {
                    delayed_frame = vk_frame_counter();
                    SDL_DelayPrecise((Uint64)(delay * 1e6));
                }
            }
            if (mode == 2 && type == 0) {  // (type 0: timestamp)
                VkCommandBuffer cb = vk_cmd();  // (first: may begin the buffer, which resets the slot's queries)
                CounterQ *q = &cq[cur_sub];
                pthread_mutex_lock(&cq_lock);
                if (q->pool && q->n < CQ_MAX) {
                    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, q->pool, q->n);
                    q->out[q->n++] = out;
                }
                pthread_mutex_unlock(&cq_lock);
            }
            break;
        }
        case CMD_COPY_TEXTURE_TO_TEXTURE: { const TexRefRec *t = x; copy_texture(&t[0], &t[1], r->x[6]); break; }
        case CMD_DOWNSAMPLE: { const TexRefRec *t = x; copy_texture(&t[0], &t[1], 1); break; }
        case CMD_COPY_BUFFER_TO_TEXTURE: copy_buffer_to_texture(r->x[0], x); break;
        case CMD_COPY_TEXTURE_TO_BUFFER: copy_texture_to_buffer(x, r->x[3]); break;
        case CMD_COPY_BUFFER_TO_BUFFER:
            wait_why = "buffer -> buffer copy";
            renderer_flush();
            watch_mem_note("buffer copy", r->x[1], r->x[2]);
            memmove((void *)(uintptr_t)r->x[1], (void *)(uintptr_t)r->x[0], r->x[2]);
            break;
        default: {  // [XCDE] commands this executor does not implement: say so (once per type) instead of dropping them silently
            static uint32_t told[4]; static uint64_t cnt[128];
            uint32_t id = (uint32_t)r->op;
            if (id < 128) {
                cnt[id]++;
                if (!(told[id >> 5] >> (id & 31) & 1)) { told[id >> 5] |= 1u << (id & 31); LOG("[vk] UNIMPLEMENTED command %u (first seen; counted below)\n", id); }
            }
            break;
        }
        }
        p += r->size;
    }
}

// ---- [XCDE] perf: async render worker. The game's render thread built its command lists AND ran this whole backend
// (translation to Vulkan, ~3-11 ms a frame) on one core: in heavy views at 4K that thread took ~19 ms a frame while
// the GPU needed 13. Now QueueSubmitCommands / FenceSync / PresentTexture / TextureFinalize only queue a job and a
// worker thread runs them in order, so the game builds the next frame while the worker translates this one. The
// command lists live in the game's command memory, which it reuses only after waiting on its fences: a sync's wait
// first waits for the worker to have run the job that set it (tickets; fences inside lists are found when the list is
// queued). At most 2 presents queued (the game can't run further ahead). Off in deterministic runs and with
// XCDE_ASYNC_RENDER=0.
typedef struct { int kind; uint64_t a, result_seq; } RJob;
enum { RJ_EXEC, RJ_PRESENT, RJ_FENCE, RJ_DESTROY };
#define RQ_N 4096
static RJob rq[RQ_N];
static uint64_t rq_head, rq_tail, rq_done;  // jobs run / queued; ticket of a job = its index + 1
static int rq_presents;
static pthread_mutex_t rq_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t rq_cv = PTHREAD_COND_INITIALIZER, rq_done_cv = PTHREAD_COND_INITIALIZER;
void backend_texture_destroyed(NvnTexture *t);
void backend_present(NvnTexture *t);

// [XCDE] T240: texture / sampler pool registrations in command order. The game registers a texture into a pool slot
// (often one a released texture left) right after submitting draws that still sample that slot's old texture: on the
// Switch the GPU has read them by then or reads them soon; the worker may be frames behind (a pipeline compile, a cold
// driver cache) and recorded those draws with the new texture - a face with another material's texture, black where a
// shader samples a cube / 3D unit that now holds a 2D texture. A registration is applied when the worker reaches the
// first job queued after it (pw_tag = jobs queued when it was made), so every list sees the pool as the game had it
// when that list was submitted. Without the worker: applied at once. XCDE_POOL_IMMEDIATE=1: the old behaviour.
typedef struct { uint64_t tag; void *dst; uint32_t n, kind; uint64_t data[16]; } PoolWrite;
static PoolWrite *pwq;
static size_t pwq_n, pwq_cap, pwq_done;  // (queued / applied; reset when all are applied)
static void pool_write_apply(const PoolWrite *w) {
    if (w->kind == 1) vk_tex_desc_changed();  // [XCDE] (stencil_unit_mask memo)
    if (w->kind == 1) {  // a texture pool entry: the glitch watch's slot history
        NvnTexEntry *e = w->dst;
        NvnTexEntry v;
        memcpy(&v, w->data, sizeof v);
        v.prev_target = e->tex ? e->tex->target : 0xff;
        memcpy(e, &v, sizeof v);
    } else {
        memcpy(w->dst, w->data, w->n);
    }
}
static void pool_writes_until(uint64_t job) {  // (worker, rq_lock held) the writes made before job `job` was queued
    while (pwq_done < pwq_n && pwq[pwq_done].tag <= job) pool_write_apply(&pwq[pwq_done++]);
    if (pwq_done == pwq_n) pwq_n = pwq_done = 0;
}
bool backend_async(void);
void backend_pool_write(void *dst, const void *src, size_t n, int kind) {
    static int imm = -1;
    if (imm < 0) imm = getenv("XCDE_POOL_IMMEDIATE") != NULL;
    PoolWrite w = {.dst = dst, .n = (uint32_t)n, .kind = (uint32_t)kind};
    if (n > sizeof w.data) { memcpy(dst, src, n); if (kind == 1) vk_tex_desc_changed(); return; }
    memcpy(w.data, src, n);
    if (imm || !backend_async()) { pool_write_apply(&w); return; }
    pthread_mutex_lock(&rq_lock);
    w.tag = rq_tail;
    if (pwq_n == pwq_cap) {
        size_t cap = pwq_cap ? pwq_cap * 2 : 4096;
        PoolWrite *q = realloc(pwq, cap * sizeof *q);
        if (!q) { pthread_mutex_unlock(&rq_lock); pool_write_apply(&w); return; }
        pwq = q; pwq_cap = cap;
    }
    pwq[pwq_n++] = w;
    pthread_mutex_unlock(&rq_lock);
}

static void *rq_worker(void *arg) {
    (void)arg;
    tl_rq_worker = true;
    plat_ov_set_worker();  // [XCDE] (XCDE_OVERLAP_STATS)
    for (;;) {
        pthread_mutex_lock(&rq_lock);
        while (rq_head == rq_tail) {
            uint64_t t0 = plat_now_ns();
            pthread_cond_wait(&rq_cv, &rq_lock);
            g_blocked_ns += plat_now_ns() - t0;  // (idle, not busy: vk_pace_frame measures this thread's frames)
        }
        RJob j = rq[rq_head % RQ_N];
        pool_writes_until(rq_head);
        pthread_mutex_unlock(&rq_lock);
        static const char *const kind_name[] = {"recording a command list", "presenting", "setting a fence", "freeing a texture"};
        vk_rq_at(j.kind >= 0 && j.kind < 4 ? kind_name[j.kind] : "a job");
        switch (j.kind) {
        case RJ_EXEC: backend_execute(j.a); break;
        case RJ_PRESENT:
            backend_present((NvnTexture *)(uintptr_t)j.a);
            if (j.result_seq) { renderer_submit(); *(uint64_t *)(uintptr_t)j.result_seq = renderer_fence_seq(); }
            break;
        case RJ_FENCE: { extern void nvn_sync_set_seq(uint64_t guest, uint64_t seq); nvn_sync_set_seq(j.a, renderer_fence_seq()); break; }
        case RJ_DESTROY: {
            NvnTexture *t = (NvnTexture *)(uintptr_t)j.a;
            extern void nvn_pools_forget_texture(NvnTexture *t);
            nvn_pools_forget_texture(t);  // [XCDE] T240 (objects.c)
            backend_texture_destroyed(t);
            free(t);
            break;
        }
        }
        vk_rq_at("idle (queue empty)");
        pthread_mutex_lock(&rq_lock);
        rq_head++;
        if (j.kind == RJ_PRESENT) rq_presents--;
        rq_done = rq_head;
        pthread_cond_broadcast(&rq_done_cv);
        pthread_mutex_unlock(&rq_lock);
    }
    return NULL;
}

// [XCDE] decided and started once, whichever game thread asks first: an unguarded "on < 0" let two first callers start
// two render workers (LASRecomp bd76094: two render threads on one queue = a driver deadlock, a boot hang with no log)
static bool g_async_on;
static void backend_async_init(void) {
    const char *e = getenv("XCDE_ASYNC_RENDER");
    g_async_on = !diag_deterministic() && !(e && *e == '0');
    if (g_async_on) {
        pthread_t th;
        pthread_create(&th, NULL, rq_worker, NULL);
        LOG("[vk] async render worker on\n");
    }
}
bool backend_async(void) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, backend_async_init);
    return g_async_on;
}

static void scan_fences(uint64_t handle, uint64_t ticket, int depth);
static uint64_t rq_push_result(int kind, uint64_t a, uint64_t result_seq) {
    pthread_mutex_lock(&rq_lock);
    while (rq_tail - rq_head >= RQ_N || (kind == RJ_PRESENT && rq_presents >= 2)) pthread_cond_wait(&rq_done_cv, &rq_lock);
    if (kind == RJ_EXEC) scan_fences(a, rq_tail + 1, 0);  // (under the lock: its syncs carry this job's ticket before anyone waits)
    rq[rq_tail % RQ_N] = (RJob){kind, a, result_seq};
    uint64_t ticket = ++rq_tail;
    if (kind == RJ_PRESENT) rq_presents++;
    pthread_cond_signal(&rq_cv);
    pthread_mutex_unlock(&rq_lock);
    return ticket;
}

static uint64_t rq_push(int kind, uint64_t a) { return rq_push_result(kind, a, 0); }
uint64_t backend_queue_present_fenced(NvnTexture *t, uint64_t *seq) {
    return rq_push_result(RJ_PRESENT, (uint64_t)(uintptr_t)t, (uint64_t)(uintptr_t)seq);
}

uint64_t backend_present_fenced_sync(NvnTexture *t) {
    backend_present(t); renderer_submit(); return renderer_fence_seq();
}

// [XCDE] T240 (diag/watchdog.c): the worker's state in a stall report - no locks (the stall may be inside one)
void backend_stall_report(FILE *f) {
    double secs = g_rq_where_ns ? (SDL_GetTicksNS() - g_rq_where_ns) / 1e9 : 0;
    fprintf(f, "[stall] render worker: %s (for %.1f s); jobs run %llu, queued %llu, presents queued %d; submissions %llu;"
               " build jobs queued %d busy %d, translations queued %d\n", g_rq_where ? g_rq_where : "?", secs,
            (unsigned long long)rq_head, (unsigned long long)rq_tail, rq_presents, (unsigned long long)submit_seq,
            (int)atomic_load(&g_jobs_queued), (int)atomic_load(&g_jobs_busy), (int)atomic_load(&g_xlat_queued));
}

void backend_wait_ticket(uint64_t ticket) {  // (guest threads) the worker has run every job up to that one
    pthread_mutex_lock(&rq_lock);
    while (rq_done < ticket) pthread_cond_wait(&rq_done_cv, &rq_lock);
    pthread_mutex_unlock(&rq_lock);
}
void backend_drain(void) {
    pthread_mutex_lock(&rq_lock);
    uint64_t t = rq_tail;
    pthread_mutex_unlock(&rq_lock);
    backend_wait_ticket(t);
}

// the fences a list sets (CMD_FENCE_SYNC with a sync object), through its jumps and calls
static void scan_fences(uint64_t handle, uint64_t ticket, int depth) {
    extern void nvn_sync_mark(uint64_t guest, uint64_t ticket);
    if (nvn_list_sync_cmds(handle) == 0) return;  // [XCDE] perf: no fence / call recorded in it (cmdbuf.c)
    uint64_t p = handle;
    for (int guard = 0; guard < 10000000; guard++) {
        CmdRec *r = (CmdRec *)(uintptr_t)p;
        switch (r->op) {
        case CMD_END: return;
        case CMD_JUMP: p = r->x[0]; continue;
        case CMD_CALL: {
            const uint64_t *h = (const uint64_t *)r->extra;
            for (uint64_t i = 0; i < r->x[0] && depth < 8; i++) scan_fences(h[i], ticket, depth + 1);
            break;
        }
        case CMD_FENCE_SYNC: if (r->x[1]) nvn_sync_mark(r->x[1], ticket); break;
        default: break;
        }
        p += r->size;
    }
}

void backend_queue_execute(uint64_t handle) { rq_push(RJ_EXEC, handle); }
uint64_t backend_queue_fence(uint64_t guest_sync) { return rq_push(RJ_FENCE, guest_sync); }
void backend_queue_present(NvnTexture *t) { rq_push(RJ_PRESENT, (uint64_t)(uintptr_t)t); }
void backend_queue_destroy(NvnTexture *t) { rq_push(RJ_DESTROY, (uint64_t)(uintptr_t)t); }

void backend_execute(uint64_t handle) {
    split_since = 0;
    PROF_T0();
    exec_list(handle, 0);
    PROF_ADD(prof_exec);
    if (prof) cum_exec += SDL_GetTicksNS() - prof_t0_;
    static int sync_submit = -1;
    if (sync_submit < 0) sync_submit = getenv("LARECOMP_SYNC_SUBMIT") ? 1 : 0;
    if (sync_submit) renderer_flush();
    else renderer_submit();
}


// [XCDE] T260: opt-in buffered per-native-frame evidence; no game content or generated frames.
void vk_metrics_frame(uint64_t frame, uint64_t now, uint64_t present_ns) {
    static FILE *file; static bool initialized;
    static uint64_t last, previous[9];
    if (!initialized) {
        initialized=true; const char *path=getenv("XCDE_METRICS");
        if (!path || !*path) return;
        file=fopen(path,"w");
        if (!file) { LOG("[metrics] cannot write %s\n",path); return; }
        setvbuf(file,NULL,_IOFBF,65536);
        fprintf(file,"#gpu=%s\n#driver=%u\n#api=%u\n#preset=%d\n#resolution=%d\n#fps=%d\n#upscaler=%d\n#effects_off=%u\n",
                g_vk.props.deviceName,g_vk.props.driverVersion,g_vk.api_version,g_settings.quality_preset,
                g_settings.res_mode,g_settings.fps,g_settings.upscaler,g_settings.effects_off);
        fprintf(file,"#output=%ux%u\n#dynamic=%d:%d\n#aa=%d\n#dof_off=%d\n#game_blur=%d\n#upscale_quality=%d\n#frame_gen=%d\n#cutscenes=%d\n#vsync=%d\n#low_memory=%d\n#ram_mib=%d\n#logical_cpus=%d\n",
                g_vk.sc_extent.width,g_vk.sc_extent.height,g_settings.dynamic_min_height,g_settings.dynamic_max_height,
                g_settings.antialias,g_settings.dof_off,g_settings.game_blur,g_settings.upscale_quality,g_settings.frame_gen,
                g_settings.cutscene_mode,g_settings.vsync,g_settings.gpu_performance,SDL_GetSystemRAM(),SDL_GetNumLogicalCPUCores());
        fprintf(file,"frame,frame_ms,gpu_ms,exec_ms,wait_ms,present_ms,draws,submits,descriptor_hits,descriptor_updates,upload_mib,gmem_upload_mib,gmem_readback_mib,compile_jobs\n");
    }
    if (!file) return;
    uint64_t values[]={cum_exec,cum_wait,cum_draws,cum_submits,metric_desc_hits,metric_desc_updates,metric_gmem_upload,metric_gmem_readback,hc_upload};
    fprintf(file,"%llu,%.4f,%.4f,%.4f,%.4f,%.4f,%llu,%llu,%llu,%llu,%.4f,%.4f,%.4f,%d\n",
            (unsigned long long)frame,last ? (now-last)/1e6 : 0,vk_native_gpu_ms(),
            (values[0]-previous[0])/1e6,(values[1]-previous[1])/1e6,present_ns/1e6,
            (unsigned long long)(values[2]-previous[2]),(unsigned long long)(values[3]-previous[3]),
            (unsigned long long)(values[4]-previous[4]),(unsigned long long)(values[5]-previous[5]),
            (values[8]-previous[8])/1048576.0,(values[6]-previous[6])/1048576.0,(values[7]-previous[7])/1048576.0,
            atomic_load(&g_jobs_queued)+atomic_load(&g_jobs_busy)+atomic_load(&g_xlat_queued));
    memcpy(previous,values,sizeof values); last=now;
    if (frame%30==0 || diag_shot_path(frame+1)) fflush(file);
}
