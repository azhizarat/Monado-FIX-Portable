#include <SDL3/SDL.h>
#include "../../settings.h"
#include "../../resource_policy.h"
// [XCDE] T187: shader precompilation from the game's own shader containers (RomFS), before the first play.
//
// `xcde.exe --precompile <dump dir>` (the launcher's "Precompilar shaders" button; JUGAR runs it the first time) walks
// <dump dir>/romfs, opens every file that holds shaders (and every xbc1 zlib block inside it), finds the game's shader
// programs and builds each one the way its first draw would (exec.c vk_precompile_program): translation, SPIR-V
// (cache/spirv) and the vertex / fragment pipeline libraries, compiled by the driver into its own shader cache. The game then finds a program's pieces compiled the first
// time it draws with it: a fast link instead of a translation, a GLSL compile and two native compiles on the spot.
// Nothing is written that the game would not write itself for the same program: a wrong guess of a translation key
// only costs work nobody uses, never a different image.
//
// Where the shaders are: shader containers ("HCPS" = SPCH: monolib/shader/*.wishp, inside models .wimdo, model streams
// .wismt, maps .wismda and effects .wiefb - mostly zlib-compressed in xbc1 blocks). Header u32s [8]/[9] = code section
// offset / size, [10]/[11] = control section offset / size (from the magic). Each shader has an NVN control section
// (magic 0x98761234; +0x6FC constants size, +0x700 constants offset, +0x704 code blob size) and a code blob (0x30-byte
// NVN header with magic 0x12345678, SPH, code, constants) packed in the same order, 256-byte aligned (a blob without
// the header is a compute shader: skipped). A program is a vertex shader and the fragment shader that follows it.
// Content hashes as the runtime's (objects.c set_shaders, T125). Measured (2026-10-20): 8,124 files, 61,908 control
// sections, 11,649 distinct shaders; 593 of the 595 the game registered in a title + field + battle run are among them.
//
// Translation keys (ShaderKey) depend on draw state the containers don't have. Guesses from the 966 variants of a cold
// title + field run (cache/variants.bin): vertex keys always depth_minus_one without flips; vertex input location 1
// unsigned-integer in 15 % (one family: such vertex shaders are built both ways), other locations float; colour targets
// float in 97 %; textures read as integers in 9 % of the fragment variants (not guessed). The game's log line
// "[shader] spirv cache: N hits, M compiled" says how many it still had to compile.
//
// Progress: cache/precompile_progress.txt rewritten ~5 times a second ("<phase> <files done> <files> <programs found>
// <programs done>"); cache/precompile_done.txt at the end (the launcher's "already done" check). Threads at below-normal
// priority. XCDE_PRECOMPILE_THREADS=<n> (default CPUs - 1); XCDE_PRECOMPILE_LIST=<file>: the shader hashes and programs
// found (tests); XCDE_PRECOMPILE_LIBS=0: SPIR-V only (no Vulkan device, no pipeline libraries).
#include <dirent.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <zlib.h>

#include "../../rt.h"
#include "shaders.h"

extern void plat_thread_background(void);
extern int plat_cpu_count(void);
extern int vk_precompile_program(const NvnProgram *np, const ShaderKey *vkey, const ShaderKey *fkey, uint32_t *vs_attr);
extern void ui_window_init(void);
static uint64_t now_ns(void) {  // (plat_init, which picks the game's clock, is not run in this mode)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

// ------------------------------------------------------------------ found shaders and programs (deduplicated)
typedef struct {
    uint64_t hash;
    int stage;
    uint32_t coff, csz;  // constants offset / size (control section)
    uint8_t *code;       // copy of the blob (zero-padded up to the translation limit)
    uint32_t int_in;     // [XCDE] T211 vertex: input locations named as integers in the container (n*: nWgtIdx);
                         // fragment: texture units named as stencil (gTStencil / gTCurStencil / gTPrevStencil)
    bool named;          // the container named its inputs (int_in is known, also when 0)
} Found;
typedef struct { uint32_t vs, fs; } Pair;  // indices into found (fs = UINT32_MAX: a vertex shader alone)
static Found *found;
static size_t nfound, capfound;
static Pair *progs;
static size_t nprogs, capprogs;
static pthread_mutex_t flock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t fcond = PTHREAD_COND_INITIALIZER;
static size_t next_job;
static bool scan_done, use_libs;
static _Atomic uint64_t n_named_int;  // [XCDE] T211 (programs built with integer inputs named by the container)
static _Atomic uint64_t n_built, n_failed, n_extra, files_done, n_controls, next_file;
static _Atomic int abandon;  // [PRECOMPILE-STALL-FIX] set when a phase gives up: the workers take no new job
static uint64_t files_total;

// hash -> value + 1 (open addressing; flock held)
typedef struct { uint64_t *k; uint32_t *v; size_t cap, n; } Map;
static Map shader_map, prog_map;
static uint32_t map_get(Map *m, uint64_t h) {
    if (!m->cap) return 0;
    if (!h) h = 1;
    for (size_t j = (h ^ (h >> 29)) & (m->cap - 1); m->k[j]; j = (j + 1) & (m->cap - 1)) if (m->k[j] == h) return m->v[j];
    return 0;
}
static void map_put(Map *m, uint64_t h, uint32_t v) {
    if (!h) h = 1;
    if ((m->n + 1) * 2 > m->cap) {
        Map o = *m;
        m->cap = o.cap ? o.cap * 2 : 1 << 16;
        m->k = calloc(m->cap, 8); m->v = calloc(m->cap, 4); m->n = 0;
        for (size_t i = 0; i < o.cap; i++) if (o.k[i]) map_put(m, o.k[i], o.v[i]);
        free(o.k); free(o.v);
    }
    size_t j = (h ^ (h >> 29)) & (m->cap - 1);
    while (m->k[j] && m->k[j] != h) j = (j + 1) & (m->cap - 1);
    if (!m->k[j]) m->n++;
    m->k[j] = h; m->v[j] = v;
}

static uint32_t rd32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }

// (flock held) the index of this shader, added if new
static uint32_t add_shader(const uint8_t *code, uint32_t blob, uint32_t coff, uint32_t csz, uint64_t h, int stage) {
    uint32_t v = map_get(&shader_map, h);
    if (v) return v - 1;
    if (nfound == capfound) found = realloc(found, (capfound = capfound ? capfound * 2 : 4096) * sizeof *found);
    Found *f = &found[nfound];
    f->hash = h; f->stage = stage; f->coff = coff; f->csz = csz;
    uint32_t lim = coff ? coff : 0x10000, sz = lim > blob ? lim : blob;
    f->code = calloc(1, sz);
    memcpy(f->code, code, blob);
    map_put(&shader_map, h, (uint32_t)nfound + 1);
    return (uint32_t)nfound++;
}
static void add_prog(uint32_t vs, uint32_t fs) {
    uint64_t h = found[vs].hash * 0x9E3779B97F4A7C15ull ^ (fs == UINT32_MAX ? 0x5555 : found[fs].hash);
    if (map_get(&prog_map, h)) return;
    if (nprogs == capprogs) progs = realloc(progs, (capprogs = capprogs ? capprogs * 2 : 4096) * sizeof *progs);
    progs[nprogs++] = (Pair){vs, fs};
    map_put(&prog_map, h, 1);
    pthread_cond_signal(&fcond);
}

// [XCDE] T211: each program's vertex input names (container program data, the layout of ScanMountGoat/xc3_lib's
// public description of SPCH, tools/spch_attrs.py): which locations the game declares as integers. Monolith names
// them by type - v* float vectors, f* float scalars, n* integers (nWgtIdx, the skinning weight index: location 1 in
// 17,729 character programs, location 0 in 356). A draw binds an integer vertex format there, so the translation key
// has an unsigned input at that location: built that way instead of guessed (the guess was float + "input 1 also
// unsigned"). Returns the number of programs; ctl_of / mask_of: their vertex control sections and integer inputs.
static bool stencil_name(const uint8_t *s, size_t lim) {  // "...Stencil" but not "...ExpStencil" (a float map)
    size_t l = 0;
    while (l < lim && l < 64 && s[l]) l++;
    if (l < 7 || memcmp(s + l - 7, "Stencil", 7)) return false;
    return !(l >= 10 && !memcmp(s + l - 10, "ExpStencil", 10));
}
static size_t spch_inputs(const uint8_t *d, size_t n, size_t b, size_t *ctl_of, uint32_t *mask_of, size_t max) {
    size_t np_ = 0;
    if (b + 0x30 > n) return 0;
    uint32_t nslct = rd32(d + b + 12), list = rd32(d + b + 8), sec = rd32(d + b + 24), ctl = rd32(d + b + 40);
    for (uint32_t i = 0; i < nslct && b + list + 8 * i + 4 <= n; i++) {
        size_t sl = b + sec + rd32(d + b + list + 8 * i);
        if (sl + 48 > n || memcmp(d + sl, "SLCT", 4)) continue;
        uint32_t pn = rd32(d + sl + 16), po = rd32(d + sl + 20);
        size_t cur = rd32(d + sl + 40);
        for (uint32_t k = 0; k < pn && sl + po + 8 * k + 8 <= n; k++) {
            size_t pd = sl + rd32(d + sl + po + 8 * k);
            uint32_t psz = rd32(d + sl + po + 8 * k + 4);
            if (pd + 88 > n || pd + psz > n) return np_;
            uint32_t so = rd32(d + pd + 24), sn = rd32(d + pd + 28);
            if (!sn || so + 20ull * sn > psz) continue;
            const uint8_t *e = d + pd + so + 20 * (sn - 1);
            uint32_t vctl = rd32(e + 12), fctl = rd32(e + 16), an = rd32(d + pd + 80), ao = rd32(d + pd + 84);
            uint32_t mask = 0;
            if (an <= 32 && ao + 8ull * an <= psz)
                for (uint32_t j = 0; j < an; j++) {
                    uint32_t no = rd32(d + pd + ao + 8 * j), loc = rd32(d + pd + ao + 8 * j + 4);
                    if (no + 2 < psz && loc < 32 && d[pd + no] == 'n' && d[pd + no + 1] >= 'A' && d[pd + no + 1] <= 'Z') mask |= 1u << loc;
                }
            uint32_t smask = 0;  // (samplers: count u16 at +0x38, list at +0x3C: name, unk, handle u8, visibility u8, unk u16)
            uint32_t sc = d[pd + 0x38] | (uint32_t)d[pd + 0x39] << 8, sp = rd32(d + pd + 0x3C);
            if (sc <= 32 && sp && sp + 12ull * sc <= psz)
                for (uint32_t j = 0; j < sc; j++) {
                    const uint8_t *se = d + pd + sp + 12 * j;
                    uint32_t no = rd32(se);
                    if (no < psz && se[8] < 32 && se[9] != 0 && stencil_name(d + pd + no, psz - no)) smask |= 1u << se[8];
                }
            if (np_ + 1 < max) {
                ctl_of[np_] = b + ctl + cur; mask_of[np_] = mask; np_++;
                if (fctl) { ctl_of[np_] = b + ctl + cur + vctl; mask_of[np_] = smask; np_++; }  // (its fragment shader)
            }
            cur += vctl + fctl;
        }
    }
    return np_;
}

// one container
static void scan_spch(const uint8_t *d, size_t n, size_t b) {
    if (b + 0x30 > n) return;
    enum { MAXP = 4096 };
    size_t *ctl_of = malloc(MAXP * sizeof *ctl_of);
    uint32_t *mask_of = malloc(MAXP * sizeof *mask_of);
    size_t nnamed = ctl_of && mask_of ? spch_inputs(d, n, b, ctl_of, mask_of, MAXP) : 0, ni = 0;
    size_t code = b + rd32(d + b + 0x20), code_end = code + rd32(d + b + 0x24);
    size_t ctl = b + rd32(d + b + 0x28), ctl_end = ctl + rd32(d + b + 0x2C);
    if (code_end > n || ctl_end > n || code_end < code || ctl_end < ctl) return;
    size_t pos = code;
    uint32_t last_vs = UINT32_MAX;  // a vertex shader waiting for its fragment shader
    for (size_t c = ctl; c + 0x710 <= ctl_end; c += 4) {
        if (rd32(d + c) != 0x98761234u) continue;
        size_t c0 = c;
        uint32_t csz = rd32(d + c + 0x6FC), coff = rd32(d + c + 0x700), blob = rd32(d + c + 0x704);
        if (!blob || blob > (4u << 20)) break;
        size_t t = pos;
        pos = (pos + blob + 255) & ~(size_t)255;
        n_controls++;
        c += 0x6FC;
        if (t + blob > code_end || rd32(d + t) != 0x12345678u) continue;
        uint64_t end = (uint64_t)coff + csz, h = 0xcbf29ce484222325ull;  // (objects.c set_shaders)
        if (end < 0x30 || end > (4u << 20)) end = 0x800;
        if (t + end > n) continue;
        for (uint64_t k = 0; k < end / 8; k++) { uint64_t w; memcpy(&w, d + t + k * 8, 8); h = (h ^ w) * 0x100000001b3ull; }
        uint32_t sph0 = rd32(d + t + 0x30);
        int stage = ((sph0 >> 10) & 0xF) == 1 ? STAGE_VERTEX : ((sph0 >> 10) & 0xF) == 5 ? STAGE_FRAGMENT : -1;
        if (stage < 0) continue;  // (geometry / tessellation: their keys come from the draw; 2 in the whole game)
        pthread_mutex_lock(&flock);
        uint32_t i = add_shader(d + t, blob, coff, csz, h, stage);
        if (nnamed) {  // (programs and control sections come in the same order)
            while (ni < nnamed && ctl_of[ni] < c0) ni++;
            if (ni < nnamed && ctl_of[ni] == c0) { found[i].int_in |= mask_of[ni]; found[i].named = true; }
        }
        if (stage == STAGE_VERTEX) {
            if (last_vs != UINT32_MAX) add_prog(last_vs, UINT32_MAX);
            last_vs = i;
        } else if (last_vs != UINT32_MAX) {
            add_prog(last_vs, i);
            last_vs = UINT32_MAX;
        }
        pthread_mutex_unlock(&flock);
    }
    if (last_vs != UINT32_MAX) { pthread_mutex_lock(&flock); add_prog(last_vs, UINT32_MAX); pthread_mutex_unlock(&flock); }
    free(ctl_of);
    free(mask_of);
}

// every container in a buffer (a file or a decompressed xbc1 block)
static void scan_buf(const uint8_t *d, size_t n) {
    for (size_t i = 0; i + 0x30 <= n; i += 4)
        if (!memcmp(d + i, "HCPS", 4)) scan_spch(d, n, i);
}

#ifndef _WIN32
#define _ftelli64 ftello  // (off_t is 64-bit on x86-64 Linux)
#endif
static pthread_mutex_t big_lock = PTHREAD_MUTEX_INITIALIZER;  // (one file over 64 MB at a time: memory on 8 GB PCs)
static void scan_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return;
    fseek(f, 0, SEEK_END);
    long long n = _ftelli64(f);
    fseek(f, 0, SEEK_SET);
    bool big = n > (64 << 20);
    if (big) pthread_mutex_lock(&big_lock);
    uint8_t *d = n > 0 ? malloc((size_t)n) : NULL;
    if (!d || fread(d, 1, (size_t)n, f) != (size_t)n) { free(d); fclose(f); if (big) pthread_mutex_unlock(&big_lock); return; }
    fclose(f);
    scan_buf(d, (size_t)n);
    // xbc1 blocks: "xbc1", u32 version, u32 decompressed size, u32 compressed size, u32 hash, name; zlib data at +0x30
    for (size_t i = 0; i + 0x30 <= (size_t)n; i += 4) {
        if (memcmp(d + i, "xbc1", 4)) continue;
        uint32_t dsz = rd32(d + i + 8), csz = rd32(d + i + 12);
        if (!dsz || dsz > (1u << 30) || i + 0x30 + (size_t)csz > (size_t)n) continue;
        uint8_t *x = malloc(dsz);
        uLongf out = dsz;
        if (x && uncompress(x, &out, d + i + 0x30, csz) == Z_OK) scan_buf(x, out);
        free(x);
        i += 0x30 + (csz & ~3u) - 4;
    }
    free(d);
    if (big) pthread_mutex_unlock(&big_lock);
}

// files that hold shaders (measured over the whole RomFS); the monolib containers (title, menus, lighting, post
// effects) come first: the game needs them first
static bool wanted(const char *name) {
    const char *e = strrchr(name, '.');
    if (!e) return false;
    static const char *ext[] = {".wishp", ".wimdo", ".wismt", ".wismda", ".wiefb", NULL};
    for (int i = 0; ext[i]; i++) if (!strcmp(e, ext[i])) return true;
    return false;
}
static char **paths;
static size_t npaths, cappaths;
static void list_dir(const char *dir) {
    DIR *dd = opendir(dir);
    if (!dd) return;
    struct dirent *e;
    while ((e = readdir(dd))) {
        if (e->d_name[0] == '.') continue;
        char p[1024];
        snprintf(p, sizeof p, "%s/%s", dir, e->d_name);
        struct stat sb;
        if (stat(p, &sb)) continue;
        if (S_ISDIR(sb.st_mode)) list_dir(p);
        else if (wanted(e->d_name)) {
            if (npaths == cappaths) paths = realloc(paths, (cappaths = cappaths ? cappaths * 2 : 1024) * sizeof *paths);
            paths[npaths++] = strdup(p);
        }
    }
    closedir(dd);
}
static int path_rank(const char *p) { return strstr(p, "/monolib/") ? 0 : strstr(p, ".wishp") ? 1 : 2; }
static int path_cmp(const void *a, const void *b) { return path_rank(*(char *const *)a) - path_rank(*(char *const *)b); }

// ------------------------------------------------------------------ building
static ShaderKey vkey_guess, fkey_guess;

static void stage_of(NvnProgram *np, int s, const Found *f) {
    np->stages[s].stage = s;
    np->stages[s].code_addr = (uint64_t)(uintptr_t)f->code;
    np->stages[s].const_offset = f->coff;
    np->stages[s].const_size = f->csz;
    np->stages[s].code_hash = f->hash;
    np->stage_mask |= 1u << s;
}

// [XCDE] T240: crash-safe (idea from issn_pc): a driver that crashes on one shader / pipeline crashed every later run
// at the same place. Each thread writes the job it is building to cache/precompile_inflight.bin (8 bytes per thread,
// flushed before the driver is called, zeroed after); the next run moves what is left there to
// cache/precompile_skip.txt and skips those jobs from then on (logged). The file is removed after a clean run.
static FILE *inflight;
static pthread_mutex_t inflight_lock = PTHREAD_MUTEX_INITIALIZER;
static _Atomic int inflight_slots;
static _Thread_local int inflight_slot = -1;
static Map skip_map;
static _Atomic uint64_t n_skipped;
static uint64_t job_key(const void *a, size_t na, uint64_t salt) {
    uint64_t h = 0xcbf29ce484222325ull ^ salt;
    for (size_t i = 0; i < na; i++) { h ^= ((const uint8_t *)a)[i]; h *= 0x100000001B3ull; }
    return h ? h : 1;
}
static bool job_begin(uint64_t key) {  // false: skip it (it crashed the driver before)
    if (skip_map.n && map_get(&skip_map, key)) { atomic_fetch_add(&n_skipped, 1); return false; }
    if (!inflight) return true;
    if (inflight_slot < 0) inflight_slot = atomic_fetch_add(&inflight_slots, 1);
    pthread_mutex_lock(&inflight_lock);
    fseek(inflight, (long)inflight_slot * 8, SEEK_SET);
    fwrite(&key, 8, 1, inflight);
    fflush(inflight);
    pthread_mutex_unlock(&inflight_lock);
    {   // (tests: XCDE_TEST_PRECOMPILE_CRASH=<n> - the process dies at the n-th job, as a crashing driver would)
        static _Atomic int n;
        static int at = -1;
        if (at < 0) at = getenv("XCDE_TEST_PRECOMPILE_CRASH") ? atoi(getenv("XCDE_TEST_PRECOMPILE_CRASH")) : 0;
        if (at && atomic_fetch_add(&n, 1) + 1 == at) { fflush(stdout); _exit(3); }
    }
    return true;
}
static void job_end(void) {
    if (!inflight || inflight_slot < 0) return;
    uint64_t z = 0;
    pthread_mutex_lock(&inflight_lock);
    fseek(inflight, (long)inflight_slot * 8, SEEK_SET);
    fwrite(&z, 8, 1, inflight);
    pthread_mutex_unlock(&inflight_lock);
}
static void inflight_open(void) {
    FILE *f = fopen("cache/precompile_inflight.bin", "rb");
    if (f) {  // the last run died: what it was building goes to the skip list
        FILE *o = fopen("cache/precompile_skip.txt", "a");
        uint64_t k;
        int n = 0;
        while (fread(&k, 8, 1, f) == 1) if (k) { if (o) fprintf(o, "%016llx\n", (unsigned long long)k); n++; }
        if (o) fclose(o);
        fclose(f);
        if (n) printf("[precompile] the last run stopped while building %d jobs: they are skipped from now on (cache/precompile_skip.txt)\n", n);
    }
    FILE *sk = fopen("cache/precompile_skip.txt", "r");
    char line[64];
    while (sk && fgets(line, sizeof line, sk)) { uint64_t k = strtoull(line, NULL, 16); if (k) map_put(&skip_map, k, 1); }
    if (sk) fclose(sk);
    if (skip_map.n) printf("[precompile] %zu jobs on the skip list (they crashed the driver)\n", skip_map.n);
    inflight = fopen("cache/precompile_inflight.bin", "wb+");
    if (inflight) { static uint8_t zero[64 * 8]; fwrite(zero, 1, sizeof zero, inflight); fflush(inflight); }
}

static void build_(const Found *vs, const Found *fs);
static void build(Pair pr, const Found *vs, const Found *fs) {
    (void)pr;
    uint64_t hk[2] = {vs->hash, fs ? fs->hash : 0};
    if (!job_begin(job_key(hk, sizeof hk, 1))) { atomic_fetch_add(&n_failed, 1); return; }  // (counted: the build loop waits for all)
    build_(vs, fs);
    job_end();
}
static void build_(const Found *vs, const Found *fs) {
    NvnProgram np;
    memset(&np, 0, sizeof np);
    stage_of(&np, 0, vs);
    if (fs) stage_of(&np, 1, fs);
    uint32_t attr = 0;
    int ok = vk_precompile_program(&np, &vkey_guess, &fkey_guess, &attr);
    if (ok && (attr & 2)) {  // (vertex input 1 read: also as an unsigned integer)
        ShaderKey k = vkey_guess;
        k.attr_kind[1] = 2;
        vk_precompile_program(&np, &k, &fkey_guess, NULL);
        atomic_fetch_add(&n_extra, 1);
    }
    // [XCDE] T211: in addition, the versions the container's names point to (they say what CAN be an integer: a
    // draw without that data binds a float format there - measured: names alone missed 170 variants, the guess 25)
    if (ok && vs->named && (vs->int_in & ~2u)) {  // integer inputs other than location 1 (nWgtIdx at 0: 356 programs)
        ShaderKey k = vkey_guess;
        for (int l = 0; l < 32; l++) if (vs->int_in >> l & 1) k.attr_kind[l] = 2;
        vk_precompile_program(&np, &k, &fkey_guess, NULL);
        atomic_fetch_add(&n_named_int, 1);
    }
    if (ok && fs && fs->named && fs->int_in) {  // stencil textures: also read as integers
        ShaderKey f = fkey_guess;
        f.tex_uint = fs->int_in;
        vk_precompile_program(&np, &vkey_guess, &f, NULL);
        if (attr & 2) { ShaderKey k = vkey_guess; k.attr_kind[1] = 2; vk_precompile_program(&np, &k, &f, NULL); }
        atomic_fetch_add(&n_named_int, 1);
    }
    atomic_fetch_add(ok ? &n_built : &n_failed, 1);
}

// [XCDE] T211: the shipped variant seed (data/variants_seed.bin, exec.c T163 records: code hashes [3] u64, has_gs u8,
// gs_in u8, 6 pad, vertex ShaderKey, fragment ShaderKey, pipeline key): each distinct (shaders, keys) built exactly -
// the versions the guesses above miss (integer textures / targets / vertex inputs: they follow the data a draw binds,
// not the shader). Geometry-stage programs are skipped.
typedef struct { uint32_t vs, fs; ShaderKey vkey, fkey; } SeedJob;
static SeedJob *seed_jobs;
static size_t nseed;
static _Atomic size_t next_seed;
static _Atomic uint64_t n_seed_built;
static _Atomic size_t n_seed_done;  // [PRECOMPILE-STALL-FIX] seed jobs finished (n_seed_built counts only the successes)
// [XCDE] T240: and every recorded variant's whole optimized pipeline (exec.c vk_precompile_variant): the shipped list
// (the debugger tours of every map and cutscene, tools/tour_batch.py) plus the player's own cache/variants.bin
extern int vk_precompile_variant(const NvnProgram *np, const void *rec, size_t rec_size);
typedef struct { uint32_t vs, fs; uint8_t *rec; } VarJob;
static VarJob *var_jobs;
static size_t nvar, var_rec_size;
static _Atomic size_t next_var;
static _Atomic uint64_t n_var_built, n_var_failed;
static Map var_done;
static void load_seed_file(const char *path, Map *done);
static void load_seed(void) {
    Map done = {0};
    load_seed_file("data/variants_seed.bin", &done);
    load_seed_file("cache/variants.bin", &done);
}
static void load_seed_file(const char *path, Map *done_) {
    Map *done = done_;
    FILE *f = fopen(path, "rb");
    if (!f) return;
    uint32_t hdr[2];
    if (fread(hdr, 4, 2, f) != 2 || hdr[0] != 0x56435843u || hdr[1] < 32 + 2 * sizeof(ShaderKey)) { fclose(f); return; }
    uint8_t *rec = malloc(hdr[1]);
    while (fread(rec, hdr[1], 1, f) == 1) {
        uint64_t h[3];
        memcpy(h, rec, 24);
        if (rec[24]) continue;  // (has_gs)
        uint32_t vi = map_get(&shader_map, h[0]), fi = h[1] ? map_get(&shader_map, h[1]) : 0;
        if (!vi || (h[1] && !fi)) continue;  // (a shader the scan didn't find)
        SeedJob j = {vi - 1, fi ? fi - 1 : UINT32_MAX};
        memcpy(&j.vkey, rec + 32, sizeof(ShaderKey));
        memcpy(&j.fkey, rec + 32 + sizeof(ShaderKey), sizeof(ShaderKey));
        uint64_t k = 0xcbf29ce484222325ull;
        const uint8_t *b = (const uint8_t *)&j;
        for (size_t i = 0; i < sizeof j; i++) { k ^= b[i]; k *= 0x100000001B3ull; }
        {   // (the whole record: one pipeline each)
            uint64_t h = 0xcbf29ce484222325ull;
            for (uint32_t i = 0; i < hdr[1]; i++) { h ^= rec[i]; h *= 0x100000001B3ull; }
            if (!h) h = 1;
            if (!map_get(&var_done, h) && (!var_rec_size || var_rec_size == hdr[1])) {
                map_put(&var_done, h, 1);
                var_rec_size = hdr[1];
                var_jobs = realloc(var_jobs, (nvar + 1) * sizeof *var_jobs);
                var_jobs[nvar].vs = j.vs, var_jobs[nvar].fs = j.fs;
                var_jobs[nvar].rec = malloc(hdr[1]);
                memcpy(var_jobs[nvar++].rec, rec, hdr[1]);
            }
        }
        if (!k) k = 1;
        if (map_get(done, k)) continue;
        map_put(done, k, 1);
        seed_jobs = realloc(seed_jobs, (nseed + 1) * sizeof *seed_jobs);
        seed_jobs[nseed++] = j;
    }
    free(rec);
    fclose(f);
}
static void *var_worker(void *arg) {
    (void)arg;
    plat_thread_background();
    for (size_t i; !atomic_load(&abandon) && (i = atomic_fetch_add(&next_var, 1)) < nvar;) {
        VarJob *j = &var_jobs[i];
        NvnProgram np;
        memset(&np, 0, sizeof np);
        stage_of(&np, 0, &found[j->vs]);
        if (j->fs != UINT32_MAX) stage_of(&np, 1, &found[j->fs]);
        if (!job_begin(job_key(j->rec, var_rec_size, 3))) { atomic_fetch_add(&n_var_failed, 1); continue; }
        atomic_fetch_add(vk_precompile_variant(&np, j->rec, var_rec_size) ? &n_var_built : &n_var_failed, 1);
        job_end();
    }
    return NULL;
}
static void *seed_worker(void *arg) {
    (void)arg;
    plat_thread_background();
    for (size_t i; !atomic_load(&abandon) && (i = atomic_fetch_add(&next_seed, 1)) < nseed;) {
        SeedJob *j = &seed_jobs[i];
        NvnProgram np;
        memset(&np, 0, sizeof np);
        stage_of(&np, 0, &found[j->vs]);
        if (j->fs != UINT32_MAX) stage_of(&np, 1, &found[j->fs]);
        if (!job_begin(job_key(j, sizeof *j, 2))) { atomic_fetch_add(&n_seed_done, 1); continue; }
        if (vk_precompile_program(&np, &j->vkey, &j->fkey, NULL)) atomic_fetch_add(&n_seed_built, 1);
        job_end();
        atomic_fetch_add(&n_seed_done, 1);
    }
    return NULL;
}

// [PRECOMPILE-STALL-FIX] A driver call that never returns (NVIDIA / Intel stuck inside a whole-pipeline compile) left
// this process waiting forever at the launcher's 100 %. The phases now watch their own progress: if no job finished
// for XCDE_PRECOMPILE_STALL_S seconds (default 120, 0 = never) the run stops the others, keeps what was in flight in
// cache/precompile_inflight.bin and exits with code 2, so the launcher restarts it by itself (crash-safe, T240) and the
// new run skips the stuck jobs.
static bool stalled(const char *phase, uint64_t count) {
    static char cur[16];
    static uint64_t lastc, lastt;
    static int limit_s = -1;
    if (limit_s < 0) limit_s = getenv("XCDE_PRECOMPILE_STALL_S") ? atoi(getenv("XCDE_PRECOMPILE_STALL_S")) : 120;
    uint64_t now = now_ns();
    if (strcmp(cur, phase) || count != lastc) { snprintf(cur, sizeof cur, "%s", phase); lastc = count; lastt = now; return false; }
    return limit_s > 0 && now - lastt > (uint64_t)limit_s * 1000000000ull;
}
static void give_up(const char *phase, uint64_t count) {
    printf("[precompile] %s: no job finished for a long time (%llu done): a driver call is stuck. Stopping the others; the stuck jobs go to the skip list and the run restarts%c", phase, (unsigned long long)count, 10);
    fflush(stdout);
    atomic_store(&abandon, 1);
    struct timespec ts = {3, 0};
    nanosleep(&ts, NULL);  // (the others finish the job they are on and stop: what stays in the file is what is stuck)
    if (inflight) fflush(inflight);
    _exit(2);
}

static void *worker(void *arg) {
    (void)arg;
    plat_thread_background();
    for (;;) {
        pthread_mutex_lock(&flock);
        while (next_job >= nprogs && !scan_done) pthread_cond_wait(&fcond, &flock);
        if (next_job >= nprogs || atomic_load(&abandon)) { pthread_mutex_unlock(&flock); return NULL; }
        Pair pr = progs[next_job++];
        Found vs = found[pr.vs], fs = pr.fs != UINT32_MAX ? found[pr.fs] : (Found){0};
        pthread_mutex_unlock(&flock);
        build(pr, &vs, pr.fs != UINT32_MAX ? &fs : NULL);
    }
}

static void *scanner(void *arg) {
    (void)arg;
    plat_thread_background();
    for (uint64_t i; (i = atomic_fetch_add(&next_file, 1)) < npaths;) {
        scan_file(paths[i]);
        atomic_fetch_add(&files_done, 1);
    }
    return NULL;
}

static void progress(const char *phase, bool force) {
    static uint64_t last, last_print;
    uint64_t now = now_ns();
    if (!force && now - last < 200000000ull) return;
    last = now;
    uint64_t done = n_built + n_failed, fd = files_done, ft = files_total, found_n = nprogs;
    if (!strcmp(phase, "pipelines")) { fd = n_var_built + n_var_failed; ft = found_n = nvar; done = fd; }  // [PRECOMPILE-STALL-FIX]
    FILE *f = fopen("cache/precompile_progress.tmp", "w");
    if (f) {
        fprintf(f, "%s %llu %llu %llu %llu%c", phase, (unsigned long long)fd, (unsigned long long)ft, (unsigned long long)found_n,
                (unsigned long long)done, 10);
        fclose(f);
        remove("cache/precompile_progress.txt");
        rename("cache/precompile_progress.tmp", "cache/precompile_progress.txt");
    }
    if (force || now - last_print > 2000000000ull) {
        last_print = now;
        printf("[precompile] %s: files %llu/%llu, shaders %zu, programs %zu, built %llu (failed %llu)%c", phase,
               (unsigned long long)files_done, (unsigned long long)files_total, nfound, nprogs, (unsigned long long)n_built,
               (unsigned long long)n_failed, 10);
        fflush(stdout);
    }
}

int vk_precompile_main(const char *root) {
    uint64_t t0 = now_ns();
    char romfs[1024];
    snprintf(romfs, sizeof romfs, "%s/romfs", root);
    mkdir("cache", 0755);
    mkdir("cache/spirv", 0755);
    remove("cache/precompile_done.txt");
    vkey_guess.glsl = fkey_guess.glsl = 1;
    vkey_guess.depth_minus_one = true;  // (the game's depth mode and window origin, as exec.c sets them per draw)
    list_dir(romfs);
    files_total = npaths;
    if (!npaths) { fprintf(stderr, "[precompile] no shader files under %s%c", romfs, 10); return 1; }
    qsort(paths, npaths, sizeof *paths, path_cmp);
    { const char *e = getenv("XCDE_PRECOMPILE_LIBS"); use_libs = !e || atoi(e) != 0; }
    setenv("XCDE_PRECOMPILING", "1", 1);  // (shaders.c: no cache-miss log lines)
    setenv("LARECOMP_HIDDEN", "1", 1);
    ui_window_init();  // (the game's own Vulkan device - same driver caches - in a hidden window)
    if (!use_libs) g_vk.gpl = false;
    // The libraries go to the driver's own disk cache only, not to cache/vk_pipelines.bin: with them that file grew to
    // 698 MB, and the game reads it whole into memory at start (+0.7 GB on 8 GB PCs). The game's own builds of these
    // libraries then hit the driver's cache (measured: tasks.md T187).
    vkDestroyPipelineCache(g_vk.dev, g_vk.pcache, NULL);
    g_vk.pcache = VK_NULL_HANDLE;
    inflight_open();  // [XCDE] T240 crash-safe
    int nthreads = plat_cpu_count() - 1;
    if (getenv("XCDE_PRECOMPILE_THREADS")) nthreads = atoi(getenv("XCDE_PRECOMPILE_THREADS"));
    else nthreads = xcde_shader_workers(plat_cpu_count(), SDL_GetSystemRAM(),
        g_vk.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, g_settings.gpu_performance);
    {   // [XCDE] T240 (idea from SavageWindRecomp 66b110a): the driver compiles take memory per thread - measured here
        // ~1.1 GB peak for 15 threads on NVIDIA, but other drivers' largest shaders take far more: at most one thread
        // per 400 MB of a quarter of the RAM (8 GB: 5 threads; the min-spec 4-core CPU has 3 anyway)
        extern int SDL_GetSystemRAM(void);
        int ram = SDL_GetSystemRAM(), cap = ram > 0 ? ram / 4 / 400 : nthreads;
        if (!getenv("XCDE_PRECOMPILE_THREADS") && cap < nthreads) {
            printf("[precompile] %d threads (memory: %d MB of RAM)%c", cap < 1 ? 1 : cap, ram, 10);
            nthreads = cap;
        }
    }
    if (!getenv("XCDE_PRECOMPILE_THREADS") && g_vk.props.vendorID == 0x1002 &&
        g_vk.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU && nthreads > 2) {  // [VEGA-FIX]
        printf("[precompile] AMD integrated GPU: 2 threads%c", 10);
        nthreads = 2;
    }
    printf("[precompile] GPU: %s (vendor 0x%04x), pipeline libraries %s%c", g_vk.props.deviceName, (unsigned)g_vk.props.vendorID,
           g_vk.gpl ? "on" : "off", 10);
    fflush(stdout);
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 64) nthreads = 64;
    pthread_t th[64], sc[64];
    for (int i = 0; i < nthreads; i++) pthread_create(&th[i], NULL, worker, NULL);
    for (int i = 0; i < nthreads; i++) pthread_create(&sc[i], NULL, scanner, NULL);
    while (files_done < npaths) {
        progress("scan", false);
        struct timespec ts = {0, 50000000};
        nanosleep(&ts, NULL);
    }
    for (int i = 0; i < nthreads; i++) pthread_join(sc[i], NULL);
    uint64_t t_scan = now_ns() - t0;
    pthread_mutex_lock(&flock);
    scan_done = true;
    pthread_cond_broadcast(&fcond);
    pthread_mutex_unlock(&flock);
    while (n_built + n_failed < nprogs) {
        if (stalled("build", n_built + n_failed)) give_up("build", n_built + n_failed);
        progress("build", false);
        struct timespec ts = {0, 100000000};
        nanosleep(&ts, NULL);
    }
    for (int i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
    load_seed();  // [XCDE] T211
    if (nseed) {
        uint64_t ts0 = now_ns();
        for (int i = 0; i < nthreads; i++) pthread_create(&th[i], NULL, seed_worker, NULL);
        while (atomic_load(&n_seed_done) < nseed) {
            if (stalled("seed", atomic_load(&n_seed_done))) give_up("seed", atomic_load(&n_seed_done));
            progress("seed", false); struct timespec ts = {0, 100000000}; nanosleep(&ts, NULL);
        }
        for (int i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
        printf("[precompile] seed: %zu variants, %llu built in %.1f s%c", nseed, (unsigned long long)n_seed_built, (now_ns() - ts0) / 1e9, 10);
    }
    if (nvar) {  // [XCDE] T240: the whole optimized pipelines (XCDE_PRECOMPILE_PIPELINES=0: pieces only, as before)
        const char *e = getenv("XCDE_PRECOMPILE_PIPELINES");
        // [VEGA-FIX] AMD integrated GPUs: the driver compiles of whole pipelines are what crashed there - off unless =1
        bool amd_igpu = g_vk.props.vendorID == 0x1002 && g_vk.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
        if (amd_igpu && !(e && atoi(e) == 1)) {
            printf("[precompile] AMD integrated GPU: whole-pipeline precompile skipped (XCDE_PRECOMPILE_PIPELINES=1 to force)%c", 10);
            e = "0";
        }
        if (!(e && atoi(e) == 0)) {
            uint64_t tv0 = now_ns();
            for (int i = 0; i < nthreads; i++) pthread_create(&th[i], NULL, var_worker, NULL);
            while (atomic_load(&next_var) < nvar || n_var_built + n_var_failed < nvar) {
                static uint64_t lastp;
                if (now_ns() - lastp > 2000000000ull) {
                    lastp = now_ns();
                    printf("[precompile] pipelines: %llu/%zu%c", (unsigned long long)(n_var_built + n_var_failed), nvar, 10);
                    fflush(stdout);
                }
                if (stalled("pipelines", n_var_built + n_var_failed)) give_up("pipelines", n_var_built + n_var_failed);
                progress("pipelines", false);
                struct timespec ts = {0, 100000000};
                nanosleep(&ts, NULL);
            }
            for (int i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
            printf("[precompile] pipelines: %zu recorded variants, %llu built, %llu not (shaders not found / failed) in %.1f s%c", nvar,
                   (unsigned long long)n_var_built, (unsigned long long)n_var_failed, (now_ns() - tv0) / 1e9, 10);
        }
    }
    progress("done", true);
    if (inflight) { fclose(inflight); inflight = NULL; remove("cache/precompile_inflight.bin"); }  // (a clean run)
    if (n_skipped) printf("[precompile] %llu jobs skipped (skip list)\n", (unsigned long long)n_skipped);
    double secs = (now_ns() - t0) / 1e9;
    printf("[precompile] %zu files, %llu control sections, %zu distinct shaders, %zu programs in %.1f s (scan %.1f s): "
           "%llu built (%llu also with integer input 1, %llu more from the container's names), %llu failed; %llu SPIR-V compiled, %llu from the cache; %d threads%c",
           npaths, (unsigned long long)n_controls, nfound, nprogs, secs, t_scan / 1e9, (unsigned long long)n_built,
           (unsigned long long)n_extra, (unsigned long long)n_named_int, (unsigned long long)n_failed, (unsigned long long)vk_spirv_misses,
           (unsigned long long)vk_spirv_hits, nthreads, 10);
    if (getenv("XCDE_PRECOMPILE_LIST")) {  // (tests: "S <hash> <stage>" per shader, "P <vs hash> <fs hash>" per program)
        FILE *l = fopen(getenv("XCDE_PRECOMPILE_LIST"), "w");
        for (size_t i = 0; l && i < nfound; i++) fprintf(l, "S %016llx %d%c", (unsigned long long)found[i].hash, found[i].stage, 10);
        for (size_t i = 0; l && i < nprogs; i++)
            fprintf(l, "P %016llx %016llx%c", (unsigned long long)found[progs[i].vs].hash,
                    (unsigned long long)(progs[i].fs != UINT32_MAX ? found[progs[i].fs].hash : 0), 10);
        if (l) fclose(l);
    }
    FILE *f = fopen("cache/precompile_done.txt", "w");
    if (f) {
        fprintf(f, "programs %zu built %llu failed %llu shaders %zu seconds %.0f%c", nprogs, (unsigned long long)n_built,
                (unsigned long long)n_failed, nfound, secs, 10);
        // [XCDE] T211: the build that precompiled. A new build may translate shaders differently (every translator fix
        // changes SPIR-V): its pieces would no longer match and the game compiled them while playing (10-58 ms hitches
        // on the first draws of each area). The launcher compares this line with build/xcde.exe and precompiles again.
        struct stat es;
        if (stat("build/xcde.exe", &es) == 0) fprintf(f, "exe %lld %lld%c", (long long)es.st_size, (long long)es.st_mtime, 10);
        fclose(f);
    }
    fflush(stdout);
    _exit(0);  // (no atexit work: the pipeline cache is saved above; the game's exit handler is for the game)
}
