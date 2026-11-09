// Virtual RomFS image over a folder of extracted game files (FR-020, M3).  [XCDE]
//
// The game's own SDK code parses a raw RomFS image through IStorage::Read(offset, size). Instead of writing a
// 13.6 GB image, this builds only the tables in memory and serves file data lazily from the host files. Layout as
// the Switch's: 0x50-byte header, data from 0x200 (each file 16-byte aligned), then dir hash table, dir entries,
// file hash table, file entries. Derived from LASRecomp runtime/romfs_build.c (MIT).
#include "romfs_virtual.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../platform.h"
#include "romfs_fd_cache.h"

#define EMPTY 0xFFFFFFFFu
#define DATA_START 0x200u
#define FD_CACHE XCDE_FD_CACHE

typedef struct Dir Dir;
typedef struct { char *name, *path; uint64_t size, data_off; uint32_t off; Dir *parent; int index; uint8_t *mem; bool traced; } File;
struct Dir { char *name, *path; Dir *parent; Dir **dirs; File **files; int ndirs, nfiles; uint32_t off; };
typedef struct { Dir **dirs; File **files; int nd, nf, cd, cf; } Lists;

static uint8_t *g_hdr_tables;  // header (0x200) .. data gap .. tables are served from separate buffers below
static uint8_t g_hdr[DATA_START];
static uint8_t *g_tables;      // dir hash + dir meta + file hash + file meta, contiguous
static uint64_t g_tables_off, g_tables_size, g_image_size, g_data_end;
static File **g_files;         // sorted by data_off (table order)
static int g_nfiles;
static XcdeFdEntry g_fds[FD_CACHE];
static pthread_mutex_t g_fd_lock = PTHREAD_MUTEX_INITIALIZER;  // [XCDE] the loader threads read concurrently (T108)
static uint64_t g_stamp;

static int by_name_dir(const void *a, const void *b) { return strcmp((*(Dir *const *)a)->name, (*(Dir *const *)b)->name); }
static int by_name_file(const void *a, const void *b) { return strcmp((*(File *const *)a)->name, (*(File *const *)b)->name); }

// [XCDE] Edited game files, served from memory in place of the folder's copy. monolib/shader/lib_nx.ini is the
// game's graphics configuration ("key=value" lines): xcde.ini `game_blur = 0` turns off its temporal anti-aliasing
// (tmaa), FXAA-style AntiAliasing, motion blur (blur2) and fur blur - made for the Switch's 720p, they soften a 4K
// image (the same edit as the community "AA/Blur Removal" mod). XCDE_LIBNX="key=value,key=value": test edits.
#include "../settings.h"
#include "../quality_policy.h"
static void edit_ini(File *f, const char *edits) {
    FILE *in = fopen(f->path, "rb");
    if (!in) return;
    char *src = malloc(f->size + 1);
    size_t n = fread(src, 1, f->size, in);
    fclose(in);
    src[n] = 0;
    size_t cap = n + strlen(edits) + 64, len = 0;
    char *out = malloc(cap);
    int changed = 0;
    for (char *line = src; *line;) {
        char *eol = strchr(line, '\n');
        size_t ll = eol ? (size_t)(eol - line + 1) : strlen(line);
        char key[64] = {0};
        size_t k = 0;
        while (k < ll && k < 63 && line[k] != '=' && line[k] != ' ' && line[k] != '\r' && line[k] != '\n') key[k] = line[k], k++;
        const char *v = NULL;
        size_t vl = 0;
        if (k && k < ll && (line[k] == '=' || line[k] == ' '))
            for (const char *e = edits; *e;) {  // "key=value,key=value"
                const char *eq = strchr(e, '='), *end = strchr(e, ',');
                if (!end) end = e + strlen(e);
                if (eq && eq < end && (size_t)(eq - e) == k && !memcmp(e, key, k)) { v = eq + 1; vl = (size_t)(end - eq - 1); }
                e = *end ? end + 1 : end;
            }
        if (v) {
            len += (size_t)snprintf(out + len, cap - len, "%s=%.*s\r\n", key, (int)vl, v);
            changed++;
        } else {
            memcpy(out + len, line, ll);
            len += ll;
        }
        line += ll;
    }
    free(src);
    f->mem = (uint8_t *)out;
    f->size = len;
    fprintf(stderr, "[romfs] %s: %d settings edited (%s)\n", f->path, changed, edits);
}
static void maybe_edit(File *f) {
    extern bool kbicons_wanted(const char *path), kbicons_init(void), settings_mod_enabled(const char *name);
    extern uint8_t *kbicons_make(const char *path, uint64_t *len);
    if (kbicons_wanted(f->path)) {  // [XCDE] T234: keyboard button prompts (runtime/services/kbicons.c)
        static int loaded;
        if (!loaded++) settings_load();
        uint64_t n;
        uint8_t *m = settings_mod_enabled("icons_keyboard") && kbicons_init() ? kbicons_make(f->path, &n) : NULL;
        if (m) free(f->mem), f->mem = m, f->size = n;
        return;
    }
    size_t pl = strlen(f->path);
    static const char lib[] = "monolib/shader/lib_nx.ini";
    if (pl < sizeof lib - 1 || strcmp(f->path + pl - (sizeof lib - 1), lib)) return;
    static char edits[1024];
    edits[0] = 0;  // (a mod's lib_nx.ini is edited again after it replaces the game's)
    settings_load();  // (the romfs may be scanned before the window reads it; reading it twice is harmless)
    if (!g_settings.game_blur) strcat(edits, "AntiAliasing=off,tmaa=off,blur2=off,furBlur=off");
    // [XCDE] FSR 3 / DLSS replace the game's own anti-aliasing (its temporal AA would blur their jittered input)
    else if (g_settings.upscaler == UPSCALE_FSR3 || g_settings.upscaler == UPSCALE_DLSS) strcat(edits, "AntiAliasing=off,tmaa=off");
    // [XCDE] xcde.ini taa = 0: only the game's temporal AA (tmaa) off; its other AA and motion blur stay as set above
    if (g_settings.taa_off) { if (*edits) strcat(edits, ","); strcat(edits, "tmaa=off"); }
    // T259: opt-in, whitelisted cosmetic edits. Mods are read first; explicit diagnostic edits below win.
    static const char *effect_keys[]={"tssao=off", "ssr=off", "bloom=off", "godray=off", "shadowCloud=off"};
    for (unsigned i=0;i<5;i++) if (g_settings.effects_off&(1u<<i)) {
        if (*edits) strcat(edits,","); strcat(edits,effect_keys[i]);
    }
    const char *e = getenv("XCDE_LIBNX");
    if (e && *e) { if (*edits) strcat(edits, ","); strncat(edits, e, sizeof edits - strlen(edits) - 1); }
    if (*edits) edit_ini(f, edits);
}

static Dir *scan(const char *path, const char *name, Dir *parent, int depth) {
    if (depth > 32) return NULL;
    DIR *d = opendir(path);
    if (!d) return NULL;
    Dir *dir = calloc(1, sizeof *dir);
    dir->name = strdup(name), dir->path = strdup(path), dir->parent = parent;
    int cd = 0, cf = 0;
    struct dirent *e;
    while ((e = readdir(d))) {
        if (e->d_name[0] == '.') continue;
        size_t n = strlen(path) + strlen(e->d_name) + 2;
        char *p = malloc(n);
        snprintf(p, n, "%s/%s", path, e->d_name);
        struct stat st;
        if (stat(p, &st) != 0) { free(p); continue; }
        if (S_ISDIR(st.st_mode)) {
            Dir *c = scan(p, e->d_name, dir, depth + 1);
            free(p);
            if (!c) continue;
            if (dir->ndirs == cd) dir->dirs = realloc(dir->dirs, sizeof *dir->dirs * (cd = cd ? cd * 2 : 8));
            dir->dirs[dir->ndirs++] = c;
        } else if (S_ISREG(st.st_mode)) {
            File *f = calloc(1, sizeof *f);
            f->name = strdup(e->d_name), f->path = p, f->size = (uint64_t)st.st_size, f->parent = dir;
            maybe_edit(f);
            if (dir->nfiles == cf) dir->files = realloc(dir->files, sizeof *dir->files * (cf = cf ? cf * 2 : 16));
            dir->files[dir->nfiles++] = f;
        } else {
            free(p);
        }
    }
    closedir(d);
    if (dir->ndirs) qsort(dir->dirs, dir->ndirs, sizeof *dir->dirs, by_name_dir);
    if (dir->nfiles) qsort(dir->files, dir->nfiles, sizeof *dir->files, by_name_file);
    return dir;
}

// [XCDE] Mods: mods/<name>/romfs/... (the Ryujinx / yuzu layout) replaces or adds game files. Every folder in mods/
// with a romfs/ inside is on unless xcde.ini says `mod_<name> = 0` (the launcher's Mods tab); they apply in
// alphabetical order, so a later mod wins over an earlier one for the same file. Only data: code mods (exefs / NSO
// patches, .pchtxt) don't apply to recompiled code. XCDE_MODS_DIR=<dir>: another folder; XCDE_NO_MODS=1: none.
// The applied files are listed in the log ([mods] lines).
static int ci_eq(const char *a, const char *b) {  // (case-insensitive: mod folders made on Windows may change case)
    for (; *a && *b; a++, b++) {
        char x = *a >= 'A' && *a <= 'Z' ? *a + 32 : *a, y = *b >= 'A' && *b <= 'Z' ? *b + 32 : *b;
        if (x != y) return 0;
    }
    return *a == *b;
}
static void overlay(Dir *dst, const char *path, int depth, int *replaced, int *added) {
    if (depth > 32) return;
    DIR *d = opendir(path);
    if (!d) return;
    struct dirent *e;
    while ((e = readdir(d))) {
        if (e->d_name[0] == '.') continue;
        size_t n = strlen(path) + strlen(e->d_name) + 2;
        char *p = malloc(n);
        snprintf(p, n, "%s/%s", path, e->d_name);
        struct stat st;
        if (stat(p, &st) != 0) { free(p); continue; }
        if (S_ISDIR(st.st_mode)) {
            Dir *c = NULL;
            for (int i = 0; i < dst->ndirs && !c; i++) if (ci_eq(dst->dirs[i]->name, e->d_name)) c = dst->dirs[i];
            if (!c) {
                c = calloc(1, sizeof *c);
                c->name = strdup(e->d_name), c->path = strdup(p), c->parent = dst;
                dst->dirs = realloc(dst->dirs, sizeof *dst->dirs * (dst->ndirs + 1));
                dst->dirs[dst->ndirs++] = c;
            }
            overlay(c, p, depth + 1, replaced, added);
            free(p);
        } else if (S_ISREG(st.st_mode)) {
            File *f = NULL;
            for (int i = 0; i < dst->nfiles && !f; i++) if (ci_eq(dst->files[i]->name, e->d_name)) f = dst->files[i];
            if (f) {
                (*replaced)++;
                free(f->path);
                free(f->mem);
                f->mem = NULL;
            } else {
                (*added)++;
                f = calloc(1, sizeof *f);
                f->name = strdup(e->d_name), f->parent = dst;
                dst->files = realloc(dst->files, sizeof *dst->files * (dst->nfiles + 1));
                dst->files[dst->nfiles++] = f;
            }
            f->path = p, f->size = (uint64_t)st.st_size;
            fprintf(stderr, "[mods]   %s\n", p);
            maybe_edit(f);
        } else {
            free(p);
        }
    }
    closedir(d);
    if (dst->ndirs) qsort(dst->dirs, dst->ndirs, sizeof *dst->dirs, by_name_dir);
    if (dst->nfiles) qsort(dst->files, dst->nfiles, sizeof *dst->files, by_name_file);
}
static int by_str(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }
static void apply_mods(Dir *root) {
    if (getenv("XCDE_NO_MODS")) return;
    const char *base = getenv("XCDE_MODS_DIR") ? getenv("XCDE_MODS_DIR") : "mods";
    DIR *d = opendir(base);
    if (!d) return;
    char *names[256];
    int n = 0;
    struct dirent *e;
    while ((e = readdir(d)) && n < 256) if (e->d_name[0] != '.') names[n++] = strdup(e->d_name);
    closedir(d);
    qsort(names, n, sizeof *names, by_str);
    extern bool settings_mod_enabled(const char *name);
    for (int i = 0; i < n; i++) {
        char p[1024];
        snprintf(p, sizeof p, "%s/%s/romfs", base, names[i]);
        struct stat st;
        if (stat(p, &st) != 0 || !S_ISDIR(st.st_mode)) { free(names[i]); continue; }
        if (!settings_mod_enabled(names[i])) { fprintf(stderr, "[mods] %s: off (xcde.ini)\n", names[i]); free(names[i]); continue; }
        int rep_ = 0, add = 0;
        fprintf(stderr, "[mods] %s:\n", names[i]);
        overlay(root, p, 0, &rep_, &add);
        fprintf(stderr, "[mods] %s: %d files replaced, %d added\n", names[i], rep_, add);
        free(names[i]);
    }
}

static void flatten(Dir *d, Lists *l) {
    for (int i = 0; i < d->nfiles; i++) {
        if (l->nf == l->cf) l->files = realloc(l->files, sizeof *l->files * (l->cf = l->cf ? l->cf * 2 : 256));
        l->files[l->nf++] = d->files[i];
    }
    for (int i = 0; i < d->ndirs; i++) {
        if (l->nd == l->cd) l->dirs = realloc(l->dirs, sizeof *l->dirs * (l->cd = l->cd ? l->cd * 2 : 64));
        l->dirs[l->nd++] = d->dirs[i];
    }
    for (int i = 0; i < d->ndirs; i++) flatten(d->dirs[i], l);
}

static uint32_t name_hash(uint32_t parent, const char *name) {
    uint32_t h = parent ^ 123456789u;
    for (const unsigned char *p = (const unsigned char *)name; *p; p++) h = ((h >> 5) | (h << 27)) ^ *p;
    return h;
}
static uint32_t bucket_count(uint32_t n) {
    if (n < 3) return 3;
    if (n < 19) return n | 1;
    uint32_t c = n;
    while (c % 2 == 0 || c % 3 == 0 || c % 5 == 0 || c % 7 == 0 || c % 11 == 0 || c % 13 == 0 || c % 17 == 0) c++;
    return c;
}
static uint32_t align4(uint32_t x) { return (x + 3) & ~3u; }
static uint64_t align16(uint64_t x) { return (x + 15) & ~15ull; }
static void put32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }
static void put64(uint8_t *p, uint64_t v) { memcpy(p, &v, 8); }

bool rvfs_init(const char *dir, char *why, size_t why_size) {
    Dir *root = scan(dir, "", NULL, 0);
    if (!root) { snprintf(why, why_size, "cannot read %s", dir); return false; }
    apply_mods(root);
    Lists l = {0};
    l.dirs = malloc(sizeof *l.dirs * 64), l.cd = 64;
    l.dirs[l.nd++] = root;
    flatten(root, &l);
    if (!l.nf) { snprintf(why, why_size, "no files in %s", dir); return false; }

    uint32_t dsize = 0, fsize = 0;
    for (int i = 0; i < l.nd; i++) l.dirs[i]->off = dsize, dsize += 0x18 + align4((uint32_t)strlen(l.dirs[i]->name));
    for (int i = 0; i < l.nf; i++) l.files[i]->off = fsize, fsize += 0x20 + align4((uint32_t)strlen(l.files[i]->name));
    uint64_t data = 0;
    for (int i = 0; i < l.nf; i++) {
        data = align16(data);
        l.files[i]->data_off = data;
        l.files[i]->index = i;
        data += l.files[i]->size;
    }
    uint32_t nbd = bucket_count((uint32_t)l.nd), nbf = bucket_count((uint32_t)l.nf);
    uint64_t dhash_off = (DATA_START + data + 3) & ~3ull;
    uint64_t dmeta_off = dhash_off + 4ull * nbd, fhash_off = dmeta_off + dsize, fmeta_off = fhash_off + 4ull * nbf;
    g_tables_off = dhash_off;
    g_tables_size = fmeta_off + fsize - dhash_off;
    g_tables = calloc(1, g_tables_size);
    uint32_t *dbuckets = (uint32_t *)g_tables;
    uint8_t *dtab = g_tables + (dmeta_off - dhash_off);
    uint32_t *fbuckets = (uint32_t *)(g_tables + (fhash_off - dhash_off));
    uint8_t *ftab = g_tables + (fmeta_off - dhash_off);
    for (uint32_t i = 0; i < nbd; i++) dbuckets[i] = EMPTY;
    for (uint32_t i = 0; i < nbf; i++) fbuckets[i] = EMPTY;
    for (int i = 0; i < l.nd; i++) {
        Dir *d = l.dirs[i];
        uint8_t *e = dtab + d->off;
        uint32_t parent = d->parent ? d->parent->off : 0, nl = (uint32_t)strlen(d->name), sib = EMPTY;
        if (d->parent)
            for (int k = 0; k + 1 < d->parent->ndirs; k++)
                if (d->parent->dirs[k] == d) sib = d->parent->dirs[k + 1]->off;
        uint32_t h = name_hash(parent, d->name) % nbd;
        put32(e, parent), put32(e + 4, sib);
        put32(e + 8, d->ndirs ? d->dirs[0]->off : EMPTY);
        put32(e + 12, d->nfiles ? d->files[0]->off : EMPTY);
        put32(e + 16, dbuckets[h]), put32(e + 20, nl);
        memcpy(e + 0x18, d->name, nl);
        dbuckets[h] = d->off;
    }
    for (int i = 0; i < l.nf; i++) {
        File *f = l.files[i];
        uint8_t *e = ftab + f->off;
        uint32_t nl = (uint32_t)strlen(f->name), sib = EMPTY;
        for (int k = 0; k + 1 < f->parent->nfiles; k++)
            if (f->parent->files[k] == f) sib = f->parent->files[k + 1]->off;
        uint32_t h = name_hash(f->parent->off, f->name) % nbf;
        put32(e, f->parent->off), put32(e + 4, sib);
        put64(e + 8, f->data_off), put64(e + 16, f->size);
        put32(e + 24, fbuckets[h]), put32(e + 28, nl);
        memcpy(e + 0x20, f->name, nl);
        fbuckets[h] = f->off;
    }
    uint64_t hv[10] = {0x50, dhash_off, 4ull * nbd, dmeta_off, dsize, fhash_off, 4ull * nbf, fmeta_off, fsize, DATA_START};
    memcpy(g_hdr, hv, sizeof hv);
    g_files = l.files;
    g_nfiles = l.nf;
    g_data_end = DATA_START + data;
    g_image_size = g_tables_off + g_tables_size;
    for (int i = 0; i < FD_CACHE; i++) g_fds[i].fd = -1, g_fds[i].file = -1;
    free(l.dirs);
    (void)g_hdr_tables;
    return true;
}

uint64_t rvfs_size(void) { return g_image_size; }

// [XCDE] T108: the cache used to be unlocked, so two threads whose files shared a slot closed each other's descriptor
// mid-pread (-> read as zeros, or another file's bytes after the fd number was reused). A slot in use is never recycled:
// the colliding reader gets a private descriptor instead. `*slot` is -1 for a private one.
static int file_fd(int idx, int *slot) {
    return xcde_fd_acquire(g_fds,&g_fd_lock,&g_stamp,idx,g_files[idx]->path,O_RDONLY|O_BINARY_FLAG,slot);
}
static void file_fd_release(int fd, int slot) { xcde_fd_release(g_fds,&g_fd_lock,fd,slot); }

static void copy_mem(uint8_t *dst, uint64_t off, uint64_t n, const uint8_t *src, uint64_t src_off) {
    memcpy(dst, src + (off - src_off), (size_t)n);
}

// Reads `size` bytes at image offset `off` into dst; bytes outside any file or table read as zero. Returns bytes.
int64_t rvfs_read(uint8_t *dst, uint64_t size, uint64_t off) {
    plat_ov_guest_write((uint64_t)(uintptr_t)dst, size);  // [XCDE] (XCDE_OVERLAP_STATS: the kernel can't fault on it)
    if (off >= g_image_size) return 0;
    if (off + size > g_image_size) size = g_image_size - off;
    uint64_t done = 0;
    while (done < size) {
        uint64_t o = off + done, n = size - done;
        if (o < DATA_START) {  // header
            if (n > DATA_START - o) n = DATA_START - o;
            copy_mem(dst + done, o, n, g_hdr, 0);
        } else if (o < g_data_end) {  // file data region
            uint64_t rel = o - DATA_START;
            int lo = 0, hi = g_nfiles - 1, hit = -1;
            while (lo <= hi) {  // last file whose data_off <= rel
                int mid = (lo + hi) / 2;
                if (g_files[mid]->data_off <= rel) hit = mid, lo = mid + 1; else hi = mid - 1;
            }
            File *f = hit >= 0 ? g_files[hit] : NULL;
            {   // [XCDE] XCDE_TRACE_FILE=<a,b,..>: the first read of each file whose path holds one of the substrings is
                // logged with the game frame and the host backtrace (guest functions f_<addr>): who loads maps / events
                static const char *tf = (const char *)1;
                if (tf == (const char *)1) tf = getenv("XCDE_TRACE_FILE");
                if (tf && f && rel < f->data_off + f->size && !f->traced) {
                    char pat[256];
                    snprintf(pat, sizeof pat, "%s", tf);
                    for (char *tok = strtok(pat, ","); tok; tok = strtok(NULL, ","))
                        if (strstr(f->path, tok)) {
                            extern uint64_t vk_frame_counter(void);
                            f->traced = true;
                            fprintf(stderr, "[file] frame %llu: %s\n", (unsigned long long)vk_frame_counter(), f->path);
                            if (getenv("XCDE_TRACE_FILE_BT")) plat_print_backtrace();
                            break;
                        }
                }
            }
            if (f && rel < f->data_off + f->size && f->mem) {  // (an edited file)
                uint64_t in = rel - f->data_off, avail = f->size - in;
                if (n > avail) n = avail;
                memcpy(dst + done, f->mem + in, (size_t)n);
            } else if (f && rel < f->data_off + f->size) {
                uint64_t in = rel - f->data_off, avail = f->size - in;
                if (n > avail) n = avail;
                int slot, fd = file_fd(hit, &slot);
                ssize_t r = fd >= 0 ? pread(fd, dst + done, (size_t)n, (off_t)in) : -1;
                file_fd_release(fd, slot);
                if (r < 0) fprintf(stderr, "[romfs] read error %s +%llx (%llu bytes): zero-filled\n", g_files[hit]->path, (unsigned long long)in, (unsigned long long)n);
                if (r < 0) r = 0;
                if ((uint64_t)r < n) memset(dst + done + r, 0, (size_t)(n - r));
            } else {  // padding between files
                uint64_t next = (hit + 1 < g_nfiles) ? DATA_START + g_files[hit + 1]->data_off : g_data_end;
                if (n > next - o) n = next - o;
                memset(dst + done, 0, (size_t)n);
            }
        } else if (o < g_tables_off) {  // gap after the data
            if (n > g_tables_off - o) n = g_tables_off - o;
            memset(dst + done, 0, (size_t)n);
        } else {  // tables
            copy_mem(dst + done, o, n, g_tables, g_tables_off);
        }
        done += n;
    }
    return (int64_t)done;
}

// Path of the file containing image offset `off` (NULL if none).
const char *rvfs_file_at(uint64_t off) {
    if (off < DATA_START || off >= g_data_end) return NULL;
    uint64_t rel = off - DATA_START;
    int lo = 0, hi = g_nfiles - 1, hit = -1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (g_files[mid]->data_off <= rel) hit = mid, lo = mid + 1; else hi = mid - 1;
    }
    return hit >= 0 && rel < g_files[hit]->data_off + g_files[hit]->size ? g_files[hit]->path : NULL;
}
