// [XCDE] Keyboard button prompts (T234): the game's button icon atlases redrawn with the keys bound to each Switch
// button (runtime/hle/bindings.c), so the prompts follow any remapping. Built when the romfs is assembled
// (runtime/services/romfs_virtual.c maybe_edit) for xcde.ini mod_icons_keyboard = 1 (the launcher's "Iconos de
// botones: Teclado", or "auto" without a controller), and served in place of the game's files.
//
// The atlases (menu/image/*.wilay, LAHD): one texture (a "mibl": Tegra block-linear data + a 0x28-byte LBIM footer)
// of 8 x 4 cells in the same order in each of the three styles - icon_button (white shapes, the label cut out),
// icon_button_hud (cream keys, dark label), button (glowing keys, blue label). A key is the game's own SL key
// (cell 16) with its label erased (the pixels inside the shape that aren't the key's face) and the key's name
// written in the label's colour. The new texture is BC3 with one mip level (as GameBanana's button packs ship it).
#include <SDL3/SDL.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <zlib.h>
#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#endif

#include "../input_bind.h"

#define BCDEC_IMPLEMENTATION
#define BCDEC_STATIC
#include "../../third_party/bcdec/bcdec.h"
#define STBTT_STATIC
#define STB_TRUETYPE_IMPLEMENTATION
#include "../../third_party/imgui/imstb_truetype.h"

#define LOGK(...) fprintf(stderr, "[kbicons] " __VA_ARGS__)

// cells (8 a row): which one shows what, in the white / cream atlases and in the glowing one (button.wilay: the
// d-pad first, a second ZL / ZR, no single d-pad directions); -1 none
typedef struct {
    int sl, minus, plus, x, y, a, b, lsclick, rsclick, lstick, rstick, l, r, zl, zr, zl2, zr2;
    int up, down, left, right, updown, leftright, all;
} Layout;
static const Layout k_plain = {16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, -1, 22, 23, 24, 25, 26, 27, 28};
static const Layout k_glow = {17, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, -1, -1, -1, -1, 28, 29, 0};

static const char *const k_files[] = {"icon_button.wilay", "icon_button_hud.wilay", "button.wilay", "icon_button_deco.wilay"};
// Single button glyphs inside the screens' own layout files (found by what GameBanana's Xbox pack redraws): the
// file, the texture's index in its LAHD table, the Switch button (BIND_*). Each is its own key template.
static const struct { const char *file; int tex, slot; } k_single[] = {
    {"mnu001_btn.wilay", 0, BIND_PLUS},
    {"mnu013_cont01.wilay", 4, BIND_LSTICK},
    {"mnu038_cont01.wilay", 92, BIND_ZR}, {"mnu038_cont01.wilay", 93, BIND_ZL},
    {"mnu040_cont02.wilay", 16, BIND_ZR}, {"mnu040_cont02.wilay", 17, BIND_ZL},
    {"mnu043_cont01.wilay", 7, BIND_ZL}, {"mnu043_cont01.wilay", 8, BIND_ZR},
    {"mnu053_cont01.wilay", 5, BIND_ZR}, {"mnu053_cont01.wilay", 27, BIND_ZR},  // (27: the field's "Mission" prompt)
    {"mnu061_cont01.wilay", 4, BIND_Y}, {"mnu061_cont02.wilay", 4, BIND_Y}, {"mnu061_cont03.wilay", 4, BIND_Y},
    {"mnu069_cont01.wilay", 5, BIND_R},
    {"mnu073_cont03.wilay", 2, BIND_ZR},
};
#define N_SINGLE (sizeof k_single / sizeof *k_single)

static bool name_is(const char *path, const char *file) {
    size_t n = strlen(path);
    char want[64];
    snprintf(want, sizeof want, "menu/image/%s", file);
    size_t w = strlen(want);
    return n >= w && !strcmp(path + n - w, want) && (n == w || path[n - w - 1] == '/' || path[n - w - 1] == '\\');
}

bool kbicons_wanted(const char *path) {
    for (size_t i = 0; i < N_SINGLE; i++) if (name_is(path, k_single[i].file)) return true;
    size_t n = strlen(path);
    for (size_t i = 0; i < sizeof k_files / sizeof *k_files; i++) {
        char want[64];
        snprintf(want, sizeof want, "menu/image/%s", k_files[i]);
        size_t w = strlen(want);
        if (n >= w && !strcmp(path + n - w, want) && (n == w || path[n - w - 1] == '/' || path[n - w - 1] == '\\')) return true;
    }
    return false;
}

// ------------------------------------------------------------------ labels
static const char *key_label(int k) {
    static char buf[16];
    switch (k) {
    case 0: return "";
    case BIND_MOUSE_LEFT: return "LMB";
    case BIND_MOUSE_RIGHT: return "RMB";
    case BIND_MOUSE_MIDDLE: return "MMB";
    case BIND_MOUSE_X1: return "M4";
    case BIND_MOUSE_X2: return "M5";
    case BIND_WHEEL_UP: return "W\xe2\x86\x91";
    case BIND_WHEEL_DOWN: return "W\xe2\x86\x93";
    case SDL_SCANCODE_UP: return "\xe2\x86\x91";
    case SDL_SCANCODE_DOWN: return "\xe2\x86\x93";
    case SDL_SCANCODE_LEFT: return "\xe2\x86\x90";
    case SDL_SCANCODE_RIGHT: return "\xe2\x86\x92";
    case SDL_SCANCODE_RETURN: case SDL_SCANCODE_KP_ENTER: return "\xe2\x86\xb5";  // (keyboard symbols: drawn, see is_shape)
    case SDL_SCANCODE_ESCAPE: return "ESC";
    case SDL_SCANCODE_BACKSPACE: return "\xe2\x8c\xab";
    case SDL_SCANCODE_SPACE: return "\xe2\x90\xa3";
    case SDL_SCANCODE_TAB: return "TAB";
    case SDL_SCANCODE_LSHIFT: case SDL_SCANCODE_RSHIFT: return "\xe2\x87\xa7";
    case SDL_SCANCODE_LCTRL: case SDL_SCANCODE_RCTRL: return "CTRL";
    case SDL_SCANCODE_LALT: case SDL_SCANCODE_RALT: return "ALT";
    case SDL_SCANCODE_CAPSLOCK: return "CAPS";
    case SDL_SCANCODE_DELETE: return "DEL";
    case SDL_SCANCODE_INSERT: return "INS";
    case SDL_SCANCODE_PAGEUP: return "PGUP";
    case SDL_SCANCODE_PAGEDOWN: return "PGDN";
    default: break;
    }
    if (k >= SDL_SCANCODE_KP_1 && k <= SDL_SCANCODE_KP_0) {
        snprintf(buf, sizeof buf, "N%c", k == SDL_SCANCODE_KP_0 ? '0' : '1' + (k - SDL_SCANCODE_KP_1));
        return buf;
    }
    const char *n = k > 0 && k < SDL_SCANCODE_COUNT ? SDL_GetScancodeName((SDL_Scancode)k) : "";
    int j = 0;
    for (; n[j] && j < 6; j++) buf[j] = (char)(n[j] >= 'a' && n[j] <= 'z' ? n[j] - 32 : n[j]);
    buf[j] = 0;
    return buf;
}
static int slot_key(int slot) { return g_bind_key[slot] ? g_bind_key[slot] : g_bind_key2[slot]; }
static const char *slot_label(int slot) { return key_label(slot_key(slot)); }

// a stick's four keys: "WASD" when they are single characters (up, left, down, right as on a keyboard), else ""
static void stick_label(int up, char *out, size_t n) {
    const int order[4] = {up, up + 2, up + 1, up + 3};  // (BIND_*_UP, _LEFT, _DOWN, _RIGHT)
    out[0] = 0;
    for (int i = 0; i < 4; i++) {
        const char *l = slot_label(order[i]);
        if (strlen(l) != 1) { out[0] = 0; return; }
        strncat(out, l, n - strlen(out) - 1);
    }
}

// ------------------------------------------------------------------ text (a system font)
static stbtt_fontinfo g_font;
static unsigned char *g_font_data;
static bool font_load(void) {
    if (g_font_data) return true;
    static const char *const fonts[] = {"C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeui.ttf",
                                        // [XCDE] SteamOS / Linux
                                        "/usr/share/fonts/noto/NotoSans-Bold.ttf", "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
                                        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"};
    for (size_t i = 0; i < sizeof fonts / sizeof *fonts; i++) {
        FILE *f = fopen(fonts[i], "rb");
        if (!f) continue;
        fseek(f, 0, SEEK_END);
        long n = ftell(f);
        fseek(f, 0, SEEK_SET);
        g_font_data = malloc((size_t)n);
        if (fread(g_font_data, 1, (size_t)n, f) == (size_t)n && stbtt_InitFont(&g_font, g_font_data, 0)) { fclose(f); return true; }
        fclose(f);
        free(g_font_data);
        g_font_data = NULL;
    }
    return false;
}
static int utf8_next(const char **s) {
    const unsigned char *p = (const unsigned char *)*s;
    int c = *p++;
    if (c >= 0xe0) { c = (c & 0x0f) << 12 | (p[0] & 0x3f) << 6 | (p[1] & 0x3f); p += 2; }
    else if (c >= 0xc0) { c = (c & 0x1f) << 6 | (p[0] & 0x3f); p += 1; }
    *s = (const char *)p;
    return c;
}
// arrows (U+2190..2193) are drawn as solid triangles, as the game's own d-pad arrows (a font's are thin lines)
// arrows (U+2190..2193) and the keyboard symbols shift / enter / space / backspace (U+21E7, 21B5, 2423, 232B) are drawn as
// solid shapes, as the game's own d-pad arrows: a font's are thin lines or missing, and a long word ("SHIFT") would have
// to shrink to fit a key
enum { SH_SHIFT = 0x21e7, SH_ENTER = 0x21b5, SH_SPACE = 0x2423, SH_BKSP = 0x232b };
static bool is_arrow(int c) { return (c >= 0x2190 && c <= 0x2193) || c == SH_SHIFT || c == SH_ENTER || c == SH_SPACE || c == SH_BKSP; }
static float glyph_adv(int c, float scale, float ph) {
    if (c == SH_SPACE || c == SH_ENTER) return ph * 0.9f;
    if (is_arrow(c)) return ph * 0.62f;
    int adv, lsb;
    stbtt_GetCodepointHMetrics(&g_font, c, &adv, &lsb);
    return adv * scale;
}
static float text_width(const char *t, float scale, float ph) {
    float w = 0;
    for (const char *p = t; *p;) {
        int c = utf8_next(&p);
        w += glyph_adv(c, scale, ph);
        if (*p && !is_arrow(c)) w += stbtt_GetCodepointKernAdvance(&g_font, c, *p) * scale;
    }
    return w;
}
static void cov_add(uint8_t *cov, int cw, int ch, int x, int y, int v) {
    if (x < 0 || y < 0 || x >= cw || y >= ch) return;
    v += cov[y * cw + x];
    cov[y * cw + x] = (uint8_t)(v > 255 ? 255 : v);
}
// a triangle pointing (0 left, 1 up, 2 right, 3 down) centred at cx, cy, `s` across (4x4 samples a pixel)
// is (u, v) - the shape's box, -1..1, v down - inside the shape for codepoint c
static bool shape_in(int c, float u, float v) {
    switch (c) {
    case SH_SHIFT:  // a block arrow up
        return (v >= -0.9f && v <= 0.05f && fabsf(u) <= (v + 0.9f) * 0.95f) || (v > 0.05f && v <= 0.85f && fabsf(u) <= 0.38f);
    case SH_BKSP:   // a block arrow left
        return (u >= -0.9f && u <= -0.05f && fabsf(v) <= (u + 0.9f) * 0.95f) || (u > -0.05f && u <= 0.9f && fabsf(v) <= 0.38f);
    case SH_ENTER:  // the return arrow: down the right side, then left to an arrowhead
        return (u >= 0.5f && u <= 0.85f && v >= -0.85f && v <= 0.45f) || (u >= -0.45f && u <= 0.85f && v >= 0.1f && v <= 0.45f) ||
               (u >= -0.95f && u < -0.45f && fabsf(v - 0.275f) <= (u + 0.95f) * 1.3f);
    case SH_SPACE:  // the open box of a space
        return (v >= 0.35f && v <= 0.7f && fabsf(u) <= 0.95f) || (fabsf(u) >= 0.65f && fabsf(u) <= 0.95f && v >= -0.2f && v <= 0.7f);
    default: {      // the d-pad arrows: U+2190 left, up, right, down
        int dir = c - 0x2190;
        float a = dir == 0 ? -u : dir == 2 ? u : dir == 1 ? -v : v, b = dir == 0 || dir == 2 ? v : u;  // a along
        return a >= -0.8f && a <= 0.9f && fabsf(b) <= (0.9f - a) * 0.6f;
    }
    }
}
// shape c centred at cx, cy, `s` across (4x4 samples a pixel)
static void arrow_draw(uint8_t *cov, int cw, int ch, float cx, float cy, float s, int c) {
    for (int y = (int)(cy - s); y <= (int)(cy + s); y++)
        for (int x = (int)(cx - s); x <= (int)(cx + s); x++) {
            int in = 0;
            for (int k = 0; k < 16; k++) {
                float u = (x + (k % 4 + 0.5f) / 4 - cx) / (s / 2), v = (y + (k / 4 + 0.5f) / 4 - cy) / (s / 2);
                in += shape_in(c, u, v);
            }
            if (in) cov_add(cov, cw, ch, x, y, in * 255 / 16);
        }
}
// coverage (0..255) of `t` centred in the box, into cov (cw x ch)
static void text_draw(const char *t, uint8_t *cov, int cw, int ch, float bx, float by, float bw, float bh) {
    float ph = bh * 0.92f, scale = stbtt_ScaleForPixelHeight(&g_font, ph);
    float w = text_width(t, scale, ph);
    if (w > bw * 0.92f) { float f = bw * 0.92f / w; scale *= f, ph *= f; w = text_width(t, scale, ph); }
    int asc, desc, gap;
    stbtt_GetFontVMetrics(&g_font, &asc, &desc, &gap);
    // the caps' height centred: the baseline at the middle + half the cap height (~0.7 of the ascent)
    float base = by + bh / 2 + asc * scale * 0.36f, x = bx + (bw - w) / 2;
    for (const char *p = t; *p;) {
        int c = utf8_next(&p), x0, y0, x1, y1;
        if (is_arrow(c)) {
            float adv = glyph_adv(c, scale, ph);
            arrow_draw(cov, cw, ch, x + adv / 2, by + bh / 2, c >= 0x2190 && c <= 0x2193 ? ph * 0.52f : ph * 0.8f, c);
            x += glyph_adv(c, scale, ph);
            continue;
        }
        float fx = x - (int)x;
        stbtt_GetCodepointBitmapBoxSubpixel(&g_font, c, scale, scale, fx, 0, &x0, &y0, &x1, &y1);
        int gw = x1 - x0, gh = y1 - y0;
        if (gw > 0 && gh > 0) {
            uint8_t *g = calloc(1, (size_t)gw * gh);
            stbtt_MakeCodepointBitmapSubpixel(&g_font, g, gw, gh, gw, scale, scale, fx, 0, c);
            for (int yy = 0; yy < gh; yy++)
                for (int xx = 0; xx < gw; xx++) cov_add(cov, cw, ch, (int)x + x0 + xx, (int)base + y0 + yy, g[yy * gw + xx]);
            free(g);
        }
        x += glyph_adv(c, scale, ph);
        if (*p) x += stbtt_GetCodepointKernAdvance(&g_font, c, *p) * scale;
    }
}

// ------------------------------------------------------------------ texture (Tegra block linear, BC7 / BC3 / RGBA8)
static int block_height(int hb) {
    int bh = 16;
    while (bh > 1 && hb <= (bh / 2) * 8) bh /= 2;
    return bh;
}
// byte offset of the 16-byte unit at (xb bytes, y rows) in a block-linear surface `row` bytes wide
static size_t bl_off(int xb, int y, int row, int bh) {
    int gx = (row + 63) / 64;
    size_t gob = (size_t)(y / (8 * bh)) * gx * bh + (size_t)(xb / 64) * bh + (y % (8 * bh)) / 8;
    return gob * 512 + ((xb % 64) / 32) * 256 + ((y % 8) / 2) * 64 + ((xb % 32) / 16) * 32 + (y % 2) * 16;
}
static size_t bl_size(int row, int rows) {
    int bh = block_height(rows);
    int gx = (row + 63) / 64, gy = (rows + 8 * bh - 1) / (8 * bh) * bh;
    return (size_t)gx * gy * 512;
}

static uint8_t *decode(const uint8_t *src, size_t len, int w, int h, int fmt) {
    bool bc = fmt != 37;
    int bs = bc ? 4 : 1, bpe = fmt == 37 ? 4 : 16;
    int wb = (w + bs - 1) / bs, hb = (h + bs - 1) / bs, row = wb * bpe, bh = block_height(hb);
    if (fmt != 37 && fmt != 68 && fmt != 77) return NULL;
    if (bl_size(row, hb) > len) return NULL;
    uint8_t *px = calloc(1, (size_t)w * h * 4);
    for (int y = 0; y < hb; y++)
        for (int xb = 0; xb < row; xb += 16) {
            const uint8_t *u = src + bl_off(xb, y, row, bh);
            if (!bc) { memcpy(px + (size_t)y * w * 4 + xb, u, 16); continue; }
            uint8_t blk[64];
            if (fmt == 77) bcdec_bc7(u, blk, 16); else bcdec_bc3(u, blk, 16);
            int bx = xb / 16 * 4, by = y * 4;
            for (int r = 0; r < 4; r++) memcpy(px + ((size_t)(by + r) * w + bx) * 4, blk + r * 16, 16);
        }
    return px;
}

static uint16_t to565(const float c[3]) {
    int r = (int)(c[0] * 31 / 255 + 0.5f), g = (int)(c[1] * 63 / 255 + 0.5f), b = (int)(c[2] * 31 / 255 + 0.5f);
    return (uint16_t)(r << 11 | g << 5 | b);
}
static void from565(uint16_t v, float c[3]) {
    c[0] = (v >> 11) * 255.0f / 31, c[1] = (v >> 5 & 63) * 255.0f / 63, c[2] = (v & 31) * 255.0f / 31;
}
// one 4x4 RGBA block -> BC3 (alpha: 8-value ramp between its min / max; colour: the ramp between its darkest and
// brightest visible pixels)
static void bc3_block(const uint8_t *p, int pitch, uint8_t out[16]) {
    int amin = 255, amax = 0;
    for (int i = 0; i < 16; i++) {
        int a = p[(i / 4) * pitch + (i % 4) * 4 + 3];
        if (a < amin) amin = a;
        if (a > amax) amax = a;
    }
    uint64_t ai = 0;
    if (amax > amin)
        for (int i = 0; i < 16; i++) {
            int a = p[(i / 4) * pitch + (i % 4) * 4 + 3];
            int k = (int)((float)(amax - a) / (amax - amin) * 7 + 0.5f);
            ai |= (uint64_t)(k == 0 ? 0 : k == 7 ? 1 : k + 1) << (3 * i);
        }
    out[0] = (uint8_t)amax, out[1] = (uint8_t)amin;
    for (int i = 0; i < 6; i++) out[2 + i] = (uint8_t)(ai >> (8 * i));
    float lo[3] = {0}, hi[3] = {0}, llo = 1e9f, lhi = -1;
    for (int i = 0; i < 16; i++) {
        const uint8_t *q = p + (i / 4) * pitch + (i % 4) * 4;
        if (q[3] < 8 && amax >= 8) continue;
        float l = q[0] * 0.3f + q[1] * 0.59f + q[2] * 0.11f;
        if (l < llo) llo = l, lo[0] = q[0], lo[1] = q[1], lo[2] = q[2];
        if (l > lhi) lhi = l, hi[0] = q[0], hi[1] = q[1], hi[2] = q[2];
    }
    uint16_t c0 = to565(hi), c1 = to565(lo);
    uint32_t ci = 0;
    if (c0 < c1) { uint16_t t = c0; c0 = c1; c1 = t; }
    if (c0 != c1) {
        float e0[3], e1[3], d[3], dd = 0;
        from565(c0, e0), from565(c1, e1);
        for (int k = 0; k < 3; k++) d[k] = e1[k] - e0[k], dd += d[k] * d[k];
        for (int i = 0; i < 16; i++) {
            const uint8_t *q = p + (i / 4) * pitch + (i % 4) * 4;
            float t = ((q[0] - e0[0]) * d[0] + (q[1] - e0[1]) * d[1] + (q[2] - e0[2]) * d[2]) / dd;
            int k = (int)(t * 3 + 0.5f);
            k = k < 0 ? 0 : k > 3 ? 3 : k;
            ci |= (uint32_t)(k == 0 ? 0 : k == 3 ? 1 : k == 1 ? 2 : 3) << (2 * i);
        }
    }
    memcpy(out + 8, &c0, 2), memcpy(out + 10, &c1, 2), memcpy(out + 12, &ci, 4);
}
// RGBA -> block-linear BC3 (`*len` bytes)
static uint8_t *encode_bc3(const uint8_t *px, int w, int h, size_t *len) {
    int wb = (w + 3) / 4, hb = (h + 3) / 4, row = wb * 16, bh = block_height(hb);
    *len = bl_size(row, hb);
    uint8_t *out = calloc(1, *len);
    for (int y = 0; y < hb; y++)
        for (int x = 0; x < wb; x++) bc3_block(px + ((size_t)y * 4 * w + x * 4) * 4, w * 4, out + bl_off(x * 16, y, row, bh));
    return out;
}

// ------------------------------------------------------------------ a key in a cell
typedef struct { uint8_t *px; int w, h; } Img;
static uint8_t *at(Img *im, int x, int y) { return im->px + ((size_t)y * im->w + x) * 4; }

// the template key: a key (the atlases' SL cell, or a single glyph) with its label erased; face / label colours
typedef struct { uint8_t cell[128 * 128 * 4]; int w, h; uint8_t face[4], label[4]; int fx0, fy0, fx1, fy1; } Key;

static bool key_make(Img *im, int cx, int cy, int cw, int chh, Key *k) {
    if (cw > 128 || chh > 128) return false;
    k->w = cw, k->h = chh;
    for (int y = 0; y < chh; y++) memcpy(k->cell + (size_t)y * cw * 4, at(im, cx, cy + y), (size_t)cw * 4);
    const int W = cw, H = chh, N = cw * chh;
    // the face: the commonest opaque colour (5 bits a channel)
    static int hist[32768];
    memset(hist, 0, sizeof hist);
    int best = -1;
    for (int i = 0; i < N; i++) {
        uint8_t *p = k->cell + i * 4;
        if (p[3] < 240) continue;
        int q = (p[0] >> 3) << 10 | (p[1] >> 3) << 5 | p[2] >> 3;
        if (best < 0 || ++hist[q] > hist[best]) best = q;
    }
    if (best < 0) return false;
    long sum[3] = {0}, cnt = 0;
    for (int i = 0; i < N; i++) {
        uint8_t *p = k->cell + i * 4;
        if (p[3] >= 240 && ((p[0] >> 3) << 10 | (p[1] >> 3) << 5 | p[2] >> 3) == best) sum[0] += p[0], sum[1] += p[1], sum[2] += p[2], cnt++;
    }
    for (int c = 0; c < 3; c++) k->face[c] = (uint8_t)(sum[c] / cnt);
    k->face[3] = 255;
    // not face: reachable from the cell's edge = outside (background, outline, glow); the rest = the label
    uint8_t *cls = calloc(1, (size_t)N);  // 0 face, 1 not face, 2 outside
    for (int i = 0; i < N; i++) {
        uint8_t *p = k->cell + i * 4;
        int d = abs(p[0] - k->face[0]) + abs(p[1] - k->face[1]) + abs(p[2] - k->face[2]);
        cls[i] = p[3] >= 230 && d < 60 ? 0 : 1;
    }
    int *st = malloc(sizeof(int) * N), sp = 0;
    for (int i = 0; i < W; i++) {  // (the edges)
        int e[2] = {i, (H - 1) * W + i};
        for (int j = 0; j < 2; j++) if (cls[e[j]] == 1) cls[e[j]] = 2, st[sp++] = e[j];
    }
    for (int i = 0; i < H; i++) {
        int e[2] = {i * W, i * W + W - 1};
        for (int j = 0; j < 2; j++) if (cls[e[j]] == 1) cls[e[j]] = 2, st[sp++] = e[j];
    }
    while (sp) {
        int i = st[--sp], x = i % W, y = i / W;
        int nb[4] = {x > 0 ? i - 1 : -1, x < W - 1 ? i + 1 : -1, y > 0 ? i - W : -1, y < H - 1 ? i + W : -1};
        for (int j = 0; j < 4; j++) if (nb[j] >= 0 && cls[nb[j]] == 1) cls[nb[j]] = 2, st[sp++] = nb[j];
    }
    free(st);
    int lab = -1, labd = -1;
    k->fx0 = W, k->fy0 = H, k->fx1 = 0, k->fy1 = 0;
    for (int i = 0; i < N; i++) {
        uint8_t *p = k->cell + i * 4;
        if (cls[i] != 2) {
            int x = i % W, y = i / W;
            if (x < k->fx0) k->fx0 = x;
            if (x > k->fx1) k->fx1 = x;
            if (y < k->fy0) k->fy0 = y;
            if (y > k->fy1) k->fy1 = y;
        }
        if (cls[i] != 1) continue;
        int d = abs(p[0] - k->face[0]) + abs(p[1] - k->face[1]) + abs(p[2] - k->face[2]) + 2 * (255 - p[3]);
        if (d > labd) labd = d, lab = i, memcpy(k->label, p, 4);
        memcpy(p, k->face, 4);
    }
    if (lab < 0) {  // (a label touching the key's frame, e.g. a framed R: the opaque non-face pixels in the middle)
        for (int i = 0; i < N; i++) {
            int x = i % W, y = i / W;
            uint8_t *p = k->cell + i * 4;
            if (cls[i] != 2 || p[3] < 200 || x < W / 4 || x >= W - W / 4 || y < H / 5 || y >= H - H / 5) continue;
            int d = abs(p[0] - k->face[0]) + abs(p[1] - k->face[1]) + abs(p[2] - k->face[2]);
            if (d > labd) labd = d, lab = i, memcpy(k->label, p, 4);
            memcpy(p, k->face, 4);
        }
    }
    free(cls);
    if (lab < 0) return false;
    // (pixels a little off the face colour next to the old label: the face colour too)
    for (int i = 0; i < N; i++) {
        uint8_t *p = k->cell + i * 4;
        int d = abs(p[0] - k->face[0]) + abs(p[1] - k->face[1]) + abs(p[2] - k->face[2]);
        if (p[3] >= 230 && d < 60) memcpy(p, k->face, 4);
    }
    return true;
}

// a computer mouse, drawn in the label colour (the camera stick with the mouse camera on)
static void mouse_cov(uint8_t *cov, int cw, int chh, float bx, float by, float bw, float bh) {
    float cx = bx + bw / 2, cy = by + bh / 2, rx = bh * 0.22f, ry = bh * 0.33f, t = bh * 0.07f;
    for (int y = 0; y < chh; y++)
        for (int x = 0; x < cw; x++) {
            float c = 0;
            for (int s = 0; s < 4; s++) {  // (2x2 samples)
                float px = x + 0.25f + 0.5f * (s & 1) - cx, py = y + 0.25f + 0.5f * (s >> 1) - cy;
                float e = sqrtf(px * px / (rx * rx) + py * py / (ry * ry));
                bool ring = e <= 1 && e >= 1 - t / rx;
                bool split = py < 0 && fabsf(px) < t / 2 && e <= 1;
                bool mid = py < -ry * 0.05f && py > -ry * 0.12f - t && e <= 1;
                if (ring || split || mid) c += 0.25f;
            }
            int v = cov[y * cw + x] + (int)(c * 255);
            cov[y * cw + x] = (uint8_t)(v > 255 ? 255 : v);
        }
}

// the key at (cx, cy) of the texture: the template with `text` (or a mouse) on its face
static void key_draw_at(Img *im, const Key *k, int cx, int cy, const char *text, bool mouse) {
    int W = k->w, H = k->h;
    uint8_t *cov = calloc(1, (size_t)W * H);
    float bx = k->fx0, by = k->fy0, bw = k->fx1 - k->fx0 + 1, bh = k->fy1 - k->fy0 + 1;
    // (the face's inner area: the outline / glow aside)
    bx += bw * 0.07f, bw *= 0.86f, by += bh * 0.1f, bh *= 0.8f;
    if (mouse) mouse_cov(cov, W, H, bx, by, bw, bh);
    else if (text && *text) text_draw(text, cov, W, H, bx, by, bw, bh);
    if (getenv("XCDE_KBICONS_DUMP")) {
        long s = 0;
        for (int i = 0; i < W * H; i++) s += cov[i];
        LOGK("key at %d,%d \"%s\" box %.0f,%.0f %.0fx%.0f coverage %ld face %02x%02x%02x%02x label %02x%02x%02x%02x\n", cx, cy, text ? text : "",
             bx, by, bw, bh, s, k->face[0], k->face[1], k->face[2], k->face[3], k->label[0], k->label[1], k->label[2], k->label[3]);
    }
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            const uint8_t *s = k->cell + ((size_t)y * W + x) * 4;
            uint8_t *d = at(im, cx + x, cy + y);
            float c = cov[y * W + x] / 255.0f;
            bool inface = s[0] == k->face[0] && s[1] == k->face[1] && s[2] == k->face[2] && s[3] == k->face[3];
            if (!inface) c = 0;  // (the label stays on the face)
            for (int ch = 0; ch < 4; ch++) d[ch] = (uint8_t)(s[ch] * (1 - c) + k->label[ch] * c + 0.5f);
        }
    free(cov);
}
static int g_cell;  // (the atlas being drawn: its cell size)
static void key_draw(Img *im, const Key *k, int cellno, const char *text, bool mouse) {
    if (cellno >= 0) key_draw_at(im, k, (cellno % 8) * g_cell, (cellno / 8) * g_cell, text, mouse);
}

static void atlas_draw(Img *im, int n, const Layout *L) {
    static Key k;
    g_cell = n;
    if (!key_make(im, (L->sl % 8) * n, (L->sl / 8) * n, n, n, &k)) { LOGK("no key template\n"); return; }
    const struct { int cell, slot; } single[] = {
        {L->minus, BIND_MINUS}, {L->plus, BIND_PLUS}, {L->x, BIND_X}, {L->y, BIND_Y}, {L->a, BIND_A}, {L->b, BIND_B},
        {L->lsclick, BIND_LSTICK}, {L->rsclick, BIND_RSTICK}, {L->l, BIND_L}, {L->r, BIND_R}, {L->zl, BIND_ZL}, {L->zr, BIND_ZR},
        {L->zl2, BIND_ZL}, {L->zr2, BIND_ZR}, {L->up, BIND_UP}, {L->down, BIND_DOWN}, {L->left, BIND_LEFT}, {L->right, BIND_RIGHT},
    };
    for (size_t i = 0; i < sizeof single / sizeof *single; i++) key_draw(im, &k, single[i].cell, slot_label(single[i].slot), false);
    char t[32];
    snprintf(t, sizeof t, "%s%s", slot_label(BIND_UP), slot_label(BIND_DOWN));
    key_draw(im, &k, L->updown, t, false);
    snprintf(t, sizeof t, "%s%s", slot_label(BIND_LEFT), slot_label(BIND_RIGHT));
    key_draw(im, &k, L->leftright, t, false);
    bool arrows = slot_key(BIND_UP) == SDL_SCANCODE_UP && slot_key(BIND_DOWN) == SDL_SCANCODE_DOWN &&
                  slot_key(BIND_LEFT) == SDL_SCANCODE_LEFT && slot_key(BIND_RIGHT) == SDL_SCANCODE_RIGHT;
    if (arrows) snprintf(t, sizeof t, "\xe2\x86\x90\xe2\x86\x91\xe2\x86\x93\xe2\x86\x92");
    else snprintf(t, sizeof t, "%s%s%s%s", slot_label(BIND_UP), slot_label(BIND_LEFT), slot_label(BIND_DOWN), slot_label(BIND_RIGHT));
    key_draw(im, &k, L->all, t, false);
    stick_label(BIND_LS_UP, t, sizeof t);
    key_draw(im, &k, L->lstick, *t ? t : "MOVE", false);
    stick_label(BIND_RS_UP, t, sizeof t);
    key_draw(im, &k, L->rstick, t, g_mouse_look || !*t);
}

// ------------------------------------------------------------------ the file
static uint8_t *read_all(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    *n = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *d = malloc(*n ? *n : 1);
    if (fread(d, 1, *n, f) != *n) { free(d); d = NULL; }
    fclose(f);
    return d;
}

// The game's file at `path` with the keys drawn: a new buffer (*len bytes), or NULL (left as it is)
static uint8_t *kbicons_draw(const char *path, uint64_t *len) {
    if (!font_load()) { LOGK("no system font\n"); return NULL; }
    size_t n;
    if (getenv("XCDE_KBICONS_DUMP")) LOGK("%s: draw\n", path);
    uint8_t *d = read_all(path, &n);
    if (!d) return NULL;
    if (n >= 0x30 && !memcmp(d, "xbc1", 4)) {  // (zlib-compressed)
        uint32_t dsz, csz;
        memcpy(&dsz, d + 8, 4), memcpy(&csz, d + 12, 4);
        uLongf out = dsz;
        uint8_t *x = malloc(dsz);
        if (0x30 + (size_t)csz > n || uncompress(x, &out, d + 0x30, csz) != Z_OK) { free(x); free(d); return NULL; }
        free(d);
        d = x, n = out;
    }
    uint8_t *res = NULL;
    uint32_t t, ro, cnt, off, size, ft[9];
    if (n < 0x1000 || (memcmp(d, "LAHD", 4) && memcmp(d, "LAGP", 4))) goto done;  // (atlases: LAHD; the screens' layouts: LAGP)
    memcpy(&t, d + 0x24, 4);
    if (t + 8 > n) goto done;
    memcpy(&ro, d + t, 4), memcpy(&cnt, d + t + 4, 4);
    if (getenv("XCDE_KBICONS_DUMP")) LOGK("%s: %zu bytes, table %x +%x, %u textures\n", path, n, t, ro, cnt);
    {   // single glyphs: each texture redrawn in its own block (BC3 needs no more room than the BC7 / BC3 it replaces:
        // the same size at mip 0, without the smaller mips), so nothing else in the file moves
        bool any = false;
        for (size_t i = 0; i < N_SINGLE; i++) {
            if (!name_is(path, k_single[i].file) || (uint32_t)k_single[i].tex >= cnt || t + ro + 12ull * (k_single[i].tex + 1) > n) continue;
            const uint8_t *it = d + t + ro + 12 * k_single[i].tex;
            memcpy(&off, it + 4, 4), memcpy(&size, it + 8, 4);
            if ((size_t)t + off + size > n || size < 0x28 || memcmp(d + t + off + size - 4, "LBIM", 4)) { LOGK("%s #%d: no texture\n", path, k_single[i].tex); continue; }
            uint8_t *blk = d + t + off;
            memcpy(ft, blk + size - 0x28, 36);
            int w = (int)ft[2], h = (int)ft[3], fmt = (int)ft[6];
            if (w > 128 || h > 128) continue;
            uint8_t *px = decode(blk, size - 0x28, w, h, fmt);
            if (!px) { LOGK("%s #%d: %dx%d format %d not decoded\n", path, k_single[i].tex, w, h, fmt); continue; }
            Img im = {px, w, h};
            static Key k;
            if (key_make(&im, 0, 0, w, h, &k)) key_draw_at(&im, &k, 0, 0, slot_label(k_single[i].slot), false);
            size_t bl;
            uint8_t *bc = encode_bc3(px, w, h, &bl);
            free(px);
            if (bl + 0x28 <= size) {
                memset(blk, 0, size);
                memcpy(blk, bc, bl);
                uint32_t nf[10] = {(uint32_t)bl, ft[1], (uint32_t)w, (uint32_t)h, 1, 1, 68, 1, ft[8], 0};
                memcpy(nf + 9, "LBIM", 4);
                memcpy(blk + size - 0x28, nf, 0x28);
                any = true;
                LOGK("%s #%d: %dx%d key drawn (format %d -> BC3)\n", path, k_single[i].tex, w, h, fmt);
            }
            free(bc);
        }
        if (any) { res = d, d = NULL, *len = n; goto done; }
    }
    if (cnt != 1 || t + ro + 12 > n) goto done;
    memcpy(&off, d + t + ro + 4, 4), memcpy(&size, d + t + ro + 8, 4);
    if ((size_t)t + off + size > n || size < 0x28 || memcmp(d + t + off + size - 4, "LBIM", 4)) goto done;
    memcpy(ft, d + t + off + size - 0x28, 36);  // data size, ?, width, height, depth, dims, format, mips, version
    {
        int w = (int)ft[2], h = (int)ft[3], fmt = (int)ft[6];
        int cell = w / 8;
        if (w > 1024 || h > 1024 || cell * 8 != w || cell * 4 != h || cell > 64) goto done;
        uint8_t *px = decode(d + t + off, size - 0x28, w, h, fmt);
        if (!px) { LOGK("%s: format %d not handled\n", path, fmt); goto done; }
        Img im = {px, w, h};
        const char *base = strrchr(path, '/');
        atlas_draw(&im, cell, base && !strcmp(base + 1, "button.wilay") ? &k_glow : &k_plain);
        const char *dump = getenv("XCDE_KBICONS_DUMP");  // (tests: <dir>/<file>_<w>x<h>.rgba)
        if (dump) {
            char p[512];
            const char *b = strrchr(path, '/');
            snprintf(p, sizeof p, "%s/%s_%dx%d.rgba", dump, b ? b + 1 : path, w, h);
            FILE *o = fopen(p, "wb");
            if (o) fwrite(px, 4, (size_t)w * h, o), fclose(o);
        }
        size_t bl;
        uint8_t *bc = encode_bc3(px, w, h, &bl);
        free(px);
        size_t data = (bl + 0xfff) & ~(size_t)0xfff, block = data + 0x1000, head = (size_t)t + off;
        res = calloc(1, head + block);
        memcpy(res, d, head);
        memcpy(res + head, bc, bl);
        free(bc);
        uint32_t nf[10] = {(uint32_t)data, 0x1000, (uint32_t)w, (uint32_t)h, 1, 1, 68, 1, ft[8], 0};
        memcpy(nf + 9, "LBIM", 4);
        memcpy(res + head + block - 0x28, nf, 0x28);
        uint32_t bsz = (uint32_t)block;
        memcpy(res + t + ro + 8, &bsz, 4);
        *len = head + block;
        LOGK("%s: %dx%d keys drawn (format %d -> BC3)\n", path, w, h, fmt);
    }
done:
    free(d);
    return res;
}

// Drawn once per set of bindings: cache/kbicons/<file>_<key>.wilay, the key a hash of what the drawing depends on (the
// key bindings, the mouse camera, the game's file, this code's version). A launch with the same bindings reads that
// file; a change draws again (and removes that file's older copies).
#define KBICONS_VERSION 4
static uint64_t fnv(uint64_t h, const void *p, size_t n) {
    for (size_t i = 0; i < n; i++) h = (h ^ ((const uint8_t *)p)[i]) * 0x100000001b3ull;
    return h;
}
uint8_t *kbicons_make(const char *path, uint64_t *len) {
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    uint64_t h = 0xcbf29ce484222325ull;
    int ver = KBICONS_VERSION;
    h = fnv(h, &ver, sizeof ver);
    h = fnv(h, g_bind_key, sizeof g_bind_key);
    h = fnv(h, g_bind_key2, sizeof g_bind_key2);
    h = fnv(h, &g_mouse_look, sizeof g_mouse_look);
    h = fnv(h, base, strlen(base));
    FILE *src = fopen(path, "rb");
    if (src) { fseek(src, 0, SEEK_END); long sz = ftell(src); fclose(src); h = fnv(h, &sz, sizeof sz); }
    char cp[512];
    snprintf(cp, sizeof cp, "cache/kbicons/%s_%016llx", base, (unsigned long long)h);
    size_t n;
    uint8_t *d = getenv("XCDE_KBICONS_DUMP") ? NULL : read_all(cp, &n);
    if (d && n > 0x1000 && (!memcmp(d, "LAHD", 4) || !memcmp(d, "LAGP", 4))) { *len = n; return d; }
    free(d);
    uint64_t t0 = SDL_GetTicksNS();
    d = kbicons_draw(path, len);
    if (!d) return NULL;
    LOGK("%s drawn in %.1f ms\n", base, (SDL_GetTicksNS() - t0) / 1e6);
#ifdef _WIN32
    _mkdir("cache"), _mkdir("cache/kbicons");
    char pat[512];  // (the older copies of this file)
    snprintf(pat, sizeof pat, "cache/kbicons/%s_*", base);
    WIN32_FIND_DATAA fd;
    HANDLE fh = FindFirstFileA(pat, &fd);
    if (fh != INVALID_HANDLE_VALUE) {
        do {
            char old[512];
            snprintf(old, sizeof old, "cache/kbicons/%s", fd.cFileName);
            if (strcmp(old, cp)) DeleteFileA(old);
        } while (FindNextFileA(fh, &fd));
        FindClose(fh);
    }
#endif
    char tmp[520];
    snprintf(tmp, sizeof tmp, "%s.tmp", cp);
    FILE *o = fopen(tmp, "wb");
    if (o) {
        bool ok = fwrite(d, 1, (size_t)*len, o) == (size_t)*len;
        ok = fclose(o) == 0 && ok;
        if (ok) rename(tmp, cp); else remove(tmp);
    }
    return d;
}

bool kbicons_init(void) { return true; }
