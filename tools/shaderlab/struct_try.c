// Offline check of translate.c's structurize(): takes a translated GLSL file that uses the branch state machine,
// replaces the state machine with structurize()'s output and writes the result, so both versions can be compiled
// (glslc) and compared (shaderlab).   struct_try <in.glsl> <out.glsl>   (exit 1: structurize refused the program)
#include "../../runtime/nvn/maxwell/translate.c"

int main(int argc, char **argv) {
    if (argc < 3) return 2;
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 2;
    static char src[1 << 22];
    size_t n = fread(src, 1, sizeof src - 1, f);
    src[n] = 0;
    fclose(f);
    const char *head = "  uint pc = 8u;", *open = "  switch (pc) {\n", *tail = "    exit_now = true; break;\n  default: exit_now = true; break;\n  }\n  if (exit_now) break;\n  }\n";
    char *h = strstr(src, head), *o = h ? strstr(h, open) : NULL, *t = o ? strstr(o, tail) : NULL;
    if (!t) { fprintf(stderr, "no state machine\n"); return 3; }
    *t = 0;
    char *s = structurize(o + strlen(open));
    if (!s) { fprintf(stderr, "refused\n"); return 1; }
    FILE *w = fopen(argv[2], "wb");
    fwrite(src, 1, (size_t)(h - src), w);
    fputs(s, w);
    fputs(t + strlen(tail), w);
    fclose(w);
    return 0;
}
