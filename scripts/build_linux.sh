#!/usr/bin/env bash
# Native Linux (Steam Deck) build: build/xcde, an x86-64 ELF against the system SDL3/Vulkan/shaderc/ffmpeg/opus.
# Runs inside scripts/linux/Dockerfile; from any host with Docker:
#   docker build -t monado-linux scripts/linux
#   docker run --rm -v "$PWD:/src" monado-linux scripts/build_linux.sh
# Needs only targets/exefs (the code generator does not read romfs). XCDE_GLIBC=2.xx picks the oldest glibc to run on.
set -e
cd "$(dirname "$0")/.."
. scripts/plat.sh
[ $PLAT = linux ] || { echo "run this inside the Linux build container (see the header)"; exit 1; }
step() { echo; echo "=== $* ==="; }
step "1/3 Recompiling the game code into C (generated/)"
[ -f generated/.done ] && [ generated/.done -nt targets/exefs/main ] || {
  (cd tools && "$PY" -X utf8 gen_all.py --out ../generated) && touch generated/.done; }
step "2/3 Compiling the game code (build/obj/)"
XCDE_SKIP_BOOT_LINK=1 JOBS=${JOBS:-$(nproc)} scripts/build_boot.sh
step "3/3 Compiling the runtime and linking build/xcde"
scripts/build_rt_gfx.sh "$@"
[ -f build/xcde ] || { echo "the link failed (see build/link_gfx.err)"; exit 1; }
echo; echo "Done: build/xcde"
