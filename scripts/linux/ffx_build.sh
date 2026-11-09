#!/usr/bin/env bash
# [XCDE] libamd_fidelityfx_vk.so: AMD FidelityFX SDK v1.1.4 (MIT; the last release with a Vulkan backend) built for
# Linux x86-64 -- the Linux counterpart of amd_fidelityfx_vk.dll (temporal.c): the five ffx* entry points of ffx-api with
# FSR 3.1 upscaling, FSR 2, frame generation (optical flow + frame interpolation) and the Vulkan frame-generation
# swapchain proxy. Run from Git Bash on the Windows PC (the SDK's shader compiler, FidelityFX_SC.exe, is Windows-only):
#   scripts/linux/ffx_build.sh            all steps -> $FFX/out/libamd_fidelityfx_vk.so (FFX default: D:/repos/monado/ffx)
#   scripts/linux/ffx_build.sh smoke      scripts/linux/ffx/smoke.c on lavapipe (image monado-lab)
#   scripts/linux/ffx_build.sh <step>     one step (below); FFX_DEBUG=1 ... compile: unstripped -O1 -g -D_DEBUG build;
#                                         GDB=1 ... smoke: the smoke test under gdb (backtrace on a crash)
# Steps:
#   1 fetch    the tag's tarball through gh ($FFX/dl/sdk.tgz; raw.githubusercontent may be blocked)
#   2 prepare  ffx-api/ and sdk/{include,src} into $FFX/src, then scripts/linux/ffx/linux.patch: Win32-only includes and
#              Win32 swapchain structs guarded, no amdxc64 driver providers, MSVC-isms (ui64 literals, temporaries bound
#              to references, static_cast of void* to a function pointer), wide "%s" -> "%ls", include path case, a
#              larger internal context size (4-byte wchar_t), 16-byte aligned backend scratch arrays (clang's SSE stores),
#              and a memory-type fallback for devices without a device-local-only type (lavapipe). The remaining Win32
#              calls (critical sections, events, threads, QPC, MSVC _s CRT) come from ffx/ffx_linux_compat.h,
#              force-included (its self-test runs before the compile).
# Callers: like temporal.c's ffx_proc, pass a vkDeviceProcAddr that falls back from KHR names to core ones (the
# backend asks for vkGetBufferMemoryRequirements2KHR & co; Mesa's loader returns NULL unless the extension is enabled).
#   3 shaders  FidelityFX_SC.exe + glslangValidator.exe from the SDK's tools/binary_store, run natively, with the
#              arguments of the SDK's CMake (CMakeShaders*.txt + CMakeCompile*Shaders.txt): GLSL -> SPIR-V permutation
#              headers in $FFX/build/shaders/vk (4 variants per pass: base, _wave64, _16bit, _wave64_16bit)
#   4 compile  in the monado-linux image with zig c++ for glibc 2.38; libc++ linked in statically; only the five
#              ffx* symbols exported (version script); links libvulkan.so.1 (the backend calls a few vk* directly)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
FFX=${FFX:-/d/repos/monado/ffx}
TAG=v1.1.4
COMMIT=c6efa6b  # (tag v1.1.4 = c6efa6bf7f2027b3ec94f28578bb5965eabb9e55; the tarball's top directory carries it)
GLIBC=${XCDE_GLIBC:-2.38}

fetch() {
    mkdir -p "$FFX/dl"
    [ -s "$FFX/dl/sdk.tgz" ] || gh api "repos/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/tarball/$TAG" > "$FFX/dl/sdk.tgz"
    local top; top=$(tar tzf "$FFX/dl/sdk.tgz" | head -1 || true)
    [[ $top == *"FidelityFX-SDK-$COMMIT/" ]] || { echo "sdk.tgz is not $TAG ($COMMIT)"; exit 1; }
}

prepare() {
    local top="GPUOpen-LibrariesAndSDKs-FidelityFX-SDK-$COMMIT"
    rm -rf "$FFX/src" "$FFX/tools" && mkdir -p "$FFX/src" "$FFX/tools"
    tar xzf "$FFX/dl/sdk.tgz" -C "$FFX/src" --strip-components=1 "$top/ffx-api" "$top/sdk/include" "$top/sdk/src"
    tar xzf "$FFX/dl/sdk.tgz" -C "$FFX/tools" --strip-components=4 "$top/sdk/tools/binary_store"
    (cd "$FFX/src" && patch -p1 --no-backup-if-mismatch < "$here/ffx/linux.patch")
}

# compile_shaders <component dir> "<base args>" "<permutation args>" <pass shaders (relative to src/backends/vk)...>
compile_shaders() {
    local comp=$1 base=$2 perm=$3; shift 3
    local gpu; gpu=$(cygpath -m "$FFX/src/sdk/include/FidelityFX/gpu")
    local out; out=$(cygpath -m "$FFX/build/shaders/vk")
    local sc="$FFX/tools/FidelityFX_SC.exe" glslang; glslang=$(cygpath -m "$FFX/tools/glslangValidator.exe")
    local api="-compiler=glslang -e CS --target-env vulkan1.2 -S comp -Os -DFFX_GLSL=1"
    local s n v
    for s in "$@"; do
        n=$(basename "$s" .glsl)
        for v in ":0" "_wave64:0" "_16bit:1" "_wave64_16bit:1"; do  # (the wave64 variants differ only for HLSL)
            # shellcheck disable=SC2086
            (cd "$FFX/src/sdk/src/backends/vk" && "$sc" $base $api $perm -name="$n${v%:*}" -DFFX_HALF="${v#*:}" \
                -I"$gpu" -I"$gpu/$comp" -output="$out" -glslangexe="$glslang" "$s" > /dev/null) \
                || { echo "shader compile failed: $s ($v)"; return 1; }
        done
    done
}
shaders() {
    rm -rf "$FFX/build/shaders" && mkdir -p "$FFX/build/shaders/vk"
    cd "$FFX/src/sdk/src/backends/vk"  # (the pass shaders are given relative to it, as in the SDK's CMake)
    local fsr3u_base="-reflection -deps=gcc -DFFX_GPU=1 -DFFX_FSR3UPSCALER_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF=0
        -DFFX_FSR3UPSCALER_OPTION_ACCUMULATE_SAMPLERS_USE_DATA_HALF=0 -DFFX_FSR3UPSCALER_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF=1
        -DFFX_FSR3UPSCALER_OPTION_POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0 -DFFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE=2"
    local fsr3u_perm="-DFFX_FSR3UPSCALER_OPTION_REPROJECT_USE_LANCZOS_TYPE={0,1} -DFFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT={0,1}
        -DFFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS={0,1} -DFFX_FSR3UPSCALER_OPTION_JITTERED_MOTION_VECTORS={0,1}
        -DFFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH={0,1} -DFFX_FSR3UPSCALER_OPTION_APPLY_SHARPENING={0,1}"
    local fsr2_base="-reflection -deps=gcc -DFFX_GPU=1 -DFFX_FSR2_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF=0
        -DFFX_FSR2_OPTION_ACCUMULATE_SAMPLERS_USE_DATA_HALF=0 -DFFX_FSR2_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF=1
        -DFFX_FSR2_OPTION_POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0 -DFFX_FSR2_OPTION_UPSAMPLE_USE_LANCZOS_TYPE=2"
    local fsr2_perm="-DFFX_FSR2_OPTION_REPROJECT_USE_LANCZOS_TYPE={0,1} -DFFX_FSR2_OPTION_HDR_COLOR_INPUT={0,1}
        -DFFX_FSR2_OPTION_LOW_RESOLUTION_MOTION_VECTORS={0,1} -DFFX_FSR2_OPTION_JITTERED_MOTION_VECTORS={0,1}
        -DFFX_FSR2_OPTION_INVERTED_DEPTH={0,1} -DFFX_FSR2_OPTION_APPLY_SHARPENING={0,1}"
    local fi_base="-DFFX_FRAMEINTERPOLATION_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF=0
        -DFFX_FRAMEINTERPOLATION_OPTION_ACCUMULATE_SAMPLERS_USE_DATA_HALF=0 -DFFX_FRAMEINTERPOLATION_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF=1
        -DFFX_FRAMEINTERPOLATION_OPTION_POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0
        -DFFX_FRAMEINTERPOLATION_OPTION_UPSAMPLE_USE_LANCZOS_TYPE=2 -reflection -deps=gcc -DFFX_GPU=1"
    local fi_perm="-DFFX_FRAMEINTERPOLATION_OPTION_LOW_RES_MOTION_VECTORS={0,1} -DFFX_FRAMEINTERPOLATION_OPTION_JITTER_MOTION_VECTORS={0,1}
        -DFFX_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH={0,1}"
    local p=shaders/fsr2/ffx_fsr2 o=shaders/opticalflow/ffx_opticalflow
    compile_shaders fsr3upscaler "$fsr3u_base" "$fsr3u_perm" shaders/fsr3upscaler/*.glsl &
    compile_shaders fsr2 "$fsr2_base" "$fsr2_perm" ${p}_accumulate_pass.glsl ${p}_autogen_reactive_pass.glsl \
        ${p}_compute_luminance_pyramid_pass.glsl ${p}_depth_clip_pass.glsl ${p}_lock_pass.glsl ${p}_rcas_pass.glsl \
        ${p}_reconstruct_previous_depth_pass.glsl ${p}_tcr_autogen_pass.glsl &
    compile_shaders frameinterpolation "$fi_base" "$fi_perm" shaders/frameinterpolation/*.glsl &
    compile_shaders opticalflow "-reflection -deps=gcc -DFFX_GPU=1" "-DFFX_OPTICALFLOW_OPTION_HDR_COLOR_INPUT={0,1}" \
        ${o}_prepare_luma_pass.glsl ${o}_compute_luminance_pyramid_pass.glsl ${o}_generate_scd_histogram_pass.glsl \
        ${o}_compute_scd_divergence_pass.glsl ${o}_compute_optical_flow_advanced_pass_v5.glsl \
        ${o}_filter_optical_flow_pass_v5.glsl ${o}_scale_optical_flow_advanced_pass_v5.glsl &
    local rc=0 j; for j in $(jobs -p); do wait "$j" || rc=1; done
    [ $rc = 0 ] && echo "shaders: $(ls "$FFX/build/shaders/vk"/*_permutations.h | wc -l) permutation headers"
    return $rc
}

docker_run() {  # docker_run <image> <command...>: /w = D:/repos/monado
    MSYS_NO_PATHCONV=1 docker run --rm -v "$(cygpath -m "$FFX/..")/:/w" -w /w "$@"
}
compile() { docker_run -e FFX_DEBUG="${FFX_DEBUG:-}" monado-linux bash /w/src/scripts/linux/ffx_build.sh in-docker-compile; }

# (inside monado-linux)
in_docker_compile() {
    local S=/w/ffx/src B=/w/ffx/build O=/w/ffx/out H=/w/src/scripts/linux/ffx
    mkdir -p "$B/obj" "$O"
    local cxx=("$PY" -m ziglang c++ -target "x86_64-linux-gnu.$GLIBC")
    local opt=(-O2 -DNDEBUG) strip=-s
    [ -n "${FFX_DEBUG:-}" ] && opt=(-O1 -g -D_DEBUG) strip=  # (FFX_DEBUG=1: symbols, the SDK's debug checks)
    local flags=(-std=c++17 "${opt[@]}" -fPIC -fvisibility=hidden -fvisibility-inlines-hidden -w
        -include "$H/ffx_linux_compat.h"
        -DFFX_BACKEND_VK -DFFX_FSR2 -DFFX_FSR3 -DFFX_FSR3UPSCALER -DFFX_FI -DFFX_OF -D_UNICODE -DUNICODE
        -I"$S/ffx-api/include" -I"$S/sdk/include" -I"$S/sdk/src/components" -I"$S/sdk/src/shared"
        -I"$S/sdk/src/backends/shared" -I"$B/shaders/vk" -isystem /opt/xinc)
    local b=$S/sdk/src/backends ba=$S/sdk/src/backends/shared/blob_accessors
    local srcs=("$S"/ffx-api/src/*.cpp "$S"/ffx-api/src/vk/*.cpp "$b"/vk/*.cpp "$b"/vk/FrameInterpolationSwapchain/*.cpp
        "$b"/shared/ffx_shader_blobs.cpp "$ba"/ffx_fsr2_shaderblobs.cpp "$ba"/ffx_fsr3upscaler_shaderblobs.cpp
        "$ba"/ffx_frameinterpolation_shaderblobs.cpp "$ba"/ffx_opticalflow_shaderblobs.cpp "$S"/sdk/src/shared/*.cpp
        "$S"/sdk/src/components/{fsr2,fsr3upscaler,frameinterpolation,opticalflow}/*.cpp)
    "${cxx[@]}" -std=c++17 -w -x c++ -DFFX_LINUX_COMPAT_SELFTEST "$H/ffx_linux_compat.h" -o "$B/compat_selftest" && "$B/compat_selftest"
    rm -f "$B"/obj/*.o
    local f objs=()
    for f in "${srcs[@]}"; do objs+=("$B/obj/$(basename "$f" .cpp).o"); done
    printf '%s\n' "${srcs[@]}" | xargs -P "$(nproc)" -I{} sh -c '"$0" "$@" -c "{}" -o "'"$B"'/obj/$(basename "{}" .cpp).o" || exit 255' \
        "${cxx[@]}" "${flags[@]}"
    printf '{ global: ffxCreateContext; ffxDestroyContext; ffxConfigure; ffxQuery; ffxDispatch; local: *; };\n' > "$B/exports.map"
    "${cxx[@]}" -shared -fPIC -o "$O/libamd_fidelityfx_vk.so" "${objs[@]}" -Wl,--version-script="$B/exports.map" \
        -Wl,-z,defs -Wl,--gc-sections -Wl,-soname,libamd_fidelityfx_vk.so $strip -L/opt/xlib -lvulkan -lpthread
    echo "== $O/libamd_fidelityfx_vk.so ($(stat -c %s "$O/libamd_fidelityfx_vk.so") bytes)"
    nm -D --defined-only "$O/libamd_fidelityfx_vk.so"
    readelf -d "$O/libamd_fidelityfx_vk.so" | grep NEEDED
    echo "GLIBC versions required:"; objdump -T "$O/libamd_fidelityfx_vk.so" | grep -o 'GLIBC_[0-9.]*' | sort -uV | tr '\n' ' '; echo
}

smoke() {
    docker_run -e GDB="${GDB:-}" monado-lab bash -c 'set -e; [ -n "${GDB:-}" ] && pacman -Sy --noconfirm gdb >/dev/null 2>&1; cd /w/ffx/build && $PY -m ziglang cc -target x86_64-linux-gnu.'"$GLIBC"' -O1 \
        -isystem /opt/xinc -I/w/src/third_party /w/src/scripts/linux/ffx/smoke.c -o smoke -L/opt/xlib -lvulkan -ldl \
        && ${GDB:+gdb -batch -ex run -ex bt -ex "x/3i \$pc" -ex "info registers" --args} ./smoke /w/ffx/out/libamd_fidelityfx_vk.so'
}

case "${1:-all}" in
    all) fetch; prepare; shaders; compile ;;
    fetch|prepare|shaders|compile|smoke) "$1" ;;
    in-docker-compile) in_docker_compile ;;
    *) echo "usage: $0 [all|fetch|prepare|shaders|compile|smoke]"; exit 2 ;;
esac
