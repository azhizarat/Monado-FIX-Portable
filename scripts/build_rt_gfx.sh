#!/usr/bin/env bash
# XCDE_EXE=build/other.exe builds/links a separate exe without killing running games.
# Compile every runtime object build/xcde.exe links (build/rt/*.o, build/nvn/*.o), then link it (scripts/link_gfx.sh).
#   scripts/build_rt_gfx.sh            # all objects
#   scripts/build_rt_gfx.sh kernel hle/movie nvn/vulkan/exec   # only these (paths under runtime/, no .c)
# The game objects (build/obj) come from scripts/build_boot.sh. The window build of main_boot gets -DXCDE_GFX.
cd "$(dirname "$0")/.."
. scripts/plat.sh; Z="$ZCC"
CF="-O1 -march=x86_64_v2 -fno-strict-aliasing -fvisibility=hidden -Iruntime -Iruntime/nvn -Ithird_party -I$INC -w"
RT="loader dispatch kernel ipc fnprof platform/$PLAT services/fs services/romfs_virtual services/am services/acc services/time
    services/hid_set services/kbicons services/misc services/audio services/hwopus hle/heaps hle/pl hle/vi hle/nvn_boot hle/fibers hle/input hle/party
    hle/bindings hle/movie hle/framepace host/stubs host/settings_ini host/stubs_gfx host/gamelib_static host/main_boot diag/diag diag/sampler diag/watchdog
    diag/crashdump"
NVN="astc cmdbuf formats maxwell/translate objects output_scale smaa_data tegra texpatch vulkan/device vulkan/exec vulkan/precompile vulkan/shaders vulkan/textures vulkan/temporal vulkan/heap_placed"
mkdir -p build/rt build/nvn
want="$*"
pick() { [ -z "$want" ] || [[ " $want " == *" $1 "* ]]; }
fail=0
for f in $RT; do
  pick "$f" || continue
  o=build/rt/$(echo $f | tr / _).o; extra=""
  [ "$f" = host/main_boot ] && { o=build/rt/host_main_boot_gfx.o; extra="-DXCDE_GFX"; }
  $Z -c runtime/$f.c -o $o.$$.tmp $CF -DGAMELIB_ID='"dev"' $extra && mv -f $o.$$.tmp $o || { echo "FAIL $f"; fail=1; }  # (tmp + rename: a parallel link never reads half an object)
done
for f in $NVN; do
  pick "nvn/$f" || continue
  extra=""
  if [ "$f" = smaa_data ]; then  # [XCDE] the SMAA reference shader (third_party/smaa, MIT) as a C string
    mkdir -p build/inc
    $PY tools/gen_smaa_inc.py third_party/smaa/SMAA.hlsl build/inc/smaa_hlsl.inc || fail=1
    extra="-Ibuild/inc -Ithird_party/smaa/Textures"
  fi
  o=build/nvn/nvn_$(echo $f | tr / _).o
  $Z -c runtime/nvn/$f.c -o $o.$$.tmp $CF -DGAMELIB_ID='"dev"' $extra && mv -f $o.$$.tmp $o || { echo "FAIL nvn/$f"; fail=1; }
done
# [XCDE] In-game settings overlay (F1): Dear ImGui + its Vulkan backend (third_party/imgui, MIT) and runtime/ui/menu_vk.cpp
# (replaces the menu stub object)
if pick ui/menu; then
  I=third_party/imgui
  rm -f build/nvn/host_menu_stub.o
  for f in $I/imgui.cpp $I/imgui_draw.cpp $I/imgui_tables.cpp $I/imgui_widgets.cpp $I/backends/imgui_impl_vulkan.cpp \
           runtime/nvn/vulkan/imgui_glue.cpp runtime/ui/menu_vk.cpp; do
    $ZCXX -c $f -o build/nvn/ui_$(basename ${f%.cpp}).o -std=c++17 -O1 -march=x86_64_v2 \
      -fvisibility=hidden -Iruntime -Iruntime/nvn -Iruntime/nvn/vulkan -I$I -I$I/backends -I$INC -w \
      || { echo "FAIL $f"; fail=1; }
  done
fi
# [XCDE] NVIDIA DLSS through Streamline (C++ headers, third_party/streamline, MIT)
if pick nvn/vulkan/dlss; then
  $ZCXX -c runtime/nvn/vulkan/dlss.cpp -o build/nvn/nvn_vulkan_dlss.o -std=c++17     -O1 -march=x86_64_v2 -fvisibility=hidden -Iruntime -Iruntime/nvn -Ithird_party/streamline/include -I$INC -w     || { echo "FAIL nvn/vulkan/dlss"; fail=1; }
fi
# [XCDE] Vulkan Memory Allocator (header-only, MSYS2 vulkan-memory-allocator) and the path helpers: built here with the
# same baseline CPU as everything else. (Built by hand before: vma.o carried AVX-512 from the build PC's CPU, so the
# game would stop with an illegal instruction on CPUs without it.)
if pick nvn/vulkan/vma; then
  $ZCXX -c runtime/nvn/vulkan/vma.cpp -o build/nvn/vma.o -std=c++17 -O1 -march=x86_64_v2 -fvisibility=hidden -I$INC -w || { echo "FAIL nvn/vulkan/vma"; fail=1; }
fi
if pick paths; then
  $Z -c runtime/paths.c -o build/nvn/paths.o $CF -DGAMELIB_ID='"dev"' || { echo "FAIL paths"; fail=1; }
fi
[ $fail = 0 ] || exit 1
[ -n "$XCDE_EXE" ] || [ $PLAT = linux ] || taskkill //F //IM xcde.exe >/dev/null 2>&1
scripts/link_gfx.sh && { [ $PLAT = linux ] || scripts/copy_upscaler_dlls.sh; }
