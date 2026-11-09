#!/usr/bin/env bash
# Link the windowed executable (M4): headless runtime + NVN/Vulkan backend + SDL3.
# --stack 256 MB: the default thread stack *reservation* (committed on use); guest threads rely on it (kernel.c thread_start).
cd "$(dirname "$0")/.."
. scripts/plat.sh; Z="$ZCXX"
RT="build/rt/loader.o build/rt/dispatch.o build/rt/kernel.o build/rt/ipc.o build/rt/fnprof.o build/rt/platform_$PLAT.o \
 build/rt/services_fs.o build/rt/services_romfs_virtual.o build/rt/services_am.o build/rt/services_acc.o build/rt/services_time.o \
 build/rt/services_hid_set.o build/rt/services_kbicons.o build/rt/services_misc.o build/rt/hle_heaps.o build/rt/hle_pl.o build/rt/hle_vi.o build/rt/hle_nvn_boot.o build/rt/hle_fibers.o \
 build/rt/host_stubs.o build/rt/host_settings_ini.o build/rt/host_stubs_gfx.o build/rt/services_audio.o build/rt/services_hwopus.o build/rt/hle_input.o build/rt/hle_party.o build/rt/hle_bindings.o build/rt/hle_movie.o build/rt/hle_framepace.o build/rt/host_gamelib_static.o build/rt/host_main_boot_gfx.o build/rt/diag_diag.o build/rt/diag_sampler.o build/rt/diag_watchdog.o build/rt/diag_crashdump.o"
NVN="$(ls build/nvn/*.o | tr '\n' ' ')"
$Z -o "${XCDE_EXE:-build/xcde$EXE_SUFFIX}" $RT $NVN build/obj/main/*.o build/obj/sdk/*.o $CPU \
  $SYSLIBS 2> build/link_gfx.err
rc=$?; grep "undefined symbol" build/link_gfx.err | sed 's/.*undefined symbol: //' | sort -u | head -60; exit $rc
