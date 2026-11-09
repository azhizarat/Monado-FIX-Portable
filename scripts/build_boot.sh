#!/usr/bin/env bash
# Build the headless boot executable (M3a) from targets/ (dump) with the project venv's zig cc.
#   1. python tools/gen_all.py            -> generated/{main,sdk}
#   2. this script                        -> build/obj/*.o, build/rt/*.o, build/xcde_boot.exe
#   3. build/xcde_boot.exe targets 15     -> boot log on stderr
set -e
cd "$(dirname "$0")/.."
. scripts/plat.sh; Z="$ZCC"; export Z
CF="$CPU -ffp-contract=off -fno-strict-aliasing -fvisibility=hidden -Iruntime $XINCF -w"; export CF
mkdir -p build/obj/main build/obj/sdk build/rt
# drop objects whose chunk no longer exists (the chunk layout shifts when the generator output changes)
for o in build/obj/*/*.o; do m=$(basename "$(dirname "$o")"); b=$(basename "$o" .o); [ -f "generated/$m/$b.c" ] || rm -f "$o"; done
cc1() { f=$1; m=$(basename "$(dirname "$f")"); b=$(basename "$f" .c); o="build/obj/$m/$b.o"; [ "$o" -nt "$f" ] || $Z -c "$f" -o "$o" $CF || echo "FAIL $f"; }
export -f cc1
ls generated/main/*.c generated/sdk/*.c | xargs -P "${JOBS:-16}" -I{} bash -c 'cc1 {}'
for f in loader dispatch kernel ipc fnprof platform/$PLAT services/fs services/romfs_virtual services/am services/acc \
         services/time services/hid_set services/misc hle/heaps hle/pl hle/vi hle/nvn_boot hle/fibers hle/framepace host/stubs host/settings_boot host/gamelib_static host/main_boot diag/diag diag/sampler diag/watchdog diag/crashdump; do
  $Z -c runtime/$f.c -o build/rt/$(echo $f | tr / _).o $CF -DGAMELIB_ID='"dev"'
done
# (the window-less boot exe is a dev leftover from M3; XCDE_SKIP_BOOT_LINK=1 builds only the game objects - scripts/build_all.sh)
[ -n "$XCDE_SKIP_BOOT_LINK" ] || scripts/link_boot.sh
