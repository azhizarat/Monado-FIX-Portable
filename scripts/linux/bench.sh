#!/usr/bin/env bash
# A/B benchmark on the Deck (Game Mode, over ssh): each argument is one run's extra env ("" = baseline).
# Uncapped (fps = 60, vsync off for the run; xcde.ini is restored after), SECS per run (default 55), the average of the
# last 8 "[vk] t=" stats lines. Logs: build/bench_<n>.log.
#   scripts/linux/bench.sh "" "XCDE_HANDHELD=1" "RADV_PERFTEST=pswave32"
cd "$(dirname "$(readlink -f "$0")")"
[ -x build/xcde ] || { echo "run from the install folder (next to build/xcde)"; exit 1; }
export DISPLAY=${DISPLAY:-:0} XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/1000} SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS=0
SECS=${SECS:-55}
cp xcde.ini build/xcde.ini.bench-backup
trap 'cp build/xcde.ini.bench-backup xcde.ini' EXIT
sed -i 's/^fps = .*/fps = 60/; s/^vsync = .*/vsync = 0/' xcde.ini
n=0
for cfg in "$@"; do
  n=$((n + 1)); log=build/bench_$n.log
  env $cfg timeout "$SECS" build/xcde targets > "$log" 2>&1
  fps=$(grep -a '^\[vk\] t=' "$log" | tail -8 | awk '{s += $5; c++} END {if (c) printf "%.1f", s / c; else print "n/a"}')
  printf '%-60s %s fps  (%s)\n' "${cfg:-baseline}" "$fps" "$log"
done
