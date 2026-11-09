#!/usr/bin/env bash
# Runtime lab without the Deck (scripts/linux/lab.Dockerfile): the Linux build on lavapipe, deterministic and headless,
# screenshots at fixed frames of the title screen. Two runs with the same frames give identical images unless a runtime
# change altered what is drawn, so: run once for a baseline, once with the change, compare the PNGs.
#   docker run --rm -v <dir with build/xcde>:/lab -v <dump with exefs/ and romfs/>:/lab/targets:ro monado-lab \
#     /lab/lab.sh <out dir> [frames, default 2500,4000] [extra env: XCDE_MEMO_CHECK=1 ...]
# ~15 min (software GPU). XCDE_OVERLAP=0: the page protection would fault inside lavapipe's own threads (its "GPU"
# writes guest memory from the CPU) and wait for itself.
set -e
cd /lab
out=${1:?out dir}; frames=${2:-2500,4000}; shift $(( $# > 2 ? 2 : $# ))
rm -rf build/frames "$out"
env XCDE_OVERLAP=0 LARECOMP_HIDDEN=1 XCDE_DETERMINISTIC=1 LARECOMP_SHOT_AT="$frames" LARECOMP_EXIT_AFTER_SHOT=1 "$@" \
    timeout 1800 build/xcde targets > "$out.log" 2>&1 || true
mkdir -p "$out" && mv build/frames/* "$out"/ 2>/dev/null || true
ls "$out"; echo "memo-check mismatches: $(grep -a -c 'memo-check\]' "$out.log")"
