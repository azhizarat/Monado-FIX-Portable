#!/usr/bin/env bash
# Start the native Linux build: <install>/play.sh (next to build/, targets/, save/, cache/, xcde.ini).
# Steam: add this file as a non-Steam game. Log: build/last_game.log (the previous one: build/last_game.prev.log).
# Extra env: play.env next to this file (sourced if present, e.g. "export XCDE_HANDHELD=1"), or on the command line.
cd "$(dirname "$(readlink -f "$0")")"
export DISPLAY="${DISPLAY:-:0}" XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/1000}"  # (started over ssh)
export SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS=0
[ -f play.env ] && . ./play.env
[ -f build/last_game.log ] && mv -f build/last_game.log build/last_game.prev.log
exec build/xcde targets "$@" > build/last_game.log 2>&1
