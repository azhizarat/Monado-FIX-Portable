#!/usr/bin/env bash
# Start the native Linux build: <install>/play.sh (next to build/, targets/, save/, cache/, xcde.ini).
# Steam: add this file as a non-Steam game. Log: build/last_game.log (the previous one: build/last_game.prev.log).
# Extra env passes through, e.g. XCDE_HANDHELD=1 RADV_DEBUG=shaderstats ./play.sh
cd "$(dirname "$(readlink -f "$0")")"
[ -f build/last_game.log ] && mv -f build/last_game.log build/last_game.prev.log
exec build/xcde targets "$@" > build/last_game.log 2>&1
