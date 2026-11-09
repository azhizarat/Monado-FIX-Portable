# Host toolchain, sourced by the build scripts. Windows (Git Bash + MSYS2 + .venv) by default;
# Linux (scripts/linux/Dockerfile: PY=/opt/venv/bin/python, headers/libs in /opt/xinc, /opt/xlib).
if [ "$(uname -s)" = Linux ]; then
  PLAT=linux
  PY=${PY:-.venv/bin/python}
  XCDE_GLIBC=${XCDE_GLIBC:-2.38}  # SteamOS 3.9 ships 2.43; older is fine, newer is not
  TGT="-target x86_64-linux-gnu.$XCDE_GLIBC"
  INC=${XINC:-/opt/xinc}
  EXE_SUFFIX=
  # (--allow-shlib-undefined: the system .so files reference the host glibc's newer symbols; the Deck resolves them)
  SYSLIBS="-L${XLIB:-/opt/xlib} -lSDL3 -lshaderc_shared -lvulkan -lopus -lavcodec -lavformat -lavutil -lswscale -lswresample -lz \
    -lpthread -ldl -lm -Wl,--no-gc-sections -Wl,--allow-shlib-undefined -Wl,-rpath,\$ORIGIN"
else
  PLAT=windows
  PY=.venv/Scripts/python
  TGT=
  INC=/c/msys64/ucrt64/include
  EXE_SUFFIX=.exe
  M=/c/msys64/ucrt64
  SYSLIBS="$M/lib/libSDL3.dll.a $M/lib/libshaderc_shared.dll.a $M/lib/libvulkan-1.dll.a $M/lib/libopus.dll.a $M/lib/libavcodec.dll.a \
    $M/lib/libavformat.dll.a $M/lib/libavutil.dll.a $M/lib/libswscale.dll.a $M/lib/libswresample.dll.a $M/lib/libz.a -lwinmm -ldbghelp \
    -Wl,--stack,268435456"
fi
# CPU code generation, both hosts: x86-64-v3 (AVX2 / FMA: the lifted code's fmaf() becomes one
# instruction instead of compiler-rt's software FMA, ~3.4 % of the game thread in Colony 9) and -O2. For CPUs before
# Haswell / Zen: XCDE_CPU="-O1 -march=x86_64_v2" (the old default; the startup check follows the level).
CPU=${XCDE_CPU:-"-O2 -march=x86_64_v3"}
XINCF=; [ $PLAT = linux ] && XINCF="-I$INC"  # (Windows boot objects never saw the MSYS2 headers)
ZCC="$PY -m ziglang cc $TGT"
ZCXX="$PY -m ziglang c++ $TGT"
