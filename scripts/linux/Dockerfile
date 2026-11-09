# Linux build environment for the native Steam Deck build (scripts/build_linux.sh runs inside it).
# Arch, like SteamOS: the same SDL3 / shaderc / opus headers and sonames the Deck has in /usr/lib.
# zig targets an older glibc (XCDE_GLIBC) so the binary runs on SteamOS even when Arch's glibc is newer.
#   docker build -t monado-linux scripts/linux
FROM archlinux:latest
RUN pacman -Syu --noconfirm --needed python python-pip git sdl3 vulkan-headers vulkan-icd-loader shaderc opus zlib binutils \
    && pacman -Scc --noconfirm
# ffmpeg must match the Deck's sonames (SteamOS 3.9: ffmpeg 8.1, libavcodec.so.62), not Arch's current one:
# only its headers and .so files, unpacked aside (nothing runs from them here).
ARG FFMPEG=ffmpeg-2%3A8.1.2-9-x86_64.pkg.tar.zst
RUN mkdir -p /opt/ff && curl -fsSL "https://archive.archlinux.org/packages/f/ffmpeg/$FFMPEG" | bsdtar -x -C /opt/ff usr/include usr/lib
# Vulkan Memory Allocator (header-only; not in Arch's repos)
# (fetched from GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator v3.3.0, MIT: raw.githubusercontent.com is not always reachable)
COPY vk_mem_alloc.h /usr/include/
RUN python -m venv /opt/venv && /opt/venv/bin/pip install -q --disable-pip-version-check lz4 capstone unicorn ziglang
# Only these headers / libs are visible to zig: the system glibc headers stay out (zig brings its own for XCDE_GLIBC).
RUN mkdir -p /opt/xinc /opt/xlib && \
    for h in SDL3 vulkan vk_video shaderc opus zlib.h zconf.h vk_mem_alloc.h; do ln -s /usr/include/$h /opt/xinc/; done && \
    for h in libavcodec libavformat libavutil libswscale libswresample; do ln -s /opt/ff/usr/include/$h /opt/xinc/; done && \
    for l in SDL3 vulkan shaderc_shared opus z; do ln -s /usr/lib/lib$l.so /opt/xlib/; done && \
    for l in avcodec avformat avutil swscale swresample; do ln -s /opt/ff/usr/lib/lib$l.so /opt/xlib/; done
ENV PY=/opt/venv/bin/python
WORKDIR /src
