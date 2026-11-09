# Runtime lab on a PC without the Deck: the Linux build on lavapipe (Mesa's software Vulkan) with SDL's offscreen video
# driver, for screenshot comparisons of runtime changes (scripts/linux/lab.sh). Slow, but deterministic enough for A/B.
#   docker build -t monado-lab -f scripts/linux/lab.Dockerfile scripts/linux
FROM monado-linux
# (ffmpeg: for its dependencies; the 8.1 libraries themselves come from /opt/ff)
RUN pacman -Sy --noconfirm --needed vulkan-swrast ffmpeg && pacman -Scc --noconfirm
# the ffmpeg 8.1 the build links against (as on SteamOS 3.9) needs these older sonames: unpacked aside, like ffmpeg
RUN for p in l/libjxl/libjxl-0.11.2-2 x/x265/x265-4.2-2 l/libbluray/libbluray-1.4.1-1; do \
      curl -fsSL "https://archive.archlinux.org/packages/$p-x86_64.pkg.tar.zst" | bsdtar -x -C /opt/ff usr/lib; done
ENV LD_LIBRARY_PATH=/opt/ff/usr/lib SDL_VIDEODRIVER=offscreen VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
