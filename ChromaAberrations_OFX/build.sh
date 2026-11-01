#!/usr/bin/env bash
# ChromaAberrations_OFX - OFX plug-in for Nuke, tools and tests (Linux)
#   ./build.sh          build (GPU version if nvcc is found), then run the tests
#   ./build.sh cpu      CPU-only build
#   ./build.sh clean    delete the build folder, then build
#   ./build.sh install  also install the plug-in into /usr/OFX/Plugins (sudo)
# Requirements: g++ 9+ (or clang), CMake 3.20+, Git, CUDA Toolkit 12.x for the GPU version.
set -euo pipefail
cd "$(dirname "$0")"
WITH_CUDA=ON
INSTALL=0
for a in "$@"; do
  case "$a" in
    clean) rm -rf build ;;
    cpu) WITH_CUDA=OFF ;;
    install) INSTALL=1 ;;
  esac
done
ARGS=()
if [ "$WITH_CUDA" = ON ] && command -v nvidia-smi >/dev/null 2>&1; then
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .' || true)
  if [[ "$CC" =~ ^[0-9]+$ ]] && [ "$CC" -ge 75 ] && [ "$CC" != 75 ] && [ "$CC" != 86 ] && [ "$CC" != 89 ]; then
    ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=75-real;86-real;89-real;${CC}-real;75-virtual")
  fi
fi
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCA_WITH_CUDA=$WITH_CUDA "${ARGS[@]}"
cmake --build build --parallel "$(nproc)"
./build/test_psf
./build/test_render
./build/test_lens tests/reference
echo
echo "Plugin: build/ChromaAberrations_OFX.ofx.bundle"
if [ "$INSTALL" = 1 ]; then
  sudo mkdir -p /usr/OFX/Plugins
  sudo rm -rf /usr/OFX/Plugins/ChromaAberrations_OFX.ofx.bundle
  sudo cp -r build/ChromaAberrations_OFX.ofx.bundle /usr/OFX/Plugins/
  echo "Installed into /usr/OFX/Plugins - restart Nuke."
else
  echo "To install it: ./build.sh install  (or copy the folder into /usr/OFX/Plugins, or into a folder of OFX_PLUGIN_PATH)"
fi
