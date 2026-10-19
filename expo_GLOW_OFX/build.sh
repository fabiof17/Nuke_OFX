#!/usr/bin/env bash
# Build expo_GLOW_OFX.ofx for Linux (CPU + optional CUDA).
#   ./build.sh           CPU + CUDA (if nvcc is found)
#   ./build.sh cpu       CPU only
#   ./build.sh clean     delete the build folder and rebuild
#   ./build.sh install   build, then copy the bundle to /usr/OFX/Plugins (sudo)
set -u
cd "$(dirname "$0")"
LOG="$PWD/build.log"
echo "expo_GLOW_OFX build - $(date)" > "$LOG"

fail() { echo; echo "[ERROR] $1"; echo "Full details: $LOG"; exit 1; }

echo "=== expo_GLOW_OFX : build ==="
CUDA_OPT="-DEXPOGLOW_WITH_CUDA=ON"
INSTALL=0
for a in "$@"; do
  case "$a" in
    cpu)     CUDA_OPT="-DEXPOGLOW_WITH_CUDA=OFF" ;;
    clean)   echo "Deleting the build folder..."; rm -rf build ;;
    install) INSTALL=1 ;;
  esac
done

command -v cmake >/dev/null || fail "CMake not found: sudo apt install cmake"
command -v git   >/dev/null || fail "Git not found: sudo apt install git"
command -v g++   >/dev/null || fail "g++ not found: sudo apt install build-essential"
echo "CMake : $(cmake --version | head -1 | awk '{print $3}')"
echo "g++   : $(g++ -dumpfullversion)"

# nvcc is often not in PATH after installing the toolkit
if ! command -v nvcc >/dev/null; then
  for d in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
    [ -x "$d/nvcc" ] && export PATH="$d:$PATH" && break
  done
fi
if [ "$CUDA_OPT" = "-DEXPOGLOW_WITH_CUDA=ON" ]; then
  if command -v nvcc >/dev/null; then
    echo "CUDA  : $(nvcc --version | grep release | sed 's/.*release //')"
    CUDA_OPT="$CUDA_OPT -DCMAKE_CUDA_COMPILER=$(command -v nvcc)"
  else
    echo "CUDA  : nvcc not found - CPU-only build"
    CUDA_OPT="-DEXPOGLOW_WITH_CUDA=OFF"
  fi
fi
echo

echo "[1/2] CMake configuration..."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release $CUDA_OPT >> "$LOG" 2>&1 || { tail -30 "$LOG"; fail "CMake configuration failed (if you changed compiler: ./build.sh clean)"; }
echo "[2/2] Compiling..."
cmake --build build --parallel "$(nproc)" >> "$LOG" 2>&1 || { grep -i "error" "$LOG" | tail -25; fail "Compilation failed"; }

BUNDLE="$PWD/build/expo_GLOW_OFX.ofx.bundle"
[ -f "$BUNDLE/Contents/Linux-x86-64/expo_GLOW_OFX.ofx" ] || fail "expo_GLOW_OFX.ofx not found after the build"
grep "expo_GLOW_OFX: CUDA" "$LOG" | sed 's/^-- //'

if [ "$INSTALL" = 1 ]; then
  echo "Installing into /usr/OFX/Plugins (sudo password)..."
  sudo mkdir -p /usr/OFX/Plugins && sudo rm -rf /usr/OFX/Plugins/expo_GLOW_OFX.ofx.bundle \
    && sudo cp -r "$BUNDLE" /usr/OFX/Plugins/ || fail "Copy failed"
  echo "Installed: /usr/OFX/Plugins/expo_GLOW_OFX.ofx.bundle"
else
  echo
  echo "=== OK ==="
  echo "Plug-in : $BUNDLE"
  echo "Install it with: ./build.sh install   (or copy it into /usr/OFX/Plugins)"
fi
