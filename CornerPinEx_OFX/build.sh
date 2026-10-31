#!/usr/bin/env bash
# ===========================================================================
#  CornerPinEx_OFX - Linux build (GCC + CUDA)
#
#    ./build.sh            build (GPU version if the CUDA Toolkit is found)
#    ./build.sh cpu        CPU-only build (no CUDA)
#    ./build.sh clean      delete the build folder, then build
#    ./build.sh install    build, then copy the bundle into /usr/OFX/Plugins (sudo)
#
#  Requirements: GCC/G++ (build-essential or gcc-toolset), CMake >= 3.20, Git,
#                CUDA Toolkit 12.x or 13.x for the GPU version.
#  Result      : build/CornerPinEx_OFX.ofx.bundle
#  Build log   : build.log
# ===========================================================================
set -u
cd "$(dirname "$0")"
LOG="$PWD/build.log"
echo "CornerPinEx_OFX build - $(date)" > "$LOG"

fail() { echo; echo "[ERROR] $1"; echo "Full details: $LOG"; exit 1; }

WITH_CUDA=ON
INSTALL=0
for a in "$@"; do
  case "$a" in
    cpu)     WITH_CUDA=OFF ;;
    clean)   echo "Deleting the build folder..."; rm -rf build ;;
    install) INSTALL=1 ;;
    *)       echo "Unknown option: $a (cpu, clean, install)"; exit 1 ;;
  esac
done

echo
echo "=== CornerPinEx_OFX : build ==="
echo

# --- 1. tools ----------------------------------------------------------------
command -v cmake >/dev/null || fail "CMake not found (Ubuntu: sudo apt install cmake, Rocky: sudo dnf install cmake)"
command -v git   >/dev/null || fail "Git not found (needed to download the OFX headers)"
command -v g++   >/dev/null || fail "g++ not found (Ubuntu: sudo apt install build-essential, Rocky: sudo dnf groupinstall 'Development Tools')"
CMAKE_VER=$(cmake --version | head -1 | awk '{print $3}')
echo "[1/5] CMake $CMAKE_VER, g++ $(g++ -dumpfullversion), Git : OK"
GENERATOR=()
command -v ninja >/dev/null && GENERATOR=(-G Ninja)

# OpenGL headers (overlay drawing)
if [ ! -f /usr/include/GL/gl.h ]; then
  fail "OpenGL headers not found (Ubuntu: sudo apt install libgl1-mesa-dev, Rocky: sudo dnf install mesa-libGL-devel)"
fi

# --- 2. CUDA -----------------------------------------------------------------
CUDA_ARGS=()
if [ "$WITH_CUDA" = ON ]; then
  # nvcc is often not in PATH after installing the toolkit
  if ! command -v nvcc >/dev/null; then
    for d in /usr/local/cuda/bin /usr/local/cuda-*/bin /opt/cuda/bin; do
      [ -x "$d/nvcc" ] && export PATH="$d:$PATH" && break
    done
  fi
  if command -v nvcc >/dev/null; then
    echo "[2/5] CUDA $(nvcc --version | grep release | sed 's/.*release \([0-9.]*\).*/\1/') : $(command -v nvcc)"
    CUDA_ARGS+=(-DCMAKE_CUDA_COMPILER="$(command -v nvcc)")
    # Turing, Ampere, Ada + PTX; adds the installed GPU if it is newer (e.g. RTX 50xx)
    ARCHS="75-real;86-real;89-real;75-virtual"
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')
    if [[ "$CC" =~ ^[0-9]+$ ]] && [ "$CC" -ge 75 ] && [[ ";$ARCHS;" != *";$CC-real;"* ]]; then
      ARCHS="$ARCHS;$CC-real"
      echo "      GPU detected: sm_$CC (added to the build)"
    fi
    CUDA_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="$ARCHS")
  else
    echo "[2/5] CUDA Toolkit not found: building the CPU-only version."
    echo "      For the GPU version, install the CUDA Toolkit, then run './build.sh clean'."
    WITH_CUDA=OFF
  fi
else
  echo "[2/5] CPU-only version, no CUDA"
fi

# --- 3. configure ------------------------------------------------------------
echo "[3/5] Configuring..."
cmake -S . -B build "${GENERATOR[@]}" -DCMAKE_BUILD_TYPE=Release \
      -DCORNERPINEX_OFX_WITH_CUDA=$WITH_CUDA -DCORNERPINEX_OFX_BUILD_TESTS=ON "${CUDA_ARGS[@]}" >> "$LOG" 2>&1 \
  || { tail -30 "$LOG"; fail "configuration failed (if you changed compiler: ./build.sh clean)"; }

# --- 4. build ----------------------------------------------------------------
echo "[4/5] Building..."
cmake --build build --parallel "$(nproc)" >> "$LOG" 2>&1 \
  || { grep -iE "error" "$LOG" | tail -25; fail "the build failed"; }

BUNDLE="$PWD/build/CornerPinEx_OFX.ofx.bundle"
OFX="$BUNDLE/Contents/Linux-x86-64/CornerPinEx_OFX.ofx"
[ -f "$OFX" ] || fail "CornerPinEx_OFX.ofx not found after the build"

# --- 5. tests ----------------------------------------------------------------
echo "[5/5] Running tests..."
TESTOUT=$(./build/test_engine 2>&1); TESTRC=$?
echo "$TESTOUT" >> "$LOG"
echo "$TESTOUT" | head -1 | sed 's/^/      /'          # GPU found (or why not)
echo "$TESTOUT" | tail -1 | sed 's/^/      /'          # checks / failures
if [ $TESTRC -ne 0 ]; then
  echo "$TESTOUT" | grep FAIL | head -20
  echo "WARNING: some tests failed. The plugin is built, but check it before using it."
fi

echo
echo "Build succeeded."
echo "Plugin : $BUNDLE"

# --- install -----------------------------------------------------------------
if [ "$INSTALL" = 1 ]; then
  echo "Installing into /usr/OFX/Plugins (sudo password)..."
  sudo mkdir -p /usr/OFX/Plugins \
    && sudo rm -rf /usr/OFX/Plugins/CornerPinEx_OFX.ofx.bundle \
    && sudo cp -r "$BUNDLE" /usr/OFX/Plugins/ \
    || fail "the copy failed"
  echo "Installed: /usr/OFX/Plugins/CornerPinEx_OFX.ofx.bundle - restart Nuke."
else
  echo
  echo "To install it: ./build.sh install"
  echo "  (or copy the folder into /usr/OFX/Plugins, or add its parent folder to OFX_PLUGIN_PATH)"
fi
