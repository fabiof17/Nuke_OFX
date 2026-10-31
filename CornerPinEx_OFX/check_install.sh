#!/usr/bin/env bash
# ===========================================================================
#  CornerPinEx_OFX - installation check for Nuke (Linux)
#  Close Nuke, then run: ./check_install.sh
# ===========================================================================
set -u
cd "$(dirname "$0")"
PROBLEMS=0
NAME=CornerPinEx_OFX
REL="Contents/Linux-x86-64/$NAME.ofx"

echo
echo "=== $NAME : installation check ==="
echo

echo "[1] Build"
if [ -f "build/$NAME.ofx.bundle/$REL" ]; then
  echo "    OK: the plugin has been built."
else
  echo "    NO: build/$NAME.ofx.bundle/$REL does not exist. Run ./build.sh first"
  PROBLEMS=1
fi

echo
echo "[2] Installation"
FOUND=""
DIRS="/usr/OFX/Plugins"
[ -n "${OFX_PLUGIN_PATH:-}" ] && DIRS="$DIRS:${OFX_PLUGIN_PATH}"
IFS=':' read -ra LIST <<< "$DIRS"
for d in "${LIST[@]}"; do
  [ -z "$d" ] && continue
  if [ -f "$d/$NAME.ofx.bundle/$REL" ]; then
    FOUND="$d/$NAME.ofx.bundle/$REL"
    echo "    OK: $FOUND"
  elif [ -f "$d/$NAME.ofx" ]; then
    echo "    $d contains $NAME.ofx on its own: copy the whole $NAME.ofx.bundle FOLDER."
  fi
done
if [ -z "$FOUND" ]; then
  echo "    NO: the plugin is neither in /usr/OFX/Plugins nor in OFX_PLUGIN_PATH (${OFX_PLUGIN_PATH:-not set})."
  echo "    Install it with: ./build.sh install"
  PROBLEMS=1
fi
if [ -n "${OFX_PLUGIN_PATH:-}" ] && [[ "$OFX_PLUGIN_PATH" == :* || "$OFX_PLUGIN_PATH" == *: ]]; then
  echo "    WARNING: OFX_PLUGIN_PATH starts or ends with ':' (Nuke can hang at startup)."
fi

echo
echo "[3] Loading the plugin (as Nuke does)"
TEST="${FOUND:-build/$NAME.ofx.bundle/$REL}"
if [ -f "$TEST" ]; then
  MISSING=$(ldd "$TEST" 2>/dev/null | grep "not found")
  if [ -n "$MISSING" ]; then
    echo "    FAILED: missing libraries:"
    echo "$MISSING" | sed 's/^/      /'
    PROBLEMS=1
  fi
  if command -v python3 >/dev/null; then
    if python3 - "$TEST" <<'PY'
import ctypes, sys
try:
    lib = ctypes.CDLL(sys.argv[1])
    n = lib.OfxGetNumberOfPlugins()
    print("    OK: the library loads and declares %d plugins." % n)
except Exception as e:
    print("    FAILED: %s" % e)
    sys.exit(1)
PY
    then :; else PROBLEMS=1; fi
  fi
else
  echo "    skipped (plugin not found)"
fi

echo
echo "[4] GPU"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/    /'
else
  echo "    nvidia-smi not found: no NVIDIA driver, rendering will run on the CPU."
fi

echo
echo "[5] Nuke OFX plugin cache"
CACHE="/var/tmp/nuke-u$(id -u)/ofxplugincache"
[ -d "$CACHE" ] || CACHE="/var/tmp/nuke-$(id -u)/ofxplugincache"
if [ -d "$CACHE" ]; then
  if pgrep -f -i "nuke" >/dev/null; then
    echo "    Nuke is running: close it, then run this check again to clear the cache."
    PROBLEMS=1
  else
    rm -rf "$CACHE" && echo "    Cache deleted: $CACHE (Nuke rebuilds the plugin list at next launch)."
  fi
else
  echo "    no cache found"
fi

echo
echo "==========================================================================="
if [ "$PROBLEMS" = 0 ]; then
  echo "Everything is in place. Start Nuke, press Tab and type \"CornerPinEx\"."
else
  echo "Problems were found: fix them (see above), then run this check again."
fi
echo " - Nuke Non-commercial does not load third-party OFX plugins."
echo "==========================================================================="
