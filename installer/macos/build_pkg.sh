#!/bin/bash
# Build a self-contained macOS package containing the Namsel daemon.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python3.12}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/dist}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "ERROR: Python 3.12 is required (set PYTHON to its executable path)." >&2
    exit 1
fi
if ! command -v pkgbuild >/dev/null 2>&1; then
    echo "ERROR: pkgbuild is unavailable. Install the Xcode command line tools." >&2
    exit 1
fi

if ! "$PYTHON" -c "import PyInstaller, cv2, joblib, lxml, numpy, onnxruntime, PIL, scipy, skimage, sklearn"; then
    echo "ERROR: Install the project requirements and PyInstaller in this Python environment." >&2
    echo "       See installer/macos/README.md." >&2
    exit 1
fi

VERSION="$(sed -nE "s/^[[:space:]]*version[[:space:]]*=[[:space:]]*'([^']+)'.*/\1/p" "$ROOT_DIR/setup.py" | head -n 1)"
if [ -n "${1:-}" ]; then
    VERSION="$1"
fi
if [ -z "$VERSION" ]; then
    echo "ERROR: Could not determine a package version from setup.py." >&2
    exit 1
fi

ARCH="$(uname -m)"
case "$ARCH" in
    arm64|x86_64) ;;
    *) echo "ERROR: Unsupported macOS build architecture: $ARCH" >&2; exit 1 ;;
esac

BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/namsel-macos.XXXXXX")"
trap 'rm -rf "$BUILD_DIR"' EXIT
mkdir -p "$BUILD_DIR/dist" "$BUILD_DIR/work" "$OUTPUT_DIR"

"$PYTHON" -m PyInstaller \
    --clean \
    --noconfirm \
    --distpath "$BUILD_DIR/dist" \
    --workpath "$BUILD_DIR/work" \
    "$SCRIPT_DIR/namsel_daemon.spec"

BUNDLE="$BUILD_DIR/dist/namsel_daemon"
if [ ! -x "$BUNDLE/namsel_daemon" ]; then
    echo "ERROR: PyInstaller did not produce the daemon executable." >&2
    exit 1
fi

PAYLOAD="$BUILD_DIR/payload"
mkdir -p "$PAYLOAD/NamselOCR"
cp -R "$BUNDLE" "$PAYLOAD/NamselOCR/namsel_daemon"
cp "$ROOT_DIR/LICENSE" "$PAYLOAD/NamselOCR/LICENSE"

PACKAGE="$OUTPUT_DIR/NamselOCR-${VERSION}-macos-${ARCH}.pkg"
pkgbuild \
    --root "$PAYLOAD" \
    --identifier "org.namsel.ocr" \
    --version "$VERSION" \
    --install-location "/Applications" \
    "$PACKAGE"

echo "Package created: $PACKAGE"
echo "Installed daemon: /Applications/NamselOCR/namsel_daemon/namsel_daemon"
