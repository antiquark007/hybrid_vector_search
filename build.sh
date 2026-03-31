#!/usr/bin/env bash
# build.sh — compile C++ module and run tests
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$ROOT/build"

echo "=== Hybrid Vector Search — Build Script ==="

# ── 1. C++ build ─────────────────────────────────────────────────────────────
echo ""
echo "[1/4] Configuring CMake..."
cmake -B "$BUILD_DIR" -S "$ROOT" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

echo "[2/4] Building ($(nproc) cores)..."
cmake --build "$BUILD_DIR" --parallel "$(nproc)"

echo "[3/4] Running C++ unit tests..."
cd "$BUILD_DIR" && ctest --output-on-failure
cd "$ROOT"

# ── 2. Copy .so to project root for Python import ─────────────────────────────
echo "[4/4] Copying hvs_core module..."
SO_FILE=$(find "$BUILD_DIR" -name "hvs_core*.so" | head -1)
if [[ -n "$SO_FILE" ]]; then
    cp "$SO_FILE" "$ROOT/"
    echo "Copied: $SO_FILE → $ROOT/"
fi

echo ""
echo "=== Build complete ==="
echo "Run: uvicorn src.api.main:app --reload"
