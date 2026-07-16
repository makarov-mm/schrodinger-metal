#!/usr/bin/env bash
#
# Build helper for the Split-Step Schrodinger project.
#
# Usage:
#   ./build.sh            build in release
#   ./build.sh run        build and run in release
#   ./build.sh debug      build in debug
#   ./build.sh run debug  build and run in debug
#   ./build.sh clean      remove build artifacts
#
set -euo pipefail

cd "$(dirname "$0")"

# Sanity checks.
if [[ "$(uname)" != "Darwin" ]]; then
    echo "error: this project needs macOS and Metal." >&2
    exit 1
fi

if ! command -v swift >/dev/null 2>&1; then
    echo "error: swift not found. Install the Xcode command line tools:" >&2
    echo "       xcode-select --install" >&2
    exit 1
fi

# Parse arguments in any order.
ACTION="build"
CONFIG="release"
for arg in "$@"; do
    case "$arg" in
        run)     ACTION="run" ;;
        build)   ACTION="build" ;;
        clean)   ACTION="clean" ;;
        debug)   CONFIG="debug" ;;
        release) CONFIG="release" ;;
        *)
            echo "error: unknown argument '$arg'" >&2
            echo "usage: ./build.sh [run|build|clean] [debug|release]" >&2
            exit 1
            ;;
    esac
done

if [[ "$ACTION" == "clean" ]]; then
    echo "==> swift package clean"
    swift package clean
    rm -rf .build
    echo "done."
    exit 0
fi

echo "==> swift build -c $CONFIG"
swift build -c "$CONFIG"

if [[ "$ACTION" == "run" ]]; then
    echo "==> swift run -c $CONFIG SchrodingerMetal"
    exec swift run -c "$CONFIG" SchrodingerMetal
fi

echo "build succeeded (config: $CONFIG)."
echo "binary: $(swift build -c "$CONFIG" --show-bin-path)/SchrodingerMetal"
