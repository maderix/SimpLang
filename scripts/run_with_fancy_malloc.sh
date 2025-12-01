#!/bin/bash
# run_with_fancy_malloc.sh
# Run SimpleLang MLIR compiler/runner with FancyAllocator via LD_PRELOAD
#
# Usage:
#   ./scripts/run_with_fancy_malloc.sh <command> [args...]
#
# Examples:
#   ./scripts/run_with_fancy_malloc.sh ./build_mlir/src/simplang examples/matmul.sl --emit-mlir -o /tmp/out.o
#   ./scripts/run_with_fancy_malloc.sh ./build_mlir/tests/test_runner ./tests/obj/test_matmul.so
#
# Debug mode (with memory checks):
#   FANCY_DEBUG=1 ./scripts/run_with_fancy_malloc.sh <command> [args...]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Find the library
if [[ -n "$FANCY_DEBUG" ]]; then
    LIB_NAME="libfancymalloc_debug.so"
else
    LIB_NAME="libfancymalloc.so"
fi

# Check common build directories
for BUILD_DIR in "$PROJECT_ROOT/build_mlir" "$PROJECT_ROOT/build"; do
    LIB_PATH="$BUILD_DIR/thirdparty/fancy_allocator/$LIB_NAME"
    if [[ -f "$LIB_PATH" ]]; then
        break
    fi
done

if [[ ! -f "$LIB_PATH" ]]; then
    echo "Error: $LIB_NAME not found. Build it first:"
    echo "  cd build_mlir && cmake --build . --target fancymalloc"
    exit 1
fi

echo "Using FancyAllocator: $LIB_PATH"
if [[ -n "$FANCY_DEBUG" ]]; then
    echo "Debug mode: Memory safety checks ENABLED"
fi
echo "Running: $@"
echo "---"

# Run with LD_PRELOAD
LD_PRELOAD="$LIB_PATH" "$@"

exit_code=$?

echo "---"
echo "Exit code: $exit_code"

exit $exit_code
