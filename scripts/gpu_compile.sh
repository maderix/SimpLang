#!/bin/bash
#===----------------------------------------------------------------------===//
# gpu_compile.sh - Compile SimpLang files for GPU backend with cuBLAS
#
# Usage:
#   ./scripts/gpu_compile.sh <source.sl> [runner.cpp] [output_name]
#
# Examples:
#   # Compile only (no runner):
#   ./scripts/gpu_compile.sh tests/bench_gpu_f32_matmul.sl
#
#   # Compile with runner:
#   ./scripts/gpu_compile.sh tests/bench_gpu_f32_matmul.sl tests/bench_gpu_f32_matmul_runner.cpp
#
#   # Compile with custom output name:
#   ./scripts/gpu_compile.sh tests/bench_gpu_f32_matmul.sl tests/bench_gpu_f32_matmul_runner.cpp my_bench
#
# Output files are placed in tmp_docker/ directory.
#===----------------------------------------------------------------------===//

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse arguments
SOURCE_FILE="$1"
RUNNER_FILE="$2"
OUTPUT_NAME="$3"

# Validate input
if [ -z "$SOURCE_FILE" ]; then
    echo -e "${RED}Error: No source file specified${NC}"
    echo "Usage: $0 <source.sl> [runner.cpp] [output_name]"
    exit 1
fi

if [ ! -f "$SOURCE_FILE" ]; then
    echo -e "${RED}Error: Source file not found: $SOURCE_FILE${NC}"
    exit 1
fi

# Derive output name from source file if not provided
if [ -z "$OUTPUT_NAME" ]; then
    OUTPUT_NAME=$(basename "$SOURCE_FILE" .sl)
fi

# Output directory
OUTPUT_DIR="$PROJECT_ROOT/tmp_docker"
mkdir -p "$OUTPUT_DIR"

# Compiler path
SIMPLANG="$PROJECT_ROOT/build_mlir/src/simplang"
if [ ! -f "$SIMPLANG" ]; then
    echo -e "${RED}Error: simplang compiler not found at $SIMPLANG${NC}"
    echo "Please build the project first: cd build_mlir && cmake --build ."
    exit 1
fi

# GPU runtime source
GPU_RUNTIME="$PROJECT_ROOT/runtime/src/gpu_runtime.cpp"
if [ ! -f "$GPU_RUNTIME" ]; then
    echo -e "${RED}Error: GPU runtime not found at $GPU_RUNTIME${NC}"
    exit 1
fi

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          SimpLang GPU Compiler (cuBLAS Backend)                  ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Compile .sl to .o with GPU support
echo -e "${YELLOW}[1/3] Compiling $SOURCE_FILE → ${OUTPUT_NAME}.o${NC}"
echo "      Flags: --emit-mlir --emit-gpu"
$SIMPLANG "$SOURCE_FILE" --emit-mlir --emit-gpu -o "$OUTPUT_DIR/${OUTPUT_NAME}.o"
echo -e "${GREEN}      ✓ Object file created: $OUTPUT_DIR/${OUTPUT_NAME}.o${NC}"
echo ""

# Step 2: Link to .so with cuBLAS
echo -e "${YELLOW}[2/3] Linking ${OUTPUT_NAME}.o + gpu_runtime → ${OUTPUT_NAME}.so${NC}"

# Check if we're in Docker with CUDA
if command -v nvcc &> /dev/null; then
    echo "      Using nvcc with cuBLAS"
    nvcc -shared -Xcompiler -fPIC \
        -o "$OUTPUT_DIR/${OUTPUT_NAME}.so" \
        "$OUTPUT_DIR/${OUTPUT_NAME}.o" \
        "$GPU_RUNTIME" \
        -DUSE_CUDA -lcublas -lcudart
    echo -e "${GREEN}      ✓ Shared library created: $OUTPUT_DIR/${OUTPUT_NAME}.so (CUDA enabled)${NC}"
else
    echo "      nvcc not found, using gcc with CPU fallback"
    gcc -shared -fPIC \
        -o "$OUTPUT_DIR/${OUTPUT_NAME}.so" \
        "$OUTPUT_DIR/${OUTPUT_NAME}.o" \
        "$GPU_RUNTIME" \
        -lm -lstdc++
    echo -e "${YELLOW}      ⚠ Shared library created: $OUTPUT_DIR/${OUTPUT_NAME}.so (CPU fallback only)${NC}"
fi
echo ""

# Step 3: Compile runner (if provided)
if [ -n "$RUNNER_FILE" ]; then
    if [ ! -f "$RUNNER_FILE" ]; then
        echo -e "${RED}Error: Runner file not found: $RUNNER_FILE${NC}"
        exit 1
    fi

    echo -e "${YELLOW}[3/3] Compiling runner: $RUNNER_FILE → ${OUTPUT_NAME}_runner${NC}"
    g++ -O3 -o "$OUTPUT_DIR/${OUTPUT_NAME}_runner" "$RUNNER_FILE" -ldl -std=c++17
    echo -e "${GREEN}      ✓ Runner created: $OUTPUT_DIR/${OUTPUT_NAME}_runner${NC}"
    echo ""

    echo -e "${BLUE}══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}Build complete!${NC}"
    echo ""
    echo "Output files:"
    echo "  Object:  $OUTPUT_DIR/${OUTPUT_NAME}.o"
    echo "  Library: $OUTPUT_DIR/${OUTPUT_NAME}.so"
    echo "  Runner:  $OUTPUT_DIR/${OUTPUT_NAME}_runner"
    echo ""
    echo "Run with:"
    echo "  $OUTPUT_DIR/${OUTPUT_NAME}_runner $OUTPUT_DIR/${OUTPUT_NAME}.so"
else
    echo -e "${YELLOW}[3/3] Skipping runner compilation (no runner file provided)${NC}"
    echo ""

    echo -e "${BLUE}══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}Build complete!${NC}"
    echo ""
    echo "Output files:"
    echo "  Object:  $OUTPUT_DIR/${OUTPUT_NAME}.o"
    echo "  Library: $OUTPUT_DIR/${OUTPUT_NAME}.so"
    echo ""
    echo "To use the library, link against it from your host program."
fi

echo -e "${BLUE}══════════════════════════════════════════════════════════════════${NC}"
