#!/bin/bash
#
# INT8 MatMul Benchmark Runner
#
# Compiles and runs INT8 matmul benchmarks comparing:
# - SimpLang tensor_matmul (current implementation)
# - Eigen INT8 matmul
# - VNNI-optimized reference
#
# Usage: ./run_int8_benchmarks.sh [--quick]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

QUICK_MODE=false
if [[ "$1" == "--quick" ]]; then
    QUICK_MODE=true
fi

echo "═══════════════════════════════════════════════════════════════════"
echo "   INT8 MatMul Benchmark Suite"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

# Check for MLIR compiler
if [[ ! -f "./build_mlir/src/simplang" ]]; then
    echo "ERROR: MLIR compiler not found at ./build_mlir/src/simplang"
    echo "Please build the MLIR backend first."
    exit 1
fi

# Step 1: Compile SimpLang INT8 benchmark
echo "Step 1: Compiling SimpLang INT8 benchmarks..."
./build_mlir/src/simplang simptensor/benchmarks/bench_int8_matmul.sl --emit-mlir -o /tmp/bench_int8_matmul.o 2>&1

if [[ $? -ne 0 ]]; then
    echo "ERROR: Failed to compile INT8 benchmarks"
    exit 1
fi

# Step 2: Link shared library
echo "Step 2: Linking shared library..."
gcc -shared -o /tmp/bench_int8_matmul.so /tmp/bench_int8_matmul.o -lm

if [[ $? -ne 0 ]]; then
    echo "ERROR: Failed to link shared library"
    exit 1
fi
echo "Created: /tmp/bench_int8_matmul.so"

# Step 3: Compile benchmark runner
echo "Step 3: Compiling benchmark runner..."

# Try clang first, then gcc
CXX_CMD=""
if command -v clang++-14 &> /dev/null; then
    CXX_CMD="clang++-14"
elif command -v clang++ &> /dev/null; then
    CXX_CMD="clang++"
elif command -v g++ &> /dev/null; then
    CXX_CMD="g++"
else
    echo "ERROR: No C++ compiler found"
    exit 1
fi

echo "Using compiler: $CXX_CMD"

$CXX_CMD -O3 -march=native -mavx512vnni \
    -I "$PROJECT_ROOT/thirdparty/eigen" \
    "$PROJECT_ROOT/simptensor/benchmarks/bench_int8_matmul_runner.cpp" \
    -o /tmp/bench_int8_matmul_runner -ldl

if [[ $? -ne 0 ]]; then
    echo "ERROR: Failed to compile benchmark runner"
    exit 1
fi
echo "Created: /tmp/bench_int8_matmul_runner"

# Step 4: Run benchmarks
echo ""
echo "Step 4: Running benchmarks..."
echo ""

/tmp/bench_int8_matmul_runner /tmp/bench_int8_matmul.so

echo ""
echo "Benchmark complete!"
