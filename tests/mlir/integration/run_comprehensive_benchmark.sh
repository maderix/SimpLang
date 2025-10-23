#!/bin/bash
# Comprehensive benchmark runner with GCC and Clang baselines

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="/home/maderix/simple-lang/build_mlir"

cd "$BUILD_DIR"

echo "=========================================="
echo "  COMPREHENSIVE ARRAY PATTERN BENCHMARK"
echo "  Compiling SimpLang + GCC + Clang"
echo "=========================================="
echo ""

# 1. Compile SimpLang benchmark
echo "[1/5] Compiling SimpLang benchmark..."
./src/simplang ../tests/mlir/integration/bench_array_patterns.sl --emit-mlir -o /tmp/bench_patterns.o
g++ -shared /tmp/bench_patterns.o -o /tmp/bench_patterns.so
echo "  ✓ SimpLang compiled"

# 2. Compile C++ baseline with GCC
echo "[2/5] Compiling C++ baseline with GCC -O3 -march=native..."
g++ -O3 -march=native -c ../tests/mlir/integration/bench_array_patterns_cpp.cpp -o /tmp/bench_patterns_gcc.o
echo "  ✓ GCC baseline compiled"

# 3. Compile C++ baseline with Clang
echo "[3/5] Compiling C++ baseline with Clang -O3 -march=native..."
clang++ -O3 -march=native -c ../tests/mlir/integration/bench_array_patterns_cpp.cpp -o /tmp/bench_patterns_clang.o
echo "  ✓ Clang baseline compiled"

# 4. Build host runners
echo "[4/5] Building benchmark runners..."
g++ -O3 ../tests/mlir/integration/bench_array_patterns_host_dual.cpp /tmp/bench_patterns_gcc.o /tmp/bench_patterns_clang.o -o /tmp/bench_patterns_runner -ldl
echo "  ✓ Runners built"

# 5. Run benchmarks
echo "[5/5] Running benchmarks (this will take ~2-3 minutes)..."
echo ""
/tmp/bench_patterns_runner /tmp/bench_patterns.so

echo ""
echo "Benchmark results saved to: /tmp/benchmark_results.csv"
echo ""
echo "Generating visualizations..."
python3 ../tests/mlir/integration/plot_benchmark_results.py
echo ""
echo "✓ Charts saved to /tmp/benchmark_*.png"
