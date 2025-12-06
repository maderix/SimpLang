#!/bin/bash
#
# VNNI Standalone Validation Tests
#
# This script compiles and runs the VNNI validation tests with both GCC and Clang:
# 1. Availability test - Check if VNNI is supported
# 2. Correctness test - Verify vpdpbusd produces correct results
# 3. Throughput benchmark - Measure raw VNNI performance
#
# Usage: ./run_vnni_tests.sh [--skip-bench] [--gcc-only] [--clang-only]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
SKIP_BENCH=false
GCC_ONLY=false
CLANG_ONLY=false

for arg in "$@"; do
    case $arg in
        --skip-bench) SKIP_BENCH=true ;;
        --gcc-only) GCC_ONLY=true ;;
        --clang-only) CLANG_ONLY=true ;;
    esac
done

# Output directory
OUT_DIR="/tmp/vnni_tests"
mkdir -p "$OUT_DIR"

echo "=============================================="
echo "  VNNI Standalone Validation Tests"
echo "=============================================="
echo ""
echo "Output directory: $OUT_DIR"
echo ""

# Detect available compilers
GCC_AVAILABLE=false
CLANG_AVAILABLE=false
GCC_CMD="g++"
CLANG_CMD="clang++-14"

# Check for g++
if command -v g++ &> /dev/null; then
    if g++ -mavx512vnni -E -x c++ /dev/null &> /dev/null 2>&1; then
        GCC_AVAILABLE=true
        GCC_VERSION=$(g++ --version | head -n1)
        echo "Found GCC: $GCC_VERSION"
    else
        echo "GCC found but does not support -mavx512vnni"
    fi
else
    echo "GCC (g++) not found"
fi

# Check for clang++-14
if command -v clang++-14 &> /dev/null; then
    if clang++-14 -mavx512vnni -E -x c++ /dev/null &> /dev/null 2>&1; then
        CLANG_AVAILABLE=true
        CLANG_VERSION=$(clang++-14 --version | head -n1)
        echo "Found Clang: $CLANG_VERSION"
    else
        echo "Clang-14 found but does not support -mavx512vnni"
    fi
else
    # Try generic clang++
    if command -v clang++ &> /dev/null; then
        if clang++ -mavx512vnni -E -x c++ /dev/null &> /dev/null 2>&1; then
            CLANG_CMD="clang++"
            CLANG_AVAILABLE=true
            CLANG_VERSION=$(clang++ --version | head -n1)
            echo "Found Clang: $CLANG_VERSION"
        fi
    else
        echo "Clang (clang++-14 or clang++) not found"
    fi
fi

echo ""

# Determine which compilers to use
COMPILERS=()
if [[ "$GCC_ONLY" == true ]]; then
    if [[ "$GCC_AVAILABLE" == true ]]; then
        COMPILERS+=("gcc")
    else
        echo "ERROR: --gcc-only specified but GCC is not available"
        exit 1
    fi
elif [[ "$CLANG_ONLY" == true ]]; then
    if [[ "$CLANG_AVAILABLE" == true ]]; then
        COMPILERS+=("clang")
    else
        echo "ERROR: --clang-only specified but Clang is not available"
        exit 1
    fi
else
    # Use both if available
    if [[ "$GCC_AVAILABLE" == true ]]; then
        COMPILERS+=("gcc")
    fi
    if [[ "$CLANG_AVAILABLE" == true ]]; then
        COMPILERS+=("clang")
    fi
fi

if [[ ${#COMPILERS[@]} -eq 0 ]]; then
    echo "ERROR: No suitable compiler found with AVX-512 VNNI support."
    echo "Please install GCC 9+ or Clang 9+."
    exit 1
fi

echo "Using compilers: ${COMPILERS[*]}"
echo ""

# Function to compile and test with a specific compiler
run_with_compiler() {
    local COMPILER_NAME=$1
    local CXX_CMD=$2
    local SUFFIX=$3

    echo "=============================================="
    echo "  Compiling with $COMPILER_NAME"
    echo "=============================================="
    echo ""

    echo "[1/3] Compiling test_vnni_availability.cpp..."
    $CXX_CMD -O3 -march=native \
        test_vnni_availability.cpp \
        -o "$OUT_DIR/test_vnni_availability_$SUFFIX"
    echo "      -> $OUT_DIR/test_vnni_availability_$SUFFIX"

    echo "[2/3] Compiling test_vnni_correctness.cpp..."
    $CXX_CMD -O3 -march=native -mavx512vnni \
        test_vnni_correctness.cpp \
        -o "$OUT_DIR/test_vnni_correctness_$SUFFIX"
    echo "      -> $OUT_DIR/test_vnni_correctness_$SUFFIX"

    echo "[3/3] Compiling bench_vnni_throughput.cpp..."
    $CXX_CMD -O3 -march=native -mavx512vnni \
        bench_vnni_throughput.cpp \
        -o "$OUT_DIR/bench_vnni_throughput_$SUFFIX"
    echo "      -> $OUT_DIR/bench_vnni_throughput_$SUFFIX"

    echo ""
    echo "All tests compiled successfully with $COMPILER_NAME."
    echo ""

    # Run tests
    echo "=============================================="
    echo "  Running Tests ($COMPILER_NAME)"
    echo "=============================================="

    # Test 1: Availability
    echo ""
    echo "=== [1/3] VNNI Availability Test ($COMPILER_NAME) ==="
    echo ""
    if ! "$OUT_DIR/test_vnni_availability_$SUFFIX"; then
        echo ""
        echo "ERROR: VNNI is not available on this CPU."
        echo "The INT8/INT4 matmul benchmarks require AVX-512 VNNI support."
        return 1
    fi

    # Test 2: Correctness
    echo ""
    echo "=== [2/3] VNNI Correctness Test ($COMPILER_NAME) ==="
    echo ""
    if ! "$OUT_DIR/test_vnni_correctness_$SUFFIX"; then
        echo ""
        echo "ERROR: VNNI correctness tests failed with $COMPILER_NAME!"
        echo "This indicates a hardware or compiler bug."
        return 1
    fi

    # Test 3: Throughput (optional)
    if [[ "$SKIP_BENCH" != true ]]; then
        echo ""
        echo "=== [3/3] VNNI Throughput Benchmark ($COMPILER_NAME) ==="
        echo ""
        "$OUT_DIR/bench_vnni_throughput_$SUFFIX"
    else
        echo ""
        echo "=== [3/3] VNNI Throughput Benchmark ($COMPILER_NAME) (SKIPPED) ==="
        echo "   Use without --skip-bench to run performance benchmark."
    fi

    echo ""
    return 0
}

# Track results
GCC_RESULT="N/A"
CLANG_RESULT="N/A"

# Run with each compiler
for compiler in "${COMPILERS[@]}"; do
    if [[ "$compiler" == "gcc" ]]; then
        if run_with_compiler "GCC" "$GCC_CMD" "gcc"; then
            GCC_RESULT="PASS"
        else
            GCC_RESULT="FAIL"
        fi
    elif [[ "$compiler" == "clang" ]]; then
        if run_with_compiler "Clang" "$CLANG_CMD" "clang"; then
            CLANG_RESULT="PASS"
        else
            CLANG_RESULT="FAIL"
        fi
    fi
done

# Final summary
echo "=============================================="
echo "  Final Summary"
echo "=============================================="
echo ""
echo "Compiler Results:"
echo "  GCC:   $GCC_RESULT"
echo "  Clang: $CLANG_RESULT"
echo ""

# Check if any failed
if [[ "$GCC_RESULT" == "FAIL" ]] || [[ "$CLANG_RESULT" == "FAIL" ]]; then
    echo "Some tests FAILED. Please check the output above."
    exit 1
fi

echo "All VNNI validation tests PASSED!"
echo ""
echo "Next steps:"
echo "  1. Proceed with Session 1: VNNI-based INT8 MatMul implementation"
echo "  2. The throughput baseline can be used to validate matmul performance"
echo ""
