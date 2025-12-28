#!/bin/bash
# MLIR Test Suite with Category Filtering
# Usage: ./run_mlir_tests.sh [category] [options]
# Categories: all, core, arrays, tensor, vnni, llama
# Options: --verbose, --help

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PASSED=0
FAILED=0
SKIPPED=0
TOTAL=0

# Set VERBOSE=1 to see full output
VERBOSE=${VERBOSE:-0}

# Parse arguments
CATEGORY="all"
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [category] [options]"
            echo ""
            echo "Categories:"
            echo "  all     - Run all tests (default)"
            echo "  core    - Core language tests (arithmetic, control flow, functions)"
            echo "  arrays  - Array tests (1D, multi-dimensional, dtypes)"
            echo "  tensor  - Tensor/simptensor tests (matmul, reductions, elementwise)"
            echo "  vnni    - VNNI INT8 optimization tests"
            echo "  llama   - LLaMA/transformer kernel tests"
            echo ""
            echo "Options:"
            echo "  --verbose, -v  Show full compilation/execution output"
            echo "  --help, -h     Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0              # Run all tests"
            echo "  $0 core         # Run only core tests"
            echo "  $0 vnni -v      # Run VNNI tests with verbose output"
            exit 0
            ;;
        all|core|arrays|tensor|vnni|llama)
            CATEGORY=$1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build MLIR compiler if needed
if [ ! -f "build_mlir/src/simplang" ]; then
    echo -e "${BLUE}Building MLIR compiler...${NC}"
    cmake -B build_mlir -DENABLE_MLIR=ON -DLLVM_DIR="$(llvm-config-21 --cmakedir || llvm-config --cmakedir)"
    cmake --build build_mlir --target simplang -j8
fi

# Build generic runner if needed
GENERIC_RUNNER="/tmp/generic_runner_$$"
g++ -o "$GENERIC_RUNNER" tests/generic_runner.cpp -ldl -std=c++14 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to build generic runner${NC}"
    exit 1
fi

# Helper: compile SimpLang kernel to .so
compile_kernel() {
    local kernel=$1
    local output=$2
    local flags=${3:-""}

    ./build_mlir/src/simplang "$kernel" --emit-mlir $flags -o "$output.o" > /dev/null 2>&1 && \
    gcc -shared -o "$output.so" "$output.o" -lm > /dev/null 2>&1
}

# Helper: compile C++ runner
compile_runner() {
    local cpp=$1
    local output=$2
    local extra_flags=${3:-""}

    g++ -o "$output" "$cpp" -ldl -std=c++14 $extra_flags > /dev/null 2>&1
}

# Helper: run test with custom runner
run_test() {
    local name=$1
    local kernel=$2
    local runner=$3
    local flags=${4:-""}
    local show_metrics=${5:-false}
    local timeout_sec=${6:-10}
    local extra_args=${7:-""}

    TOTAL=$((TOTAL + 1))
    echo -ne "${YELLOW}[$TOTAL]${NC} $name ... "

    local tmp_kernel="/tmp/mlir_test_$$_$TOTAL"
    local tmp_runner="/tmp/mlir_runner_$$_$TOTAL"
    local log_file="/tmp/mlir_log_$$_$TOTAL.txt"

    # Compile kernel
    if [ "$VERBOSE" = "1" ]; then
        ./build_mlir/src/simplang "$kernel" --emit-mlir $flags -o "$tmp_kernel.o" 2>&1 | tee "$log_file"
        compile_result=${PIPESTATUS[0]}
    else
        ./build_mlir/src/simplang "$kernel" --emit-mlir $flags -o "$tmp_kernel.o" > "$log_file" 2>&1
        compile_result=$?
    fi

    if [ $compile_result -ne 0 ]; then
        echo -e "${RED}FAIL${NC} (kernel compilation)"
        if [ "$VERBOSE" != "1" ]; then
            echo -e "  ${RED}Error:${NC} $(tail -5 $log_file)"
        fi
        FAILED=$((FAILED + 1))
        return
    fi

    # Link to .so
    gcc -shared -o "$tmp_kernel.so" "$tmp_kernel.o" -lm >> "$log_file" 2>&1
    if [ $? -ne 0 ]; then
        echo -e "${RED}FAIL${NC} (linking)"
        FAILED=$((FAILED + 1))
        rm -f "$tmp_kernel.o"
        return
    fi

    # Compile runner
    g++ -o "$tmp_runner" "$runner" -ldl -std=c++14 >> "$log_file" 2>&1
    if [ $? -ne 0 ]; then
        echo -e "${RED}FAIL${NC} (runner compilation)"
        FAILED=$((FAILED + 1))
        rm -f "$tmp_kernel.o" "$tmp_kernel.so"
        return
    fi

    # Run test
    if [ "$VERBOSE" = "1" ]; then
        timeout $timeout_sec "$tmp_runner" "$tmp_kernel.so" $extra_args 2>&1 | tee -a "$log_file"
        test_result=${PIPESTATUS[0]}
    else
        timeout $timeout_sec "$tmp_runner" "$tmp_kernel.so" $extra_args >> "$log_file" 2>&1
        test_result=$?
    fi

    if [ $test_result -eq 0 ]; then
        echo -e "${GREEN}PASS${NC}"
        if [ "$show_metrics" = "true" ]; then
            local throughput=$(grep -i "throughput\|tok/s\|GFLOPS\|GOPS" "$log_file" | head -3)
            if [ -n "$throughput" ]; then
                echo -e "  ${CYAN}$throughput${NC}"
            fi
        fi
        PASSED=$((PASSED + 1))
        rm -f "$log_file"
    else
        echo -e "${RED}FAIL${NC} (execution)"
        if [ "$VERBOSE" != "1" ]; then
            echo -e "  ${RED}Output:${NC} $(tail -5 $log_file)"
        fi
        FAILED=$((FAILED + 1))
    fi

    rm -f "$tmp_kernel.o" "$tmp_kernel.so" "$tmp_runner"
}

# Helper: run simple test with generic runner (just compile and check return value)
run_simple_test() {
    local name=$1
    local kernel=$2
    local func_name=$3
    local expected=$4
    local tolerance=${5:-0.001}
    local flags=${6:-""}

    TOTAL=$((TOTAL + 1))
    echo -ne "${YELLOW}[$TOTAL]${NC} $name ... "

    local tmp_kernel="/tmp/mlir_test_$$_$TOTAL"
    local log_file="/tmp/mlir_log_$$_$TOTAL.txt"

    # Compile kernel
    ./build_mlir/src/simplang "$kernel" --emit-mlir $flags -o "$tmp_kernel.o" > "$log_file" 2>&1
    if [ $? -ne 0 ]; then
        echo -e "${RED}FAIL${NC} (kernel compilation)"
        if [ "$VERBOSE" != "1" ]; then
            echo -e "  ${RED}Error:${NC} $(tail -3 $log_file)"
        fi
        FAILED=$((FAILED + 1))
        return
    fi

    # Link to .so
    gcc -shared -o "$tmp_kernel.so" "$tmp_kernel.o" -lm >> "$log_file" 2>&1
    if [ $? -ne 0 ]; then
        echo -e "${RED}FAIL${NC} (linking)"
        FAILED=$((FAILED + 1))
        rm -f "$tmp_kernel.o"
        return
    fi

    # Run with generic runner
    "$GENERIC_RUNNER" "$tmp_kernel.so" "$func_name" "$expected" "$tolerance" >> "$log_file" 2>&1
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}PASS${NC}"
        PASSED=$((PASSED + 1))
        rm -f "$log_file"
    else
        echo -e "${RED}FAIL${NC}"
        if [ "$VERBOSE" != "1" ]; then
            echo -e "  ${RED}Output:${NC} $(tail -3 $log_file)"
        fi
        FAILED=$((FAILED + 1))
    fi

    rm -f "$tmp_kernel.o" "$tmp_kernel.so"
}

# Helper: run compile-only test (just check if kernel compiles)
run_compile_test() {
    local name=$1
    local kernel=$2
    local flags=${3:-""}

    TOTAL=$((TOTAL + 1))
    echo -ne "${YELLOW}[$TOTAL]${NC} $name (compile) ... "

    local tmp_kernel="/tmp/mlir_test_$$_$TOTAL"
    local log_file="/tmp/mlir_log_$$_$TOTAL.txt"

    ./build_mlir/src/simplang "$kernel" --emit-mlir $flags -o "$tmp_kernel.o" > "$log_file" 2>&1
    if [ $? -ne 0 ]; then
        echo -e "${RED}FAIL${NC}"
        if [ "$VERBOSE" != "1" ]; then
            echo -e "  ${RED}Error:${NC} $(tail -3 $log_file)"
        fi
        FAILED=$((FAILED + 1))
        return
    fi

    gcc -shared -o "$tmp_kernel.so" "$tmp_kernel.o" -lm >> "$log_file" 2>&1
    if [ $? -ne 0 ]; then
        echo -e "${RED}FAIL${NC} (linking)"
        FAILED=$((FAILED + 1))
    else
        echo -e "${GREEN}PASS${NC}"
        PASSED=$((PASSED + 1))
        rm -f "$log_file"
    fi

    rm -f "$tmp_kernel.o" "$tmp_kernel.so"
}

# ============================================================================
# TEST CATEGORIES
# ============================================================================

run_core_tests() {
    echo -e "\n${BLUE}[Core Language Tests]${NC}"

    # Arithmetic
    run_compile_test "Arithmetic" "tests/test_arithmetic.sl"

    # Control flow
    run_compile_test "Control Flow" "tests/test_control_flow.sl"

    # Functions
    run_compile_test "Functions" "tests/test_functions.sl"

    # Loops
    run_compile_test "Loop Accumulation" "tests/test_loop_accumulation.sl"

    # Return types
    run_compile_test "Return Types" "tests/test_return_types.sl"

    # Bitwise ops
    run_compile_test "Bitwise Operations" "tests/test_bitwise.sl"

    # Static types
    run_compile_test "Static Types" "tests/test_static_types.sl"

    # Conditions
    run_compile_test "Conditions" "tests/test_conditions.sl"
}

run_array_tests() {
    echo -e "\n${BLUE}[Array Tests]${NC}"

    # Array initialization
    run_compile_test "Array Init" "tests/test_array_init.sl"

    # Basic arrays
    run_compile_test "Arrays Basic" "tests/test_arrays.sl"

    # Multi-dimensional arrays
    run_simple_test "Multi-Dim Arrays" "tests/test_multidim_arrays.sl" "kernel_main" "1179.0" "1.0"

    # Dtype support
    run_simple_test "Dtype Support" "tests/test_dtypes_basic.sl" "kernel_main" "10823.858" "1.0"

    # Array comprehensive
    run_compile_test "Array Comprehensive" "tests/test_array_comprehensive.sl"
}

run_tensor_tests() {
    echo -e "\n${BLUE}[Tensor/SimPTensor Tests]${NC}"

    # Basic tensor ops
    run_compile_test "Tensor Basic" "tests/test_tensor_basic.sl"

    # Tensor matmul with dot product
    run_compile_test "Tensor MatMul Dot" "tests/test_tensor_matmul_dot.sl"

    # Tensor reductions
    run_compile_test "Tensor Reductions" "tests/test_tensor_reductions.sl"

    # Axis reductions
    run_compile_test "Tensor Axis Reductions" "tests/test_tensor_axis_reductions.sl"

    # Elementwise ops
    run_compile_test "Tensor Elementwise" "tests/test_tensor_elementwise.sl"

    # Tensor matmul (main)
    run_compile_test "Tensor MatMul" "tests/test_tensor_matmul.sl"
}

run_vnni_tests() {
    echo -e "\n${BLUE}[VNNI INT8 Tests]${NC}"

    # VNNI annotated
    run_compile_test "VNNI Annotated" "tests/test_vnni_annotated.sl" "--llvm-vectorize"

    # VNNI debug
    run_compile_test "VNNI Debug" "tests/test_vnni_debug.sl" "--llvm-vectorize"

    # VNNI K=32
    run_compile_test "VNNI K=32" "tests/test_vnni_32.sl" "--llvm-vectorize"

    # VNNI full
    run_compile_test "VNNI Full" "tests/test_vnni_full.sl" "--llvm-vectorize"

    # VNNI with OpenMP
    run_compile_test "VNNI OpenMP" "tests/test_vnni_omp.sl" "--llvm-vectorize"

    # VNNI sweep (all sizes)
    run_compile_test "VNNI Sweep All" "tests/vnni_sweep_all.sl" "--llvm-vectorize"

    # VNNI MNK sweep
    run_compile_test "VNNI MNK Sweep" "tests/vnni_mnk_sweep.sl" "--llvm-vectorize"

    # INT8 VNNI annotated benchmark
    run_compile_test "INT8 VNNI Benchmark" "tests/bench_int8_vnni_annotated.sl" "--llvm-vectorize"

    # Tile sweep benchmark
    run_compile_test "Tile Sweep Benchmark" "tests/tile_sweep_benchmark.sl" "--llvm-vectorize"

    # Run VNNI sweep benchmark with runner if exists
    if [ -f "tests/vnni_sweep_runner.cpp" ]; then
        # Note: VNNI sweep needs OpenMP runtime, compile and link manually
        TOTAL=$((TOTAL + 1))
        echo -ne "${YELLOW}[$TOTAL]${NC} VNNI Sweep Benchmark ... "
        local tmp="/tmp/vnni_bench_$$"
        ./build_mlir/src/simplang tests/vnni_sweep_all.sl --emit-mlir --llvm-vectorize -o "$tmp.o" > "$tmp.log" 2>&1
        if [ $? -ne 0 ]; then
            echo -e "${RED}FAIL${NC} (compile)"
            FAILED=$((FAILED + 1))
        else
            gcc -shared -o "$tmp.so" "$tmp.o" -lm /lib/x86_64-linux-gnu/libomp.so.5 >> "$tmp.log" 2>&1
            g++ -O3 -fopenmp -o "$tmp.runner" tests/vnni_sweep_runner.cpp -ldl -std=c++14 >> "$tmp.log" 2>&1
            if timeout 120 "$tmp.runner" "$tmp.so" >> "$tmp.log" 2>&1; then
                echo -e "${GREEN}PASS${NC}"
                grep -i "GOPS\|throughput" "$tmp.log" | head -3 | while read line; do echo -e "  ${CYAN}$line${NC}"; done
                PASSED=$((PASSED + 1))
            else
                echo -e "${RED}FAIL${NC} (execution)"
                tail -5 "$tmp.log"
                FAILED=$((FAILED + 1))
            fi
        fi
        rm -f "$tmp.o" "$tmp.so" "$tmp.runner" "$tmp.log"
    fi
}

run_llama_tests() {
    echo -e "\n${BLUE}[LLaMA/Transformer Kernel Tests]${NC}"

    # RMSNorm
    if [ -f "tests/llama_rmsnorm_runner.cpp" ]; then
        run_test "RMSNorm" "tests/llama_rmsnorm.sl" "tests/llama_rmsnorm_runner.cpp"
    else
        run_compile_test "RMSNorm" "tests/llama_rmsnorm.sl"
    fi

    # Softmax
    if [ -f "tests/llama_softmax_runner.cpp" ]; then
        run_test "Softmax" "tests/llama_softmax.sl" "tests/llama_softmax_runner.cpp"
    else
        run_compile_test "Softmax" "tests/llama_softmax.sl"
    fi

    # SiLU
    if [ -f "tests/llama_silu_runner.cpp" ]; then
        run_test "SiLU" "tests/llama_silu.sl" "tests/llama_silu_runner.cpp"
    else
        run_compile_test "SiLU" "tests/llama_silu.sl"
    fi

    # Attention
    run_compile_test "Attention" "tests/llama_attention.sl"

    # Stories110M (if available)
    if [ -f "examples/llama2/stories110M.sl" ] && [ -f "examples/llama2/test_stories110M.cpp" ]; then
        echo -e "\n${BLUE}[Stories110M Integration]${NC}"
        run_test "Stories110M" \
            "examples/llama2/stories110M.sl" \
            "examples/llama2/test_stories110M.cpp" \
            "" \
            "true" \
            "30"
    fi
}

# ============================================================================
# MAIN
# ============================================================================

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SimpLang MLIR Test Suite${NC}"
echo -e "${BLUE}Category: ${CYAN}$CATEGORY${NC}"
echo -e "${BLUE}========================================${NC}"

case $CATEGORY in
    all)
        run_core_tests
        run_array_tests
        run_tensor_tests
        run_vnni_tests
        run_llama_tests
        ;;
    core)
        run_core_tests
        ;;
    arrays)
        run_array_tests
        ;;
    tensor)
        run_tensor_tests
        ;;
    vnni)
        run_vnni_tests
        ;;
    llama)
        run_llama_tests
        ;;
esac

# Cleanup
rm -f "$GENERIC_RUNNER"

# Summary
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Results${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Total:  $TOTAL"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"

if [ $FAILED -eq 0 ]; then
    echo -e "\n${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "\n${RED}Some tests failed!${NC}"
    exit 1
fi
