#!/bin/bash
# Don't use set -e because tests are expected to fail sometimes

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASSED=0
FAILED=0
TOTAL=0

# Set VERBOSE=1 to see full output
VERBOSE=${VERBOSE:-0}

# Build MLIR compiler if needed
if [ ! -f "build_mlir/src/simplang" ]; then
    echo -e "${BLUE}Building MLIR compiler...${NC}"
    cmake -B build_mlir -DENABLE_MLIR=ON -DLLVM_DIR="$(llvm-config-21 --cmakedir || llvm-config --cmakedir)"
    cmake --build build_mlir --target simplang -j8
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

    g++ -o "$output" "$cpp" -ldl -std=c++14 > /dev/null 2>&1
}

# Helper: run test
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
            echo -e "  ${YELLOW}Log saved to:${NC} $log_file"
        fi
        FAILED=$((FAILED + 1))
        return
    fi

    # Link to .so
    if [ "$VERBOSE" = "1" ]; then
        gcc -shared -o "$tmp_kernel.so" "$tmp_kernel.o" -lm 2>&1 | tee -a "$log_file"
        link_result=${PIPESTATUS[0]}
    else
        gcc -shared -o "$tmp_kernel.so" "$tmp_kernel.o" -lm >> "$log_file" 2>&1
        link_result=$?
    fi

    if [ $link_result -ne 0 ]; then
        echo -e "${RED}FAIL${NC} (linking)"
        if [ "$VERBOSE" != "1" ]; then
            echo -e "  ${RED}Error:${NC} $(tail -5 $log_file)"
            echo -e "  ${YELLOW}Log saved to:${NC} $log_file"
        fi
        FAILED=$((FAILED + 1))
        rm -f "$tmp_kernel.o"
        return
    fi

    # Compile runner
    if [ "$VERBOSE" = "1" ]; then
        g++ -o "$tmp_runner" "$runner" -ldl -std=c++14 2>&1 | tee -a "$log_file"
        runner_result=${PIPESTATUS[0]}
    else
        g++ -o "$tmp_runner" "$runner" -ldl -std=c++14 >> "$log_file" 2>&1
        runner_result=$?
    fi

    if [ $runner_result -ne 0 ]; then
        echo -e "${RED}FAIL${NC} (runner compilation)"
        if [ "$VERBOSE" != "1" ]; then
            echo -e "  ${RED}Error:${NC} $(tail -5 $log_file)"
            echo -e "  ${YELLOW}Log saved to:${NC} $log_file"
        fi
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

        # Show performance metrics if requested
        if [ "$show_metrics" = "true" ]; then
            local throughput=$(grep -i "throughput\|tok/s" "$log_file" | head -1)
            local gflops=$(grep -i "GFLOPS" "$log_file" | head -1)
            local time=$(grep -i "time per" "$log_file" | head -1)

            if [ -n "$throughput" ]; then
                echo -e "  ${BLUE}$throughput${NC}"
            fi
            if [ -n "$gflops" ]; then
                echo -e "  ${BLUE}$gflops${NC}"
            fi
            if [ -n "$time" ]; then
                echo -e "  ${BLUE}$time${NC}"
            fi
        fi

        PASSED=$((PASSED + 1))
        rm -f "$log_file"
    else
        echo -e "${RED}FAIL${NC} (execution)"
        if [ "$VERBOSE" != "1" ]; then
            echo -e "  ${RED}Error:${NC} $(tail -5 $log_file)"
            echo -e "  ${YELLOW}Log saved to:${NC} $log_file"
        fi
        FAILED=$((FAILED + 1))
    fi

    rm -f "$tmp_kernel.o" "$tmp_kernel.so" "$tmp_runner"
}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SimpLang MLIR Test Suite${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Matmul tests
echo -e "${BLUE}[Matmul Benchmarks]${NC}"
run_test "Matmul (simple)" \
    "tests/mlir/integration/bench_matmul_simple.sl" \
    "tests/mlir/integration/bench_matmul_baseline.cpp" \
    "" \
    "true"

run_test "Matmul (full)" \
    "tests/mlir/integration/bench_matmul.sl" \
    "tests/mlir/integration/bench_matmul_runner.cpp" \
    "" \
    "true"

run_test "Matmul (tile=8)" \
    "tests/mlir/integration/bench_matmul.sl" \
    "tests/mlir/integration/bench_matmul_runner.cpp" \
    "--tile-size 8" \
    "true"

run_test "Matmul (tile=16)" \
    "tests/mlir/integration/bench_matmul.sl" \
    "tests/mlir/integration/bench_matmul_runner.cpp" \
    "--tile-size 16" \
    "true"

# Conv2D tests
echo -e "\n${BLUE}[Conv2D Tests]${NC}"
run_test "Conv2D (simple)" \
    "tests/mlir/integration/test_conv2d_simple.sl" \
    "tests/mlir/integration/test_conv2d_host.cpp"

run_test "Conv2D (fp32 only)" \
    "tests/mlir/integration/test_conv2d_fp32_only.sl" \
    "tests/mlir/integration/test_conv2d_host.cpp"

run_test "Conv2D (4k benchmark)" \
    "tests/mlir/integration/bench_conv2d_4k.sl" \
    "tests/mlir/integration/bench_conv2d_4k_host.cpp"

# Transformer kernels
echo -e "\n${BLUE}[Transformer Kernels]${NC}"
run_test "RMSNorm" \
    "examples/llama2/kernels/test_rmsnorm.sl" \
    "examples/llama2/kernels/test_rmsnorm_host.cpp"

run_test "Softmax" \
    "examples/llama2/kernels/test_softmax.sl" \
    "examples/llama2/kernels/test_softmax_host.cpp"

run_test "SiLU" \
    "examples/llama2/kernels/test_silu.sl" \
    "examples/llama2/kernels/test_silu_host.cpp"

run_test "SwiGLU" \
    "examples/llama2/kernels/test_swiglu.sl" \
    "examples/llama2/kernels/test_swiglu_host.cpp"

# Stories110M (tiny stories model)
echo -e "\n${BLUE}[Stories110M (TinyStories)]${NC}"
run_test "Stories110M (standard)" \
    "examples/llama2/stories110M.sl" \
    "examples/llama2/test_stories110M.cpp" \
    "" \
    "true"

run_test "Stories110M (tensor)" \
    "examples/llama2/stories110M_tensor.sl" \
    "examples/llama2/test_stories110M.cpp" \
    "" \
    "true"

run_test "Stories110M (tile=8)" \
    "examples/llama2/stories110M.sl" \
    "examples/llama2/test_stories110M.cpp" \
    "--tile-size 8" \
    "true"

run_test "Stories110M (tile=16)" \
    "examples/llama2/stories110M.sl" \
    "examples/llama2/test_stories110M.cpp" \
    "--tile-size 16" \
    "true"

# LLaMA models
echo -e "\n${BLUE}[LLaMA Models]${NC}"
run_test "llama2_1B" \
    "examples/llama2/llama2_1B.sl" \
    "examples/llama2/test_llama2_host.cpp" \
    "--tile-size 8" \
    "true" \
    "60" \
    "1B"

run_test "llama2_3B" \
    "examples/llama2/llama2_3B.sl" \
    "examples/llama2/test_llama2_host.cpp" \
    "--tile-size 8" \
    "true" \
    "120" \
    "3B"

# Summary
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Results${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Total:  $TOTAL"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"

if [ $FAILED -eq 0 ]; then
    echo -e "\n${GREEN}All tests passed! ✓${NC}"
    exit 0
else
    echo -e "\n${RED}Some tests failed! ✗${NC}"
    exit 1
fi
