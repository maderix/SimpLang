#!/bin/bash
#===- test_mlir_backend.sh - MLIR Backend Integration Tests -------------===//
#
# Part of the SimpLang Project
#
# This script tests the end-to-end MLIR compilation pipeline by compiling
# SimpLang programs with both LLVM and MLIR backends and comparing results.
#
#===----------------------------------------------------------------------===//

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build_mlir"
TEST_DIR="$PROJECT_ROOT/tests"
TEMP_DIR="/tmp/mlir_integration_tests"

# Create temp directory
mkdir -p "$TEMP_DIR"

# Compiler and test runner paths
SIMPLANG="$BUILD_DIR/src/simplang"

# Test counter
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

echo "========================================"
echo "MLIR Backend Integration Tests"
echo "========================================"
echo ""

# Function to run a single test
run_test() {
    local test_name="$1"
    local test_file="$2"
    local runner="$3"
    local expected="$4"

    TESTS_RUN=$((TESTS_RUN + 1))
    echo -n "Testing $test_name... "

    # Compile with MLIR backend
    local mlir_obj="$TEMP_DIR/${test_name}_mlir.o"
    local mlir_so="$TEMP_DIR/${test_name}_mlir.so"

    if ! "$SIMPLANG" "$test_file" --emit-mlir -o "$mlir_obj" > "$TEMP_DIR/${test_name}_mlir.log" 2>&1; then
        echo -e "${RED}FAILED${NC} (compilation failed)"
        echo "  Log: $TEMP_DIR/${test_name}_mlir.log"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi

    # Create shared library
    if ! g++ -shared "$mlir_obj" -o "$mlir_so" 2>&1 > "$TEMP_DIR/${test_name}_link.log"; then
        echo -e "${RED}FAILED${NC} (linking failed)"
        echo "  Log: $TEMP_DIR/${test_name}_link.log"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi

    # Run the test
    local result=$("$runner" "$mlir_so" 2>&1 | grep "Result:" | awk '{print $2}')

    # Check result
    if [ "$result" = "$expected" ]; then
        echo -e "${GREEN}PASSED${NC} (result: $result)"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${RED}FAILED${NC} (expected: $expected, got: $result)"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Run tests for basic arithmetic
echo "=== Basic Tests ==="
echo ""

if [ -f "$BUILD_DIR/tests/test_arithmetic_runner" ]; then
    run_test "arithmetic" \
        "$TEST_DIR/test_arithmetic.sl" \
        "$BUILD_DIR/tests/test_arithmetic_runner" \
        "72"
fi

if [ -f "$BUILD_DIR/tests/test_main_runner" ]; then
    run_test "main" \
        "$TEST_DIR/test_main.sl" \
        "$BUILD_DIR/tests/test_main_runner" \
        "30"
fi

if [ -f "$BUILD_DIR/tests/test_loop_runner" ]; then
    run_test "loop" \
        "$TEST_DIR/test_loop.sl" \
        "$BUILD_DIR/tests/test_loop_runner" \
        "135"
fi

if [ -f "$BUILD_DIR/tests/test_fibonacci_runner" ]; then
    run_test "fibonacci" \
        "$TEST_DIR/test_fibonacci.sl" \
        "$BUILD_DIR/tests/test_fibonacci_runner" \
        "110"
fi

if [ -f "$BUILD_DIR/tests/test_return_runner" ]; then
    run_test "return" \
        "$TEST_DIR/test_return.sl" \
        "$BUILD_DIR/tests/test_return_runner" \
        "85"
fi

# Summary
echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo "Total tests run:    $TESTS_RUN"
echo -e "Tests passed:       ${GREEN}$TESTS_PASSED${NC}"
if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "Tests failed:       ${RED}$TESTS_FAILED${NC}"
else
    echo -e "Tests failed:       ${GREEN}$TESTS_FAILED${NC}"
fi
echo ""

# Clean up temp files on success
if [ $TESTS_FAILED -eq 0 ]; then
    echo "All tests passed! Cleaning up..."
    rm -rf "$TEMP_DIR"
    echo -e "${GREEN}✓ MLIR backend integration tests PASSED${NC}"
    exit 0
else
    echo "Some tests failed. Logs available in: $TEMP_DIR"
    echo -e "${RED}✗ MLIR backend integration tests FAILED${NC}"
    exit 1
fi
