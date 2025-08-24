#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

BUILD_DIR="build"

# Function to compile kernel
compile_kernel() {
    local kernel=$1
    echo -e "\nCompiling kernel: ${kernel}..."
    if ./build/src/simplang "tests/${kernel}.sl"; then
        echo -e "${GREEN}Successfully compiled ${kernel}${NC}"
    else
        echo -e "${RED}Failed to compile ${kernel}${NC}"
        exit 1
    fi
}

# Build everything
echo "Building SimpLang compiler and host programs..."
./build.sh

# Build all test programs
echo "Building test runners..."
cmake --build $BUILD_DIR --target all_test_runners

# Compile all test kernels
echo "Compiling test kernels..."
for kernel in arithmetic fibonacci loop conditions main return array_comprehensive; do  # Added array_comprehensive - it's working
    compile_kernel "test_${kernel}"
done

# Compile new SIMD array tests
echo "Compiling SIMD array tests..."
compile_kernel "test_simd_arrays"
compile_kernel "test_simple_simd"

# Compile performance test kernels
echo -e "\n${BLUE}Compiling performance test kernels...${NC}"
compile_kernel "test_baseline_perf"  # This one works
compile_kernel "perf_array"  # New array performance test

# Run regular tests
echo -e "\nRunning tests..."
cd $BUILD_DIR && ctest --output-on-failure

# Run performance test separately
echo -e "\n${BLUE}Running SIMD performance test...${NC}"
./tests/perf_simd_runner ./tests/obj/perf_simd.so

cd ..
