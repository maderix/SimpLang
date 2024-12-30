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
for kernel in arithmetic fibonacci loop conditions simd main return; do
    compile_kernel "test_${kernel}"
done

# Compile performance test kernel
echo -e "\n${BLUE}Compiling performance test kernel...${NC}"
compile_kernel "perf_simd"

# Run regular tests
echo -e "\nRunning tests..."
cd $BUILD_DIR && ctest --output-on-failure

# Run performance test separately
echo -e "\n${BLUE}Running SIMD performance test...${NC}"
./tests/perf_simd_runner ./tests/obj/perf_simd.so

cd ..
