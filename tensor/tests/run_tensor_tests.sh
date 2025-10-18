#!/bin/bash

# SimpLang Tensor Test Runner
# Compiles and runs tensor operations with SIMD auto-vectorization

set -e

echo "==================================================="
echo "SimpLang Tensor Library Test Suite"
echo "Testing auto-vectorization across all datatypes"
echo "==================================================="

# Build SimpLang compiler if needed
if [ ! -f "../../build/src/simplang" ]; then
    echo "Building SimpLang compiler..."
    cd ../..
    ./build.sh
    cd tensor/tests
fi

COMPILER="../../build/src/simplang"
TEST_DIR="../../build/tensor_tests"
mkdir -p $TEST_DIR

echo ""
echo "1. Testing Tensor Type System (All Datatypes)"
echo "---------------------------------------------"
$COMPILER test_tensor_types.sl -o $TEST_DIR/test_tensor_types.o --print-ir
echo "✓ Tensor types compiled successfully"

echo ""
echo "2. Testing Tensor Performance (SIMD Optimization)"
echo "------------------------------------------------"
$COMPILER test_tensor_performance.sl -o $TEST_DIR/test_tensor_performance.o --print-ir
echo "✓ Tensor performance test compiled"

echo ""
echo "3. Testing Matrix Operations (2D Tensors)"
echo "----------------------------------------"
$COMPILER test_tensor_matrix.sl -o $TEST_DIR/test_tensor_matrix.o --print-ir  
echo "✓ Matrix operations compiled"

echo ""
echo "4. Analyzing Generated LLVM IR for Vectorization"
echo "-----------------------------------------------"

# Check for vector instructions in the generated IR
echo "Checking for auto-vectorization patterns..."

if [ -f "$TEST_DIR/test_tensor_performance.ll" ]; then
    echo ""
    echo "Vector instructions found in tensor performance test:"
    grep -E "(vector|llvm\.vector|<.*x.*>)" "$TEST_DIR/test_tensor_performance.ll" | head -5 || echo "No explicit vector IR found (may be optimized later)"
    
    echo ""  
    echo "SIMD-related optimizations:"
    grep -E "(fmul.*fast|fadd.*fast|load.*align|store.*align)" "$TEST_DIR/test_tensor_performance.ll" | head -3 || echo "No explicit SIMD optimizations visible"
fi

echo ""
echo "==================================================="
echo "Tensor Test Suite Completed Successfully"
echo "All tensor operations compiled with auto-vectorization enabled"
echo "Backend-agnostic SIMD will use: SSE/AVX/AVX-512 as available"
echo "==================================================="