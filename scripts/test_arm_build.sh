#!/bin/bash

# ARM Build Validation Script
# This script tests SimpleLang compiled for ARM architecture

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}SimpleLang ARM Cross-Compilation Test${NC}"
echo "========================================"

# Check if ARM build exists
if [ ! -f "./build_mlir_aarch64/src/simplang" ]; then
    echo -e "${RED}Error: ARM build not found!${NC}"
    echo "Please run: ./build.sh --target=aarch64 --mlir"
    exit 1
fi

# Test 1: Check binary architecture
echo -e "\n${YELLOW}Test 1: Checking binary architecture...${NC}"
ARCH_INFO=$(file ./build_mlir_aarch64/src/simplang)
echo "Binary info: $ARCH_INFO"

if echo "$ARCH_INFO" | grep -q "aarch64\|ARM aarch64"; then
    echo -e "${GREEN}✓ Binary is correctly built for ARM aarch64${NC}"
else
    echo -e "${RED}✗ Binary is not ARM aarch64!${NC}"
    exit 1
fi

# Test 2: Compile test program
echo -e "\n${YELLOW}Test 2: Compiling test program for ARM...${NC}"
./build_mlir_aarch64/src/simplang tests/arm_validation.sl --emit-mlir -o /tmp/arm_test.o

if [ -f "/tmp/arm_test.o" ]; then
    echo -e "${GREEN}✓ Test program compiled successfully${NC}"

    # Check object file architecture
    OBJ_INFO=$(file /tmp/arm_test.o)
    echo "Object info: $OBJ_INFO"

    if echo "$OBJ_INFO" | grep -q "aarch64\|ARM aarch64"; then
        echo -e "${GREEN}✓ Object file is correctly built for ARM${NC}"
    else
        echo -e "${RED}✗ Object file is not ARM!${NC}"
    fi
else
    echo -e "${RED}✗ Compilation failed!${NC}"
    exit 1
fi

# Test 3: Link to shared library
echo -e "\n${YELLOW}Test 3: Linking to shared library...${NC}"
aarch64-linux-gnu-gcc -shared -o /tmp/arm_test.so /tmp/arm_test.o -lm

if [ -f "/tmp/arm_test.so" ]; then
    echo -e "${GREEN}✓ Shared library created successfully${NC}"

    # Check library architecture
    LIB_INFO=$(file /tmp/arm_test.so)
    echo "Library info: $LIB_INFO"

    if echo "$LIB_INFO" | grep -q "aarch64\|ARM aarch64"; then
        echo -e "${GREEN}✓ Shared library is correctly built for ARM${NC}"
    else
        echo -e "${RED}✗ Shared library is not ARM!${NC}"
    fi
else
    echo -e "${RED}✗ Linking failed!${NC}"
    exit 1
fi

# Test 4: Try to run with qemu (if available)
echo -e "\n${YELLOW}Test 4: Testing execution with qemu-aarch64...${NC}"
if command -v qemu-aarch64-static >/dev/null 2>&1; then
    echo -e "${BLUE}qemu-aarch64-static is available${NC}"

    # Try to run the compiler itself
    echo "Testing compiler execution..."
    if qemu-aarch64-static ./build_mlir_aarch64/src/simplang --help >/dev/null 2>&1; then
        echo -e "${GREEN}✓ ARM compiler runs successfully under qemu${NC}"
    else
        echo -e "${YELLOW}⚠ Compiler cannot run under qemu (may need ARM libs)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ qemu-aarch64-static not found, skipping execution test${NC}"
fi

# Summary
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}ARM Cross-Compilation Validation Complete!${NC}"
echo ""
echo "Summary of generated files:"
echo "  - Compiler: ./build_mlir_aarch64/src/simplang (aarch64)"
echo "  - Object:   /tmp/arm_test.o (aarch64)"
echo "  - Library:  /tmp/arm_test.so (aarch64)"
echo ""
echo "To deploy on ARM device:"
echo "  1. Copy ./build_mlir_aarch64/src/simplang to ARM device"
echo "  2. Copy test files and run on target hardware"
echo "  3. Or use full QEMU system emulation for testing"

# Optional: Compare with native build
if [ -f "./build_mlir/src/simplang" ]; then
    echo -e "\n${BLUE}Native vs ARM compiler comparison:${NC}"
    echo -n "Native: "
    file ./build_mlir/src/simplang | cut -d: -f2
    echo -n "ARM:    "
    file ./build_mlir_aarch64/src/simplang | cut -d: -f2
fi