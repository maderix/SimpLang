#!/bin/bash
set -e

# Determine LLVM cmake directory using llvm-config (prefers versioned binary)
LLVM_DIR="$(llvm-config-14 --cmakedir 2>/dev/null || llvm-config --cmakedir)"

mkdir -p build
cd build
cmake .. -DLLVM_DIR="${LLVM_DIR}"
make
