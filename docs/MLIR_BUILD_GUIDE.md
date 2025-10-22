# MLIR Build Guide for SimpLang

This guide provides step-by-step instructions for building LLVM/MLIR from source and integrating it with SimpLang. **Custom LLVM/MLIR build is REQUIRED** - system packages will NOT work due to ABI incompatibilities.

## Why Custom Build is Required

**Problem**: System LLVM packages (e.g., `llvm-14` from apt) are built with different ABI settings than MLIR requires:
- Missing `llvm::EnableABIBreakingChecks` symbol
- Different RTTI (Run-Time Type Information) settings
- Incompatible `mlir::Dialect` typeinfo

**Solution**: Build LLVM/MLIR from source with consistent settings.

---

## Option 1: Git Submodule Setup (Recommended for New Clones)

### Step 1: Add LLVM as Submodule

```bash
# In your simple-lang repository root
cd /path/to/simple-lang

# Add LLVM project as a submodule
git submodule add https://github.com/llvm/llvm-project.git external/llvm-project

# Initialize and update submodule
git submodule update --init --recursive

# Commit the submodule
git add .gitmodules external/llvm-project
git commit -m "Add LLVM/MLIR as git submodule"
```

### Step 2: Checkout Specific LLVM Version

```bash
cd external/llvm-project

# Checkout LLVM 14.0.0 (matches system LLVM for compatibility)
git checkout llvmorg-14.0.0

cd ../..
```

### Step 3: Build LLVM/MLIR

```bash
# Create build directory
mkdir -p external/llvm-project/build
cd external/llvm-project/build

# Configure LLVM/MLIR with CMake
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir" \
   -DLLVM_TARGETS_TO_BUILD="Native" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=OFF \
   -DLLVM_ENABLE_EH=OFF \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_INSTALL_UTILS=ON \
   -DCMAKE_C_COMPILER=clang \
   -DCMAKE_CXX_COMPILER=clang++ \
   -DLLVM_ENABLE_LLD=ON

# Build (use -j to parallelize, e.g., -j8 for 8 cores)
ninja -j$(nproc)

# This will take 30-60 minutes depending on your machine
```

### Step 4: Verify LLVM/MLIR Build

```bash
# Check that key libraries exist
ls -lh lib/libMLIRIR.a
ls -lh lib/libLLVMSupport.a
ls -lh bin/mlir-tblgen

# Check for the critical ABI symbol
nm lib/libLLVMSupport.a | grep EnableABIBreakingChecks
# Should output: 0000000000000000 B llvm::EnableABIBreakingChecks
```

---

## Option 2: Manual LLVM Build (Existing Setup)

If you already have LLVM/MLIR built elsewhere:

### Step 1: Verify Your LLVM Build Location

```bash
# Check your existing LLVM build
ls $HOME/llvm-project/build/lib/libMLIRIR.a

# Verify the ABI symbol exists
nm $HOME/llvm-project/build/lib/libLLVMSupport.a | grep EnableABIBreakingChecks
```

### Step 2: Update SimpLang CMakeLists.txt

Your `CMakeLists.txt` should already have:

```cmake
# Prefer custom LLVM build if MLIR is enabled
if(USE_MLIR)
    set(LLVM_DIR "$ENV{HOME}/llvm-project/build/lib/cmake/llvm" CACHE PATH "Path to LLVM cmake config")
endif()
find_package(LLVM REQUIRED CONFIG)

# ... later ...

if(USE_MLIR)
    set(MLIR_DIR "$ENV{HOME}/llvm-project/build/lib/cmake/mlir" CACHE PATH "Path to MLIR cmake config")
    find_package(MLIR REQUIRED CONFIG)

    # CRITICAL: Include LLVM build options
    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
    include(TableGen)
    include(AddLLVM)
    include(AddMLIR)
    include(HandleLLVMOptions)  # Sets -fno-rtti, -fno-exceptions
endif()
```

---

## Building SimpLang with MLIR

### Step 1: Configure SimpLang Build

```bash
# In simple-lang root directory
mkdir -p build_mlir
cd build_mlir

# Configure with MLIR enabled
cmake -DUSE_MLIR=ON ..

# Verify it found the correct LLVM/MLIR:
# Look for lines like:
#   "Using LLVMConfig.cmake in: /path/to/llvm-project/build/lib/cmake/llvm"
#   "Using MLIRConfig.cmake in: /path/to/llvm-project/build/lib/cmake/mlir"
```

### Step 2: Build SimpLang

```bash
# Build everything
cmake --build . -j$(nproc)

# Or build specific targets:
cmake --build . --target simp_dialect        # MLIR dialect library
cmake --build . --target test_simp_dialect   # Dialect tests
cmake --build . --target simplang            # Main compiler
```

### Step 3: Run Tests

```bash
# Run MLIR dialect tests
./tests/mlir/test_simp_dialect

# Expected output: All 15 tests should pass
# ✅ Test 1: SimpDialect loaded successfully!
# ...
# 🎉 All 15 tests passed!
```

---

## Troubleshooting

### Error: `undefined reference to llvm::EnableABIBreakingChecks`

**Cause**: Linking against system LLVM instead of custom build

**Fix**:
1. Verify `LLVM_DIR` is set correctly in CMake configuration
2. Check CMake output shows custom LLVM path, not `/usr/lib/llvm-14`
3. Clean and reconfigure: `rm -rf build_mlir && cmake -B build_mlir -DUSE_MLIR=ON`

### Error: `undefined reference to typeinfo for mlir::Dialect`

**Cause**: Missing `include(HandleLLVMOptions)` in CMakeLists.txt

**Fix**: Add to CMakeLists.txt:
```cmake
include(HandleLLVMOptions)
```

### Error: `Could not find include file 'SimpBase.td'`

**Cause**: TableGen include paths not set

**Fix**: Ensure all `mlir_tablegen()` calls in `src/mlir/CMakeLists.txt` have:
```cmake
mlir_tablegen(SimpOps.h.inc -gen-op-decls
    -I ${CMAKE_SOURCE_DIR}/include/mlir/Dialects/Simp)
```

### LLVM Build Fails with Disk Space Error

**Cause**: LLVM build requires ~30GB of disk space

**Fix**:
- Use `Release` build type instead of `Debug` to reduce size
- Build only MLIR: `-DLLVM_ENABLE_PROJECTS="mlir"` (not all projects)
- Clean intermediate files: `ninja clean` after successful build

---

## Updating LLVM/MLIR Version

### For Submodule Setup:

```bash
cd external/llvm-project

# Update to latest release
git fetch --tags
git checkout llvmorg-15.0.0  # or desired version

# Rebuild
cd build
ninja clean
cmake ..  # Same flags as before
ninja -j$(nproc)
```

### For Manual Setup:

```bash
cd $HOME/llvm-project

# Pull latest changes
git pull origin main
git checkout llvmorg-15.0.0

# Rebuild
cd build
ninja clean
cmake ..
ninja -j$(nproc)
```

---

## Build Configuration Reference

### Minimal MLIR Build (Faster)

For development iterations, use a minimal build:

```bash
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir" \
   -DLLVM_TARGETS_TO_BUILD="Native" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=OFF \
   -DLLVM_ENABLE_RTTI=OFF \
   -DLLVM_BUILD_EXAMPLES=OFF \
   -DLLVM_BUILD_TESTS=OFF \
   -DLLVM_INCLUDE_TESTS=OFF
```

Build time: ~15-20 minutes

### Full MLIR Build (Recommended)

For complete functionality:

```bash
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir;clang" \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=OFF \
   -DLLVM_BUILD_EXAMPLES=ON
```

Build time: ~45-60 minutes

---

## Verification Checklist

After building LLVM/MLIR and SimpLang, verify:

- [ ] `libMLIRIR.a` exists in LLVM build lib directory
- [ ] `libLLVMSupport.a` contains `EnableABIBreakingChecks` symbol
- [ ] `mlir-tblgen` executable exists in LLVM build bin directory
- [ ] CMake finds custom LLVM, not system LLVM
- [ ] `test_simp_dialect` runs and passes all 15 tests
- [ ] No undefined reference errors during linking

---

## Quick Reference Commands

```bash
# Full build from scratch (submodule approach):
git submodule add https://github.com/llvm/llvm-project.git external/llvm-project
cd external/llvm-project && git checkout llvmorg-14.0.0 && cd ../..
mkdir -p external/llvm-project/build && cd external/llvm-project/build
cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_TARGETS_TO_BUILD="Native" \
    -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_RTTI=OFF
ninja -j$(nproc)
cd ../../..

# Build SimpLang:
cmake -B build_mlir -DUSE_MLIR=ON
cmake --build build_mlir --target test_simp_dialect
./build_mlir/tests/mlir/test_simp_dialect
```

---

## Additional Resources

- MLIR Getting Started: https://mlir.llvm.org/getting_started/
- LLVM CMake Guide: https://llvm.org/docs/CMake.html
- MLIR Toy Tutorial: https://mlir.llvm.org/docs/Tutorials/Toy/
- Standalone Dialect Example: `llvm-project/mlir/examples/standalone/`
