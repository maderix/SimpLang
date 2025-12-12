# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
DO NOT POLLUTE THE WORKSPACE WITH UNNECESSARY FILES WHILE DEBUGGING. CREATE NEW FILES ONLY WHEN THERE IS NO OTHER OPTION.USE INLINE CODE WHEREEVER POSSIBLE
SKIP ATTRIBUTION WHILE COMMITTING WITH GIT.
ALWAYS CHECK THE CURRENT DIRECTORY WHEN RUNNING COMMANDS

## Project Overview

SimpLang is a domain-specific language (DSL) designed for SIMD hardware optimization, particularly for deep learning applications. It provides high-level abstractions for SIMD operations, coupled with robust debugging and runtime infrastructure.

## Common Development Commands

### Building the Project
```bash
# Quick build using build script (recommended)
./build.sh

# Manual CMake build with SIMD debugging
cmake -B build -DSIMD_DEBUG=ON -DLLVM_DIR="$(llvm-config-14 --cmakedir || llvm-config --cmakedir)"
cmake --build build --target simplang
```

**CRITICAL: Two Build Directories**
- **./build/** - Standard LLVM backend (basic compilation)
- **./build_mlir/** - MLIR backend (advanced features: rmsnorm, softmax, silu, conv2d, matmul, etc.)

**For transformer/ML workloads, ALWAYS use ./build_mlir/**

### Running Tests
```bash
# Run all tests (compiles kernels and runs test suite)
./run_tests.sh

# Run tests in Docker
./dev.sh test

# Run individual test manually
./build/tests/test_arithmetic_runner ./build/tests/obj/test_arithmetic.so
```

### Development Workflow Commands
```bash
# Docker-based development (recommended)
./dev.sh setup     # One-time setup
./dev.sh build     # Rebuild after changes
./dev.sh test      # Run test suite
./dev.sh debug     # Start interactive debugger
./dev.sh shell     # Open development shell
./dev.sh clean     # Clean build artifacts
./dev.sh rebuild   # Clean and rebuild

# Compile a SimpLang kernel (LLVM backend - basic)
./build/src/simplang my_kernel.sl

# Compile with MLIR backend (for transformers/ML - REQUIRED for rmsnorm, softmax, etc.)
# Step 1: Compile .sl to .o (from project root or specify full path to .sl file)
./build_mlir/src/simplang --emit-mlir examples/my_kernel.sl -o /tmp/my_kernel.o

# Step 2: Link .o to .so shared library
gcc -shared -o /tmp/my_kernel.so /tmp/my_kernel.o -lm

# Example: Complete workflow for stories110M transformer
cd build_mlir
./src/simplang ../examples/llama2/stories110M.sl --emit-mlir -o /tmp/stories110M.o
gcc -shared -o /tmp/stories110M.so /tmp/stories110M.o -lm
cd ..
g++ -o /tmp/generate examples/llama2/generate_stories110M.cpp -ldl -std=c++14
/tmp/generate assets/models/stories110M.bin assets/models/tokenizer.bin /tmp/stories110M.so 0.0

# Run a compiled kernel
./build/tests/test_loop_runner ./build/tests/obj/test_loop.so
```

### Linting and Type Checking
No specific lint/typecheck commands are configured. Tests serve as the primary validation mechanism.

## Code Architecture

### Compiler Pipeline
```
Source Code (.sl) → Lexer (Flex) → Parser (Bison) → AST → CodeGen → LLVM IR → Object Code (.o/.so)
```

**Key Components:**
- **Frontend**: `src/lexer.l` (tokenization), `src/parser.y` (AST generation)
- **Backend**: `src/codegen.cpp` (LLVM IR generation), `src/ast.cpp` (AST node implementations)
- **Main Compiler**: `src/main.cpp` - orchestrates the compilation pipeline

### Runtime Architecture
**Host-Kernel Model**: C++ host applications dynamically load compiled SimpLang kernels (.so files) for execution.

**Core Runtime Components:**
- **KernelRunner** (`runtime/include/kernel_runner.hpp`): Loads and executes compiled kernels
- **SIMD Operations** (`src/simd_ops.cpp`, `src/simd_interface.cpp`): SSE/AVX vector operations
- **Memory Management**: Aligned allocation for SIMD operations

### Debugging Infrastructure
**Interactive Debugger** (`runtime/src/kernel_debugger/`):
- **KernelDebugger**: Core debugging engine with breakpoints and stepping
- **MemoryTracker**: Tracks allocations, detects leaks and out-of-bounds access
- **CallStack**: Function call tracking with local variable inspection
- **CommandProcessor + UIHelper**: GDB-like command-line interface using readline
- **SourceManager**: Maps execution to source code locations

### SIMD Support
- **SSE (128-bit)** and **AVX (256-bit)** vector operations
- **Slice Types**: `SSESlice` (2 doubles), `AVXSlice` (8 doubles)
- **Vector Operations**: Addition, subtraction, multiplication, division with intrinsics
- **Alignment**: Automatic memory alignment for optimal SIMD performance

## Development Setup

### Prerequisites
- LLVM 14 or later
- CMake 3.20+
- C++17 compatible compiler
- Boost libraries
- readline library

### Docker Development (Recommended)
The project includes comprehensive Docker support with live file mounting for efficient development.

### VS Code Dev Container
Full IDE support available via `.devcontainer/devcontainer.json` with pre-configured extensions and settings.

## File Organization

### Source Structure
- `src/`: Compiler implementation (lexer, parser, codegen)
- `include/`: Public headers for AST, codegen, SIMD interfaces
- `runtime/`: Runtime library and debugging infrastructure
- `tests/`: SimpLang test kernels (.sl files) and host runners

### Important Files
- `CMakeLists.txt`: Main build configuration with SIMD_DEBUG and ENABLE_DEBUGGER options
- `build.sh`: Quick build script with LLVM detection
- `run_tests.sh`: Comprehensive test runner
- `dev.sh`: Docker development helper script

## SimpLang Language Features

### Basic Syntax
```simplang
fn kernel_main() {
    var x = 10.0;
    var y = 5.0;
    return x + y;
}
```

### SIMD Operations
```simplang
fn kernel_main(var out_sse SSESlice, var out_avx AVXSlice) {
    var sse1 = make(SSESlice, 1);
    sse1[0i] = sse(1.0, 2.0);  // Create SSE vector
    
    var avx1 = make(AVXSlice, 1);
    avx1[0i] = avx(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);  // Create AVX vector
    
    slice_set_sse(out_sse, 0i, slice_get_sse(sse1, 0i) + slice_get_sse(sse1, 0i));
    return 1.0;
}
```

## Testing Strategy

### Test Types
- **Unit Tests**: Individual .sl kernels testing specific language features
- **SIMD Tests**: Vector operation validation
- **Performance Tests**: Benchmarking SIMD vs scalar operations
- **Debug Tests**: Debugger functionality validation

### Running Specific Tests
```bash
# Compile and run individual test
./build/src/simplang tests/test_arithmetic.sl
./build/tests/test_arithmetic_runner ./build/tests/obj/test_arithmetic.so

# Performance testing
./tests/perf_simd_runner ./tests/obj/perf_simd.so
```

## Performance Characteristics
- **JIT Compilation**: < 1ms overhead
- **SIMD Performance**: ~1.3x slower than optimized native C++ (acceptable for abstraction benefits)
- **Memory Overhead**: ~2MB per kernel instance
- **Debug Mode**: ~5% performance impact when enabled

## Common Development Patterns

### Adding New Language Features
1. Update `src/lexer.l` for new tokens
2. Modify `src/parser.y` for grammar rules
3. Add AST nodes in `include/ast.hpp` and `src/ast.cpp`
4. Implement codegen in `src/codegen.cpp`
5. Add test cases in `tests/`

### SIMD Development
- Use aligned memory allocation for vector data
- Test both SSE and AVX paths
- Enable SIMD_DEBUG for detailed vector operation insights
- Verify alignment requirements are met

### Debugging Workflow
- Use `./dev.sh debug` for interactive debugging
- Set breakpoints with `break <line>`
- Step through code with `step`, `next`, `continue`
- Inspect memory with integrated MemoryTracker
- Check SIMD registers with `printVectorState()`
- NOOO STOP saying simpler test, do it only if it's impossible to proceed with the harder tests. sue simpler test only for debugging
- we don't commit checkpoint files, ever!

## INT8/INT4 VNNI Optimization Commands (CRITICAL!)

### Compiling INT8 Benchmarks with VNNI Pass
```bash
# ALWAYS use BOTH flags together for INT8/INT4 workloads:
cd build_mlir
./src/simplang ../simptensor/benchmarks/bench_int8_matmul.sl --emit-mlir --llvm-vectorize -o /tmp/bench_int8.o

# Link to shared library
gcc -shared -o /tmp/bench_int8.so /tmp/bench_int8.o -lm

# Compile runner (Eigen is in thirdparty/)
g++ -O3 -march=native -o /tmp/bench_runner simptensor/benchmarks/bench_int8_matmul_runner.cpp -ldl -std=c++17 -I thirdparty/eigen

# Run benchmark
/tmp/bench_runner /tmp/bench_int8.so
```

### Key Flags
- `--emit-mlir` - Use MLIR backend (REQUIRED for tensor ops)
- `--llvm-vectorize` - Enable LLVM vectorization + VNNIPass (for INT8/INT4)
- Both flags MUST be used together for INT8 matmul optimization

### Third-party Libraries
- **Eigen**: Located at `thirdparty/eigen` - use `-I thirdparty/eigen` for includes

## GPU Backend (CUDA/cuBLAS) Commands

### Build Directories
- **./build_gpu/** - GPU-enabled MLIR backend (for CUDA/cuBLAS operations)
- Uses `simplang-cuda:dev` Docker image for compilation and execution
- Source mounted at `/app`, LLVM/MLIR at `/usr/lib/llvm-14`

### Rebuilding GPU Runtime (after modifying gpu_runtime.cpp)
```bash
# Clean CMake cache and rebuild with correct paths
docker run --rm --gpus all -v /home/maderix/simple-lang:/app simplang-cuda:dev bash -c \
  "rm -rf /app/build_gpu/CMakeCache.txt /app/build_gpu/CMakeFiles && \
   cd /app/build_gpu && \
   cmake .. -DUSE_CUDA=ON -DUSE_MLIR=ON \
     -DMLIR_DIR=/usr/lib/llvm-14/lib/cmake/mlir \
     -DLLVM_DIR=/usr/lib/llvm-14/lib/cmake/llvm && \
   make simplang_runtime -j4"
```

### Compiling GPU Kernels (MUST use Docker)
```bash
# Step 1: Compile .sl to .o
docker run --rm --gpus all -v /home/maderix/simple-lang:/app simplang-cuda:dev \
  /app/build_gpu/src/simplang /app/tests/my_gpu_kernel.sl --emit-mlir -o /app/tmp_docker/my_kernel.o

# Step 2: Link to .so with CUDA libraries
docker run --rm --gpus all -v /home/maderix/simple-lang:/app simplang-cuda:dev \
  gcc -shared -o /app/tmp_docker/my_kernel.so /app/tmp_docker/my_kernel.o \
  -L/app/build_gpu/runtime -lsimplang_runtime -lm \
  -L/usr/local/cuda/lib64 -lcublas -lcudart

# Step 3: Compile C++ runner
docker run --rm --gpus all -v /home/maderix/simple-lang:/app simplang-cuda:dev \
  g++ -O3 -o /app/tmp_docker/my_runner /app/tests/my_runner.cpp -ldl -std=c++17

# Step 4: Run with proper library paths
docker run --rm --gpus all -v /home/maderix/simple-lang:/app \
  -e LD_LIBRARY_PATH=/app/build_gpu/runtime:/usr/local/cuda/lib64 \
  simplang-cuda:dev /app/tmp_docker/my_runner /app/tmp_docker/my_kernel.so
```

### GPU Operations
- `tensor_matmul(A, B)` - Uses cuBLAS SGEMM for f32 matrix multiplication
- `tensor_from_array(array, offset)` - Create tensor from host array (avoids fill overhead)
- Persistent weight caching: Weights (>64KB) stay on GPU between calls
- Pinned memory + async streams for faster H2D/D2H transfers

### GPU Performance (RTX 4090)
- **Peak cuBLAS SGEMM**: ~60 TFLOPS (72% of theoretical 83 TFLOPS)
- **stories110M inference**: ~240-300 tok/s
- **Optimal matrix sizes**: 4096x4096 and larger for best GPU utilization

### Example: GPU MatMul Benchmark
```bash
# Complete workflow for GPU f32 matmul benchmark
docker run --rm --gpus all -v /home/maderix/simple-lang:/app simplang-cuda:dev bash -c \
  "/app/build_gpu/src/simplang /app/tests/bench_gpu_matmul_only.sl --emit-mlir -o /app/tmp_docker/bench_gpu.o && \
   gcc -shared -o /app/tmp_docker/bench_gpu.so /app/tmp_docker/bench_gpu.o \
     -L/app/build_gpu/runtime -lsimplang_runtime -lm \
     -L/usr/local/cuda/lib64 -lcublas -lcudart && \
   g++ -O3 -o /app/tmp_docker/bench_gpu_runner /app/tests/bench_gpu_matmul_only_runner.cpp -ldl -std=c++17"

docker run --rm --gpus all -v /home/maderix/simple-lang:/app \
  -e LD_LIBRARY_PATH=/app/build_gpu/runtime:/usr/local/cuda/lib64 \
  simplang-cuda:dev /app/tmp_docker/bench_gpu_runner /app/tmp_docker/bench_gpu.so
```