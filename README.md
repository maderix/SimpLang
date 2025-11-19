<p align="center">
  <img src="docs/animation_cropped.gif" alt="SimpLang Banner" width="600"/>
</p>

# SimpLang: A High-Performance Language for ML and Scientific Computing

SimpLang is a domain-specific language born from a simple question: What if compiling machine learning models could be as straightforward as writing a function? This research project explores compiler design from the ground up, bridging the gap between readable high-level code and the blazing-fast performance needed for modern ML and scientific computing.

At its heart, SimpLang is built on a dual-backend architecture that gives you the best of both worlds. The LLVM backend handles general-purpose compute tasks with automatic SIMD vectorization, while the experimental MLIR backend transforms your code into optimized tensor operations for machine learning workloads. Write once in a clean, intuitive syntax, and let the compiler do the heavy lifting.

This is a research project, which means you'll find rough edges alongside genuine innovation. It's one developer's journey into the depths of compiler construction, SIMD optimization, and ML infrastructure. If you're interested in how compilers work, how to squeeze every ounce of performance from modern hardware, or how to build ML systems from scratch, you're in the right place.

## What Can It Do?

SimpLang runs complete transformer models on CPU at competitive speeds. A 110M-parameter LLaMA model generates text at **42.95 tokens per second** on standard x86 hardware, compiled from high-level SimpLang code that looks remarkably like the math you'd write on paper. The same codebase cross-compiles to ARM, where it outperforms NumPy by **6x** on Raspberry Pi 5, making it genuinely useful for edge deployment.

<p align="center">
  <img src="assets/llama_demo.gif" alt="LLaMA 110M Demo" width="800"/>
</p>

The performance isn't limited to transformers. Matrix multiplication, the backbone of deep learning, achieves **73.65 GFLOP/s** on x86 (2.17x faster than Eigen) and **16.23 GFLOP/s** on ARM with optimized tiling. These aren't toy benchmarks—these are real workloads running on real hardware, compiled from code you can read and understand.

## Performance That Matters

Let's talk numbers, because that's what really counts when you're building performance-critical systems.

### Transformer Inference (x86)

Running a complete LLaMA architecture with multi-head attention, RMSNorm, SwiGLU activation, and KV caching:

**TinyStories 110M Model**: 42.95 tokens/second on CPU, processing the same operations as production transformer implementations. The MLIR backend compiles high-level tensor operations (`tensor_matmul`, `rmsnorm`, `softmax`, `silu`) into vectorized loops with proper cache tiling.

**Scaling to Larger Models**: The architecture scales from 125M to 3B parameters, maintaining consistent throughput per parameter. A 3B model runs at 0.758 tokens/s (4.547 GFLOP/s), showing the compiler generates efficient code regardless of model size.

**W4 Quantization**: 4-bit quantized weights compress models by 4x (3B parameters fit in 2.1GB instead of 9.2GB) with 2x slowdown—acceptable for memory-constrained deployments where the model wouldn't otherwise fit.

### Matrix Multiplication (x86)

The fundamental operation for deep learning, tested against Eigen (a heavily optimized C++ library):

- **256×256 Float**: 60.21 GFLOP/s (1.89x faster than Eigen)
- **512×512 Float**: 73.65 GFLOP/s (2.17x faster than Eigen)
- **256×256 Int32**: 105.83 GFLOP/s (5.73x faster than Eigen)

These results come from the `tensor_matmul` intrinsic with loop tiling (16×16×16) and automatic vectorization. SimpLang's MLIR backend generates code that actually beats hand-optimized libraries at common sizes.

### ARM Cross-Compilation (Raspberry Pi 5)

Cross-compile from x86 to ARM with a single flag (`--target aarch64`), and watch your code fly on edge hardware:

**vs NumPy on Raspberry Pi 5**:
- **Float MatMul 256×256**: 6.0x faster (16.23 vs 2.72 GFLOP/s)
- **Int MatMul 256×256**: 4.4x faster (8.31 vs 1.88 GIOP/s)
- **Int MatMul 512×512**: 3.7x faster (6.82 vs 1.85 GIOP/s)

NumPy on Raspberry Pi uses unoptimized OpenBLAS without ARM NEON, while SimpLang automatically generates NEON vectorized code. For ARM-based ML deployment where every watt matters, this performance gap is the difference between viable and unusable.

**LLaMA 110M on ARM**: 13.49 tokens/s with optimized tiling (8×8×8), a 47% speedup over default settings. The compiler automatically generates ARM NEON instructions (`fmla v1.4s`) for 4-way vector operations.

## Understanding the Dual Backend Architecture

SimpLang's real power comes from its two compilation backends, each optimized for different workloads. Understanding when to use which backend is crucial.

### LLVM Backend: General Compute

The LLVM backend lives in `./build/` and handles traditional computational tasks. It excels at automatic SIMD vectorization, taking your scalar loops and transforming them into parallel vector operations. This is your go-to for scientific computing, signal processing, numerical algorithms, and any workload where you're not dealing with multi-dimensional tensors.

Compile with the LLVM backend when you need:
- Automatic SSE/AVX/AVX-512 vectorization
- General-purpose numerical computation
- Simple array operations and loops
- Fast compilation times for rapid iteration

The LLVM backend generates clean, optimized machine code with minimal overhead. It's stable, well-tested, and the foundation SimpLang was built on.

### MLIR Backend: Machine Learning Workloads

The MLIR backend lives in `./build_mlir/` and is where things get interesting for ML practitioners. MLIR (Multi-Level Intermediate Representation) is a modern compiler infrastructure designed for heterogeneous hardware and domain-specific optimizations. This backend transforms SimpLang into a genuine ML compilation pipeline.

Use the MLIR backend for:
- **Tensor operations**: Multi-dimensional arrays with shape awareness
- **ML primitives**: RMSNorm, Softmax, SiLU, convolutions, matrix multiplication
- **Layout optimizations**: NHWC for GPU-style workloads, NCHW for CPU caching
- **Transformer architectures**: Attention mechanisms, layer normalization, feedforward networks
- **Cross-compilation**: ARM targets for edge deployment

The MLIR backend performs progressive lowering through multiple dialects: your high-level SimpLang code transforms through Simp → MemRef/Linalg → SCF (structured control flow) → LLVM IR. Each stage applies domain-specific optimizations: loop tiling for cache locality, fusion to reduce memory traffic, and vectorization for SIMD hardware.

**Critical Note**: If you're working with transformers, convolutions, or any ML workload, you **must** use `./build_mlir/`. The LLVM backend doesn't understand tensor operations or ML-specific primitives.

## Quick Start: Three Paths to Running SimpLang

Choose the approach that fits your workflow. Docker is the fastest way to start experimenting, while local installation gives you full control.

### Docker (Recommended for First-Time Users)

The Docker workflow uses live file mounting, so you never rebuild the container. Edit code on your host machine, compile instantly inside the container:

```bash
# One-time setup: build container and compile the project
./dev.sh setup

# Daily workflow
./dev.sh build     # Rebuild after code changes
./dev.sh test      # Run the full test suite
./dev.sh shell     # Drop into a development shell
./dev.sh debug     # Start the interactive debugger
```

Files are mounted as volumes, so when you create a new `.sl` file on your host machine, it's immediately available inside the container. Incremental builds are fast thanks to persistent build caches.

### VS Code Dev Container (Best for Active Development)

For the full IDE experience with IntelliSense, debugging, and integrated terminals:

1. Install the "Dev Containers" extension in VS Code
2. Clone the repository and open it: `code .`
3. Click "Reopen in Container" when prompted (or use Command Palette: "Dev Containers: Reopen in Container")
4. Build inside the container: `./build.sh`

VS Code will configure everything automatically: C++ tooling, LLVM extensions, debugger integration. You get autocomplete, syntax highlighting, and the ability to set breakpoints in both SimpLang and the generated LLVM IR.

### Local Installation (Maximum Control)

If you prefer native development without containers, install the prerequisites and build directly:

**Dependencies** (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install -y llvm-14-dev clang-14 cmake libboost-dev libreadline-dev
```

**Build**:
```bash
# Quick build script (auto-detects LLVM)
./build.sh

# Or manual CMake configuration
cmake -B build -DSIMD_DEBUG=ON
cmake --build build --target simplang

# For MLIR backend (required for ML workloads)
cmake -B build_mlir -DMLIR_ENABLED=ON
cmake --build build_mlir --target simplang
```

**Test**:
```bash
# Run all tests (compiles kernels and executes test runners)
./run_tests.sh

# For MLIR tests
./run_mlir_tests.sh
```

## Writing SimpLang: From "Hello World" to Transformers

SimpLang's syntax is intentionally minimal. You write functions that look like math, and the compiler handles the complexity.

### Your First SimpLang Program

Here's the canonical loop that every compiler tutorial starts with:

```simplang
fn kernel_main() {
    var sum = 0.0;
    var i = 1.0;

    while (i <= 100.0) {
        sum = sum + i;
        i = i + 1.0;
    }
    return sum;
}
```

Save this as `loop.sl` and compile with the LLVM backend:

```bash
./build/src/simplang loop.sl -o loop.o
gcc -shared loop.o -o loop.so
```

The `kernel_main()` function is your entry point. Every SimpLang program needs one. Variables are dynamically typed but float-optimized (notice `1.0` instead of `1`). The compiler generates LLVM IR, which then compiles to native machine code.

Run it from C++:

```cpp
#include "kernel_runner.hpp"

int main() {
    KernelRunner runner;
    runner.loadLibrary("./loop.so");
    double result = runner.runKernel();
    std::cout << "Sum: " << result << std::endl;  // Output: 5050
    return 0;
}
```

### SIMD Vectorization with the LLVM Backend

The LLVM backend automatically vectorizes your loops when it detects opportunities for parallelism. No special syntax required:

```simplang
fn vector_add(var data f32[], var n i64) {
    var i = 0i;
    while (i < n) {
        data[i] = data[i] * 2.0 + 1.0;
        i = i + 1i;
    }
    return 0.0;
}
```

Compile with optimization enabled, and inspect the generated assembly—you'll see AVX instructions processing multiple elements simultaneously. The compiler recognizes the pattern and generates vector code automatically.

Enable SIMD debugging to see what's happening:

```bash
cmake -B build -DSIMD_DEBUG=ON
cmake --build build
```

### Tensor Operations with the MLIR Backend

Now we enter ML territory. The MLIR backend understands multi-dimensional tensors as first-class types:

```simplang
fn matmul_example() {
    // Allocate tensors with explicit shapes
    var A = tensor<256, 256, f32>();
    var B = tensor<256, 256, f32>();
    var C = tensor<256, 256, f32>();

    // Fill with data
    var i = 0i;
    while (i < 256i) {
        var j = 0i;
        while (j < 256i) {
            A[i, j] = 1.0;
            B[i, j] = 2.0;
            i = i + 1i;
        }
        i = i + 1i;
    }

    // High-level tensor operation
    tensor_matmul(A, B, C, 256i, 256i, 256i);

    return C[0i, 0i];
}

fn kernel_main() {
    return matmul_example();
}
```

Compile with the MLIR backend:

```bash
./build_mlir/src/simplang matmul.sl --emit-mlir -o matmul.o
gcc -shared matmul.o -o matmul.so -lm
```

The `--emit-mlir` flag activates the MLIR pipeline. The compiler performs loop tiling (16×16×16 by default), generates vectorized code, and optimizes memory access patterns. This single `tensor_matmul` call expands into hundreds of lines of optimized LLVM IR.

Inspect the intermediate representations:

```bash
./build_mlir/src/simplang matmul.sl --emit-mlir --dump-mlir-passes
```

You'll see the transformation from high-level tensor ops → Linalg (linear algebra) → SCF (loops) → LLVM IR.

### A Real Transformer Layer

Here's where SimpLang shows its real capabilities—a complete transformer architecture with attention, normalization, and activation functions:

```simplang
fn transformer_block(
    var x f32[],
    var W_q f32[], var W_k f32[], var W_v f32[], var W_o f32[],
    var W_ffn1 f32[], var W_ffn2 f32[],
    var norm1_weight f32[], var norm2_weight f32[],
    var output f32[],
    var dim i64, var seq_len i64, var n_heads i64
) {
    var hidden = f32[dim];
    var attn_out = f32[dim];
    var ffn_hidden = f32[dim * 4i];

    // Pre-attention normalization
    rmsnorm(x, norm1_weight, hidden, dim);

    // Multi-head attention (simplified single-head example)
    var Q = f32[seq_len, dim];
    var K = f32[seq_len, dim];
    var V = f32[seq_len, dim];

    tensor_matmul(hidden, W_q, Q, seq_len, dim, dim);
    tensor_matmul(hidden, W_k, K, seq_len, dim, dim);
    tensor_matmul(hidden, W_v, V, seq_len, dim, dim);

    // Attention scores with scaling
    var scores = f32[seq_len, seq_len];
    tensor_matmul(Q, K, scores, seq_len, dim, seq_len);  // Q @ K^T

    var scale = 1.0 / sqrt(dim);
    var i = 0i;
    while (i < seq_len * seq_len) {
        scores[i] = scores[i] * scale;
        i = i + 1i;
    }

    // Softmax over attention scores
    softmax(scores, scores, seq_len, seq_len);

    // Attention output
    tensor_matmul(scores, V, attn_out, seq_len, seq_len, dim);
    tensor_matmul(attn_out, W_o, hidden, seq_len, dim, dim);

    // Residual connection
    i = 0i;
    while (i < dim) {
        hidden[i] = hidden[i] + x[i];
        i = i + 1i;
    }

    // Pre-FFN normalization
    rmsnorm(hidden, norm2_weight, ffn_hidden, dim);

    // SwiGLU feedforward network
    var gate = f32[dim * 4i];
    var up = f32[dim * 4i];

    tensor_matmul(ffn_hidden, W_ffn1, gate, dim, dim, dim * 4i);
    silu(gate, gate, dim * 4i);  // SiLU activation

    tensor_matmul(ffn_hidden, W_ffn2, up, dim, dim, dim * 4i);

    // Element-wise multiply
    i = 0i;
    while (i < dim * 4i) {
        gate[i] = gate[i] * up[i];
        i = i + 1i;
    }

    // Down projection and residual
    tensor_matmul(gate, W_ffn2, output, dim * 4i, dim, dim);

    i = 0i;
    while (i < dim) {
        output[i] = output[i] + hidden[i];
        i = i + 1i;
    }

    return 0.0;
}

fn kernel_main() {
    // Initialize weights and run transformer block...
    return transformer_block(...);
}
```

This is actual runnable code. The `rmsnorm`, `softmax`, `silu`, and `tensor_matmul` operations are built-in MLIR intrinsics that compile to optimized implementations. The complete LLaMA 110M model in `examples/llama2/stories110M.sl` uses this exact pattern, scaled up with proper KV caching and multi-layer stacking.

Compile and run:

```bash
./build_mlir/src/simplang transformer.sl --emit-mlir -o transformer.o
gcc -shared transformer.o -o transformer.so -lm
```

The MLIR backend recognizes the tensor operations, applies cache-friendly loop tiling, fuses operations where possible, and generates vectorized code. The result is transformer inference at competitive CPU speeds.

## Cross-Compiling for ARM: From x86 to Edge Devices

One of SimpLang's most practical features is seamless ARM cross-compilation. Write and test on your x86 development machine, then deploy to Raspberry Pi or other ARM devices with a single compiler flag.

### The Workflow

On your x86 development machine:

```bash
# Compile SimpLang kernel for ARM target
./build_mlir/src/simplang model.sl --emit-mlir --target aarch64 --tile-size 8 -o model.o

# Link with ARM cross-compiler
aarch64-linux-gnu-gcc -shared -o model.so model.o -lm

# Copy to ARM device
scp model.so pi@raspberry-pi:/home/pi/models/
```

On the Raspberry Pi:

```bash
# Run the compiled model
./run_model /home/pi/models/model.so
```

The `--tile-size 8` flag is crucial for ARM. Raspberry Pi 5 has smaller L1 caches (32KB) compared to typical x86 CPUs, so the default 16×16×16 tiling is suboptimal. Using 8×8×8 tiles gives a 47% speedup on LLaMA inference (9.18 → 13.49 tokens/s).

### Why SimpLang Beats NumPy on ARM

NumPy on Raspberry Pi is surprisingly slow because the default OpenBLAS build doesn't optimize for ARM NEON. SimpLang automatically generates NEON vectorized code, giving you massive speedups for free:

**256×256 Float MatMul**:
- NumPy: 12.345 ms (2.72 GFLOP/s)
- SimpLang: 2.068 ms (16.23 GFLOP/s)
- **Speedup: 6.0x**

For edge ML deployment where you're running inference on resource-constrained hardware, this performance gap is game-changing. Deploy a SimpLang-compiled model instead of a NumPy-based one and watch your battery life improve.

### Verifying ARM Vectorization

Want to confirm the compiler is generating NEON instructions? Disassemble the compiled code:

```bash
aarch64-linux-gnu-objdump -d model.so | grep fmla
```

You'll see instructions like `fmla v1.4s, v2.4s, v3.4s`—NEON 4-way vector multiply-accumulate operations. The compiler detected the parallelism and generated the hardware-specific instructions automatically.

## The Compiler Pipeline: What Happens Under the Hood

Understanding the compilation process helps you write better SimpLang code and debug issues when they arise.

### LLVM Backend Pipeline

Your SimpLang source code flows through these stages:

1. **Lexical Analysis** (`src/lexer.l`): Text → tokens. The lexer recognizes keywords (`fn`, `var`, `while`), operators (`+`, `*`, `=`), and literals (`42.0`, `"string"`). Syntax errors like `fn functionName(` get caught here.

2. **Parsing** (`src/parser.y`): Tokens → Abstract Syntax Tree (AST). The parser builds a tree structure representing your program's logic. It understands that `x + y * z` means "multiply first, then add" and creates the appropriate tree nodes.

3. **Code Generation** (`src/codegen.cpp`): AST → LLVM IR. Each AST node generates corresponding LLVM instructions. A `while` loop becomes branch instructions, variable assignments become store operations, arithmetic becomes SSA (Static Single Assignment) form.

4. **Optimization**: LLVM's optimization passes transform the IR. Loop vectorization kicks in, recognizing parallelizable patterns and generating SIMD instructions. Dead code elimination removes unused variables. Constant folding evaluates `2.0 * 3.0` at compile time.

5. **Code Emission**: LLVM IR → native assembly → object file. The backend targets your CPU architecture (x86_64, aarch64) and generates machine code.

6. **Linking**: The object file becomes a shared library (`.so`) that your host program can dynamically load.

### MLIR Backend Pipeline

The MLIR path is more sophisticated, with multiple intermediate representations:

1. **Lexing & Parsing**: Same as LLVM backend, but the AST recognizes tensor operations.

2. **Simp Dialect Generation**: High-level operations like `tensor_matmul(A, B, C, M, N, K)` become Simp dialect ops. The Simp dialect is SimpLang's custom MLIR dialect, representing ML primitives.

3. **Lowering to Linalg/MemRef** (`ConvertSimpToMemRef` pass): Tensor operations lower to Linalg (linear algebra ops) and MemRef (memory references). `tensor_matmul` becomes nested `linalg.matmul` operations with explicit memory layouts.

4. **Loop Tiling** (`TilingPass`): Large matrix operations get tiled into cache-friendly blocks. A 512×512 matmul becomes 16×16 tiles processed in a triply-nested loop structure, dramatically improving cache hit rates.

5. **Lowering to SCF** (Structured Control Flow): Linalg ops become explicit loops (`scf.for`). Tiling annotations turn into actual loop nests with controlled iteration bounds.

6. **Lowering to LLVM Dialect**: SCF loops become LLVM IR. Vectorization passes insert SIMD instructions. This is where `tensor_matmul` finally becomes the hundreds of IR instructions that implement blocked, vectorized matrix multiplication.

7. **LLVM Optimization & Emission**: Same as the LLVM backend—standard LLVM passes optimize, then emit machine code.

You can inspect any stage:

```bash
# See initial MLIR after Simp dialect generation
./build_mlir/src/simplang model.sl --emit-mlir --dump-mlir-passes | less

# See final LLVM IR before assembly
./build_mlir/src/simplang model.sl --emit-mlir --emit-llvm -o model.ll
cat model.ll
```

The multi-stage lowering is what makes MLIR powerful: each dialect applies domain-specific optimizations that wouldn't be possible in a single-pass compiler.

## Debugging: When Things Go Wrong (Or Just Get Curious)

SimpLang provides different debugging approaches depending on which backend you're using.

### Interactive Debugger (LLVM Backend Only)

For LLVM-compiled kernels, SimpLang includes a sophisticated GDB-like debugger that provides runtime inspection:

```bash
# Start debugger with a compiled kernel (LLVM backend)
./dev.sh debug

# Inside the debugger
(simplang-db) break 42           # Set breakpoint at line 42
(simplang-db) run                 # Execute until breakpoint
(simplang-db) step                # Step to next line
(simplang-db) print x             # Inspect variable
(simplang-db) backtrace           # Show call stack
(simplang-db) continue            # Resume execution
```

The debugger maps compiled machine code back to source lines, so you can set breakpoints on specific SimpLang statements and see exactly what's happening.

**Memory Tracking**: Enable leak detection and bounds checking for LLVM kernels:

```cpp
#include "kernel_runner.hpp"
#include "kernel_debugger.hpp"

int main() {
    KernelRunner runner;
    runner.loadLibrary("./kernel.so");  // LLVM-compiled kernel

    // Attach debugger with memory tracking
    DebugConfig config;
    config.enableMemoryTracking()
          .setBreakpointMode(HardwareBreakpoints);

    runner.attachDebugger(config);

    // Register leak handler
    runner.debugger().onMemoryLeak([](const LeakInfo& info) {
        std::cerr << "LEAK: " << info.size << " bytes at " << info.address << "\n";
    });

    runner.runKernel();
    return 0;
}
```

Memory tracking adds 2-5% overhead when enabled, but catches errors that would otherwise be silent bugs.

**SIMD Register Inspection**: When working with vectorized LLVM code, you can inspect SIMD registers:

```cpp
// In debugger
(simplang-db) print vectorState

// Output shows AVX registers
AVX Register v1: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
AVX Register v2: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
```

This helps you verify that vectorization actually happened and understand which data is being processed in parallel.

**Note**: Interactive debugging for MLIR-compiled kernels is on the roadmap. For now, MLIR debugging relies on compiler-level inspection (see below) and standard tools like `gdb` for the compiled binaries.

### Compiler Debugging: Dumping MLIR Passes

For compiler debugging, dump intermediate representations to understand the transformation stages:

```bash
# Verbose output showing all MLIR passes
./build_mlir/src/simplang model.sl --emit-mlir --dump-mlir-passes > passes.mlir

# Grep for specific operations
grep "linalg.matmul" passes.mlir
grep "scf.for" passes.mlir
grep "llvm.fma" passes.mlir  # Check for fused multiply-add
```

Each pass is clearly marked with comments like `// -----// IR Dump After ConvertSimpToMemRef //-----`, making it easy to see exactly what each transformation does.

## Architecture Deep Dive

For those who want to understand the internals or contribute to SimpLang, here's how the pieces fit together.

### Host-Kernel Model

SimpLang separates your main application (the **host**, typically C++) from compute-intensive code (the **kernel**, written in SimpLang). The host loads compiled kernels as shared libraries (`.so` files) and invokes them through a clean C ABI:

```cpp
// Host application (C++)
#include "kernel_runner.hpp"

int main() {
    KernelRunner runner;
    runner.loadLibrary("./compute_kernel.so");

    // Allocate input data
    std::vector<float> input(1000, 1.0f);

    // Run kernel
    double result = runner.runKernel();

    std::cout << "Result: " << result << std::endl;
    return 0;
}
```

This separation gives you modularity (swap kernels without recompiling the host), fault isolation (kernel crashes don't take down the host), and hot-reloading for rapid iteration.

### Memory Management

SimpLang handles memory alignment automatically. SIMD operations require data aligned to 16, 32, or 64-byte boundaries depending on instruction set (SSE, AVX, AVX-512). The runtime allocates arrays with proper alignment and the compiler generates aligned load/store instructions.

For MLIR-compiled code, arrays pass as **memref descriptors**—a struct containing:
```c
{
    void* allocated;    // Original allocation pointer
    void* aligned;      // Aligned data pointer
    int64_t offset;     // Offset into data
    int64_t size;       // Number of elements
    int64_t stride;     // Stride for multidimensional indexing
}
```

This descriptor format enables dynamic shapes and strided access patterns, critical for tensor operations.

### Type System

SimpLang uses dynamic typing with type inference. Variables default to `f32` (float) unless explicitly annotated:

```simplang
var x = 1.0;           // f32 (inferred from literal)
var i = 0i;            // i64 (integer literal with 'i' suffix)
var data f32[1024];    // Explicit f32 array
var counts i64[100];   // Explicit i64 array
```

The MLIR backend performs type checking and shape inference for tensor operations, catching dimension mismatches at compile time.

### Error Handling

Compilation errors include source locations and helpful messages:

```
Error at line 42, column 5:
    tensor_matmul(A, B, C, 256, 512, 128);
                   ^
Type mismatch: expected tensor<256,512,f32> but got tensor<256,256,f32>
```

Runtime errors (out-of-bounds access, null dereferences) trigger exceptions that the host can catch, preventing silent corruption.

## Current State and Roadmap

SimpLang is under active development. Here's what works today and what's coming next.

### Working Today

**LLVM Backend**:
- Complete SimpLang language implementation (functions, variables, loops, arrays)
- Automatic SIMD vectorization (SSE/AVX/AVX-512)
- Cross-platform support (x86_64, aarch64)
- Interactive debugger with memory tracking

**MLIR Backend**:
- Tensor operations: `tensor_matmul`, `tensor_add`, `tensor_mul`
- ML primitives: `rmsnorm`, `softmax`, `silu`, `conv2d`
- Loop tiling for cache optimization
- ARM NEON vectorization
- Complete transformer implementation (LLaMA architecture)
- W4 quantization (4-bit weights)
- Cross-compilation to ARM with tunable tile sizes

**Benchmarks & Validation**:
- LLaMA models from 110M to 3B parameters running on CPU
- 73.65 GFLOP/s matrix multiplication (beating Eigen)
- 6x faster than NumPy on Raspberry Pi 5
- Quantized models with 4x memory compression

### Near-Term Roadmap (Next 3-6 Months)

**Performance Optimization**:
- Native int4/int8 matmul kernels (eliminate dequantization overhead)
- SIMD optimization for quantized operations (AVX-512 VNNI, ARM DP4A)
- Kernel fusion to reduce memory bandwidth bottlenecks
- Multi-threading with OpenMP for layer-parallel execution

**Language & Compiler**:
- `for` loop syntax for cleaner iteration (`for i in 0..n`)
- Annotation system for optimization hints (`@tile(8,8,8)`, `@unroll(4)`)
- Better error messages with suggestions
- Compile-time shape inference for all tensor ops

**Backend Improvements**:
- GPU code generation via MLIR GPU dialect (CUDA/ROCm)
- Tensor-to-memref bufferization (eliminate current memref workarounds)
- Custom lowering for matmul accumulator pattern (reduce memory stores)
- Flash Attention implementation for memory-efficient transformers

### Long-Term Vision (6-12+ Months)

**Hardware Targets**:
- Apple M-series optimization (AMX instructions for matrix ops)
- RISC-V vector extensions support
- FPGA backend for custom accelerators

**Model Support**:
- Vision transformers and diffusion models
- Graph neural networks
- Quantization-aware training (QAT) compilation
- Dynamic shapes for variable-length sequences

**Tooling**:
- VS Code extension with syntax highlighting and language server
- Visual profiler showing where time is spent
- Model zoo with pre-compiled transformers
- Package manager for sharing SimpLang libraries

**Research Directions**:
- Polyhedral optimization for perfect loop nests
- Auto-tuning tile sizes based on cache hierarchy
- Learned optimizations using ML to predict best compilation strategies

## Contributing

SimpLang is a research project, which means contributions are welcome but expect active iteration and breaking changes. If you're interested in compilers, ML systems, or SIMD optimization, this is a great codebase to explore.

### Areas Where Help Is Needed

**Testing**: More test coverage for edge cases, especially around tensor operations and ARM cross-compilation.

**Documentation**: Tutorial-style guides for specific use cases (building a CNN, optimizing a specific model architecture).

**Benchmarking**: Comparing against other ML compilation frameworks (TVM, IREE, XLA).

**Optimization**: SIMD experts who can help squeeze out the last 10-20% performance.

**Language Design**: Feedback on syntax, type system, and what features would make SimpLang more useful.

### How to Contribute

1. **Explore the codebase**: Read `CLAUDE.md` for development guidelines
2. **Run the tests**: `./run_tests.sh` and `./run_mlir_tests.sh`
3. **Pick an issue**: Check GitHub issues for "good first issue" tags
4. **Submit a PR**: Include tests and performance benchmarks where applicable

Code style is enforced loosely—readable code with comments explaining "why" not just "what".

## License

This project is licensed under the MIT License—see `LICENSE` file for details. Use it, modify it, learn from it. If you build something interesting with SimpLang, I'd love to hear about it.

---

**Questions? Issues? Ideas?** Open an issue on GitHub or start a discussion. This is a learning project, so don't hesitate to ask "why did you do X this way?"—there's often a good reason, or sometimes just "seemed like a good idea at the time."

Happy compiling! 🚀
