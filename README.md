# **SimpLang: A Multi-Level Compilation Framework for Machine Learning and High-Performance Numerical Computation**

<p align="center">
  <img src="docs/animation_cropped.gif" alt="SimpLang Banner" width="600"/>
</p>

## **1 Introduction**

SimpLang is a domain-specific language and compiler infrastructure designed to investigate a unified representation for numerical computation and machine-learning workloads. The project evaluates whether high-level mathematical expressions can be lowered through a structured, multi-stage compilation strategy to generate performant code for CPUs and ARM-based edge devices.

The system adopts a **dual-backend architecture**, combining a classical LLVM code generation pipeline with an MLIR-based path for tensor computations. The first backend emphasizes general-purpose compute with automatic SIMD vectorization; the second progressively lowers tensor operations through the MLIR stack (Simp → Linalg → SCF → LLVM), enabling domain-specific optimizations essential for contemporary ML models.

SimpLang is an intentionally exploratory implementation. It includes fully optimized kernels as well as incomplete or experimental components, reflecting the incremental development process commonly observed in research compilers. The implementation therefore provides a practical environment for studying scanner/parser design, SSA IR construction, tiling transformations, vectorization strategies, and heterogeneous lowering pipelines in a single coherent system.

---
# **See it in action**
<p align="center">
  <img src="assets/llama_demo.gif" alt="LLaMA 110M Demo" width="800"/>
</p>

# **2 Execution Characteristics**

SimpLang has been evaluated on representative machine-learning and numerical workloads. Experimental results illustrate the effectiveness of a multi-level IR approach in achieving competitive performance relative to hand-optimized or production libraries.

### **2.1 Transformer Inference on CPU**

A 110M-parameter LLaMA model executes at **42.95 tokens/s** on x86 hardware. The model is implemented entirely in SimpLang and lowered through the MLIR backend, illustrating that high-level tensor operations such as `tensor_matmul`, `rmsnorm`, `softmax`, and `silu` can be compiled into efficient vectorized kernels.

Scaling behavior remains consistent across model sizes up to 3B parameters. For example, the 3B-parameter model sustains **0.758 tokens/s** (4.547 GFLOP/s), demonstrating that the generated code maintains efficiency despite increased problem size.

Quantization is supported through **4-bit (W4)** weights, providing 4× memory reduction (e.g., 3B parameters → 2.1 GB instead of 9.2 GB) with only a ~2× slowdown. This enables models to run on devices that could not otherwise accommodate uncompressed model weights.

---

### **2.2 Matrix Multiplication**

Matrix multiplication, the central kernel behind most ML workloads, exhibits competitive throughput:

| Size    | Precision | SimpLang      | Eigen | Speedup |
| ------- | --------- | ------------- | ----- | ------- |
| 256×256 | f32       | 60.21 GFLOP/s | —     | 1.89×   |
| 512×512 | f32       | 73.65 GFLOP/s | —     | 2.17×   |
| 256×256 | int32     | 105.83 GIOP/s | —     | 5.73×   |

These results arise from explicit tiling (16×16×16), shape-aware lowering via MLIR, and target-specific vectorization (AVX/AVX2/AVX-512).

---

### **2.3 ARM Cross-Compilation**

Cross-compilation to ARM uses the same high-level SimpLang program with a target switch (`--target aarch64`). On Raspberry Pi 5, results show substantial improvements over NumPy:

* f32 matmul 256×256: **6.0× faster**
* int32 matmul 256×256: **4.4× faster**
* int32 matmul 512×512: **3.7× faster**

The gap primarily reflects that NumPy’s default OpenBLAS build lacks NEON support, while SimpLang lowers to NEON via MLIR’s vectorizer.

Transformer inference achieves **13.49 tokens/s** with tuned 8×8×8 tiling for ARM caches.

---

# **3 Dual-Backend Architecture**

SimpLang adopts two compilation paths, unified by a single high-level source language but optimized for different classes of workloads.

---

## **3.1 LLVM Backend**

The LLVM backend is oriented toward general numerical computation and traditional scalar/vector loops.

Key characteristics:

* Employs LLVM’s mature vectorizer and loop optimizations
* Generates SSE/AVX/AVX2/AVX-512 instructions depending on target
* Suitable for scalar algorithms, array loops, DSP-style kernels
* Provides rapid compilation times for iterative development

This backend underlies the baseline performance of SimpLang and served as the initial implementation prior to MLIR integration.

---

## **3.2 MLIR Backend**

The MLIR backend is specialized for tensor-centric ML workloads. It introduces a custom Simp dialect representing high-level tensor operations and ML primitives such as:

* `tensor_matmul`
* `rmsnorm`
* `softmax`
* `silu`
* `conv2d`

Programs lower through a structured sequence of dialects:

```
Simp Dialect
   ↓ ConvertSimpToMemRef / ConvertSimpToLinalg
Linalg on Tensors
   ↓ Tiling / Fusion
Linalg on Buffers
   ↓ Bufferization → SCF
SCF (structured loops)
   ↓ Vectorization
LLVM Dialect
   ↓ LLVM IR → Machine Code
```

This multi-level approach exposes optimization opportunities—tiling, fusion, vectorization—that are difficult to express or detect in a single low-level IR.

For ML workloads (e.g., attention mechanisms, feed-forward networks, convolutional layers), this backend is required.

---

# **4 Programming Interface**

SimpLang adopts a minimal, math-oriented syntax. Programs define a single entry function `kernel_main`, and variables use inferred types unless explicitly annotated.

Below we summarize representative examples within the MLIR paper style.

---

## **4.1 Scalar Example**

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

This lowers to SSA form in LLVM IR, subsequently optimized by the LLVM pipeline.

---

## **4.2 Vectorizable Loop**

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

The LLVM vectorizer transforms this loop into SIMD operations (e.g., AVX2), with no special syntax required from the programmer.

---

## **4.3 Tensor Example (MLIR Backend)**

```simplang
fn matmul_example() {
    var A = tensor<256,256,f32>();
    var B = tensor<256,256,f32>();
    var C = tensor<256,256,f32>();

    tensor_matmul(A, B, C, 256i, 256i, 256i);
    return C[0i,0i];
}
```

The lowering sequence constructs a tiled and vectorized Linalg→SCF loop nest.

---

## **4.4 Transformer Block**

A simplified transformer block is expressible directly in SimpLang:

* RMSNorm
* Multi-head attention decomposition (`Q`, `K`, `V`)
* Attention scaling and softmax
* Feed-forward network with SiLU
* Residual connections

The complete model (as used in evaluation) follows this pattern and lowers exclusively through MLIR dialects until producing optimized LLVM IR.

---

# **5 Cross-Compilation to ARM**

Cross-compilation requires only changing the target on the MLIR command line:

```bash
./simplang model.sl --emit-mlir --target aarch64 --tile-size 8
```

An ARM cross-compiler links the object file:

```bash
aarch64-linux-gnu-gcc -shared model.o -o model.so -lm
```

The generated code employs ARM NEON instructions (e.g., `fmla v1.4s, v2.4s, v3.4s`). Tile size selection affects locality and throughput.

---

# **6 Compilation Pipeline**

SimpLang exposes two independent but conceptually parallel pipelines.

---

## **6.1 LLVM Pipeline**

1. **Lexing** → tokenization
2. **Parsing** → abstract syntax tree (AST)
3. **AST to LLVM IR** → SSA construction
4. **LLVM Optimizations** → vectorization, dead-code elimination, scalar replacements
5. **Code Emission** → machine code
6. **Linking** → shared objects (`.so`) for host invocation

---

## **6.2 MLIR Pipeline**

1. **Front-end** parses SimpLang into high-level ops
2. **Simp Dialect** captures tensor-level semantics
3. **ConvertSimpToLinalg/MemRef**
4. **Tiling and Fusion Passes** optimize locality
5. **Lower to SCF** explicitly materializes loop nests
6. **Vectorization** transforms SCF loops to vector ops
7. **Lower to LLVM Dialect**
8. **LLVM Optimization and Codegen** → executable machine code

Intermediate representations can be inspected at any stage via `--dump-mlir-passes`.

---

# **7 Debugging and Inspection**

### **7.1 LLVM Interactive Debugger**

SimpLang integrates a debugger analogous to GDB for the LLVM backend. It supports:

* Source-level breakpoints
* Variable inspection
* Backtraces
* Memory tracking and leak detection
* Inspection of SIMD register state

### **7.2 MLIR Debugging**

Inspection is performed via IR dumps:

```bash
./simplang model.sl --emit-mlir --dump-mlir-passes
```

Developers can search for `linalg.matmul`, `scf.for`, or `llvm.fma` to trace lowering results.

---

# **8 System Architecture**

### **8.1 Host/Kernels**

SimpLang kernels compile into `.so` libraries with a C ABI. A C++ host program loads them dynamically and invokes `kernel_main()`. This boundary isolates compute kernels from application code and supports rapid iteration.

### **8.2 Memory Representation**

Aligned allocations ensure compatibility with SIMD load/store semantics. MLIR-generated code uses **memref descriptors**, enabling:

* Dynamic shapes
* Multidimensional views
* Strided accesses

### **8.3 Type and Shape System**

Types default to `f32`. Integer literals use the `i` suffix. Tensor operations employ static shapes; shape inference validates dimensions and emits informative errors.

---

# **9 Current Implementation and Roadmap**

### **9.1 Supported Today**

* LLVM backend with full language coverage
* MLIR backend with tensor and ML primitives
* LLaMA models up to 3B parameters
* 4-bit quantized matmuls
* ARM NEON vectorization
* Competitive matrix multiplication
* Docker/Dev-container workflows

### **9.2 Short-Term Development**

* int4/int8 native kernels
* AVX-512 VNNI / ARM DP4A support
* Kernel fusion
* `for` loop syntax, annotations (`@tile`, `@unroll`)
* Enhanced error diagnostics
* Multi-threading for ML workloads

### **9.3 Long-Term Objectives**

* GPU lowering via MLIR GPU dialects
* Apple AMX support
* RISC-V vectorization
* Polyhedral optimizations
* Auto-tuning for tile sizes
* Vision models, diffusion, dynamic shapes
* Integrated profiler and model zoo

---

# **10 Contributing**

The project welcomes contributions in:

* Test coverage
* Documentation
* Benchmarking
* Optimization kernels
* Language design feedback

Pull requests should include test cases and performance evaluations where relevant.

---

# **11 License**

SimpLang is released under the Apache2.0 License.
