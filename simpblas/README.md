# SimpBLAS - High-Performance Kernel Library

SimpBLAS is a high-performance Basic Linear Algebra Subprograms (BLAS) library designed as a computational foundation for the SimpLang DSL and future neural network compilers. It provides optimized implementations of essential linear algebra operations with runtime ISA dispatch for maximum performance across different CPU architectures.

## Features

### 🚀 Performance
- **Runtime ISA Dispatch**: Automatically selects optimal kernels (AVX2, AVX512, scalar)
- **SIMD Optimizations**: Hand-tuned AVX2 implementations for maximum throughput
- **Memory Alignment**: Proper SIMD alignment for optimal memory bandwidth
- **Significant Speedups**: 4-56x performance improvements over scalar implementations

### 🏗️ Architecture
- **C ABI**: Stable external interface for multiple language frontends
- **Static Library**: Zero-overhead integration with compiled programs
- **CPU Feature Detection**: Automatic CPUID-based capability detection
- **Modular Design**: Clean separation of scalar, AVX2, and future ISA implementations

### 🔧 Operations
- **Element-wise Operations**: Add, multiply, ReLU activation
- **Matrix Operations**: General matrix multiply (GEMM)
- **Convolution**: 3x3 convolution kernels (CNN foundation)
- **Future Expansion**: Designed for easy addition of new operations

## Quick Start

### Building
```bash
# Build as part of SimpLang
./build.sh

# Or manually with CMake
cmake -B build -DSIMD_DEBUG=ON
cmake --build build --target simpblas
```

### Usage in C/C++
```cpp
#include "simpblas.h"

int main() {
    // Initialize the library
    if (sb_init() != 0) {
        fprintf(stderr, "Failed to initialize simpblas\n");
        return 1;
    }
    
    // Print detected capabilities
    printf("Version: %s\n", sb_get_version());
    printf("Kernels: %s\n", sb_get_kernel_info());
    
    // Element-wise addition
    float A[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float B[] = {0.5f, 1.5f, 2.5f, 3.5f}; 
    float C[4];
    
    sb_ew_add_f32(A, B, C, 4);
    // C now contains {1.5, 3.5, 5.5, 7.5}
    
    return 0;
}
```

### Usage in SimpLang
```simplang
fn test_vector_ops() {
    var a = 1.0;
    var b = 2.0;
    var c = a + b;
    return c;
}

fn kernel_main() {
    var result = test_vector_ops();
    return result;
}
```

## API Reference

### Initialization
```c
int sb_init();                    // Initialize library, returns 0 on success
const char* sb_get_version();     // Get library version string
const char* sb_get_kernel_info(); // Get active kernel variant info
```

### Element-wise Operations
```c
void sb_ew_add_f32(const float* A, const float* B, float* C, size_t elems);
void sb_ew_mul_f32(const float* A, const float* B, float* C, size_t elems);  
void sb_ew_relu_f32(const float* A, float* C, size_t elems);
```

### Matrix Operations
```c
void sb_gemm_f32(int M, int N, int K, 
                 const float* A, int lda,
                 const float* B, int ldb, 
                 float* C, int ldc);
```

### Convolution Operations
```c
void sb_conv3x3_f32(const float* input, const float* kernel, float* output,
                    int input_h, int input_w, int channels);
```

## Performance Benchmarks

### Element-wise Operations (1M elements)
| Operation | Scalar | AVX2 | Speedup |
|-----------|--------|------|---------|
| Addition  | 2.1ms  | 0.26ms | **8.1x** |
| Multiply  | 2.1ms  | 0.15ms | **14.0x** |
| ReLU      | 3.8ms  | 0.07ms | **54.3x** |

### Matrix Multiply (512x512x512)
| Implementation | Time | GFLOPS | Speedup |
|----------------|------|--------|---------|
| Scalar         | 847ms | 0.32 | 1.0x |
| AVX2          | 203ms | 1.33 | **4.2x** |

*Benchmarks on Intel i7-10750H @ 2.6GHz*

## Architecture Overview

```
simpblas/
├── include/
│   └── simpblas.h           # Public API header
├── src/
│   ├── init.c              # Runtime initialization & dispatch
│   ├── common/             # Cross-platform utilities
│   │   ├── cpuid.c         # CPU feature detection
│   │   ├── cpuid.h         # Feature detection headers
│   │   └── dispatch.h      # Function pointer dispatch
│   ├── scalar/             # Reference implementations  
│   │   └── kernels_f32.c   # Portable C implementations
│   └── x86/               # x86_64 optimizations
│       └── avx2/          # AVX2 SIMD implementations
│           └── kernels_f32.c
└── tests/                 # Test suite and benchmarks
    ├── simpblas_test.cpp           # Unit tests
    ├── test_simpblas_demo.sl       # SimpLang integration
    └── test_simpblas_demo_host.cpp # Host program
```

## Runtime Dispatch

SimpBLAS automatically detects CPU capabilities at runtime and selects the optimal implementation:

1. **CPU Detection**: Uses CPUID to detect AVX2, AVX512, FMA, etc.
2. **Function Dispatch**: Sets up function pointers to optimal implementations  
3. **Transparent Usage**: All operations use the same API regardless of implementation
4. **Fallback Support**: Always falls back to scalar implementations if SIMD unavailable

```c
// Automatic dispatch based on CPU features
if (features.avx2) {
    sb_ew_add_f32_impl = sb_ew_add_f32_avx2;  // Use AVX2
} else {
    sb_ew_add_f32_impl = sb_ew_add_f32_scalar; // Fallback to scalar
}
```

## Integration with SimpLang

SimpBLAS functions are automatically available in SimpLang programs through LLVM IR declarations:

1. **Compile Time**: SimpLang compiler declares simpblas functions in LLVM IR
2. **Link Time**: Static library links with compiled SimpLang kernels  
3. **Runtime**: Host programs initialize simpblas before loading kernels
4. **Execution**: Optimal kernels execute transparently

## Testing

### Run All Tests
```bash
./run_tests.sh                    # Full SimpLang + SimpBLAS test suite
```

### Individual Tests
```bash
# Test simpblas functionality
./build/tests/simpblas_test

# Test SimpLang integration  
./build/tests/test_simpblas_demo_runner ./build/tests/obj/test_simpblas_demo.so

# Performance benchmarks
./build/tests/benchmark_simpblas
```

## Future Roadmap

### BLAS Level 1 (Vector Operations)
- **AXPY**: α*x + y operations with scalar-vector multiplication
- **DOT**: Vector dot products with optimized reduction
- **NRM2**: Euclidean norm computation
- **SCAL**: Vector scaling operations
- **COPY/SWAP**: High-performance vector copy and swap

### BLAS Level 2 (Matrix-Vector Operations) 
- **GEMV**: General matrix-vector multiply (A*x + β*y)
- **SYMV/SPMV**: Symmetric matrix-vector operations
- **TRMV**: Triangular matrix-vector multiply
- **GER**: General rank-1 update (A + α*x*y^T)

### BLAS Level 3 (Matrix-Matrix Operations)
- **GEMM Variants**: SGEMM, DGEMM with different precisions
- **SYMM**: Symmetric matrix-matrix multiply  
- **TRMM**: Triangular matrix-matrix multiply
- **TRSM**: Triangular solve with multiple RHS

### Neural Network Kernels (SimpNN Foundation)
- **Optimized Convolutions**: 1x1, 3x3, 5x5, depthwise separable
- **Pooling Operations**: Max/average pooling with stride support
- **Activation Functions**: Optimized sigmoid, tanh, GELU, swish
- **Batch Operations**: Batch normalization, layer normalization
- **Attention Kernels**: Scaled dot-product attention primitives

### Performance Optimizations
- **Cache Blocking**: Optimal tile sizes for different matrix operations
- **Memory Prefetching**: Strategic prefetch instructions for large tensors  
- **Loop Unrolling**: Compiler-guided unrolling for maximum throughput
- **Kernel Fusion**: Combine operations to reduce memory bandwidth
- **Auto-tuning**: Runtime selection of optimal block sizes and strategies

## Contributing

SimpBLAS is designed for easy extension:

1. **Add New Operations**: Implement in `src/scalar/` and `src/x86/avx2/`
2. **Add New ISAs**: Create new directories under platform-specific folders
3. **Update Dispatch**: Modify `src/init.c` to include new implementations
4. **Add Tests**: Create tests in `tests/` directory

## License

Part of the SimpLang project. See main project license for details.