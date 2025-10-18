# SimpLang Tensor Library

Pure mathematical tensor operations for N-dimensional arrays with automatic SIMD vectorization.

## Architecture

```
tensor/
├── core/           # Basic tensor operations (elementwise, reductions)
├── linalg/         # Linear algebra operations (matmul, transpose)  
├── tests/          # Tensor-specific performance and correctness tests
└── README.md       # This file

simpnn/
├── ops/            # ML-specific operations built on tensor primitives
├── tests/          # Neural network layer tests
└── README.md       # ML operations documentation
```

## Design Principles

1. **1D Buffer + Metadata**: All tensors are contiguous 1D arrays with shape/stride metadata
2. **Auto-Vectorization First**: All loops designed for SimpLang's SIMD passes  
3. **Pure Math**: No ML assumptions, just N-dimensional array operations
4. **Explicit Dimensions**: No hidden shape inference for predictable performance
5. **Composable**: Higher-level operations built from primitive functions

## Performance Target

- **Unit-stride operations**: Automatic AVX-512 vectorization (16x parallel)
- **Cache-friendly**: Row-major layout with blocked operations
- **Memory-aligned**: 64-byte alignment for optimal SIMD performance

## Quick Start

```simplang
// Create tensors
var A = tensor_create_2d(1024, 512);
var B = tensor_create_2d(1024, 512);
var C = tensor_create_2d(1024, 512);

// Element-wise addition (auto-vectorized)
tensor_add(A, B, C, 1024 * 512);

// Reduction
var sum = tensor_sum(C, 1024 * 512);
```