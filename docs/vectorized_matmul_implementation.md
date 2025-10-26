# Vectorized MatMul Implementation - Session Summary

## Performance Results
- **Original (manual loops)**: 15.47 tok/s (64.6 ms/token)
- **Final (vectorized matmul)**: 18.7 tok/s (53.3 ms/token)
- **Improvement**: +21% faster

## Key Changes

### 1. Extended MatMul with Offsets
**File**: `include/mlir/Dialects/Simp/SimpOps.td`
- Added 3 offset parameters: `lhs_offset`, `rhs_offset`, `output_offset`
- Enables layer-specific weight access without copying

### 2. Static Shape Detection
**File**: `src/mlir/lowering/ConvertSimpToMemRef.cpp`
- Detects compile-time constant dimensions
- Creates static MemRef types for vectorization
- Falls back to dynamic shapes when needed

### 3. Mandatory Tiling Before Vectorization
**File**: `src/mlir/mlir_pipeline.cpp`
- Tile size: 32x32x32 (optimized for L1 cache)
- Prevents IR explosion with large matrices
- **Key fix**: Without tiling, 768x768 matrix → 1.6GB memory, hangs
- With tiling, compilation takes seconds

### 4. Complete Vectorization Pipeline
**File**: `src/mlir/mlir_pipeline.cpp`
```
Simp dialect → MemRef + Linalg
    ↓ Tiling (32x32x32)
    ↓ Linalg vectorization strategy
    ↓ Vector lowering
    ↓ LLVM optimization
    → 32-wide SIMD code
```

### 5. Zero Initialization Fix
**File**: `examples/llama2/stories110M.sl`
- `linalg.matmul` performs accumulation (C += A*B), not assignment
- Required adding zero init loops before each matmul call
- Without this, logits become NaN after position 2

## Vectorization Statistics
- **32-wide SIMD vectors**: 16,744 instances
- **Total fmuladd ops**: 7,169
- **Vector type**: `<32 x float>` (AVX-256 compatible when scalarized)

## Code Changes in stories110M.sl

All 7 matrix multiplications converted to builtin with compile-time constant shapes:

1. **QKV projections** (3x):
   ```
   q = matmul(wq, xb, q, 768i, 768i, 1i, qkv_offset, 0i, 0i);
   k = matmul(wk, xb, k, 768i, 768i, 1i, qkv_offset, 0i, 0i);
   v = matmul(wv, xb, v, 768i, 768i, 1i, qkv_offset, 0i, 0i);
   ```

2. **Output projection**:
   ```
   xb2 = matmul(wo, xb, xb2, 768i, 768i, 1i, wo_offset, 0i, 0i);
   ```

3. **FFN w1, w3** (2x):
   ```
   hb = matmul(w1, xb, hb, 2048i, 768i, 1i, ffn_offset, 0i, 0i);
   xb2 = matmul(w3, xb, xb2, 2048i, 768i, 1i, ffn_offset, 0i, 0i);
   ```

4. **FFN w2**:
   ```
   xb = matmul(w2, hb, xb, 768i, 2048i, 1i, w2_offset, 0i, 0i);
   ```

5. **Classifier**:
   ```
   logits = matmul(wcls, xb, logits, 32000i, 768i, 1i, 0i, 0i, 0i);
   ```

## Zero Initialization Pattern

Before each matmul call:
```c
// Zero output buffer (matmul does accumulation C += A*B)
i = 0i;
while (i < dim) {
    output[i] = 0.0;
    i = i + 1i;
}
output = matmul(...);
```

## Critical Learnings

1. **Tiling is mandatory** for large matrix vectorization
   - Direct vectorization of 768x768 causes memory explosion
   - 32x32 tiles vectorize efficiently

2. **Compile-time shapes** enable better vectorization
   - Runtime dimensions → dynamic shapes → no vectorization
   - Constant dimensions → static shapes → 32-wide SIMD

3. **Accumulation semantics** of linalg.matmul require zero init
   - C += A*B (accumulation) vs C = A*B (assignment)
   - Missing zero init causes NaN propagation

4. **MemRef ABI** passes arrays as 5-tuples
   - Each array: `(allocated_ptr, aligned_ptr, offset, size, stride)`
   - Host code must match this calling convention

## Performance Breakdown

| Change | Tokens/sec | ms/token | Improvement |
|--------|------------|----------|-------------|
| Original manual loops | 15.47 | 64.6 | baseline |
| Transformer vectorized | 16.1 | 62.0 | +4% |
| Classifier vectorized | 18.7 | 53.3 | +21% |

The classifier vectorization gave the biggest boost because:
- Called once per token
- Large matrix (32000 × 768)
- Very compute-intensive

## Next Steps (Optional)

- **Tile size tuning**: Experiment with 16x16x16 vs 32x32x32 vs 64x64x64
- **Cache profiling**: Use `perf` to measure L1/L2/L3 hit rates
- **Polyhedral optimizations**: Enable MLIR's affine dialect transformations
- **Multi-threading**: Parallelize across transformer layers
- **FP16/BF16**: Lower precision for additional speedup

## Files Modified

1. `include/mlir/Dialects/Simp/SimpOps.td` - MatMulOp definition
2. `src/mlir/lowering/ConvertSimpToMemRef.cpp` - Static shape detection
3. `src/mlir/mlir_pipeline.cpp` - Mandatory tiling
4. `src/mlir/mlir_codegen.cpp` - Dialect registration, 9-arg matmul handler
5. `src/parser.y` - 6-arg and 9-arg matmul parsing
6. `include/ast/expr/matmul_expr.hpp` - MatMulExprAST with offsets
7. `examples/llama2/stories110M.sl` - All matmuls converted to builtin
