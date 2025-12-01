# Register Blocking Optimization Analysis for SimpTensor

**Date**: 2025-11-11
**Context**: tensor_matmul performance optimization
**Current Performance**: 45.68 tokens/s @ 8×8×8 tiling (+18% vs baseline)

---

## Current Implementation Analysis

### Pipeline Overview
```
Source (tensor_matmul)
  → linalg.matmul
  → linalg.tiling (8×8×8)
  → linalg.vectorization
  → vector.lowering
  → LLVM IR (vector<8xf32>)
  → LLVM optimization passes (O3)
```

### What's Currently Working Well

1. **Cache Tiling** ✅
   - 8×8×8 tiles fit perfectly in L1 cache (256 bytes)
   - Minimizes cache misses
   - Enables spatial locality

2. **SIMD Vectorization** ✅
   - MLIR vectorization generates `vector<8xf32>` operations
   - Uses AVX/AVX2 256-bit registers (8 floats)
   - Auto-vectorized by LLVM

3. **Loop Optimizations** ✅
   - Loop-invariant code motion (LICM)
   - Common subexpression elimination (CSE)
   - Dead code elimination

---

## Register Blocking: The Missing Optimization

### What is Register Blocking?

**Register blocking** (also called **register tiling** or **micro-kernel optimization**) keeps multiple accumulator values in CPU registers during the innermost loop, reducing memory traffic.

### Current vs Register-Blocked Approach

#### Current Implementation (Conceptual)
```cpp
// Simplified current behavior after tiling
for (i = 0; i < 8; i++) {
  for (j = 0; j < 8; j++) {
    for (k = 0; k < 8; k++) {
      C[i,j] += A[i,k] * B[k,j];  // 1 accumulator
      // Load C[i,j], compute, store back EVERY iteration
    }
  }
}
```
**Problem**: Each iteration loads/stores C[i,j] from/to memory

#### Register-Blocked Approach
```cpp
// Compute 4×4 block of C in registers
for (ii = 0; ii < 8; ii += 4) {
  for (jj = 0; jj < 8; jj += 4) {
    // Allocate 16 registers for C block
    float c00=0, c01=0, c02=0, c03=0;
    float c10=0, c11=0, c12=0, c13=0;
    float c20=0, c21=0, c22=0, c23=0;
    float c30=0, c31=0, c32=0, c33=0;

    for (k = 0; k < 8; k++) {
      // Load 1 row of A (4 elements)
      float a0 = A[ii+0, k];
      float a1 = A[ii+1, k];
      float a2 = A[ii+2, k];
      float a3 = A[ii+3, k];

      // Load 1 column of B (4 elements)
      float b0 = B[k, jj+0];
      float b1 = B[k, jj+1];
      float b2 = B[k, jj+2];
      float b3 = B[k, jj+3];

      // Compute 4×4 outer product (16 FMAs)
      c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
      c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
      c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
      c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
    }

    // Store 4×4 block back to memory ONCE
    C[ii+0, jj+0] = c00; C[ii+0, jj+1] = c01; ...
  }
}
```
**Benefit**: 16 accumulators stay in registers, memory ops reduced by 8×

---

## Why Register Blocking Matters

### Memory vs Compute Bottleneck

| Operation | Latency (cycles) | Throughput |
|-----------|------------------|------------|
| FP32 FMA | 4-6 | 2 per cycle (AVX2) |
| L1 cache load | 4-5 | 64 bytes/cycle |
| L1 cache store | 1 | 32 bytes/cycle |

**Problem**: For 8×8×8 tile with 512 FMAs:
- Current: 512 stores to C (2048 bytes written)
- Register-blocked: 64 stores (256 bytes written)
- **8× reduction in memory traffic**

### Register Availability

**x86-64 AVX2 Registers**:
- 16× YMM registers (256-bit each) = 16× `vector<8xf32>`
- Can hold 16×8 = 128 float32 values simultaneously
- For 4×4 register block: uses 16 registers (perfect fit!)

---

## Performance Impact Estimation

### Current Performance Breakdown (8×8×8 tiling)
```
45.68 tokens/s @ Stories110M (768×768 matmuls)
= ~138 GFLOP/s actual throughput
```

**Theoretical Peak** (AVX2 @ 3.5 GHz):
- 8 FP32 ops/cycle × 2 FMA units × 3.5 GHz = 56 GFLOP/s per core
- 16 cores = 896 GFLOP/s theoretical

**Current Efficiency**: 138 / 896 = **15.4%** of peak

### Register Blocking Expected Gains

**Conservative Estimate**:
- Memory bottleneck: 30-40% of runtime
- Register blocking reduces memory ops by 8×
- Expected speedup: **1.3-1.4×** (30-40% faster)

**Optimistic Estimate**:
- With perfect ILP (instruction-level parallelism)
- CPU can overlap 16 FMAs in flight
- Expected speedup: **1.8-2.0×** (80-100% faster)

**Predicted Performance**:
- Conservative: 45.68 × 1.35 = **61.7 tokens/s**
- Optimistic: 45.68 × 1.9 = **86.8 tokens/s**

---

## Implementation Strategy

### Option 1: MLIR linalg.batch_matmul with Register Tiling

**Approach**: Use MLIR's `linalg.batch_matmul` with explicit 4×4 register tile
```mlir
// Pseudo-MLIR
%tile_size = constant 4 : index
linalg.tiled_loop [%i, %j] = [0, 0] to [8, 8] step [4, 4] {
  // Allocate 4×4 register block
  %C_reg = vector.splat 0.0 : vector<4x4xf32>

  scf.for %k = 0 to 8 step 1 {
    // Load 4×1 from A, 1×4 from B
    %A_vec = vector.load %A[%i, %k] : vector<4xf32>
    %B_vec = vector.load %B[%k, %j] : vector<4xf32>

    // Outer product: 4×4 matrix
    %outer = vector.outerproduct %A_vec, %B_vec : vector<4x4xf32>
    %C_reg = arith.addf %C_reg, %outer
  }

  // Store 4×4 block
  vector.store %C_reg, %C[%i, %j]
}
```

**Status**: MLIR 14 supports this, needs custom lowering pattern

---

### Option 2: Custom TensorMatMulOpLowering with Register Blocking

**Approach**: Bypass linalg.matmul, generate optimized SCF loops directly

```cpp
// In ConvertSimpToMemRef.cpp
struct TensorMatMulOpLowering : public OpConversionPattern<simp::TensorMatMulOp> {
  LogicalResult matchAndRewrite(...) {
    // For 2D matmul with specific sizes
    if (M <= 768 && N <= 768 && K <= 768) {
      // Generate register-blocked version
      return lowerRegisterBlockedMatmul(op, adaptor, rewriter);
    }
    // Fall back to linalg.matmul
    return lowerLinalgMatmul(op, adaptor, rewriter);
  }

private:
  LogicalResult lowerRegisterBlockedMatmul(...) {
    // Cache tile loop (8×8×8)
    auto tileI = rewriter.create<scf.ForOp>(loc, c0, M, c8);
    auto tileJ = rewriter.create<scf.ForOp>(loc, c0, N, c8);

    // Register block loop (4×4 micro-kernel)
    auto regI = rewriter.create<scf.ForOp>(loc, tileI.iv, ..., c4);
    auto regJ = rewriter.create<scf.ForOp>(loc, tileJ.iv, ..., c4);

    // Allocate 4×4 register block (16 vector<4xf32>)
    SmallVector<Value, 16> C_regs;
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        C_regs[i*4+j] = rewriter.create<vector.SplatOp>(loc, zero);
      }
    }

    // K loop
    auto k_loop = rewriter.create<scf.ForOp>(loc, c0, K, c1);
    {
      // Load 4×1 column from A
      Value A0 = rewriter.create<memref.LoadOp>(loc, A, {regI.iv+0, k});
      Value A1 = rewriter.create<memref.LoadOp>(loc, A, {regI.iv+1, k});
      Value A2 = rewriter.create<memref.LoadOp>(loc, A, {regI.iv+2, k});
      Value A3 = rewriter.create<memref.LoadOp>(loc, A, {regI.iv+3, k});

      // Load 1×4 row from B
      Value B0 = rewriter.create<memref.LoadOp>(loc, B, {k, regJ.iv+0});
      Value B1 = rewriter.create<memref.LoadOp>(loc, B, {k, regJ.iv+1});
      Value B2 = rewriter.create<memref.LoadOp>(loc, B, {k, regJ.iv+2});
      Value B3 = rewriter.create<memref.LoadOp>(loc, B, {k, regJ.iv+3});

      // 16 FMAs (4×4 outer product)
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          Value prod = rewriter.create<arith.MulFOp>(loc, A[i], B[j]);
          C_regs[i*4+j] = rewriter.create<arith.AddFOp>(loc, C_regs[i*4+j], prod);
        }
      }
    }

    // Store 4×4 block
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        rewriter.create<memref.StoreOp>(loc, C_regs[i*4+j], C, {regI.iv+i, regJ.iv+j});
      }
    }
  }
};
```

**Status**: More control, but more complex implementation

---

### Option 3: LLVM IR Post-Processing (Simplest)

**Approach**: Add LLVM pass to identify and optimize matmul patterns

```cpp
// In mlir_pipeline.cpp, before LLVM codegen
if (enableRegisterBlocking) {
  llvmModule->addPass(createLoopUnrollPass(4)); // Unroll by 4
  llvmModule->addPass(createSLPVectorizerPass()); // Vectorize straight-line code
  llvmModule->addPass(createLoadStoreVectorizerPass()); // Group memory ops
}
```

**Status**: Easiest to try first, may not achieve full potential

---

## Recommended Action Plan

### Phase 1: Baseline Measurement (1 hour)
1. Extract current LLVM IR for 768×768 matmul
2. Count actual memory operations (loads/stores)
3. Measure IPC (instructions per cycle) with `perf`
4. Identify bottleneck: memory-bound vs compute-bound

### Phase 2: Quick Win - LLVM Optimization (2 hours)
1. Enable aggressive loop unrolling (`-unroll-count=4`)
2. Enable SLP vectorizer for register reuse
3. Measure performance gain
4. If <20% gain, proceed to Phase 3

### Phase 3: Custom Register Blocking (1-2 days)
1. Implement `lowerRegisterBlockedMatmul` in ConvertSimpToMemRef.cpp
2. Generate 4×4 register-blocked micro-kernel
3. Benchmark against current implementation
4. Tune block size (2×2, 4×4, 8×8) based on results

### Phase 4: Integration & Testing (1 day)
1. Test on Stories110M transformer (all 12 layers)
2. Benchmark across different matrix sizes (64, 128, 256, 512, 768, 1024)
3. Profile with `perf` to verify register usage
4. Document results

---

## Success Criteria

✅ **Minimum Viable**: 55 tokens/s (+21% from 45.68)
🎯 **Target**: 62 tokens/s (+36%)
🚀 **Stretch Goal**: 70+ tokens/s (+53%)

---

## Next Steps

1. **Run `perf` analysis** on current Stories110M to identify bottleneck
2. **Extract LLVM IR** for one 768×768 matmul and count memory ops
3. **Implement Option 3** (LLVM passes) as quickest test
4. **If promising, implement Option 2** (custom lowering)

---

## References

- [BLIS: Register Blocking for GEMM](https://github.com/flame/blis/wiki/Multithreading)
- [Anatomy of High-Performance Matrix Multiplication](http://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf)
- [MLIR Vector Dialect](https://mlir.llvm.org/docs/Dialects/Vector/)
