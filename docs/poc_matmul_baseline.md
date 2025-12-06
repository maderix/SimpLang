# POC: Pure SimpleLang MatMul - Baseline Performance

## Goal
Replace `tensor_matmul` intrinsic with pure SimpleLang implementation using annotations while maintaining ±5% performance.

## Baseline: tensor_matmul Intrinsic (Current Implementation)

### Test Environment
- CPU: AMD Ryzen (znver3)
- Compiler: SimpleLang MLIR backend with LLVM 14
- Optimization: O3, loop tiling (16×16×16)
- Date: 2025-01-17

### F32 Performance (Target Sizes: 256×256, 512×512)

| Size | Time (ms) | GFLOP/s | vs Eigen | Status |
|------|-----------|---------|----------|--------|
| 64×64 | 0.010 | **53.69** | 1.31× faster | ✓ |
| 128×128 | 0.085 | **49.48** | 1.52× faster | ✓ |
| **256×256** | **0.557** | **60.21** | **1.89× faster** | **✓ TARGET** |
| **512×512** | **3.645** | **73.65** | **2.17× faster** | **✓ TARGET** |
| 1024×1024 | 48.163 | **44.59** | 1.09× faster | ✓ |

### I32 Performance

| Size | Time (ms) | GFLOP/s | vs Eigen | Status |
|------|-----------|---------|----------|--------|
| 64×64 | 0.007 | **80.23** | 5.86× faster | ✓ |
| 128×128 | 0.041 | **103.25** | 6.17× faster | ✓ |
| 256×256 | 0.317 | **105.83** | 5.73× faster | ✓ |
| 512×512 | 2.756 | **97.40** | 5.06× faster | ✓ |
| 1024×1024 | 46.548 | **46.14** | 2.49× faster | ✓ |

### I64 Performance
- 256×256: 33.36 GFLOP/s (5.24× faster than Eigen)

## Performance Target for Pure SimpleLang Implementation

To match the current intrinsic within ±5%:

| Size | Target Range (GFLOP/s) | Target Time (ms) |
|------|------------------------|------------------|
| **256×256** | **57.20 - 63.22** | **0.529 - 0.585** |
| **512×512** | **69.97 - 77.33** | **3.463 - 3.827** |

## Implementation Plan

### Phase 2: Annotation System
1. Add `@rewrite(pattern)` annotation for compute optimization hints
2. Add `@memory(placement)` annotation for memory layout hints
3. Implement `ApplyAnnotations` MLIR pass to process annotations

### Phase 3: Pure SimpleLang MatMul
1. Add `for` loop syntax to SimpleLang
2. Implement `tensor_matmul()` in `simptensor/tensor_core.sl` using:
   - Explicit loops (i, j, k)
   - `@rewrite("tile(16,16,16)")` for cache blocking
   - `@memory("align(64)")` for SIMD-friendly alignment

### Phase 4: Verification
1. Compile and benchmark pure SimpleLang version
2. Verify performance within target range (±5%)
3. Run correctness tests to ensure numerical accuracy

## Next Steps
1. ✅ Baseline established (60.21 GFLOP/s @ 256×256, 73.65 GFLOP/s @ 512×512)
2. → Implement annotation system (@rewrite, @memory)
3. → Add for loop syntax
4. → Implement pure SimpleLang matmul with annotations
5. → Benchmark and verify ±5% target met
