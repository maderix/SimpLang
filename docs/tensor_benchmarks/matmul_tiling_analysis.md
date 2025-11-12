# Matrix Multiplication Tiling Strategy Analysis

**Date**: 2025-11-10
**Task**: Optimize 1024×1024 f32 matmul performance (currently 9% of NumPy)

## Performance Benchmark Results

### 1024×1024 f32 Matrix Multiplication

| Configuration | GFLOP/s | vs NumPy | Instructions | Cycles | IPC | L1 Miss % | Cache Miss % | Compile Time |
|--------------|---------|----------|--------------|--------|-----|-----------|--------------|--------------|
| **NumPy (OpenBLAS)** | 211.66 | 100% | - | - | - | - | - | - |
| **16×16×16 single-level** | **43.23** | **20.4%** | 391M | 867M | 0.45 | 35.31% | 2.76% | ~60s |
| **64×64×64 → 16×16×16** | 37.17 | 17.6% | 413M | 1,024M | 0.40 | 38.85% | 0.62% | ~50s |
| **24×24×24 → 8×8×8** | 4.95 | 2.3% | 34,182M | 9,435M | 3.62 | 4.57% | 0.92% | ~20s |
| **8×8×8 default** | ~28 | 13.2% | - | - | - | - | - | ~30s |

### 512×512 f32 Matrix Multiplication

| Configuration | GFLOP/s | Instructions | Notes |
|--------------|---------|--------------|-------|
| **NumPy (OpenBLAS)** | 430.32 | - | Highly optimized BLAS |
| **16×16×16 single-level** | 76.14 | 1.92M | Best single-level |
| **64×64×64 → 16×16×16** | 70.30 | 2.09M | Hierarchical |

## Root Cause Analysis

### Why 24×24→8×8 Failed Catastrophically (8.8× slower)

**Perf counter analysis revealed:**

```
Metric                    16×16×16    24×24→8×8    Ratio
─────────────────────────────────────────────────────────
Instructions              391M        34,182M      87.3× MORE ⚠️
Cycles                    867M        9,435M       10.9× MORE
Loop iterations           262,144     2,146,689    8.2× MORE
Inner work per iteration  4,096 ops   512 ops      8× LESS
```

**Problem**: Inner loops only iterate **3×3×3 = 27 times**, creating massive loop control overhead:
- Each of 79,507 outer iterations spawns only 27 inner iterations
- Inner iterations do only 512 operations each
- Loop overhead (bounds checks, index arithmetic, jumps) dominates
- 87× more instructions executing mostly useless work

### Why 64×64→16×16 is Better But Still Loses

**Fixed loop overhead:**
- Inner loops iterate 4×4×4 = 64 times (enough to amortize overhead)
- Instruction count similar to single-level (413M vs 391M) ✓
- Better cache miss rate (0.62% vs 2.76%) ✓

**But still slower due to:**
- **IPC degradation**: 0.40 vs 0.45 (nested loops prevent instruction-level parallelism)
- **LLVM optimization barrier**: Nested structure prevents aggressive vectorization
- **More cycles**: 1,024M vs 867M despite better cache behavior

## Loop Iteration Analysis

```
1024×1024 matmul loop structure:

16×16×16 single-level:
  Outer loops: 64³ = 262,144 iterations
  Inner work: 16³ = 4,096 operations per iteration
  Loop overhead: 262,144 × ~10 instructions = 2.6M

24×24→8×8 hierarchical:
  Outer loops: 43³ = 79,507 iterations
  Inner loops: 3³ = 27 per outer
  Total iterations: 79,507 × 27 = 2,146,689
  Inner work: 8³ = 512 operations per iteration
  Loop overhead: 2,146,689 × ~10 instructions = 21.5M ⚠️

64×64→16×16 hierarchical (optimized):
  Outer loops: 16³ = 4,096 iterations
  Inner loops: 4³ = 64 per outer
  Total iterations: 4,096 × 64 = 262,144 (same as single!)
  Inner work: 16³ = 4,096 operations per iteration
  Loop overhead: Similar to single-level ✓
```

## Key Lessons Learned

### 1. Loop Overhead Dominates Cache Benefits
- **Theory**: Hierarchical tiling should improve cache locality → faster performance
- **Reality**: Loop control overhead >> cache benefits for modern CPUs
- **Conclusion**: Simple loop structures win

### 2. Critical Ratio: Inner Iterations per Outer Iteration
- **< 27 iterations**: Loop overhead dominates (catastrophic)
- **64 iterations**: Balanced, but still hurts IPC
- **4,096+ operations**: Amortizes overhead best

### 3. LLVM Optimization Capability
- **Single-level tiling**: LLVM can fully vectorize and optimize
- **Nested tiling**: Creates optimization barriers
- **Result**: Simplicity enables better code generation

### 4. IPC Tells the Real Story
- High IPC (3.62) executing loop overhead = wasted work
- Moderate IPC (0.45) executing actual computation = productive work
- **IPC must be measured against instruction mix, not in isolation**

## Compilation Performance

### Code Explosion with Aggressive Tiling

| Tile Configuration | LLVM IR Size | Compilation Result |
|-------------------|--------------|-------------------|
| 8×8×8 | 2.3 MB | ✅ ~30s |
| 16×16×16 | 14 MB | ✅ ~60s |
| 32×32×32 | 93 MB | ❌ OOM killed |
| 128→32→8 (3-level) | 93 MB | ❌ OOM killed |
| 24→8 (2-level) | 452 KB | ✅ ~20s |
| 64→16 (2-level) | ~1 MB | ✅ ~50s |

**Key finding**: Number of tiling levels matters less than tile size for code explosion.

## Recommendations

### Production Default: **16×16×16 single-level tiling**

**Rationale:**
- ✅ Best overall performance: **43.23 GFLOP/s** (20.4% of NumPy)
- ✅ Simple structure enables LLVM optimization
- ✅ Reasonable compilation time (~60s)
- ✅ Acceptable cache performance
- ✅ Highest IPC efficiency

### Experimental: Keep hierarchical tiling available

The `--hierarchical-tiling` flag can remain for:
- Research and experimentation
- Future compiler improvements might handle nested loops better
- Different workload characteristics might benefit

### Not Recommended:
- Small tiles (< 16): Poor performance/compilation tradeoff
- 3+ level tiling: Code explosion without benefit
- Tiles < 4× inner subdivision: Loop overhead dominates

## Future Work

### Why NumPy is 5× Faster

NumPy (via OpenBLAS/MKL) uses techniques we don't:
1. **Hand-written assembly kernels** for critical inner loops
2. **Register blocking** and micro-kernel optimization
3. **Data packing/copying** to ensure contiguous memory access
4. **Tail handling** optimizations
5. **Multi-threading** (we benchmark single-threaded)

### Potential Improvements
1. **Better MLIR vectorization**: Target AVX2/AVX-512 explicitly
2. **MLIR affine loop transformations**: Interchange, unroll-and-jam
3. **Polyhedral optimization**: Use Pluto/ISL for automatic tiling
4. **Compare against Halide/TVM**: Learn from other compiler frameworks

## Conclusion

**Simplicity wins.** Single-level 16×16×16 tiling achieves 20% of NumPy performance, which is respectable for a DSL compiler. The hierarchical tiling investigation taught us that:

1. Modern CPU cache hierarchies are complex - simple strategies often work best
2. LLVM's optimization capabilities are key - enable them with simple IR
3. Performance analysis requires holistic view - IPC, cache, instructions, cycles
4. Loop overhead is not free - amortization matters

The journey from 9% (8×8×8) → 20% (16×16×16) of NumPy performance demonstrates that thoughtful benchmarking and profiling yields concrete improvements.
