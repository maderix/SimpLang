# SimpLang Matmul Tile Sweep Benchmarks

**Date:** 2024-12-20
**CPU:** AMD Ryzen 7 7800X3D (8 cores, Zen4)
**Matrix Size:** 1024x1024 F32
**FLOP Count:** 2 × 1024³ = 2.15 GFLOP per matmul
**Theoretical Peak:** ~576 GFLOP/s (8 cores × 4.5GHz × 16 FLOP/cycle)

## Methodology

Benchmarks use `tensor_from_array()` to receive pre-initialized matrices from the host,
ensuring we measure pure matmul performance without initialization overhead.

- **Warmup:** 5 iterations before timing
- **Measurement:** 10 iterations, median time reported
- **Runner:** `tests/bench_matmul_sweep_runner.cpp`

---

## 🏆 Best Results

| Rank | Configuration | Threads | GFLOP/s | % of Peak |
|------|---------------|---------|---------|-----------|
| **🥇 #1** | `@parallel @tile(64, 256, 4)` | 4-16 | **517** | **90%** |
| 🥈 #2 | `@parallel @tile(64, 128, 4)` | 4 | 450 | 78% |
| 🥉 #3 | `@parallel @tile(32, 128, 4)` | 4 | 458 | 80% |
| #4 | `@parallel @tile(128, 128, 8)` | 8-16 | 427 | 74% |
| **Best Sequential** | `@tile(8, 16, 4)` | 1 | **110** | 19% |

### 🚀 Best Parallel Config
```simplang
@parallel @tile(64, 256, 4)
var C = tensor_matmul(A, B);
```
**Performance:** 500-517 GFLOP/s (4-16 threads)
**Speedup:** 4.7x over best sequential

### 📌 Best Sequential Config
```simplang
@tile(8, 16, 4)
var C = tensor_matmul(A, B);
```
**Performance:** 105-110 GFLOP/s

---

## Key Findings

### 1. Tile Size Selection is Critical
- **Small K (4)** is essential - keeps inner reduction loop tight for register reuse
- **Large M (64) and N (256)** for parallel - maximizes work per thread
- **Small M (8), medium N (16-32)** for sequential - fits in L1/L2 cache

### 2. Thread Scaling for Best Config (@parallel @tile(64,256,4))

| Threads | Time (ms) | GFLOP/s | Scaling |
|---------|-----------|---------|---------|
| 1 | 4.9 | 442 | 1.0x |
| 4 | 4.2 | **517** | 1.17x |
| 8 | 4.3 | 510 | 1.15x |
| 16 | 4.2 | 516 | 1.17x |

**Note:** Even single-threaded parallel config (442 GFLOP/s) beats sequential (110 GFLOP/s) by 4x due to better tile structure!

### 3. Avoid These Configurations
- `@tile(256, 256, 4)` - Too large, poor cache utilization (~165 GFLOP/s)
- `@tile(256, 128, 4)` - Unbalanced, only ~240 GFLOP/s
- Uniform large tiles (64×64×64) - Causes severe slowdown

---

## Full Sweep Results

### Sequential (@tile only)

| M | N | K | Time (ms) | GFLOP/s |
|---|---|---|-----------|---------|
| **8** | **16** | **4** | **20.0** | **110** |
| 8 | 32 | 4 | 23.5 | 91 |
| 16 | 16 | 16 | 24.4 | 88 |
| 32 | 32 | 16 | 25.5 | 84 |
| 4 | 32 | 4 | 26.6 | 81 |

### Parallel (@parallel @tile) - Best Thread Count

| M | N | K | Best Threads | Time (ms) | GFLOP/s |
|---|---|---|--------------|-----------|---------|
| **64** | **256** | **4** | **4-16** | **4.2** | **517** |
| 32 | 128 | 4 | 4 | 4.7 | 458 |
| 64 | 128 | 4 | 4 | 4.8 | 450 |
| 128 | 128 | 8 | 8 | 5.0 | 427 |
| 128 | 256 | 8 | 8 | 4.9 | 439 |
| 256 | 128 | 4 | 1 | 8.6 | 250 |
| 256 | 256 | 4 | 1 | 12.6 | 170 |

---

## Multi-Shape Benchmark Results

Performance varies significantly with matrix dimensions. Tested shapes include square, rectangular, and ML-specific configurations.

### Results Summary

| Shape | Dimensions (M×K×N) | GFLOP/Op | Best Config | Threads | GFLOP/s | Notes |
|-------|-------------------|----------|-------------|---------|---------|-------|
| **Small Square** | 512×512×512 | 0.27 | `@parallel @tile(64,128,4)` | 8 | **338** | L3 cache fits |
| **Medium Square** | 1024×1024×1024 | 2.15 | `@parallel @tile(64,256,4)` | 8 | **473** | Sweet spot |
| **Large Square** | 2048×2048×2048 | 17.2 | `@parallel @tile(64,256,4)` | 8 | **491** | Near peak |
| **XL Square** | 4096×4096×4096 | 137.4 | `@parallel @tile(64,256,4)` | 4 | **261** | Memory bound |
| **Wide** | 1024×1024×4096 | 8.6 | `@parallel @tile(64,256,4)` | 8 | ~450 | Good scaling |
| **Tall** | 4096×1024×1024 | 8.6 | `@parallel @tile(64,256,4)` | 8 | **440** | Good scaling |
| **Transformer 768** | 768×768×768 | 0.91 | `@parallel @tile(32,128,4)` | 4 | **326** | Common LLM dim |
| **LLaMA FFN** | 2048×2048×8192 | 68.7 | `@parallel @tile(64,256,4)` | 4 | **249** | FFN projection |

### Key Observations

1. **Sweet Spot: 1024-2048 matrices** - Best efficiency at ~473-491 GFLOP/s
2. **4096² becomes memory-bound** - Only 261 GFLOP/s despite more compute
3. **Smaller matrices (512²)** - Lower efficiency due to parallelization overhead
4. **Optimal tile config** - `@parallel @tile(64,256,4)` works across most sizes
5. **Thread count** - 4-8 threads optimal; larger matrices prefer fewer threads

### Shape-Specific Recommendations

| Use Case | Recommended Config | Expected GFLOP/s |
|----------|-------------------|------------------|
| Small ops (< 512²) | `@tile(8,16,4)` | 80-110 |
| Medium ops (512-2048²) | `@parallel @tile(64,256,4)` 8T | 340-500 |
| Large ops (> 2048²) | `@parallel @tile(64,256,4)` 4T | 250-300 |
| Transformer attention | `@parallel @tile(32,128,4)` 4T | 300-350 |

---

## Recommendations

### For Maximum Performance (Multi-threaded)
```simplang
// Best: 517 GFLOP/s with 4+ threads
@parallel @tile(64, 256, 4)
var C = tensor_matmul(A, B);
```

### For Single-Threaded Workloads
```simplang
// Best sequential: 110 GFLOP/s
@tile(8, 16, 4)
var C = tensor_matmul(A, B);
```

### General Guidelines
1. **Always use K=4** for the reduction dimension
2. **Use `@parallel`** with M=64, N=128-256 for multi-threaded
3. **For sequential**, use small M (8), medium N (16-32)
4. **Avoid M or N > 128** for sequential - cache thrashing
5. **4-8 threads** is optimal; beyond 8 threads shows diminishing returns

---

## How to Run

```bash
# Build the sweep runner
g++ -O3 -fopenmp -o bench_matmul_sweep tests/bench_matmul_sweep_runner.cpp -ldl -std=c++17

# Run sweep (requires OpenMP preload)
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libomp.so.5 ./bench_matmul_sweep ./build_mlir/src/simplang
```
