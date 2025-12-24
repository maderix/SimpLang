# SimpLang MatMul Annotation Sweep Benchmarks

## Platform
- **CPU:** AMD Ryzen 7 7800X3D (8 cores, Zen 4)
- **L3 Cache:** 96MB (3D V-Cache)
- **Frequency:** ~5.0 GHz boost

---

## INT8 VNNI MatMul (2024-12-24)

### Configuration
- **Data Types:** INT8 inputs, INT32 accumulator
- **Instruction:** AVX-VNNI (`vpdpbusd`)
- **Theoretical Peak:** ~2560 GIOP/s (adjusted)
- **Parallelization:** 8 threads (OpenMP)
- **API:** `tensor_matmul_out(A, B, C)` - zero-copy direct output

### Visualization
![VNNI Sweep Plot](vnni_sweep_plot.png)

### Best Results by Matrix Size

| N | Best Tile | GIOP/s | % Peak |
|------|-----------|--------|--------|
| 512 | 128x128x16 | 600 | 23.4% |
| 768 | 128x128x16 | 1083 | 42.3% |
| 1024 | 128x128x16 | 1063 | 41.5% |
| 1536 | 64x64x16 | 1749 | 68.3% |
| 2048 | 128x128x16 | 873 | 34.1% |
| 3072 | 64x64x16 | 1528 | 59.7% |
| 4096 | 64x64x16 | 917 | 35.8% |
| 5120 | **64x64x16** | **1974** | **77.1%** |
| 6144 | 64x64x16 | 1306 | 51.0% |
| 7168 | 64x64x16 | 1852 | 72.3% |
| 8192 | 32x32x16 | 1126 | 44.0% |
| 10240 | 64x64x16 | 1228 | 48.0% |
| 12288 | 64x64x16 | 1160 | 45.3% |
| 14336 | 256x256x16 | 1316 | 51.4% |
| 16384 | 256x256x16 | 1242 | 48.5% |

### Key Findings (INT8)
1. **64x64x16 tiles optimal for most sizes** - good balance of parallelism and cache reuse
2. **Non-power-of-2 sizes outperform** (1536, 5120, 7168) - likely cache alignment effects
3. **Peak: 1974 GIOP/s** at N=5120 with `tensor_matmul_out` (77.1% of theoretical)
4. **17.5% faster** than previous with-copy version (1974 vs 1680 GIOP/s)
5. **Large matrices (8K+) become memory-bound** - drop to ~1100-1300 GIOP/s

### tensor_matmul_out: Zero-Copy Output

The new `tensor_matmul_out(A, B, C)` API writes directly to user-provided buffer,
eliminating the ~15% copy overhead from the previous `tensor_matmul` + copy loop approach.

| Version | N=5120 | Improvement |
|---------|--------|-------------|
| tensor_matmul + copy | 1680 GIOP/s | baseline |
| **tensor_matmul_out** | **1974 GIOP/s** | **+17.5%** |

### Recommended INT8 Config
```simplang
// Zero-copy output (recommended): 1974 GIOP/s @ N=5120
i32<N, N> C = tensor_from_array(C_arr, 0i);
@parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
tensor_matmul_out(A, B, C);

// For very large matrices (8K+), consider 256x256 tiles
@parallel @tile(256, 256, 16) @lower("vnni.i8_matmul")
tensor_matmul_out(A, B, C);
```

---

## M,N,K Tile Sweep (2024-12-24)

### Configuration
- **Matrix Sizes:** 1536, 5120, 7168 (best performers from square tile sweep)
- **M Tiles:** 32, 64, 128, 256
- **N Tiles:** 32, 64, 128, 256
- **K Tiles:** 8, 16, 32, 64
- **Total Configurations:** 192

### Visualization
![MNK Sweep Plot](mnk_sweep_plot.png)

### Best Configuration per Matrix Size

| N | M Tile | N Tile | K Tile | GIOP/s | % Peak |
|------|--------|--------|--------|--------|--------|
| 1536 | 32 | 64 | 16 | 1810 | 70.7% |
| 5120 | 32 | 64 | 16 | 1967 | 76.8% |
| 7168 | 128 | 64 | 32 | 1837 | 71.8% |

### Key Findings (M,N,K Sweep)

1. **K=16 optimal for most sizes** - matches VNNI's 4-element processing (16 = 4×4 vectorization)
2. **Asymmetric tiles perform well** - 32×64 outperforms 64×64 square tiles
3. **Smaller M tiles preferred** - M=32 often beats M=64,128,256 (better row parallelism)
4. **N=64 consistent winner** - good balance for column blocking
5. **Larger K helps at larger matrix sizes** - K=32 wins at N=7168

### K Tile Analysis

| Matrix Size | Best K | Max GIOP/s |
|-------------|--------|------------|
| 1536 | 16 | 1810 |
| 5120 | 16 | 1967 |
| 7168 | 32 | 1837 |

### Recommended Asymmetric Config
```simplang
// For N <= 5120: use M=32, N=64, K=16
@parallel @tile(32, 64, 16) @lower("vnni.i8_matmul")
tensor_matmul_out(A, B, C);

// For N > 5120: consider larger K
@parallel @tile(128, 64, 32) @lower("vnni.i8_matmul")
tensor_matmul_out(A, B, C);
```

---

## F32 MatMul (2024-12-20)

### Configuration
- **Data Types:** F32 inputs and outputs
- **Matrix Size:** 1024x1024
- **FLOP Count:** 2 × 1024³ = 2.15 GFLOP per matmul
- **Theoretical Peak:** ~576 GFLOP/s (8 cores × 4.5GHz × 16 FLOP/cycle)

### Best Results

| Rank | Configuration | Threads | GFLOP/s | % Peak |
|------|---------------|---------|---------|--------|
| #1 | `@parallel @tile(64, 256, 4)` | 4-16 | **517** | **90%** |
| #2 | `@parallel @tile(64, 128, 4)` | 4 | 450 | 78% |
| #3 | `@parallel @tile(32, 128, 4)` | 4 | 458 | 80% |
| Best Sequential | `@tile(8, 16, 4)` | 1 | **110** | 19% |

### Thread Scaling (F32, @tile(64,256,4))

| Threads | Time (ms) | GFLOP/s | Scaling |
|---------|-----------|---------|---------|
| 1 | 4.9 | 442 | 1.0x |
| 4 | 4.2 | **517** | 1.17x |
| 8 | 4.3 | 510 | 1.15x |
| 16 | 4.2 | 516 | 1.17x |

### Multi-Shape Results (F32)

| Shape | Dimensions | Best Config | GFLOP/s |
|-------|-----------|-------------|---------|
| Small Square | 512³ | `@parallel @tile(64,128,4)` | 338 |
| Medium Square | 1024³ | `@parallel @tile(64,256,4)` | 473 |
| Large Square | 2048³ | `@parallel @tile(64,256,4)` | 491 |
| XL Square | 4096³ | `@parallel @tile(64,256,4)` | 261 |
| Transformer 768 | 768³ | `@parallel @tile(32,128,4)` | 326 |

### Recommended F32 Config
```simplang
// Multi-threaded: 517 GFLOP/s
@parallel @tile(64, 256, 4)
var C = tensor_matmul(A, B);

// Single-threaded: 110 GFLOP/s
@tile(8, 16, 4)
var C = tensor_matmul(A, B);
```

---

## General Guidelines

### Tile Selection Rules
1. **K dimension small (4-16)** - keeps inner reduction tight for register reuse
2. **M and N larger (64-256)** for parallel - maximizes work per thread
3. **Smaller tiles for small matrices** - more parallelism
4. **Larger tiles for large matrices** - better cache reuse

### Thread Count
- **4-8 threads optimal** for most sizes
- Larger matrices prefer fewer threads (memory bound)
- Diminishing returns beyond 8 threads

---

## Files
- `vnni_sweep_plot.png` - INT8 VNNI square tile visualization
- `sweep_results.csv` - INT8 VNNI square tile raw data
- `mnk_sweep_plot.png` - INT8 VNNI M,N,K 3-way heatmap visualization
- `mnk_sweep_results.csv` - INT8 VNNI M,N,K sweep raw data
- `tile_sweep_1024x1024.csv` - F32 1024x1024 sweep data

## How to Run
```bash
# INT8 VNNI benchmark
./build_mlir/src/simplang test.sl --emit-mlir -o test.o
gcc -shared -fopenmp -o test.so test.o -lm -lgomp
LD_PRELOAD=/lib/x86_64-linux-gnu/libiomp5.so ./bench_runner test.so
```
