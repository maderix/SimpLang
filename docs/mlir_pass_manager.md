# MLIR Pass Manager Architecture

**Status:** ✅ Production (refactored Nov 2024)
**Performance:** 45.66 tok/s on Stories110M (optimal)

This document describes SimpLang's modern MLIR pass manager architecture, featuring a unified pipeline with composable builders and named presets.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Available Passes](#available-passes)
- [Named Pipelines](#named-pipelines)
- [CLI Usage](#cli-usage)
- [Programmatic Usage](#programmatic-usage)
- [Performance Tuning](#performance-tuning)

---

## Overview

### Design Principles

1. **Single PassManager**: One unified pass manager instead of multiple instances
2. **Composable Builders**: Pipeline phases as reusable building blocks
3. **Named Pipelines**: Pre-configured pipelines for common use cases
4. **CLI Integration**: All passes accessible via `mlir-opt` style commands

### Key Benefits

- **Performance**: Maintains 45.66 tok/s on Stories110M (8×8×8 tiling)
- **Simplicity**: Reduced code complexity (-3 lines net, better organization)
- **Flexibility**: Easy to configure and extend
- **Modern**: Follows MLIR best practices (PassWrapper, PassRegistration)

---

## Architecture

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Single PassManager                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Phase 1: Simp Lowering                                          │
│  ├─ ConvertSimpToMemRef (tensor_alloc → memref.alloc)          │
│  └─ ConvertSimpToLinalg (tensor_matmul → linalg.matmul)        │
│                                                                   │
│  Phase 2: Linalg Optimization                                    │
│  ├─ SimpLinalgTiling (8×8×8 default, configurable)             │
│  ├─ SCFToOpenMP (optional, for parallelization)                │
│  ├─ LinalgVectorization (SSE/AVX code generation)              │
│  ├─ ConvertLinalgToLoops (fallback for non-vectorized)         │
│  ├─ LoopOptimizations (unroll, LICM)                           │
│  └─ Canonicalization + CSE                                      │
│                                                                   │
│  Phase 2.5: Buffer Management                                    │
│  ├─ BufferHoisting (move allocations out of loops)             │
│  ├─ BufferLoopHoisting (cache-aware placement)                 │
│  ├─ BufferDeallocation (automatic memory management)            │
│  └─ BufferizationToMemRef (cleanup bufferization ops)          │
│                                                                   │
│  Phase 3: LLVM Dialect Lowering                                 │
│  ├─ ConvertVectorToSCF (high-level vector ops)                 │
│  ├─ LowerAffine (affine.min → standard)                        │
│  ├─ LowerToCFG (scf.for/while/if → cf.br/cond_br)             │
│  ├─ ConvertOpenMPToLLVM (optional, omp.* → LLVM)              │
│  ├─ ArithmeticExpandOps (maxf/minf → cmpf+select)             │
│  └─ ReconcileUnrealizedCasts (cleanup)                         │
│                                                                   │
│  Post-Pipeline: Pattern-Based Conversion                         │
│  ├─ VectorLoweringPatterns (contract, transpose, broadcast)    │
│  ├─ LLVMConversionPatterns (arith/math/memref → LLVM)         │
│  └─ PartialConversion (preserve OpenMP ops for translation)    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Code Organization

```
src/mlir/
├── Passes.cpp                      # Central pass registration
├── passes/
│   ├── LinalgTiling.cpp           # Configurable tiling pass
│   └── Pipelines.cpp              # Named pipeline builders
├── lowering/
│   ├── ConvertSimpToMemRef.cpp    # Simp → MemRef+Arith
│   ├── ConvertSimpToLinalg.cpp    # MatMul lowering
│   └── SpecializeShapes.cpp       # Dynamic → static shapes
└── mlir_pipeline.cpp              # Main pipeline orchestration
```

---

## Available Passes

### Core Lowering Passes

#### `convert-simp-to-memref`
**Purpose:** Lower Simp dialect to MemRef, Arith, and Linalg dialects
**Input:** Simp dialect operations
**Output:** MemRef allocations, arithmetic ops, linalg operations

**Key Conversions:**
- `simp.tensor_alloc` → `memref.alloc`
- `simp.tensor_matmul` → `linalg.matmul`
- `simp.rmsnorm` → `linalg.generic`
- `simp.softmax` → `linalg.generic`
- `simp.tensor_add/mul` → `arith.addf/mulf` loops

**CLI Usage:**
```bash
mlir-opt input.mlir --convert-simp-to-memref
```

#### `convert-simp-to-linalg`
**Purpose:** Convert high-level Simp matmul to linalg.matmul
**Input:** `simp.tensor_matmul`
**Output:** `linalg.matmul` (optimizable by MLIR)

**CLI Usage:**
```bash
mlir-opt input.mlir --convert-simp-to-linalg
```

#### `specialize-shapes`
**Purpose:** Convert dynamic memrefs to static shapes for optimization
**Input:** `memref<?x?xf32>` (dynamic)
**Output:** `memref<768x768xf32>` (static, enables affine optimizations)

**CLI Usage:**
```bash
mlir-opt input.mlir --specialize-shapes
```

### Optimization Passes

#### `simp-linalg-tiling`
**Purpose:** Cache-aware tiling for matmul operations
**Default:** 8×8×8 tiling (optimal for transformers)

**Options:**
- `--tile-size N` - Set tile size (default: 8)
- `--hierarchical` - Enable two-level tiling (64→16)
- `--parallel` - Generate scf.parallel for OpenMP

**Performance Impact:**
- 8×8×8: **45.68 tok/s** ✅ (optimal for Stories110M)
- 16×16×16: 38.71 tok/s (baseline)
- 4×4×4: 32.42 tok/s (too many iterations)
- 32×32×32: 17.80 tok/s (cache thrashing)

**CLI Usage:**
```bash
simplang input.sl --emit-mlir --tile-size 8 -o output.o
```

**Programmatic Usage:**
```cpp
#include "mlir/Passes.h"

// Default 8×8×8 tiling
pm.addNestedPass<FuncOp>(createSimpLinalgTilingPass());

// Custom configuration
pm.addNestedPass<FuncOp>(createSimpLinalgTilingPass(16, false, false));
```

---

## Named Pipelines

Pre-configured pipelines for common scenarios. Use via `--pass-pipeline` flag.

### `simp-default`
**Purpose:** Production pipeline for transformer workloads
**Configuration:**
- 8×8×8 tiling (optimal)
- Vectorization enabled
- No OpenMP (single-threaded)

**Performance:** 45.68 tok/s on Stories110M

**Usage:**
```bash
mlir-opt input.mlir --pass-pipeline='simp-default'
```

### `simp-transformer`
**Purpose:** Alias for `simp-default`, explicitly documented for transformers
**Use Case:** LLaMA, Stories110M, GPT-style models

**Why 8×8×8 is optimal:**
- Perfect L1 cache fit (256 bytes per tile)
- Optimal for 768×768 matrices (typical transformer dimensions)
- 18% faster than 16×16×16 baseline

### `simp-debug`
**Purpose:** Fast compilation with minimal optimizations
**Configuration:**
- No tiling
- No vectorization
- Basic lowering only

**Use Case:** Development, debugging, quick iteration

**Usage:**
```bash
mlir-opt input.mlir --pass-pipeline='simp-debug'
```

### `simp-high-perf`
**Purpose:** Maximum performance for large matrices
**Configuration:**
- Hierarchical 64→16 two-level tiling
- Vectorization enabled
- Cache-aware blocking

**Use Case:** Large matrix multiplication (>2048×2048)

**Usage:**
```bash
mlir-opt input.mlir --pass-pipeline='simp-high-perf'
```

### `simp-parallel`
**Purpose:** Multi-core parallelization with OpenMP
**Configuration:**
- 16×16×16 tiling
- OpenMP parallelization enabled
- scf.parallel → omp.parallel conversion

**Use Case:** Multi-core CPUs, batch processing

**Usage:**
```bash
mlir-opt input.mlir --pass-pipeline='simp-parallel'
```

---

## CLI Usage

### Basic Compilation

```bash
# Default pipeline (8×8×8 tiling)
./build_mlir/src/simplang input.sl --emit-mlir -o output.o

# Custom tile size
./build_mlir/src/simplang input.sl --emit-mlir --tile-size 16 -o output.o

# Enable OpenMP
./build_mlir/src/simplang input.sl --emit-mlir --enable-openmp -o output.o

# Hierarchical tiling
./build_mlir/src/simplang input.sl --emit-mlir --enable-hierarchical-tiling -o output.o

# Dump intermediate IR
./build_mlir/src/simplang input.sl --emit-mlir --dump-mlir-passes -o output.o
```

### Using Named Pipelines

```bash
# Apply transformer-optimized pipeline
mlir-opt input.mlir --pass-pipeline='simp-transformer'

# Debug mode (fast compilation)
mlir-opt input.mlir --pass-pipeline='simp-debug'

# High-performance mode
mlir-opt input.mlir --pass-pipeline='simp-high-perf'
```

### Individual Passes

```bash
# Run specific pass
mlir-opt input.mlir --convert-simp-to-memref

# Chain passes manually
mlir-opt input.mlir \
  --convert-simp-to-memref \
  --simp-linalg-tiling \
  --canonicalize
```

---

## Programmatic Usage

### Using the MLIRCompilationPipeline

```cpp
#include "mlir/mlir_pipeline.hpp"

// Create pipeline
mlir::simp::MLIRCompilationPipeline pipeline(module);

// Configure options
pipeline.setTileSize(8);               // Default: 8 (optimal)
pipeline.setEnableTiling(true);        // Default: true
pipeline.setEnableHierarchicalTiling(false);  // Default: false
pipeline.setEnableOpenMP(false);       // Default: false
pipeline.setDumpIntermediateIR(false); // Default: false

// Run pipeline
if (!pipeline.runPasses()) {
  llvm::errs() << "Pipeline failed\n";
  return nullptr;
}

// Translate to LLVM IR
auto llvmModule = pipeline.translateToLLVMIR(llvmContext);
```

### Custom Pipeline Builder

```cpp
#include "mlir/Passes.h"
#include "mlir/Pass/PassManager.h"

// Register all SimpLang passes
mlir::simp::registerSimpPasses();
mlir::simp::registerSimpPipelines();

// Create custom pipeline
mlir::PassManager pm(context);

// Phase 1: Lowering
pm.addPass(mlir::simp::createConvertSimpToMemRefPass());

// Phase 2: Optimization
pm.addNestedPass<FuncOp>(mlir::simp::createSimpLinalgTilingPass(8, false, false));
pm.addPass(mlir::createCanonicalizerPass());

// Phase 3: LLVM lowering
pm.addPass(mlir::createLowerAffinePass());
pm.addPass(mlir::createLowerToCFGPass());

// Run
if (failed(pm.run(module))) {
  llvm::errs() << "Custom pipeline failed\n";
}
```

---

## Performance Tuning

### Tile Size Selection

| Matrix Size | Optimal Tile | Rationale |
|-------------|-------------|-----------|
| 768×768 (transformers) | **8×8×8** | L1 cache fit, minimal overhead |
| 1024×1024 | 16×16×16 | Balance cache/iterations |
| 2048×2048+ | 64→16 (hierarchical) | Multi-level cache blocking |

### Performance Guidelines

1. **Start with 8×8×8**: Optimal for most transformer workloads
2. **Use hierarchical tiling for large matrices** (>2048×2048)
3. **Enable OpenMP for batch processing** (multiple sequences)
4. **Disable tiling for debugging** (use `simp-debug` pipeline)

### Benchmarking

```bash
# Compile with different tile sizes
for tile in 4 8 16 32; do
  ./build_mlir/src/simplang stories110M.sl --tile-size $tile -o /tmp/test_$tile.o
  gcc -shared -o /tmp/test_$tile.so /tmp/test_$tile.o -lm
  ./test_stories110M /tmp/test_$tile.so | grep "Throughput"
done
```

### Expected Results (Stories110M)

```
Tile 4×4×4:   32.42 tok/s
Tile 8×8×8:   45.68 tok/s  ✅ OPTIMAL
Tile 16×16×16: 38.71 tok/s
Tile 32×32×32: 17.80 tok/s
```

---

## Troubleshooting

### Common Issues

#### Low Performance
**Symptom:** <40 tok/s on Stories110M
**Solution:** Ensure tile size is 8 (`--tile-size 8`)

#### Compilation Errors
**Symptom:** Pass verification failed
**Solution:** Check MLIR version compatibility (requires MLIR 14+)

#### OpenMP Errors
**Symptom:** Index type mismatches
**Solution:** Ensure `ConvertOpenMPToLLVMPass` runs AFTER `LowerToCFGPass`

### Debug Mode

```bash
# Dump IR at each stage
./build_mlir/src/simplang input.sl --emit-mlir --dump-mlir-passes -o output.o

# Check intermediate files
ls /tmp/*.mlir /tmp/*.ll
```

---

## References

- [MLIR Pass Infrastructure](https://mlir.llvm.org/docs/PassManagement/)
- [Linalg Tiling](https://mlir.llvm.org/docs/Dialects/Linalg/#linalg-tiling)
- [SimpTensor Checkpoint](../simptensor/TENSOR_CHECKPOINT.md) - Performance data

---

**Last Updated:** November 2024
**Maintainer:** SimpLang Team
