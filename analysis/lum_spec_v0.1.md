# Lum Language Specification v0.1

> **Lum** — A multi-level rewrite and orchestration DSL for MLIR/LLVM backends
>
> *"Blaze defines the kernels. Lum orchestrates the fire."*

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Core Concepts](#3-core-concepts)
4. [Language Syntax](#4-language-syntax)
5. [Pattern Matching](#5-pattern-matching)
6. [Scheduling Primitives](#6-scheduling-primitives)
7. [Cost Model Integration](#7-cost-model-integration)
8. [Multi-Dialect Support](#8-multi-dialect-support)
9. [Pipeline Orchestration](#9-pipeline-orchestration)
10. [Extension Points](#10-extension-points)
11. [Standard Library](#11-standard-library)
12. [Compilation Target](#12-compilation-target)
13. [Examples](#13-examples)

---

## 1. Overview

### 1.1 Purpose

Lum is a domain-specific language for:
- **Graph rewriting**: Pattern matching and transformation at multiple IR levels
- **Scheduling**: Tiling, vectorization, fusion, and memory optimization
- **Orchestration**: Composing passes into optimization pipelines
- **Cost-driven optimization**: Pluggable cost models for auto-tuning

### 1.2 Design Principles

1. **Multi-level**: Operate on ONNX, Linalg, SCF, Vector, LLVM dialects seamlessly
2. **Composable**: Small primitives that combine into complex transformations
3. **Extensible**: Plugin system for cost models, patterns, and backends
4. **Readable**: Human-friendly syntax that maps to Transform Dialect
5. **Type-safe**: Static verification of pattern and schedule validity
6. **Cost-aware**: First-class support for cost-driven decisions

### 1.3 Relationship to Blaze

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Code                                 │
├─────────────────────────────────────────────────────────────────┤
│  Blaze (.blaze)           │  Lum (.lum)                         │
│  - Kernel definitions      │  - Rewrite rules                    │
│  - Tensor operations       │  - Scheduling strategies            │
│  - SIMD primitives         │  - Fusion policies                  │
│  - Type annotations        │  - Cost models                      │
├─────────────────────────────────────────────────────────────────┤
│                    MLIR (Multiple Dialects)                      │
│  Linalg │ Tensor │ Vector │ SCF │ Arith │ MemRef │ GPU │ ...    │
├─────────────────────────────────────────────────────────────────┤
│                         LLVM IR                                  │
├─────────────────────────────────────────────────────────────────┤
│                    Target (x86, ARM, GPU)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Architecture

### 2.1 Compilation Pipeline

```
Lum Source (.lum)
      │
      ▼
┌─────────────┐
│   Parser    │ → Lum AST
└─────────────┘
      │
      ▼
┌─────────────┐
│  Analyzer   │ → Validated AST + Type Info
└─────────────┘
      │
      ▼
┌─────────────┐
│  Lowering   │ → Transform Dialect IR
└─────────────┘
      │
      ▼
┌─────────────┐
│  Executor   │ → Applies transforms to target MLIR
└─────────────┘
```

### 2.2 Runtime Components

```
┌────────────────────────────────────────────────────────────────┐
│                      Lum Runtime                                │
├────────────────┬───────────────┬───────────────┬───────────────┤
│  Pattern       │   Schedule    │    Cost       │   Dialect     │
│  Matcher       │   Executor    │    Oracle     │   Bridge      │
├────────────────┼───────────────┼───────────────┼───────────────┤
│ - Graph match  │ - Tiling      │ - HW model    │ - Linalg ops  │
│ - Constraint   │ - Vectorize   │ - Profiling   │ - SCF loops   │
│ - Capture      │ - Fuse        │ - Heuristics  │ - Vector ops  │
│ - Dataflow     │ - Interchange │ - ML models   │ - LLVM intrin │
└────────────────┴───────────────┴───────────────┴───────────────┘
```

---

## 3. Core Concepts

### 3.1 Levels

Lum operates at multiple abstraction levels:

| Level | Dialect(s) | Operations |
|-------|------------|------------|
| `graph` | ONNX, TOSA | High-level ops (Conv, MatMul, Softmax) |
| `tensor` | Linalg, Tensor | Structured ops with iteration semantics |
| `loop` | SCF, Affine | Explicit loop nests |
| `vector` | Vector | SIMD operations |
| `memory` | MemRef | Memory allocation and access |
| `llvm` | LLVM | Low-level instructions |

### 3.2 Entities

```lum
// Pattern: matches IR structure
pattern MatMulBias {
    matmul: linalg.matmul
    bias: linalg.generic [broadcast_add]
    where matmul.result -> bias.input[0]
}

// Schedule: transforms matched patterns
schedule TileAndFuse for MatMulBias {
    tile matmul [M=64, N=64, K=32]
    fuse bias into matmul.loop[0]
    vectorize matmul.inner [N=8, K=4]
}

// Cost: estimates performance
cost model HardwareCost {
    flops_per_cycle: 32
    l1_size: 32KB
    l2_size: 512KB
    vector_width: 512
}

// Pipeline: orchestrates transformations
pipeline OptimizeGEMM {
    match MatMulBias
    apply TileAndFuse with cost HardwareCost
    lower to vector
}
```

### 3.3 Type System

```lum
// Tensor types
type Tensor<dtype, shape> = tensor<shape x dtype>
type Matrix<T> = Tensor<T, [?, ?]>
type Vector<T> = Tensor<T, [?]>

// Constraint types
type Divisible<N, D> = N where N % D == 0
type PowerOf2<N> = N where (N & (N-1)) == 0

// Hardware types
type CacheLevel = L1 | L2 | L3 | HBM
type VectorWidth = 128 | 256 | 512
```

---

## 4. Language Syntax

### 4.1 Lexical Structure

```ebnf
(* Identifiers *)
identifier = letter , { letter | digit | "_" } ;
qualified_id = identifier , { "." , identifier } ;

(* Literals *)
integer = digit , { digit } ;
float = integer , "." , integer ;
string = '"' , { character } , '"' ;
boolean = "true" | "false" ;

(* Keywords *)
keyword = "pattern" | "schedule" | "cost" | "pipeline"
        | "match" | "where" | "for" | "with" | "apply"
        | "tile" | "fuse" | "vectorize" | "interchange"
        | "parallel" | "unroll" | "peel" | "pipeline"
        | "if" | "else" | "when" | "derive" | "minimize" | "maximize"
        | "level" | "dialect" | "lower" | "raise" | "import" ;

(* Operators *)
operator = "->" | "=>" | "::" | ":" | "=" | "==" | "!="
         | "<" | ">" | "<=" | ">=" | "+" | "-" | "*" | "/"
         | "&&" | "||" | "!" | "|" | "&" | "%" ;
```

### 4.2 Module Structure

```lum
// Module declaration
module my_optimizations

// Imports
import std.patterns.gemm
import std.schedules.cache_aware
import hw.x86.avx512

// Dialect configuration
dialect linalg, scf, vector, llvm

// Hardware target
target x86_64 {
    features: [avx512, vnni]
    cache: [L1=32KB, L2=512KB, L3=16MB]
    vector_width: 512
}

// Module contents
pattern ...
schedule ...
pipeline ...
```

### 4.3 Comments and Documentation

```lum
// Single line comment

/* Multi-line
   comment */

/// Documentation comment for next item
/// @param M - rows of output
/// @returns optimized schedule
pattern Documented { ... }

#[deprecated("use PatternV2 instead")]
pattern OldPattern { ... }
```

---

## 5. Pattern Matching

### 5.1 Basic Patterns

```lum
// Match single operation
pattern SingleMatMul {
    mm: linalg.matmul
}

// Match operation chain
pattern MatMulChain {
    mm1: linalg.matmul
    mm2: linalg.matmul
    where mm1.result -> mm2.input[0]
}

// Match with type constraints
pattern Int8MatMul {
    mm: linalg.matmul
    where mm.input[0].dtype == i8
      and mm.input[1].dtype == i8
      and mm.result.dtype == i32
}
```

### 5.2 Pattern Composition

```lum
// Base pattern
pattern Linear {
    matmul: linalg.matmul
    bias: linalg.generic [broadcast_add]
    where matmul.result -> bias.input[0]
}

// Extended pattern (inherits + adds)
pattern LinearReLU extends Linear {
    relu: linalg.generic [max_zero]
    where bias.result -> relu.input[0]
}

// Alternative patterns (match any)
pattern Activation = ReLU | GELU | SiLU | Tanh

// Sequential composition
pattern LinearActivation {
    linear: Linear
    act: Activation
    where linear.bias.result -> act.input[0]
}
```

### 5.3 Wildcards and Groups

```lum
// Single wildcard (any one op)
pattern ConvAnyReLU {
    conv: linalg.conv_*          // Wildcard in op name
    _: *                          // Any single op
    relu: linalg.generic [relu]
    where conv.result ->* relu.input[0]  // Transitive connection
}

// Multi-wildcard (chain of ops)
pattern ConvBlockReLU {
    conv: linalg.conv_*
    middle: **[0..5]              // 0 to 5 intermediate ops
    relu: linalg.generic [relu]
    where conv.result ->* relu.input[0]
}

// Operation groups
pattern ComputeOp {
    op: (linalg.matmul | linalg.conv_* | linalg.batch_matmul)
}
```

### 5.4 Attribute and Shape Constraints

```lum
pattern SquareMatMul {
    mm: linalg.matmul
    where mm.input[0].shape[0] == mm.input[0].shape[1]  // Square A
}

pattern LargeMatMul {
    mm: linalg.matmul
    let M = mm.input[0].shape[0]
    let N = mm.input[1].shape[1]
    let K = mm.input[0].shape[1]
    where M * N * K > 1_000_000  // > 1M FLOPs
}

pattern VNNICompatible {
    mm: linalg.matmul
    where mm.input[0].dtype == i8
      and mm.input[0].shape[1] % 4 == 0  // K divisible by 4
}
```

### 5.5 Dataflow Constraints

```lum
// Single use (no other consumers)
pattern FusableMatMul {
    mm: linalg.matmul
    consumer: *
    where mm.result -> consumer.input[*]
      and mm.result.uses == 1           // Only one consumer
}

// Residual connection (skip connection)
pattern ResidualBlock {
    input: tensor<*>
    block: Linear
    add: linalg.generic [add]
    where input -> block.matmul.input[0]
      and block.bias.result -> add.input[0]
      and input -> add.input[1]          // Skip connection
}

// Multi-consumer pattern
pattern SharedWeight {
    weight: tensor<*>
    users: linalg.matmul[2..]            // 2 or more matmuls
    where weight -> users[*].input[1]    // Weight shared by all
}
```

### 5.6 Named Captures

```lum
pattern AttentionQKV {
    q_proj: linalg.matmul as "query"
    k_proj: linalg.matmul as "key"
    v_proj: linalg.matmul as "value"

    // Same input, different weights
    where q_proj.input[0] == k_proj.input[0]
      and k_proj.input[0] == v_proj.input[0]

    capture {
        hidden_dim: q_proj.input[0].shape[1]
        head_dim: q_proj.result.shape[1]
    }
}
```

### 5.7 Multi-Level Patterns

```lum
// Match at linalg level, constrain at loop level
pattern TiledMatMul {
    level tensor {
        mm: linalg.matmul
    }
    level loop {
        outer: scf.for
        inner: scf.for[3]    // Exactly 3 nested loops
        where mm lowered_to inner.body
    }
}

// Cross-level constraints
pattern VectorizedOp {
    level tensor {
        op: linalg.generic
    }
    level vector {
        vec_ops: vector.* [1..]
        where op lowered_to vec_ops
    }
}
```

---

## 6. Scheduling Primitives

### 6.1 Tiling

```lum
schedule TileMatMul for MatMulPattern {
    // Static tile sizes
    tile mm [M=64, N=64, K=32]

    // Named loop handles
    tile mm [M=64, N=64, K=32]
        -> (m_loop, n_loop, k_loop)

    // Multi-level tiling
    tile mm [M=64, N=64, K=32] as l2_tile
    tile l2_tile [M=8, N=8, K=4] as l1_tile
}

// Dynamic tile sizes
schedule DynamicTile for MatMulPattern {
    let M = mm.shape[0]
    let N = mm.shape[1]

    // Derived tile sizes
    let tile_m = min(64, M)
    let tile_n = min(64, N)

    tile mm [M=tile_m, N=tile_n, K=32]
}

// Tile with constraints
schedule ConstrainedTile for MatMulPattern {
    tile mm [M, N, K]
    where M % 8 == 0           // Vectorization alignment
      and K % 4 == 0           // VNNI alignment
      and M * K * 4 <= 32768   // L1 fit for A tile
}
```

### 6.2 Fusion

```lum
schedule FuseChain for LinearReLU {
    tile matmul [M=64, N=64, K=32] -> (m_loop, n_loop, k_loop)

    // Fuse into outermost loop
    fuse bias into m_loop
    fuse relu into m_loop

    // Order matters: bias before relu
    fuse [bias, relu] into m_loop ordered
}

// Conditional fusion
schedule ConditionalFuse for MatMulBias {
    tile matmul [M=64, N=64, K=32] -> (m_loop, n_loop, k_loop)

    // Only fuse if profitable
    if cost.fusion_benefit(bias, m_loop) > 0 {
        fuse bias into m_loop
    }
}

// Producer-consumer fusion
schedule ProducerConsumerFuse for Pattern {
    // Automatically fuse producer into consumer's loop
    fuse_producers mm depth=1    // One level up
}
```

### 6.3 Vectorization

```lum
schedule VectorizeMatMul for MatMulPattern {
    tile mm [M=1, N=16, K=4] -> inner_tile

    // Basic vectorization
    vectorize inner_tile [N=16, K=4]

    // With vector type specification
    vectorize inner_tile [N=16, K=4] as vector<16xf32>

    // Hardware-specific
    vectorize inner_tile for avx512
}

// VNNI vectorization
schedule VNNIVectorize for Int8MatMul {
    tile mm [M=1, N=16, K=4]

    // VNNI requires specific layout and sizes
    vectorize mm [N=16, K=4] using vnni {
        pack B [K, N] -> [K/4, N, 4]
        intrinsic: vpdpbusd
    }
}

// Masked vectorization
schedule MaskedVectorize for MatMulPattern {
    tile mm [M=1, N=16, K=4]
    vectorize mm [N=16] with_mask    // Handle non-divisible N
}
```

### 6.4 Loop Transformations

```lum
schedule LoopOpt for MatMulPattern {
    tile mm [M=64, N=64, K=32] -> (m_loop, n_loop, k_loop)

    // Interchange
    interchange [n_loop, m_loop, k_loop]  // N outermost

    // Unroll
    unroll k_loop factor=4

    // Peel (for remainder handling)
    peel m_loop

    // Coalesce (merge loops)
    coalesce [m_loop, n_loop] -> mn_loop
}

// Pipeline (software pipelining)
schedule PipelineK for MatMulPattern {
    tile mm [M=64, N=64, K=64] -> (m_loop, n_loop, k_loop)

    pipeline k_loop stages=2 {
        stage 0: load next_A, next_B
        stage 1: compute current_A * current_B
    }
}

// Parallelization
schedule ParallelMatMul for MatMulPattern {
    tile mm [M=64, N=64, K=32] -> (m_loop, n_loop, k_loop)

    parallel m_loop         // OpenMP parallel
    parallel n_loop simd    // SIMD parallelism
    // k_loop stays sequential (reduction)
}
```

### 6.5 Memory Operations

```lum
schedule MemoryOpt for MatMulPattern {
    tile mm [M=64, N=64, K=32] -> (m_loop, n_loop, k_loop)

    // Promote to local memory (stack allocation)
    promote mm.input[0] to stack    // A tile
    promote mm.input[1] to stack    // B tile

    // Prefetch
    prefetch mm.input[0] distance=2 locality=L1
    prefetch mm.input[1] distance=1 locality=L2

    // Double buffering
    buffer mm.input[0] double
}

// Data layout transformation
schedule PackForVNNI for Int8MatMul {
    // Pack B matrix for VNNI layout
    pack mm.input[1] {
        from: [K, N]
        to: [K/4, N, 4]
        tile_sizes: [4]
        inner_dims: [0]      // Pack K dimension
    }
}

// Explicit copy for GPU
schedule GPUMemory for MatMulPattern {
    copy mm.input[0] to gpu.shared
    copy mm.input[1] to gpu.shared
    sync

    compute mm on gpu.thread

    copy mm.result to host
}
```

### 6.6 Decomposition

```lum
schedule DecomposeConv for ConvPattern {
    // Decompose conv2d to im2col + matmul
    decompose conv {
        im2col: tensor.pack conv.input
        matmul: linalg.matmul im2col, conv.filter
        reshape: tensor.expand_shape matmul
    }

    // Then schedule the matmul
    tile matmul [M=64, N=64, K=32]
    fuse_producers matmul     // Fuse im2col into matmul loops
}

// Decompose activation
schedule DecomposeGELU for GELUPattern {
    // Keep decomposed but fuse
    decompose gelu {
        pow: arith.powf x, 3
        mul1: arith.mulf pow, 0.044715
        add1: arith.addf x, mul1
        mul2: arith.mulf add1, sqrt_2_pi
        tanh: math.tanh mul2
        add2: arith.addf 1.0, tanh
        mul3: arith.mulf x, add2
        mul4: arith.mulf mul3, 0.5
    }

    // Fuse all element-wise ops
    fuse_elementwise [pow, mul1, add1, mul2, tanh, add2, mul3, mul4]
}
```

---

## 7. Cost Model Integration

### 7.1 Hardware Model Definition

```lum
cost model X86Skylake {
    // Compute capabilities
    cores: 8
    threads_per_core: 2
    frequency: 3.0 GHz
    flops_per_cycle: 32        // AVX-512: 16 FMA = 32 FLOPS

    // Memory hierarchy
    cache L1 {
        size: 32 KB
        line_size: 64 B
        associativity: 8
        latency: 4 cycles
        bandwidth: 128 GB/s
    }

    cache L2 {
        size: 512 KB
        latency: 12 cycles
        bandwidth: 64 GB/s
    }

    cache L3 {
        size: 16 MB
        latency: 40 cycles
        bandwidth: 40 GB/s
    }

    memory DRAM {
        bandwidth: 50 GB/s
        latency: 100 ns
    }

    // Vector units
    vector {
        width: 512 bits
        units: 2
        types: [f32, f64, i32, i16, i8]
        features: [fma, vnni]
    }
}

// GPU model
cost model NvidiaA100 {
    sms: 108
    threads_per_sm: 2048
    frequency: 1.4 GHz
    tensor_cores: 432

    memory HBM {
        size: 80 GB
        bandwidth: 2 TB/s
    }

    memory shared {
        size: 164 KB per SM
        bandwidth: 19 TB/s
    }

    memory registers {
        size: 256 KB per SM
    }
}
```

### 7.2 Cost Functions

```lum
cost model MyCost extends X86Skylake {
    // Custom cost functions

    fn compute_cost(op: Operation) -> Cycles {
        match op {
            linalg.matmul => {
                let M, N, K = op.dimensions
                let flops = 2 * M * N * K
                return flops / self.flops_per_cycle
            }
            linalg.generic => {
                let elements = product(op.output.shape)
                let ops_per_element = count_arithmetic(op.body)
                return elements * ops_per_element / self.flops_per_cycle
            }
            _ => 0
        }
    }

    fn memory_cost(access: MemoryAccess) -> Cycles {
        let size = access.size
        let cache_level = self.find_cache_level(size)
        return size / cache_level.bandwidth * self.frequency
    }

    fn tile_cost(op: Operation, tiles: TileConfig) -> Cycles {
        let compute = self.compute_cost(op)

        // Memory traffic estimate
        let a_tile = tiles.M * tiles.K * sizeof(op.input[0].dtype)
        let b_tile = tiles.K * tiles.N * sizeof(op.input[1].dtype)
        let c_tile = tiles.M * tiles.N * sizeof(op.output.dtype)

        // Determine cache residency
        let working_set = a_tile + b_tile + c_tile
        let cache = self.find_cache_level(working_set)

        let memory = (a_tile + b_tile + c_tile) / cache.bandwidth * self.frequency

        return max(compute, memory)  // Bottleneck
    }

    fn fusion_benefit(producer: Operation, consumer: Operation) -> Cycles {
        // Benefit = avoided memory round-trip
        let intermediate_size = product(producer.output.shape) * sizeof(producer.output.dtype)
        return intermediate_size / self.cache.L2.bandwidth * self.frequency
    }
}
```

### 7.3 Cost-Driven Decisions

```lum
schedule AutoTile for MatMulPattern {
    // Let cost model derive tile sizes
    tile mm derive [M, N, K] using cost MyCost {
        minimize total_cycles
        subject to {
            M * K * 4 + K * N * 4 + M * N * 4 <= L1.size    // L1 fit
            M % 8 == 0 and N % 8 == 0                        // Alignment
            K % 4 == 0                                        // VNNI
        }
    }
}

// Compare strategies
schedule BestTile for MatMulPattern {
    let strategies = [
        tile mm [M=32, N=32, K=32],
        tile mm [M=64, N=64, K=16],
        tile mm [M=128, N=32, K=32],
    ]

    // Pick best according to cost model
    apply best_of strategies using cost MyCost
}

// Conditional based on cost
schedule AdaptiveSchedule for MatMulPattern {
    let M, N, K = mm.dimensions

    if cost.is_compute_bound(mm) {
        // Maximize parallelism
        tile mm [M=64, N=64, K=large]
        parallel m_loop
        parallel n_loop
    } else {
        // Memory bound: maximize reuse
        tile mm [M=32, N=32, K=256]
        pipeline k_loop
    }
}
```

### 7.4 Auto-Tuning Interface

```lum
// Define tuning space
tuning_space MatMulSpace for MatMulPattern {
    param tile_m: [16, 32, 64, 128, 256]
    param tile_n: [16, 32, 64, 128, 256]
    param tile_k: [16, 32, 64, 128, 256]
    param unroll_k: [1, 2, 4, 8]
    param vectorize_n: [8, 16, 32]

    constraints {
        tile_m * tile_k * 4 <= 32768    // A tile in L1
        tile_k * tile_n * 4 <= 32768    // B tile in L1
        tile_n % vectorize_n == 0       // Vectorization alignment
    }
}

// Auto-tune schedule
schedule AutoTuned for MatMulPattern {
    search MatMulSpace using {
        method: bayesian_optimization
        budget: 100 trials
        objective: minimize latency
        cost_model: MyCost           // Initial estimates
    }
}

// Use pre-tuned parameters
schedule PreTuned for MatMulPattern {
    load_params "tuned_matmul_skylake.json"

    tile mm [M=param.tile_m, N=param.tile_n, K=param.tile_k]
    unroll k_loop factor=param.unroll_k
    vectorize inner [N=param.vectorize_n]
}
```

---

## 8. Multi-Dialect Support

### 8.1 Dialect Declaration

```lum
// Declare which dialects this module operates on
module multi_level_opt

dialect {
    source: [onnx, linalg]     // Input dialects
    target: [vector, llvm]      // Output dialects
    intermediate: [scf, arith]  // Used during lowering
}
```

### 8.2 Level-Specific Patterns

```lum
// Pattern at ONNX level
pattern ONNXMatMulAdd at level graph {
    mm: onnx.MatMul
    add: onnx.Add
    where mm.result -> add.input[0]
}

// Pattern at Linalg level
pattern LinalgMatMulBias at level tensor {
    mm: linalg.matmul
    bias: linalg.generic
    where mm.result -> bias.input[0]
}

// Pattern at loop level
pattern TriplyNestedLoop at level loop {
    outer: scf.for
    middle: scf.for
    inner: scf.for
    where outer.body contains middle
      and middle.body contains inner
}

// Pattern at vector level
pattern VectorFMA at level vector {
    mul: vector.fma | (vector.mulf followed_by vector.addf)
}
```

### 8.3 Cross-Level Transformations

```lum
// Raise: go up in abstraction
transform RaiseToLinalg {
    at level loop {
        match TriplyNestedLoop
        where inner.body is matmul_body
    }

    raise to level tensor {
        emit linalg.matmul(...)
    }
}

// Lower: go down in abstraction
transform LowerToLoops {
    at level tensor {
        match LinalgMatMulBias
    }

    lower to level loop {
        // Use standard linalg-to-loops lowering
        apply mlir::linalg::lowerToLoops
    }
}

// Mixed-level transformation
transform FuseAcrossLevels {
    // Match at tensor level
    at level tensor {
        mm: linalg.matmul
        bias: linalg.generic
    }

    // Lower to loops
    lower mm to level loop -> mm_loops

    // Fuse at loop level
    at level loop {
        fuse bias into mm_loops.outer
    }

    // Vectorize
    lower to level vector
    vectorize mm_loops.inner [N=16]
}
```

### 8.4 Dialect Bridges

```lum
// Define how to translate between dialects
bridge onnx_to_linalg {
    onnx.MatMul(A, B) => linalg.matmul(A, B, init_zero())
    onnx.Add(X, Y) => linalg.generic { ^bb: add } (X, Y, init)
    onnx.Relu(X) => linalg.generic { ^bb: max(0, x) } (X, init)

    // Complex lowering
    onnx.Softmax(X) => {
        max_val = linalg.reduce [max] (X)
        exp_val = linalg.generic [exp(x - max)] (X, max_val)
        sum_val = linalg.reduce [add] (exp_val)
        result = linalg.generic [x / sum] (exp_val, sum_val)
    }
}

bridge linalg_to_vector {
    // Use MLIR's standard lowering
    use mlir::linalg::vectorize

    // Custom overrides
    linalg.matmul where dtype == i8 and target.has_vnni => {
        // VNNI-specific vectorization
        pack B for vnni
        emit vector.contract with vnni_combiner
    }
}
```

---

## 9. Pipeline Orchestration

### 9.1 Pipeline Definition

```lum
pipeline OptimizeTransformer {
    // Input: ONNX or Linalg IR
    input: module at level tensor

    // Phase 1: Graph-level optimizations
    phase graph_opt {
        match ONNXMatMulAdd
        apply FuseToGemm

        match AttentionPattern
        apply FuseAttention
    }

    // Phase 2: Tensor-level scheduling
    phase tensor_schedule {
        match all linalg.matmul
        apply TileForCache using cost X86Skylake

        match all linalg.generic [elementwise]
        apply FuseWithProducer
    }

    // Phase 3: Loop-level optimization
    phase loop_opt {
        lower to level loop

        match TriplyNestedLoop
        apply Interchange for locality
        apply Unroll inner factor=4
    }

    // Phase 4: Vectorization
    phase vectorize {
        lower to level vector

        match all vectorizable
        apply Vectorize width=512
    }

    // Phase 5: Final lowering
    phase finalize {
        lower to level llvm
        apply mlir::standard_passes
    }
}
```

### 9.2 Phase Control

```lum
pipeline ConditionalPipeline {
    input: module at level tensor

    phase analyze {
        // Analysis pass - no transformation
        let model_size = count_ops(input)
        let has_attention = exists AttentionPattern
        let target_is_gpu = config.target == "gpu"
    }

    phase optimize {
        if target_is_gpu {
            apply GPUSchedule
        } else if model_size > 1000 {
            apply LargeModelSchedule
        } else {
            apply SmallModelSchedule
        }

        when has_attention {
            match AttentionPattern
            apply FlashAttention
        }
    }
}
```

### 9.3 Pipeline Composition

```lum
// Reusable sub-pipelines
pipeline TileAndVectorize {
    param tile_sizes: [int, int, int]
    param vector_width: int

    phase tile {
        match all linalg.matmul
        apply TileMatMul [M=tile_sizes.0, N=tile_sizes.1, K=tile_sizes.2]
    }

    phase vectorize {
        match all tiled_op
        apply Vectorize width=vector_width
    }
}

// Compose pipelines
pipeline FullOptimization {
    // Run sub-pipeline with parameters
    run TileAndVectorize {
        tile_sizes: [64, 64, 32]
        vector_width: 512
    }

    // Then additional phases
    phase fuse {
        match FusionCandidate
        apply FuseElementwise
    }
}

// Conditional pipeline selection
pipeline AdaptivePipeline {
    let hw = detect_hardware()

    match hw {
        X86 with avx512 => run X86_AVX512_Pipeline
        X86 with avx2 => run X86_AVX2_Pipeline
        ARM with neon => run ARM_NEON_Pipeline
        GPU nvidia => run CUDA_Pipeline
        _ => run GenericPipeline
    }
}
```

### 9.4 Error Handling

```lum
pipeline RobustPipeline {
    phase risky_opt {
        try {
            match ComplexPattern
            apply AggressiveTransform
        } catch TransformFailed {
            // Fallback to simpler approach
            apply ConservativeTransform
        }
    }

    phase must_succeed {
        match RequiredPattern
        apply RequiredTransform
        or fail "Critical transformation failed"
    }

    // Validation
    phase validate {
        assert module.valid
        assert no_op linalg.* remaining   // All should be lowered
    }
}
```

---

## 10. Extension Points

### 10.1 Custom Operations

```lum
// Define custom operation for matching
define op simp.vnni_matmul {
    inputs: [A: tensor<i8>, B: tensor<i8>, C: tensor<i32>]
    outputs: [result: tensor<i32>]
    attributes: {
        layout: string
    }

    // Semantics for cost model
    semantics {
        flops: 2 * M * N * K
        memory_reads: M * K + K * N
        memory_writes: M * N
    }
}

// Pattern using custom op
pattern VNNIMatMul {
    vnni: simp.vnni_matmul
}
```

### 10.2 Custom Constraints

```lum
// Define reusable constraint
constraint AlignedForVectorization(shape, dim, width) {
    shape[dim] % width == 0
}

constraint FitsInCache(sizes, cache_level) {
    let total = sum(sizes.map(s => s * 4))  // Assume f32
    total <= cache_level.size
}

// Use in pattern
pattern VectorFriendly {
    mm: linalg.matmul
    where AlignedForVectorization(mm.output.shape, 1, 16)
      and FitsInCache([mm.M * mm.K, mm.K * mm.N], L1)
}
```

### 10.3 Custom Schedules

```lum
// Parameterized schedule template
template schedule TiledGemm<M, N, K, VEC> for MatMulPattern {
    tile mm [M=M, N=N, K=K]
    vectorize inner [N=VEC]
    unroll k_loop factor=4
}

// Instantiate
schedule MyGemm = TiledGemm<64, 64, 32, 16>

// Generic schedule with type constraints
schedule GenericTile<T: Operation> for T {
    require T has_trait Tileable

    let dims = T.iteration_dims
    tile T dims.map(d => min(32, d))
}
```

### 10.4 Plugins

```lum
// Plugin interface
plugin interface CostModelPlugin {
    fn estimate_cycles(op: Operation) -> Cycles
    fn estimate_memory(op: Operation) -> Bytes
    fn tile_recommendation(op: Operation) -> TileConfig
}

// Register plugin
register plugin MyCostModel implements CostModelPlugin {
    // Implementation provided externally (C++/Python)
    external "libmycost.so"
}

// Use plugin
schedule PluginDriven for MatMulPattern {
    let tiles = plugin MyCostModel.tile_recommendation(mm)
    tile mm tiles
}
```

### 10.5 External Integration

```lum
// Call external C++ pass
external pass mlir::createLinalgFusionPass

// Call Python function for analysis
external python "analyze.py" {
    fn compute_optimal_tiles(op) -> TileConfig
}

// Integrate with auto-tuner
external tuner "autotvm" {
    fn tune(schedule, input_shapes) -> Parameters
}

// Use in schedule
schedule AutoTunedSchedule for MatMulPattern {
    let params = external tuner.tune(this, mm.shapes)
    tile mm [M=params.tile_m, N=params.tile_n, K=params.tile_k]
}
```

---

## 11. Standard Library

### 11.1 std.patterns

```lum
import std.patterns

// Provides:
// - std.patterns.gemm.MatMul, MatMulBias, MatMulBiasReLU
// - std.patterns.conv.Conv2D, Conv2DBNReLU, DepthwiseConv
// - std.patterns.attention.SelfAttention, MultiHeadAttention
// - std.patterns.norm.LayerNorm, BatchNorm, RMSNorm
// - std.patterns.activation.ReLU, GELU, SiLU, Softmax
```

### 11.2 std.schedules

```lum
import std.schedules

// Provides:
// - std.schedules.cache_aware.L1Tiled, L2Tiled, CacheOblivious
// - std.schedules.parallel.OpenMP, ThreadPool
// - std.schedules.vector.AVX2, AVX512, VNNI, NEON
// - std.schedules.gpu.CUDA, TensorCore
```

### 11.3 std.cost_models

```lum
import std.cost_models

// Provides:
// - std.cost_models.x86.Skylake, CascadeLake, IceLake
// - std.cost_models.arm.CortexA76, AppleM1
// - std.cost_models.nvidia.A100, H100
// - std.cost_models.generic.RooflineModel
```

### 11.4 std.pipelines

```lum
import std.pipelines

// Provides:
// - std.pipelines.inference.CPUInference, GPUInference
// - std.pipelines.training.DataParallel, ModelParallel
// - std.pipelines.mobile.TFLiteCompat, CoreMLCompat
```

---

## 12. Compilation Target

### 12.1 Transform Dialect Output

```lum
// Lum source
schedule TileMatMul for MatMulPattern {
    tile mm [M=64, N=64, K=32] -> (m_loop, n_loop, k_loop)
    fuse bias into m_loop
    vectorize mm.inner [N=16]
}

// Compiles to MLIR Transform Dialect:
```

```mlir
transform.named_sequence @TileMatMul(%arg0: !transform.any_op) {
  // Pattern matching
  %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0
  %bias = transform.structured.match ops{["linalg.generic"]} in %arg0

  // Tiling
  %tiled, %m_loop, %n_loop, %k_loop =
      transform.structured.tile_using_for %mm tile_sizes [64, 64, 32]

  // Fusion
  %fused = transform.structured.fuse_into_containing_op %bias into %m_loop

  // Vectorization
  transform.structured.vectorize %tiled vector_sizes [1, 16, 4]

  transform.yield
}
```

### 12.2 Standalone Executable

```bash
# Compile Lum to standalone optimizer
lumc --emit-pass my_opt.lum -o my_opt_pass.so

# Use with mlir-opt
mlir-opt input.mlir --load-pass-plugin=my_opt_pass.so --apply-my-opt
```

### 12.3 Library Integration

```cpp
// C++ API
#include "lum/runtime.h"

lum::Pipeline pipeline = lum::compile("my_pipeline.lum");
mlir::ModuleOp optimized = pipeline.apply(input_module);

// Python API
import lum

pipeline = lum.compile("my_pipeline.lum")
optimized = pipeline.apply(input_module)
```

---

## 13. Examples

### 13.1 Complete GEMM Optimization

```lum
module gemm_opt

import std.patterns.gemm
import std.cost_models.x86

target x86_64 {
    features: [avx512, vnni]
}

pattern GemmBiasReLU extends std.patterns.gemm.MatMulBias {
    relu: linalg.generic [relu]
    where bias.result -> relu.input[0]
}

cost model MyCost extends std.cost_models.x86.IceLake {
    // Override with measured values
    flops_per_cycle: 64  // With 2x FMA units
}

schedule OptimizedGemm for GemmBiasReLU {
    // Derive optimal tiles
    tile mm derive [M, N, K] using cost MyCost {
        minimize cycles
        subject to {
            working_set <= L2.size * 0.8
            M % 8 == 0 and N % 16 == 0 and K % 4 == 0
        }
    }

    // Multi-level tiling
    tile mm [M=derived.M, N=derived.N, K=derived.K] as l2_tile
    tile l2_tile [M=8, N=16, K=4] as l1_tile

    // Fuse pointwise ops
    fuse [bias, relu] into l2_tile.m_loop ordered

    // Vectorize micro-kernel
    vectorize l1_tile [N=16] for avx512

    // Unroll for ILP
    unroll l1_tile.k_loop factor=4

    // Parallelize outer loops
    parallel l2_tile.m_loop
}

pipeline OptimizeAllGemms {
    phase find_and_optimize {
        match all GemmBiasReLU
        apply OptimizedGemm
    }

    phase lower {
        lower to level llvm
    }
}
```

### 13.2 Flash Attention

```lum
module flash_attention

import std.patterns.attention

pattern FlashAttentionCandidate {
    q: tensor<B, H, S, D>
    k: tensor<B, H, S, D>
    v: tensor<B, H, S, D>

    qk: linalg.batch_matmul(q, k.transpose)
    scale: linalg.generic [div sqrt_d]
    mask: linalg.generic [add mask] optional
    softmax: linalg.softmax
    out: linalg.batch_matmul(softmax, v)

    where q.H == k.H == v.H  // Same number of heads
}

schedule FlashAttention for FlashAttentionCandidate {
    // Block over sequence length
    tile qk [B=1, H=1, S_q=64, S_k=64] -> (batch_loop, head_loop, q_loop, k_loop)

    // Online softmax: maintain running max and sum
    transform softmax to online {
        state: [max_so_far, sum_so_far, output_accumulator]
        update: {
            new_max = max(max_so_far, block_max)
            correction = exp(max_so_far - new_max)
            sum_so_far = sum_so_far * correction + block_sum * exp(block_max - new_max)
            output_accumulator = output_accumulator * correction + block_output
        }
    }

    // Fuse everything into tiled loop
    fuse [scale, mask, softmax.online, out] into k_loop

    // Use shared memory for Q, K, V blocks (GPU)
    when target.is_gpu {
        promote q_block to gpu.shared
        promote k_block to gpu.shared
        promote v_block to gpu.shared
    }
}
```

### 13.3 INT8 Quantized Inference

```lum
module int8_inference

pattern QuantizedMatMul {
    mm: linalg.matmul
    where mm.input[0].dtype == i8
      and mm.input[1].dtype == i8
      and mm.result.dtype == i32
      and mm.K % 4 == 0  // VNNI alignment
}

pattern QuantizedMatMulWithScale {
    mm: QuantizedMatMul
    scale: linalg.generic [mul_scale]
    requant: linalg.generic [clamp_and_cast_to_i8] optional
    where mm.result -> scale.input[0]
}

schedule VNNIQuantized for QuantizedMatMulWithScale
    when target.has_feature("vnni") {

    // Pack B for VNNI layout: [K, N] -> [K/4, N, 4]
    pack mm.input[1] {
        inner_dims: [0]
        inner_tiles: [4]
    }

    // Tile for cache
    tile mm [M=32, N=32, K=64] -> (m_loop, n_loop, k_loop)

    // VNNI vectorization (vpdpbusd)
    vectorize mm [M=1, N=16, K=4] using vnni {
        A: vector<4xi8>      // unsigned
        B: vector<64xi8>     // signed, packed [16, 4]
        C: vector<16xi32>    // accumulator
        intrinsic: x86.avx512.vpdpbusd
    }

    // Fuse scale into outer loop
    fuse scale into m_loop

    // Fuse requantization if present
    when requant.exists {
        fuse requant into m_loop
    }
}

pipeline Int8InferencePipeline {
    phase quantize {
        match all linalg.matmul where dtype == f32
        apply QuantizeDynamic { scheme: symmetric, dtype: i8 }
    }

    phase optimize {
        match all QuantizedMatMulWithScale
        apply VNNIQuantized
    }

    phase lower {
        lower to level llvm
    }
}
```

---

## Appendix A: Grammar (EBNF)

```ebnf
program = module_decl , { import } , { dialect_decl } , { target_decl } ,
          { definition } ;

module_decl = "module" , identifier ;

import = "import" , qualified_id ;

dialect_decl = "dialect" , identifier , { "," , identifier } ;

target_decl = "target" , identifier , "{" , target_body , "}" ;

definition = pattern_def | schedule_def | cost_def | pipeline_def
           | template_def | plugin_def ;

pattern_def = "pattern" , identifier , [ extends_clause ] ,
              [ at_level ] , "{" , pattern_body , "}" ;

schedule_def = "schedule" , identifier , "for" , identifier ,
               [ when_clause ] , "{" , schedule_body , "}" ;

cost_def = "cost" , "model" , identifier , [ extends_clause ] ,
           "{" , cost_body , "}" ;

pipeline_def = "pipeline" , identifier , "{" , pipeline_body , "}" ;

(* ... complete grammar continues ... *)
```

---

## Appendix B: Comparison with Alternatives

| Feature | Lum | PDLL | Transform Dialect | ByteInfer DSL |
|---------|-----|------|-------------------|---------------|
| Pattern matching | ✅ Rich | ✅ Good | ⚠️ Basic | ✅ Good |
| Multi-level | ✅ Native | ❌ Single | ⚠️ Manual | ❌ ONNX only |
| Scheduling | ✅ Full | ❌ None | ✅ Full | ❌ None |
| Cost model | ✅ Pluggable | ❌ None | ❌ None | ❌ None |
| Auto-derivation | ✅ Built-in | ❌ None | ❌ None | ❌ None |
| Readability | ✅ High | ⚠️ Medium | ❌ Low | ✅ High |
| MLIR native | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No |

---

*Lum Specification v0.1 — Draft for Review*
