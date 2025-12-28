# Lum Language Specification v0.2

> Minimal. Expressive. Verifiable.

## Design Goals

1. **1:1 Transform Dialect mapping** — Every Lum construct has a direct MLIR equivalent
2. **Minimal syntax** — Few keywords, consistent patterns, no redundancy
3. **Debug-first** — Built-in tracing, breakpoints, and IR inspection
4. **Verified correctness** — Mandatory accuracy gates before/after every transform

---

## 1. Core Syntax

### 1.1 Minimal Keyword Set

```
Transform Keywords (25):
  # Tiling & Layout
  tile, pack, peel, promote

  # Fusion
  fuse, fuse_chain, fuse_horizontal, fuse_reduction, fuse_elementwise

  # Vectorization & Parallelism
  vec, unroll, interchange, parallel, map

  # Memory & Sync
  coop, sync, prefetch

  # Control & Data
  let, if, for, derive

  # Verification & Debug
  check, trace, assert, break, snapshot, diff

Structural Keywords (9):
  pattern, schedule, pipeline, phase, gate, cost, register, apply, match
```

### 1.2 Basic Structure

```lum
# Comments start with #

# Schedule: the core unit
schedule Name(target_op) {
    # transformations
}

# Pattern: reusable matcher
pattern Name {
    # constraints
}

# Pipeline: composition
pipeline Name {
    # phases
}
```

### 1.3 Operator Mapping to Transform Dialect

| Lum | Transform Dialect | Notes |
|-----|-------------------|-------|
| `match ops{X}` | `transform.structured.match` | Direct 1:1 |
| `tile [M,N,K]` | `transform.structured.tile_using_for` | Direct 1:1 |
| `fuse A into L` | `transform.structured.fuse_into_containing_op` | Producer into loop |
| `vec [N]` | `transform.structured.vectorize` | Direct 1:1 |
| `parallel` | `transform.loop.forall` | Direct 1:1 |
| `unroll N` | `transform.loop.unroll` | Direct 1:1 |
| `interchange [i,j,k]` | `transform.structured.interchange` | Direct 1:1 |
| `pack` | `transform.structured.pack` | Direct 1:1 |
| `fuse_chain [ops]` | `transform.structured.fuse_elementwise_ops` | Element-wise chain |
| `fuse_horizontal [ops]` | Batching + `tile_using_forall` | Independent ops |
| `fuse_reduction` | Specialized reduction fusion | Keep in registers |
| `map X => gpu.block.y` | `mapping = [#gpu.block<y>]` | GPU/parallel mapping |
| `coop` | `transform.structured.gpu.map_copy_to_threads` | Cooperative load |
| `sync` | `gpu.barrier` / memory fence | Synchronization |
| `prefetch` | `memref.prefetch` | Memory prefetch |

---

## 2. SSA Semantics

Lum is SSA-compatible. Transforms **consume** handles and return new ones.

### 2.1 Implicit Rebinding (Default)

```lum
schedule Opt(linalg.matmul) {
    tile [64, 64, 32] => m, n, k   # mm rebound to tiled op
    fuse bias into m               # bias rebound to fused op
    vec [16]                       # uses current (rebound) mm
}
```

### 2.2 Explicit Binding

```lum
schedule Opt(linalg.matmul) {
    tile mm [64, 64, 32] => tiled, m, n, k   # mm consumed, tiled is new
    vec tiled [16]                            # must use tiled, not mm
}
```

### 2.3 Consumed Handle Error

```lum
tile mm [64, 64, 32] => tiled, m, n, k
vec mm [16]   # ERROR: mm consumed by tile
```

```
error[E0001]: use of consumed handle 'mm'
 --> schedule.lum:2:5
  |
1 |     tile mm [64, 64, 32] => tiled, m, n, k
  |          -- consumed here
2 |     vec mm [16]
  |         ^^ cannot use consumed handle
  |
help: use 'tiled' instead
```

### 2.4 Handle Types

| Type | Description | Example |
|------|-------------|---------|
| Op handle | Points to operation | `mm: linalg.matmul` |
| Loop handle | Points to loop | `m, n, k` from `tile => m, n, k` |
| Value handle | Points to SSA value | `mm.result` |

---

## 3. Pattern Matching

### 3.1 Basic Patterns

```lum
# Match single op
pattern Matmul {
    mm: linalg.matmul
}

# Match chain (-> means "feeds into")
pattern MatmulBias {
    mm: linalg.matmul
    bias: linalg.generic
    mm -> bias
}

# Match with constraints
pattern Int8Matmul {
    mm: linalg.matmul
    mm.dtype == i8
    mm.K % 4 == 0
}
```

### 3.2 Implicit Handle Naming

When a schedule targets a single op type, an implicit handle is created using these rules:

```lum
# Rule 1: Use the op's short name (after the dot)
schedule Opt(linalg.matmul) {
    # Implicit handle: mm (from "matmul" → "mm")
    tile [64, 64, 32]        # Operates on 'mm'
}

# Rule 2: Common abbreviations
linalg.matmul      → mm
linalg.generic     → gen
linalg.conv_2d     → conv
linalg.batch_matmul → bmm
linalg.reduce      → red
tensor.pack        → pack
tensor.unpack      → unpack

# Rule 3: For patterns, use explicit names
pattern MatmulBias {
    mm: linalg.matmul      # Explicit name 'mm'
    bias: linalg.generic   # Explicit name 'bias'
    mm -> bias
}
```

The implicit handle is **rebound** after each transform (see Section 2 SSA Semantics).

### 3.3 Constraint Grammar

Pattern constraints use a simple expression syntax:

```
constraint     := handle.property op value
               | handle.predicate
               | constraint '&&' constraint
               | constraint '||' constraint
               | '(' constraint ')'

property       := 'dtype' | 'shape' | 'rank' | dim_accessor
dim_accessor   := 'M' | 'N' | 'K' | 'shape[' index ']'

op             := '==' | '!=' | '<' | '<=' | '>' | '>=' | '%'

predicate      := 'is_' identifier   # e.g., is_broadcast_add
               | 'has_' identifier   # e.g., has_single_use

value          := literal | '?' (dynamic)
```

**Examples:**

```lum
pattern Int8Matmul {
    mm: linalg.matmul
    mm.dtype == i8           # Type constraint
    mm.K % 4 == 0            # Divisibility constraint
}

pattern DynamicBatch {
    mm: linalg.matmul
    mm.shape[0] == ?         # Dynamic dimension
    mm.rank == 2             # Rank constraint
}

pattern FusableAdd {
    add: linalg.generic
    add.is_broadcast_add     # Predicate (no == needed)
    add.has_single_use       # Only one consumer
}
```

### 3.4 Compiled Output

```lum
pattern MatmulBias {
    mm: linalg.matmul
    bias: linalg.generic
    mm -> bias
}
```

Compiles to:

```mlir
%mm = transform.structured.match ops{["linalg.matmul"]} in %arg0
%bias = transform.structured.match ops{["linalg.generic"]} in %arg0
# Connection verified via use-def chain analysis
```

---

## 4. Scheduling

### 4.1 Tiling

```lum
schedule TileMatmul(linalg.matmul) {
    tile [64, 64, 32]              # M, N, K
}

# Named loops
schedule TileMatmul(linalg.matmul) {
    tile [64, 64, 32] => m, n, k   # Capture loop handles
    unroll k 4
}

# Multi-level
schedule TwoLevel(linalg.matmul) {
    tile [64, 64, 32] => L2        # L2 tile
    tile L2 [8, 8, 4] => L1        # L1 micro-kernel
}
```

Compiles to:

```mlir
transform.named_sequence @TileMatmul(%arg0: !transform.any_op) {
  %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0
  %tiled, %m, %n, %k = transform.structured.tile_using_for %mm
      tile_sizes [64, 64, 32]
  transform.loop.unroll %k { factor = 4 }
  transform.yield
}
```

### 4.2 Fusion

Fusion eliminates intermediate memory traffic by combining operations.

#### 4.2.1 Producer-Consumer Fusion

```lum
# Fuse single producer into consumer's loop
schedule BasicFusion(MatmulBias) {
    tile mm [64, 64, 32] => m, n, k
    fuse bias into m                 # bias computed inside m loop
}

# Fuse chain (order preserved)
schedule ChainFusion(MatmulBiasRelu) {
    tile mm [64, 64, 32] => m, n, k
    fuse [bias, relu] into m         # bias first, then relu
}

# Fuse at different loop levels
schedule MultiLevelFusion(MatmulBiasRelu) {
    tile mm [64, 64, 32] => m, n, k
    fuse bias into m                 # Per M-tile
    fuse relu into n                 # Per output element
}
```

#### 4.2.2 Element-wise Chain Fusion

```lum
# GELU: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
pattern GELU {
    pow: linalg.generic [pow3]
    mul1: linalg.generic [mul]
    add1: linalg.generic [add]
    mul2: linalg.generic [mul]
    tanh: linalg.generic [tanh]
    add2: linalg.generic [add]
    mul3: linalg.generic [mul]

    pow -> mul1 -> add1 -> mul2 -> tanh -> add2 -> mul3
}

schedule FuseGELU(GELU) {
    # Fuse entire element-wise chain into single kernel
    fuse_chain [pow, mul1, add1, mul2, tanh, add2, mul3]
    vec [16]
}

# Auto-fuse all element-wise ops in matched pattern
schedule AutoFuseElementwise(SomePattern) {
    fuse_elementwise *               # Fuse all element-wise ops
}
```

#### 4.2.3 Horizontal Fusion (Independent Ops)

```lum
# Q, K, V projections share input - can be batched
pattern QKV {
    input: tensor
    q_proj: linalg.matmul
    k_proj: linalg.matmul
    v_proj: linalg.matmul

    input -> q_proj
    input -> k_proj
    input -> v_proj
}

schedule FuseQKV(QKV) {
    # Batch independent matmuls into single op
    fuse_horizontal [q_proj, k_proj, v_proj] => qkv_batched

    # Schedule the batched op
    tile qkv_batched [64, 192, 32]   # N dimension is 3x
    vec [16]
}

# Horizontal fusion with explicit batching dimension
schedule FuseParallelConvs(ParallelConvs) {
    fuse_horizontal [conv1, conv2, conv3] batch_dim=0
}
```

#### 4.2.4 Reduction Fusion

```lum
# Softmax: exp(x - max(x)) / sum(exp(x - max(x)))
pattern Softmax {
    max_reduce: linalg.reduce [max]
    sub: linalg.generic [sub]
    exp: linalg.generic [exp]
    sum_reduce: linalg.reduce [add]
    div: linalg.generic [div]

    max_reduce -> sub -> exp -> sum_reduce -> div
}

schedule FuseSoftmax(Softmax) {
    # Fuse reduction with its element-wise consumers
    fuse_reduction max_reduce with [sub, exp]
    fuse_reduction sum_reduce with [div]

    vec [16]
}

# RMSNorm: x * rsqrt(mean(x²) + eps) * weight
pattern RMSNorm {
    sq: linalg.generic [square]
    mean: linalg.reduce [add]
    rsqrt: linalg.generic [rsqrt]
    norm: linalg.generic [mul]
    scale: linalg.generic [mul]
}

schedule FuseRMSNorm(RMSNorm) {
    # Tile outer dims, fuse reduction + normalization
    tile sq [1, -1] => batch, hidden    # Tile batch, full hidden

    fuse_reduction mean with [rsqrt]    # Keep partial sum in registers
    fuse [norm, scale] into batch       # Fuse scaling

    vec [16]
}
```

#### 4.2.5 Nested Producer Fusion

```lum
# Fuse im2col into matmul for conv2d
schedule FuseIm2Col(Conv2DDecomposed) {
    tile mm [64, 64, 32] => m, n, k

    # Fuse im2col (producer) into K loop
    # im2col computed on-demand per K-tile
    fuse im2col into k

    vec [16]
}
```

#### 4.2.6 Conditional Fusion

```lum
schedule SmartFusion(MatmulBias) {
    tile mm [64, 64, 32] => m, n, k

    # Only fuse if it saves memory traffic
    let fusion_benefit = cost.memory_saved(bias, m)
    let fusion_overhead = cost.register_pressure(bias)

    if fusion_benefit > fusion_overhead {
        fuse bias into m
    }
}
```

#### 4.2.7 Fusion Primitives Summary

| Primitive | Use Case | Semantics |
|-----------|----------|-----------|
| `fuse A into L` | Producer into loop | A computed inside L |
| `fuse [A,B,C] into L` | Chain into loop | Ordered, all inside L |
| `fuse_chain [ops]` | Element-wise chain | Single fused kernel |
| `fuse_elementwise *` | All element-wise | Auto-fuse compatible ops |
| `fuse_horizontal [ops]` | Independent ops | Batch into single op |
| `fuse_reduction R with [ops]` | Reduction + consumers | Keep reduction in registers |
| `fuse A after B` | Post-op fusion | A executes after B completes |

#### 4.2.8 Fusion Constraints

```lum
schedule ConstrainedFusion(Pattern) {
    tile mm [64, 64, 32] => m, n, k

    # Fusion must preserve semantics
    fuse bias into m {
        # Constraints checked by Lum
        assert bias.uses == 1           # Single consumer only
        assert bias.deps subset m.scope # All deps available in m
    }
}
```

### 4.3 Vectorization

```lum
schedule Vectorize(linalg.matmul) {
    tile [1, 16, 4] => inner
    vec inner [16, 4]                # N=16, K=4
}

# VNNI-specific
schedule VNNI(linalg.matmul) {
    tile [1, 16, 4]
    vec [16, 4] vnni                 # Use vpdpbusd
}
```

### 4.4 Data Layout

```lum
schedule PackB(linalg.matmul) {
    pack mm.B [K, N] => [K/4, N, 4]  # VNNI layout
    tile [32, 32, 64]
}
```

### 4.5 Parallelization

```lum
schedule Parallel(linalg.matmul) {
    tile [64, 64, 32] => m, n, k
    parallel m                       # OpenMP
    parallel n simd                  # SIMD parallel
}
```

### 4.6 Cooperative Tiling & Synchronization

```lum
# GPU cooperative tiling
schedule GPUCoop(linalg.matmul) {
    # Block-level tile
    tile [128, 128, 32] => bm, bn, k
    map bm => gpu.block.y
    map bn => gpu.block.x

    # Thread-level tile
    tile [8, 8, 32] => tm, tn, _
    map tm => gpu.thread.y
    map tn => gpu.thread.x

    # Cooperative load into shared memory
    coop mm.A => shared_A [128, 32] {
        threads: [32, 4]
    }
    coop mm.B => shared_B [32, 128] {
        threads: [4, 32]
    }
    sync block                       # Barrier after load

    vec [8]
}

# CPU cooperative (thread pool sharing L3)
schedule CPUCoop(linalg.matmul) {
    tile [512, 512, 256] => l3m, l3n, l3k
    coop [l3m, l3n] threads=num_cores

    tile [64, 64, 64] => l2m, l2n, l2k
    vec [16]
}
```

### 4.7 Prefetching

```lum
schedule WithPrefetch(linalg.matmul) {
    tile [64, 64, 32] => m, n, k

    # Prefetch next tile while computing current
    prefetch mm.A distance=2 locality=L1
    prefetch mm.B distance=1 locality=L2
}
```

### 4.8 Synchronization Primitives

| Primitive | Scope | Use Case |
|-----------|-------|----------|
| `sync block` | GPU block | After shared memory load |
| `sync warp` | GPU warp | Warp-level reduction |
| `sync threads` | CPU thread pool | Barrier across threads |
| `sync memory` | All | Memory fence |

---

## 5. Custom Ops & Blaze Interop

Lum works on **any MLIR operation**, not just built-in ops. This enables scheduling of Blaze-defined kernels.

### 5.1 Interface-Based Matching

```lum
# Match ANY op with TilingInterface
pattern AnyTileable {
    op: * [TilingInterface]
}

# Match by trait
pattern AnyElementwise {
    op: * [Elementwise]
}

# Match by dialect
pattern AnySimpOp {
    op: simp.*
}

# Match by multiple interfaces
pattern TileableAndVectorizable {
    op: * [TilingInterface, VectorizableOpInterface]
}
```

### 5.2 Blaze Kernel → Lum Schedule

**Step 1: Define kernel in Blaze**

```blaze
# my_gemm.blaze
@kernel
fn my_gemm(A: tensor<M, K, f32>, B: tensor<K, N, f32>) -> tensor<M, N, f32> {
    var C = zeros<M, N, f32>()
    for i in 0..M {
        for j in 0..N {
            for k in 0..K {
                C[i, j] += A[i, k] * B[k, j]
            }
        }
    }
    return C
}
```

**Step 2: Blaze compiles to MLIR with simp dialect**

```mlir
// Auto-generated, implements TilingInterface
simp.kernel @my_gemm(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>) -> tensor<?x?xf32>
    attributes { tiling_dims = [0, 1, 2], parallel_dims = [0, 1] }
{
    // ... lowered IR
}
```

**Step 3: Schedule in Lum**

```lum
pattern MyGemm {
    k: simp.kernel ["my_gemm"]
}

schedule OptMyGemm(MyGemm) {
    tile k [64, 64, 32]
    vec [16]
    check accuracy
}
```

### 5.3 Minimal Annotations for Custom Ops

Blaze auto-derives interfaces from kernel structure. Explicit hints when needed:

```blaze
@kernel
@tile_dims(0, 1, 2)        # Which dims are tileable
@parallel_dims(0, 1)       # Which dims are parallel (no reduction)
@vector_dim(1)             # Which dim to vectorize
fn my_gemm(...) { ... }
```

**That's it.** No C++ interface implementation needed.

### 5.4 Interface Auto-Derivation Rules

| Kernel Pattern | Auto-Derived Interface |
|----------------|------------------------|
| Nested loops over output dims | `TilingInterface` |
| Independent iterations | `ParallelLoopInterface` |
| Innermost contiguous access | `VectorizableOpInterface` |
| Reduction pattern | `ReductionOpInterface` |
| Element-wise body | `ElementwiseMappable` |

```blaze
# This kernel auto-implements TilingInterface + VectorizableOpInterface
@kernel
fn add(A: tensor<N, f32>, B: tensor<N, f32>) -> tensor<N, f32> {
    var C = zeros<N, f32>()
    for i in 0..N {           # Tileable, parallel, vectorizable
        C[i] = A[i] + B[i]
    }
    return C
}
```

### 5.5 Registering External Ops

For ops not from Blaze (external libraries, custom MLIR):

```lum
# Register external op with capabilities
register op "custom.my_op" {
    tileable: [0, 1, 2]
    parallel: [0, 1]
    vectorizable: 1
    reduction: 2
}

# Now it works in patterns
pattern CustomOp {
    c: custom.my_op
}

schedule OptCustom(CustomOp) {
    tile c [64, 64]
    vec [16]
}
```

### 5.6 Generic Schedules

```lum
# Works on ANY tileable op (built-in or custom)
schedule GenericTile(* [TilingInterface]) {
    let dims = op.tiling_dims
    let sizes = dims.map(d => min(64, op.dim_size(d)))

    tile op sizes

    if op.has_interface(VectorizableOpInterface) {
        vec [16]
    }

    check accuracy
}

# Apply to all tileable ops in module
pipeline TileEverything {
    phase tile_all {
        match * [TilingInterface]
        apply GenericTile
    }
}
```

### 5.7 Quick Reference: Making an Op Schedulable

| What You Want | What To Do |
|---------------|------------|
| Tile a Blaze kernel | Just write loops → auto-derived |
| Tile with specific dims | Add `@tile_dims(0, 1)` |
| Parallelize | Add `@parallel_dims(0)` or loops are independent |
| Vectorize | Add `@vector_dim(1)` or innermost is contiguous |
| External op | Use `register op` in Lum |
| Fully custom | Implement MLIR interfaces in C++ |

**90% of cases: Write normal Blaze code, it just works.**

---

## 6. Expressions and Derivation

### 6.1 Let Bindings

```lum
schedule Dynamic(linalg.matmul) {
    let M = mm.shape[0]
    let N = mm.shape[1]
    let tile_m = min(64, M)

    tile [tile_m, 64, 32]
}
```

### 6.2 Conditionals

```lum
schedule Adaptive(linalg.matmul) {
    let K = mm.shape[2]

    if K > 512 {
        tile [64, 64, 64]
        pipeline k 2
    } else {
        tile [64, 64, K]
    }
}
```

### 6.3 Cost-Driven Derivation

```lum
schedule AutoTile(linalg.matmul) {
    derive [M, N, K] {
        minimize cycles
        M * K + K * N <= 32KB        # L1 constraint
        M % 8 == 0
        N % 16 == 0
        K % 4 == 0                   # VNNI
    }

    tile [M, N, K]
}
```

---

## 7. Debugging

### 7.1 Trace Points

```lum
schedule Debug(linalg.matmul) {
    trace "before tiling"            # Print IR
    tile [64, 64, 32] => m, n, k
    trace "after tiling"

    trace m.bounds                   # Print loop bounds
    trace mm.shape                   # Print tensor shape
}
```

### 7.2 Breakpoints

```lum
schedule StepThrough(linalg.matmul) {
    tile [64, 64, 32]
    break                            # Pause, inspect IR

    fuse bias into m
    break "check fusion"             # Named breakpoint
}
```

### 7.3 Assertions

```lum
schedule Validated(linalg.matmul) {
    tile [64, 64, 32] => m, n, k

    assert m.trip_count > 0          # Runtime check
    assert k.is_innermost            # Structure check
    assert mm.is_tiled               # Transformation check
}
```

### 7.4 IR Diff

```lum
schedule ShowChanges(linalg.matmul) {
    snapshot before

    tile [64, 64, 32]
    fuse bias into m
    vec inner [16]

    diff before                      # Show what changed
}
```

### 7.5 Debug Mode Invocation

```bash
# Run with full tracing
lum --debug schedule.lum input.mlir

# Interactive stepping
lum --step schedule.lum input.mlir

# Dump IR at each stage
lum --dump-stages schedule.lum input.mlir -o stages/
```

---

## 8. Verification Gates

### 8.1 Mandatory Accuracy Checks

Every schedule has implicit verification. Explicit control:

```lum
schedule Verified(linalg.matmul) {
    # Automatic snapshot before any transform

    tile [64, 64, 32]
    check accuracy                   # Verify numerical equivalence

    fuse bias into m
    check accuracy                   # Verify again

    vec [16]
    check accuracy rtol=1e-5         # Custom tolerance
}
```

### 8.2 Gate Types

```lum
schedule FullyVerified(linalg.matmul) {
    tile [64, 64, 32]

    # Numerical accuracy (default: rtol=1e-6, atol=1e-6)
    check accuracy

    # Structural validity
    check valid                      # No dangling refs, valid types

    # Performance sanity
    check perf >= 0.9x baseline      # No major regression

    # Memory bounds
    check memory <= 2x baseline      # Memory usage
}
```

### 8.3 Test Inputs

```lum
schedule WithTestData(linalg.matmul) {
    # Define test inputs
    test {
        inputs: [
            random<f32>[128, 256],   # A
            random<f32>[256, 512],   # B
        ]
        runs: 10                     # Multiple runs for stability
    }

    tile [64, 64, 32]
    check accuracy
}
```

### 8.4 Failure Modes

```lum
schedule StrictMode(linalg.matmul) {
    tile [64, 64, 32]

    check accuracy {
        on_fail: abort               # Stop immediately (default)
        # on_fail: warn              # Continue with warning
        # on_fail: rollback          # Undo last transform
    }
}
```

### 8.5 Verification Report

```bash
$ lum --verify schedule.lum input.mlir

=== Verification Report ===
Step 1: tile [64,64,32]
  ✓ Structural validity: PASS
  ✓ Numerical accuracy: PASS (max_diff=2.3e-7, rtol=1e-6)

Step 2: fuse bias into m
  ✓ Structural validity: PASS
  ✓ Numerical accuracy: PASS (max_diff=2.3e-7, rtol=1e-6)

Step 3: vec [16]
  ✓ Structural validity: PASS
  ✗ Numerical accuracy: FAIL
    Expected: [[1.234, 2.345, ...]]
    Got:      [[1.235, 2.346, ...]]
    Max diff: 1.2e-3 (exceeds rtol=1e-6)

VERIFICATION FAILED at step 3
```

---

## 9. Pipeline Composition

### 9.1 Basic Pipeline

```lum
pipeline Optimize {
    # Phases run in order
    phase tile_all {
        match linalg.matmul
        apply TileMatmul
    }

    phase vectorize_all {
        match linalg.* [tiled]
        apply Vectorize
    }

    phase lower {
        lower to llvm
    }
}
```

### 9.2 Conditional Phases

```lum
pipeline Adaptive {
    let has_int8 = exists linalg.matmul [dtype=i8]

    if has_int8 {
        phase vnni {
            match linalg.matmul [dtype=i8]
            apply VNNISchedule
        }
    }

    phase fallback {
        match linalg.matmul [not tiled]
        apply DefaultTile
    }
}
```

### 9.3 Verification in Pipelines

```lum
pipeline Verified {
    phase transform {
        match linalg.matmul
        apply OptSchedule
    }

    # Gate between phases
    gate {
        check accuracy
        check valid
    }

    phase lower {
        lower to llvm
    }
}
```

---

## 10. Compilation Model

### 10.1 Invocation

```bash
# Basic: schedule + input → output
lum schedule.lum input.mlir -o output.mlir

# With verification
lum --verify schedule.lum input.mlir -o output.mlir

# Debug mode
lum --debug schedule.lum input.mlir -o output.mlir

# Emit Transform Dialect (for inspection)
lum --emit-transform schedule.lum -o schedule.transform.mlir

# Then use with mlir-opt
mlir-opt input.mlir --transform-interpreter=schedule.transform.mlir
```

### 10.2 File Structure

```
project/
├── schedules/
│   ├── gemm.lum           # GEMM optimizations
│   ├── conv.lum           # Convolution optimizations
│   └── attention.lum      # Attention optimizations
├── patterns/
│   └── common.lum         # Shared patterns
├── tests/
│   ├── test_gemm.mlir     # Test inputs
│   └── golden/            # Expected outputs
└── lum.toml               # Project config
```

### 10.3 Project Config

```toml
# lum.toml
[project]
name = "my-optimizer"
version = "0.1.0"

[target]
arch = "x86_64"
features = ["avx512", "vnni"]

[verification]
rtol = 1e-6
atol = 1e-6
runs = 10

[debug]
trace_level = "info"  # none, info, verbose
dump_ir = false
```

---

## 11. Examples

### 11.1 Basic GEMM with Bias

**Problem**: Fuse `C = A @ B + bias` into single tiled kernel.

```lum
pattern GemmBias {
    mm: linalg.matmul
    bias: linalg.generic
    mm -> bias
    bias.is_broadcast_add
}

schedule OptGemmBias(GemmBias) {
    # Tile for L2 cache
    tile mm [64, 64, 32] => m, n, k

    # Fuse bias into M loop (computed per M-tile)
    fuse bias into m

    # Micro-kernel vectorization
    tile [8, 16, 4] => micro
    vec micro [16]

    # Verify
    check accuracy
}
```

**Output Transform Dialect**:

```mlir
transform.named_sequence @OptGemmBias(%arg0: !transform.any_op) {
  %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0
  %bias = transform.structured.match ops{["linalg.generic"]} in %arg0

  %tiled, %m, %n, %k = transform.structured.tile_using_for %mm
      tile_sizes [64, 64, 32]
  %fused = transform.structured.fuse_into_containing_op %bias into %m

  %micro, %loops:3 = transform.structured.tile_using_for %tiled
      tile_sizes [8, 16, 4]
  transform.structured.vectorize %micro vector_sizes [1, 16, 4]

  transform.yield
}
```

---

### 11.2 INT8 VNNI Matmul

**Problem**: Optimize INT8 matmul for AVX-512 VNNI (vpdpbusd).

```lum
pattern Int8Matmul {
    mm: linalg.matmul
    mm.A.dtype == i8
    mm.B.dtype == i8
    mm.C.dtype == i32
    mm.K % 4 == 0              # VNNI requires K % 4 == 0
}

schedule VNNIOptimized(Int8Matmul) {
    # Pack B for VNNI: [K, N] → [K/4, N, 4]
    pack mm.B [0] tiles=[4]

    # Tile for cache
    tile mm [32, 32, 64] => m, n, k

    # VNNI micro-kernel: 1×16×4
    tile [1, 16, 4] => micro
    vec micro [16, 4] vnni

    # Verify (INT8 should be exact)
    check accuracy atol=0
}
```

---

### 11.3 Conv2D → Im2Col + Matmul

**Problem**: Decompose convolution and optimize the resulting matmul.

```lum
pattern Conv2D {
    conv: linalg.conv_2d_nhwc_hwcf
}

schedule OptConv2D(Conv2D) {
    # Decompose conv → im2col + matmul
    decompose conv {
        im2col: tensor.pack [input]
        mm: linalg.matmul [im2col, filter]
        reshape: tensor.expand_shape [mm]
    }

    # Now optimize the matmul
    tile mm [64, 64, 32] => m, n, k

    # Fuse im2col into matmul loops
    fuse im2col into k

    # Vectorize
    tile [8, 16, 4]
    vec [16, 4]

    check accuracy
}
```

---

### 11.4 Attention Block

**Problem**: Fuse Q×K^T → softmax → ×V attention pattern.

```lum
pattern Attention {
    q: tensor [B, H, S, D]
    k: tensor [B, H, S, D]
    v: tensor [B, H, S, D]

    qk: linalg.batch_matmul [q, k.T]
    scale: linalg.generic [div]
    softmax: linalg.softmax
    out: linalg.batch_matmul [softmax, v]

    qk -> scale -> softmax -> out
}

schedule OptAttention(Attention) {
    # Tile over batch and heads (independent)
    tile qk [1, 1, 64, 64] => b, h, sq, sk

    # Fuse scale and softmax into sequence tile
    fuse scale into sq
    fuse softmax into sq

    # Tile the output matmul to match
    tile out [1, 1, 64, 64]
    fuse out into sq

    # Vectorize
    vec [16]

    check accuracy
}
```

---

### 11.5 RMSNorm Fusion

**Problem**: Fuse RMSNorm components: `x * rsqrt(mean(x²) + eps) * weight`

```lum
pattern RMSNorm {
    x: tensor [*, D]
    sq: linalg.generic [square]
    mean: linalg.reduce [add]
    rsqrt: linalg.generic [rsqrt]
    norm: linalg.generic [mul]
    scale: linalg.generic [mul]

    x -> sq -> mean -> rsqrt
    x -> norm <- rsqrt
    norm -> scale
}

schedule OptRMSNorm(RMSNorm) {
    # Tile over non-reduced dimensions
    let outer_dims = x.shape[:-1]
    tile sq outer_dims => outer

    # Fuse entire chain
    fuse [mean, rsqrt, norm, scale] into outer

    # Vectorize the reduction
    vec [16]

    check accuracy rtol=1e-5  # Relaxed for rsqrt
}
```

---

### 11.6 Residual Block

**Problem**: Handle skip connections in residual blocks.

```lum
pattern Residual {
    input: tensor
    block: linalg.* [+]            # One or more ops
    add: linalg.generic [add]

    input -> block -> add
    input -> add                   # Skip connection
}

schedule OptResidual(Residual) {
    # Tile the block
    tile block.last [64, 64] => m, n

    # Fuse residual add
    fuse add into m

    # The skip connection is handled automatically
    # (add reads from both block output and original input)

    check accuracy
}
```

---

### 11.7 Dynamic Shape Handling

**Problem**: Handle unknown dimensions at compile time.

```lum
pattern DynamicMatmul {
    mm: linalg.matmul
    mm.M == ?                      # Dynamic
    mm.N == ?                      # Dynamic
    mm.K == ?                      # Dynamic
}

schedule DynamicOptimized(DynamicMatmul) {
    let M = mm.M
    let N = mm.N
    let K = mm.K

    # Clamp tile sizes to actual dimensions
    let tile_m = min(64, M)
    let tile_n = min(64, N)
    let tile_k = min(32, K)

    tile [tile_m, tile_n, tile_k] => m, n, k

    # Peel remainders
    peel m
    peel n

    vec [min(16, tile_n)]

    check accuracy
}
```

---

### 11.8 Multi-Level Tiling with Cost Model

**Problem**: Auto-derive optimal tile sizes for target hardware.

```lum
cost L2Cache {
    size: 512KB
    line: 64B
}

cost L1Cache {
    size: 32KB
    line: 64B
}

schedule CacheOptimized(linalg.matmul) {
    # Derive L2 tiles
    derive [M2, N2, K2] for L2Cache {
        # Working set: A_tile + B_tile + C_tile
        M2 * K2 * 4 + K2 * N2 * 4 + M2 * N2 * 4 <= L2Cache.size * 0.8
        M2 % 8 == 0
        N2 % 8 == 0
        maximize M2 * N2 * K2      # Maximize compute per tile
    }

    tile [M2, N2, K2] => l2

    # Derive L1 micro-kernel
    derive [M1, N1, K1] for L1Cache {
        M1 * K1 * 4 + K1 * N1 * 4 <= L1Cache.size * 0.5
        N1 % 16 == 0               # Vector width
        K1 % 4 == 0                # VNNI
    }

    tile l2 [M1, N1, K1] => l1
    vec l1 [N1]

    check accuracy
}
```

---

### 11.9 GPU Schedule

**Problem**: Map computation to GPU blocks and threads.

```lum
schedule GPUMatmul(linalg.matmul) {
    # Block-level tiling
    tile [128, 128, 32] => bm, bn, k
    map bm => gpu.block.y
    map bn => gpu.block.x

    # Thread-level tiling
    tile [8, 8, 32] => tm, tn, _
    map tm => gpu.thread.y
    map tn => gpu.thread.x

    # Shared memory for tiles
    promote mm.A to gpu.shared
    promote mm.B to gpu.shared
    sync

    # Vectorize
    vec [4]

    check accuracy
}
```

---

### 11.10 Full Transformer Layer

**Problem**: Optimize complete transformer layer with all fusions.

```lum
pattern TransformerLayer {
    # Input norm
    norm1: RMSNorm

    # Attention
    qkv_proj: linalg.matmul [3]    # Q, K, V projections
    attention: Attention
    out_proj: linalg.matmul

    # Residual
    add1: linalg.generic [add]

    # FFN
    norm2: RMSNorm
    up_proj: linalg.matmul
    gate_proj: linalg.matmul
    silu: linalg.generic [silu]
    down_proj: linalg.matmul

    # Residual
    add2: linalg.generic [add]
}

schedule OptTransformer(TransformerLayer) {
    # Norms
    apply OptRMSNorm to [norm1, norm2]

    # QKV projection (can be batched)
    fuse [qkv_proj] => qkv_batched
    apply VNNIOptimized to qkv_batched if dtype == i8
    apply OptGemmBias to qkv_batched if dtype == f32

    # Attention
    apply OptAttention to attention

    # Output projection with residual
    apply OptGemmBias to out_proj
    fuse add1 after out_proj

    # FFN with gating
    apply OptGemmBias to [up_proj, gate_proj]
    fuse silu after gate_proj
    fuse up_proj.result * gate_proj.result  # Element-wise
    apply OptGemmBias to down_proj
    fuse add2 after down_proj

    # Full layer verification
    check accuracy rtol=1e-4
    check perf >= 0.8x baseline
}
```

---

## 12. Q&A Analysis

### Q1: Is the language consistent with MLIR Transform Dialect?

**Answer: Yes, by design.**

Every Lum construct has a 1:1 mapping:

| Lum | Transform Dialect |
|-----|-------------------|
| `match X` | `transform.structured.match ops{["X"]}` |
| `tile [a,b,c]` | `transform.structured.tile_using_for ... tile_sizes [a,b,c]` |
| `tile ... => x,y,z` | Returns `%x, %y, %z` loop handles |
| `fuse A into B` | `transform.structured.fuse_into_containing_op %A into %B` |
| `vec [n]` | `transform.structured.vectorize ... vector_sizes [n]` |
| `unroll N` | `transform.loop.unroll ... { factor = N }` |
| `interchange [i,j,k]` | `transform.structured.interchange ... iterator_interchange = [i,j,k]` |
| `parallel` | `transform.loop.forall` or mapping attributes |
| `pack` | `transform.structured.pack` |
| `peel` | `transform.loop.peel` |

**Lum is syntax sugar over Transform Dialect**, not a new abstraction.

The `--emit-transform` flag outputs pure Transform Dialect MLIR that can be used directly with `mlir-opt --transform-interpreter`.

---

### Q2: Is the language expressive but minimal and intuitive?

**Answer: Designed to be.**

**Minimal:**
- 25 transform keywords + 9 structural keywords
- Consistent `verb target [params]` syntax
- No redundant constructs

**Expressive:**
- Patterns with constraints and dataflow
- Derived tile sizes with cost constraints
- Conditionals for adaptive schedules
- Composition via pipelines

**Intuitive:**
- Reads like English: "tile mm [64, 64, 32]"
- Arrow notation for dataflow: `mm -> bias`
- Natural loop naming: `=> m, n, k`

**Comparison:**

```lum
# Lum (4 lines)
schedule Opt(linalg.matmul) {
    tile [64, 64, 32] => m, n, k
    fuse bias into m
    vec [16]
}
```

```mlir
# Transform Dialect (12 lines)
transform.named_sequence @Opt(%arg0: !transform.any_op) {
  %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0
  %bias = transform.structured.match ops{["linalg.generic"]} in %arg0
  %tiled, %m, %n, %k = transform.structured.tile_using_for %mm
      tile_sizes [64, 64, 32]
  %fused = transform.structured.fuse_into_containing_op %bias into %m
  %micro, %loops:3 = transform.structured.tile_using_for %tiled
      tile_sizes [1, 16, 1]
  transform.structured.vectorize %micro vector_sizes [1, 16, 1]
  transform.yield
}
```

---

### Q3: Examples Coverage

| Problem | Example | Status |
|---------|---------|--------|
| GEMM + Bias fusion | 9.1 | ✅ |
| INT8 VNNI | 9.2 | ✅ |
| Conv2D decomposition | 9.3 | ✅ |
| Attention fusion | 9.4 | ✅ |
| RMSNorm fusion | 9.5 | ✅ |
| Residual/skip connections | 9.6 | ✅ |
| Dynamic shapes | 9.7 | ✅ |
| Cost-driven tiling | 9.8 | ✅ |
| GPU mapping | 9.9 | ✅ |
| Full transformer | 9.10 | ✅ |

**Additional examples to add:**
- Depthwise convolution
- Grouped convolution
- Softmax with online algorithm
- KV-cache handling
- Mixed precision (FP16 compute, FP32 accumulate)
- Batch normalization folding
- Quantization insertion

---

### Q4: Compilation Model

```
                    ┌─────────────┐
                    │ schedule.lum │
                    └──────┬──────┘
                           │
                           ▼
┌─────────────┐     ┌─────────────┐
│  input.mlir │────▶│     lum     │────▶ output.mlir
└─────────────┘     └─────────────┘
                           │
                    (optional)
                           ▼
                    ┌─────────────────────┐
                    │ schedule.transform.mlir │
                    └─────────────────────┘
```

**Usage:**

```bash
# Direct compilation
lum schedule.lum input.mlir -o output.mlir

# With verification
lum --verify schedule.lum input.mlir -o output.mlir

# Emit intermediate Transform Dialect
lum --emit-transform schedule.lum -o schedule.transform.mlir

# Use Transform Dialect with mlir-opt
mlir-opt input.mlir \
    --transform-interpreter=schedule.transform.mlir \
    -o output.mlir
```

---

### Q5: Debugging Support

| Feature | Syntax | Purpose |
|---------|--------|---------|
| Trace | `trace "msg"` | Print message + IR |
| Trace value | `trace expr` | Print expression value |
| Breakpoint | `break` | Pause execution |
| Named breakpoint | `break "name"` | Pause with label |
| Assertion | `assert cond` | Fail if condition false |
| Snapshot | `snapshot name` | Save IR state |
| Diff | `diff name` | Compare to snapshot |

**CLI Debug Options:**

```bash
lum --debug         # Enable all tracing
lum --step          # Interactive stepping
lum --dump-stages   # Save IR after each step
lum --trace=tile    # Trace only tile operations
lum --break=fusion  # Break at specific point
```

---

### Q6: Testing Gates with Accuracy Guarantees

**Non-negotiable: Every transformation is verified.**

```lum
schedule Verified(linalg.matmul) {
    # IMPLICIT: snapshot taken before first transform

    tile [64, 64, 32]
    check accuracy                   # REQUIRED after structural change

    fuse bias into m
    check accuracy                   # REQUIRED after fusion

    vec [16]
    check accuracy                   # REQUIRED after vectorization
}
```

**Verification Levels:**

| Level | Check | Default Tolerance |
|-------|-------|-------------------|
| `accuracy` | Numerical equivalence | rtol=1e-6, atol=1e-6 |
| `valid` | Structural validity | N/A |
| `perf` | Performance regression | >= 0.9x baseline |
| `memory` | Memory usage | <= 1.5x baseline |

**Failure Handling:**

```lum
check accuracy {
    rtol: 1e-6
    atol: 1e-6
    on_fail: abort          # abort | warn | rollback
}
```

**CI Integration:**

```bash
# Returns non-zero exit code on verification failure
lum --verify --strict schedule.lum input.mlir -o output.mlir

# Generate verification report
lum --verify --report=report.json schedule.lum input.mlir
```

**Report Format:**

```json
{
  "status": "PASS",
  "steps": [
    {
      "name": "tile [64,64,32]",
      "accuracy": {
        "status": "PASS",
        "max_diff": 2.3e-7,
        "rtol": 1e-6,
        "atol": 1e-6
      },
      "validity": "PASS"
    },
    ...
  ],
  "total_time_ms": 1234,
  "output_hash": "abc123..."
}
```

---

## 13. Implementation Roadmap

### Phase 1: Core Language (MVP)
- [ ] Lexer/Parser for minimal syntax
- [ ] Pattern matching (single op, chains, constraints)
- [ ] Basic scheduling (tile, fuse, vec)
- [ ] Transform Dialect emission
- [ ] Basic verification (accuracy check)

### Phase 2: Debugging
- [ ] trace/break/assert
- [ ] IR diff and snapshots
- [ ] Interactive stepping mode

### Phase 3: Advanced Features
- [ ] Cost model integration
- [ ] Tile size derivation
- [ ] Pipeline composition
- [ ] Multi-level patterns

### Phase 4: Production
- [ ] CI/CD integration
- [ ] Performance regression tracking
- [ ] Documentation generator
- [ ] IDE support (LSP)

---

## 14. Open Questions

1. **Pattern matching expressiveness**: Is the current syntax sufficient for all patterns, or do we need regex-like extensions?

2. **Error messages**: How to provide actionable error messages when patterns don't match or transforms fail?

3. **Incremental compilation**: Should Lum support incremental transformation (apply to subset of ops)?

4. **Interop with C++/Python**: How should custom cost models and constraints be defined in host languages?

5. **Caching**: Should verified transformations be cached for faster re-runs?

---

*Lum Specification v0.2 — Minimal, Expressive, Verified*
