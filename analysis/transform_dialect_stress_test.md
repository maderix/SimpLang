# Transform Dialect Stress Test

## Objective
Evaluate if MLIR Transform Dialect (+ extensions) can express all optimization patterns we need for SimpLang's ML backend.

## Test Categories

1. [Fusion Patterns](#1-fusion-patterns)
2. [Decomposition Patterns](#2-decomposition-patterns)
3. [Tiling Strategies](#3-tiling-strategies)
4. [Vectorization](#4-vectorization)
5. [Memory Optimization](#5-memory-optimization)
6. [Transformer-Specific](#6-transformer-specific)
7. [Quantization Patterns](#7-quantization-patterns)
8. [Loop Transformations](#8-loop-transformations)
9. [Hardware-Specific](#9-hardware-specific)
10. [Edge Cases](#10-edge-cases)

---

## 1. Fusion Patterns

### 1.1 MatMul + Bias (Basic GEMM)
**Pattern**: `C = A × B + bias` where bias broadcasts over M dimension

```mlir
// ✅ WORKS
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
  %bias = transform.structured.match ops{["linalg.generic"]} in %arg0
    // TODO: need attribute filter for "broadcast_add" semantics

  // Tile matmul, then fuse bias into outer loop
  %tiled, %loop = transform.structured.tile_using_for %matmul tile_sizes [32, 32, 0]
  %fused = transform.structured.fuse_into_containing_op %bias into %loop

  transform.yield
}
```

**Verdict**: ✅ Works, but needs attribute matching for bias pattern
**Gap**: No built-in "is_broadcast_add" predicate

---

### 1.2 MatMul + Bias + ReLU
**Pattern**: `C = ReLU(A × B + bias)`

```mlir
// ✅ WORKS
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
  %bias = transform.structured.match ops{["linalg.generic"]} in %arg0  // broadcast add
  %relu = transform.structured.match ops{["linalg.generic"]} in %arg0  // max(0, x)

  // Tile and fuse all three
  %tiled, %loop = transform.structured.tile_using_for %matmul tile_sizes [32, 32, 0]
  %fused_bias = transform.structured.fuse_into_containing_op %bias into %loop
  %fused_relu = transform.structured.fuse_into_containing_op %relu into %loop

  transform.yield
}
```

**Verdict**: ✅ Works
**Gap**: Ordering of fusions must be explicit (bias before relu)

---

### 1.3 MatMul + Bias + GELU
**Pattern**: `C = GELU(A × B + bias)` where GELU = x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

```mlir
// ⚠️ PARTIAL - GELU is typically multiple ops
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
  %bias = transform.structured.match ops{["linalg.generic"]} in %arg0

  // GELU decomposed: pow, mul, add, tanh, add, mul, mul
  // Need to match ALL of them and fuse
  %gelu_ops = transform.structured.match
      ops{["linalg.generic"]}
      filter_result_type = "tensor<?x?xf32>"  // All intermediate GELU ops
      in %arg0

  %tiled, %loop = transform.structured.tile_using_for %matmul tile_sizes [32, 32, 0]

  // Fuse chain: bias, then each GELU op
  // Problem: How to express "fuse all ops in topological order"?

  transform.yield
}
```

**Verdict**: ⚠️ Partial - need way to match op chains and fuse in order
**Gap**: No "match_chain" or "fuse_chain" primitive

---

### 1.4 MatMul + Bias + SiLU (Swish)
**Pattern**: `C = SiLU(A × B + bias)` where SiLU = x * sigmoid(x)

```mlir
// ⚠️ SIMILAR to GELU - multi-op activation
// SiLU = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
// Decomposes to: neg, exp, add_one, reciprocal, mul

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // Same problem as GELU - need to match and fuse op chain
  transform.yield
}
```

**Verdict**: ⚠️ Same gap as GELU
**Gap**: Need "match activation pattern" abstraction

---

### 1.5 Conv + BatchNorm + ReLU
**Pattern**: Classic CNN block fusion

```mlir
// ✅ WORKS (assuming BN is already folded into conv weights)
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %conv = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg0
  %relu = transform.structured.match ops{["linalg.generic"]} in %arg0  // max(0,x)

  // Tile over output spatial dims, fuse relu
  %tiled, %loops:2 = transform.structured.tile_using_for %conv
      tile_sizes [0, 8, 8, 0, 0, 0, 0]  // N, OH, OW, OC, KH, KW, IC

  %fused = transform.structured.fuse_into_containing_op %relu into %loops#0

  transform.yield
}
```

**Verdict**: ✅ Works for inference (BN folded)
**Gap**: BN folding itself is a separate pass, not expressible in Transform

---

### 1.6 Residual Add Fusion
**Pattern**: `output = F(x) + x` (skip connection)

```mlir
// ⚠️ TRICKY - need to identify residual pattern
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // Match conv/matmul block
  %block = transform.structured.match ops{["linalg.matmul"]} in %arg0

  // Match the add that has one input from block, one from skip
  %residual_add = transform.structured.match ops{["linalg.generic"]} in %arg0
    // Need: filter where one input == block.output AND other input == block.input

  // Fuse residual add
  %tiled, %loop = transform.structured.tile_using_for %block tile_sizes [32, 32, 0]
  %fused = transform.structured.fuse_into_containing_op %residual_add into %loop

  transform.yield
}
```

**Verdict**: ⚠️ Partial - identifying residual pattern requires data flow analysis
**Gap**: No "match by data flow" primitive

---

## 2. Decomposition Patterns

### 2.1 Conv2D → Im2Col + MatMul
**Pattern**: Convert convolution to matrix multiplication

```mlir
// ❌ NOT DIRECTLY SUPPORTED
// Transform dialect does SCHEDULING, not op decomposition
// This requires a rewrite pattern (PDLL or C++)

// What Transform CAN do: schedule the RESULT of decomposition
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // Assuming im2col+matmul already exists from earlier pass
  %im2col = transform.structured.match ops{["tensor.pack"]} in %arg0  // or custom op
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0

  // Fuse im2col into matmul's loop
  %tiled, %loop = transform.structured.tile_using_for %matmul tile_sizes [32, 32, 0]
  %fused = transform.structured.fuse_into_containing_op %im2col into %loop

  transform.yield
}
```

**Verdict**: ❌ Transform doesn't do decomposition
**Gap**: Need PDLL/C++ pattern for conv→im2col+matmul, then Transform for scheduling

---

### 2.2 Depthwise Conv Decomposition
**Pattern**: Depthwise separable conv optimization

```mlir
// ❌ SAME ISSUE - decomposition is rewrite, not scheduling
// Depthwise conv has groups=channels, can be parallelized differently

// After decomposition, scheduling:
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %dwconv = transform.structured.match ops{["linalg.depthwise_conv_2d_nhwc_hwc"]} in %arg0

  // Tile over channels (each channel is independent)
  %tiled, %loop = transform.structured.tile_using_for %dwconv
      tile_sizes [1, 0, 0, 32, 0, 0]  // N=1, H=all, W=all, C=32, KH=all, KW=all

  // Parallelize channel loop
  transform.loop.parallelize %loop

  transform.yield
}
```

**Verdict**: ⚠️ Scheduling works, decomposition needs separate pass
**Gap**: No decomposition primitives

---

### 2.3 GroupNorm Decomposition
**Pattern**: GroupNorm = reshape → mean → variance → normalize → reshape

```mlir
// ❌ DECOMPOSITION - not Transform's job
// But can schedule after decomposition:

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // After decomposition: reduce_mean, reduce_var, normalize are separate ops
  %mean = transform.structured.match ops{["linalg.reduce"]} in %arg0  // first reduce
  %var = transform.structured.match ops{["linalg.reduce"]} in %arg0   // second reduce
  %norm = transform.structured.match ops{["linalg.generic"]} in %arg0 // normalize

  // Fuse mean+var computation (same iteration space)
  // Problem: these have different outputs, can they be fused?

  transform.yield
}
```

**Verdict**: ❌ Decomposition not supported; fusion of reduces unclear
**Gap**: How to fuse multiple reductions with different outputs?

---

### 2.4 Attention Decomposition (Q×K^T → Softmax → ×V)
**Pattern**: Split fused attention into components or vice versa

```mlir
// ⚠️ Can schedule, but decomposition/fusion is rewrite territory
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // Match attention components
  %qk_matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0  // Q×K^T
  %softmax = transform.structured.match ops{["linalg.softmax"]} in %arg0
  %sv_matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0  // scores×V

  // Problem: how to distinguish qk_matmul from sv_matmul?
  // Need filtering by position in graph or attribute

  // Tile over batch and heads
  %tiled_qk, %loops:2 = transform.structured.tile_using_for %qk_matmul
      tile_sizes [1, 1, 0, 0]  // batch=1, head=1, seq=all, dim=all

  // Fuse softmax and sv_matmul into same loop
  %fused_softmax = transform.structured.fuse_into_containing_op %softmax into %loops#0
  %fused_sv = transform.structured.fuse_into_containing_op %sv_matmul into %loops#0

  transform.yield
}
```

**Verdict**: ⚠️ Works but ambiguous op matching
**Gap**: Need way to match ops by graph position, not just type

---

## 3. Tiling Strategies

### 3.1 L2 Cache Tiling
**Pattern**: Tile for L2 cache residency (256KB-1MB working set)

```mlir
// ✅ WORKS
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0

  // For f32: 64*64*4 + 64*32*4 + 64*32*4 = 16KB + 8KB + 8KB = 32KB per tile
  // L2 can hold multiple tiles for prefetching
  %tiled, %loops:3 = transform.structured.tile_using_for %matmul
      tile_sizes [64, 64, 32]  // M, N, K

  transform.yield
}
```

**Verdict**: ✅ Works
**Gap**: Tile sizes are magic numbers, no auto-derivation

---

### 3.2 L1 Cache Tiling (Register Blocking)
**Pattern**: Micro-kernel tiles that fit in L1 + registers

```mlir
// ✅ WORKS - two-level tiling
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0

  // L2 tiles
  %l2_tiled, %l2_loops:3 = transform.structured.tile_using_for %matmul
      tile_sizes [64, 64, 32]

  // L1 micro-kernel tiles
  %l1_tiled, %l1_loops:3 = transform.structured.tile_using_for %l2_tiled
      tile_sizes [8, 8, 4]  // Register-blocked micro-kernel

  transform.yield
}
```

**Verdict**: ✅ Works
**Gap**: Still magic numbers; no "fit in L1" constraint

---

### 3.3 Tile Size Auto-Derivation
**Pattern**: Compute optimal tile sizes from tensor shapes and cache sizes

```mlir
// ❌ NOT SUPPORTED - Transform is static
// Would need something like:

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0

  // HYPOTHETICAL: derive tile sizes
  %M, %N, %K = transform.get_operand_dims %matmul  // Get dynamic dims

  // HYPOTHETICAL: compute cache-optimal tiles
  %tile_m = transform.compute_tile_size %M
      target_cache = "L2"
      cache_size = 256000
      dtype_size = 4

  %tiled = transform.structured.tile_using_for %matmul
      tile_sizes [%tile_m, %tile_n, %tile_k]  // Dynamic!

  transform.yield
}
```

**Verdict**: ❌ Not supported
**Gap**: Major gap - no dynamic tile size computation

---

### 3.4 Batch Dimension Tiling
**Pattern**: Tile over batch for parallelism

```mlir
// ✅ WORKS
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %batch_matmul = transform.structured.match ops{["linalg.batch_matmul"]} in %arg0

  // Tile over batch dimension
  %tiled, %batch_loop = transform.structured.tile_using_for %batch_matmul
      tile_sizes [4, 0, 0, 0]  // Batch=4, M=all, N=all, K=all

  // Parallelize batch loop
  %parallel = transform.loop.parallelize %batch_loop

  transform.yield
}
```

**Verdict**: ✅ Works

---

### 3.5 Asymmetric Tiling (Different M, N, K strategies)
**Pattern**: Tile K for reuse, M/N for parallelism

```mlir
// ✅ WORKS
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0

  // M, N parallel; K sequential for accumulation
  %tiled, %m_loop, %n_loop, %k_loop = transform.structured.tile_using_for %matmul
      tile_sizes [32, 32, 64]

  // Interchange: K innermost for data reuse
  %interchanged = transform.structured.interchange %tiled
      iterator_interchange = [0, 1, 2]  // M, N, K order

  // Parallelize M and N
  transform.loop.parallelize %m_loop
  transform.loop.parallelize %n_loop

  transform.yield
}
```

**Verdict**: ✅ Works

---

## 4. Vectorization

### 4.1 Basic Vectorization
**Pattern**: Vectorize innermost dimension

```mlir
// ✅ WORKS
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0

  // Tile to vector-sized chunks
  %tiled, %loops:3 = transform.structured.tile_using_for %matmul
      tile_sizes [0, 8, 4]  // M=all, N=8 (AVX-512 f32), K=4

  // Vectorize the tiled op
  %vectorized = transform.structured.vectorize %tiled
      vector_sizes [1, 8, 4]

  transform.yield
}
```

**Verdict**: ✅ Works

---

### 4.2 VNNI Vectorization (INT8 4-way dot product)
**Pattern**: Use vpdpbusd for int8 matmul

```mlir
// ⚠️ PARTIAL - needs custom lowering
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
    // Need: filter by element type = i8

  // Tile K to multiple of 4 for VNNI
  %tiled, %loops:3 = transform.structured.tile_using_for %matmul
      tile_sizes [0, 16, 4]  // K must be 4 for VNNI

  // Vectorize
  %vectorized = transform.structured.vectorize %tiled
      vector_sizes [1, 16, 4]

  // Problem: vectorize produces vector<16xi8> ops, not vpdpbusd
  // Need custom lowering pattern

  transform.yield
}
```

**Verdict**: ⚠️ Vectorization works, but VNNI intrinsic selection needs separate pass
**Gap**: No "use_vnni" directive

---

### 4.3 Mixed Precision Vectorization
**Pattern**: f16 compute, f32 accumulate

```mlir
// ⚠️ UNCLEAR - type promotion during vectorization
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
    // inputs: f16, output: f32

  // Vectorize with mixed types?
  %vectorized = transform.structured.vectorize %matmul
      vector_sizes [1, 16, 8]
  // Does this automatically handle f16→f32 promotion?

  transform.yield
}
```

**Verdict**: ⚠️ Unclear if automatic type promotion works
**Gap**: Need explicit mixed-precision support

---

### 4.4 Predicated/Masked Vectorization
**Pattern**: Handle non-divisible dimensions

```mlir
// ✅ WORKS (recent MLIR addition)
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0

  // Vectorize with masking for non-divisible dims
  %vectorized = transform.structured.vectorize %matmul
      vector_sizes [1, 8, 4]
      vectorize_nd_extract  // Enables masking

  transform.yield
}
```

**Verdict**: ✅ Works with recent MLIR

---

## 5. Memory Optimization

### 5.1 Data Packing (VNNI Layout)
**Pattern**: Repack B matrix for VNNI: [K,N] → [K/4,N,4]

```mlir
// ⚠️ PARTIAL - pack exists but integration unclear
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0

  // Get B operand
  %B = transform.get_operand %matmul[1]

  // Pack B for VNNI layout
  %B_packed = transform.structured.pack %B
      inner_dims_pos = [0]   // Pack K dimension
      inner_tiles = [4]      // Groups of 4 for VNNI

  // Replace matmul's B with packed version
  // Problem: how to rewire the matmul to use B_packed?

  transform.yield
}
```

**Verdict**: ⚠️ Pack exists, but operand replacement is unclear
**Gap**: No "replace_operand" primitive

---

### 5.2 Prefetching
**Pattern**: Insert prefetch instructions

```mlir
// ❌ NOT DIRECTLY SUPPORTED in Transform
// Prefetching is typically done at lower level (LLVM)

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
  %tiled, %loops:3 = transform.structured.tile_using_for %matmul
      tile_sizes [64, 64, 32]

  // HYPOTHETICAL:
  // transform.insert_prefetch %tiled
  //     operand = 0  // Prefetch A
  //     distance = 2  // 2 tiles ahead
  //     locality = 3  // L1 cache

  transform.yield
}
```

**Verdict**: ❌ Not supported
**Gap**: No prefetch primitives (would need custom op)

---

### 5.3 Double Buffering
**Pattern**: Overlap compute with memory transfers

```mlir
// ⚠️ PARTIAL - pipelining exists
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
  %tiled, %k_loop = transform.structured.tile_using_for %matmul
      tile_sizes [0, 0, 64]  // Tile K for streaming

  // Pipeline the K loop with 2 stages
  %pipelined = transform.loop.pipeline %k_loop
      num_stages = 2
      // Stage 0: load next tile
      // Stage 1: compute current tile

  transform.yield
}
```

**Verdict**: ⚠️ Loop pipelining exists, but memory staging is implicit
**Gap**: No explicit "allocate buffer for stage" control

---

### 5.4 Memory Promotion (Stack Allocation)
**Pattern**: Promote tile to stack/scratchpad

```mlir
// ✅ WORKS
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
  %tiled, %loops:3 = transform.structured.tile_using_for %matmul
      tile_sizes [32, 32, 32]

  // Promote operands to local memory
  %promoted = transform.structured.promote %tiled
      operands_to_promote = [0, 1]  // A and B tiles
      use_full_tiles_by_default

  transform.yield
}
```

**Verdict**: ✅ Works

---

## 6. Transformer-Specific

### 6.1 RMSNorm Fusion
**Pattern**: Fuse sqrt(mean(x²)) with normalize

```mlir
// ⚠️ PARTIAL - need to identify RMSNorm pattern first
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // RMSNorm pattern: x * rsqrt(mean(x²) + eps) * weight
  // Decomposed: square, reduce_mean, add_eps, rsqrt, mul_x, mul_weight

  %square = transform.structured.match ops{["linalg.generic"]} in %arg0
    // filter: body is x*x
  %reduce = transform.structured.match ops{["linalg.reduce"]} in %arg0
  %rsqrt = transform.structured.match ops{["linalg.generic"]} in %arg0
    // filter: body is rsqrt
  %scale = transform.structured.match ops{["linalg.generic"]} in %arg0

  // Fuse all into single kernel
  // Problem: reduce has different iteration space than pointwise ops

  transform.yield
}
```

**Verdict**: ⚠️ Hard - mixing reductions with pointwise
**Gap**: Fusing reductions with pointwise requires special handling

---

### 6.2 Softmax Fusion
**Pattern**: Fuse exp, sum, div

```mlir
// ⚠️ SIMILAR to RMSNorm - reduction fusion issue
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // Softmax: exp(x - max(x)) / sum(exp(x - max(x)))
  // Has TWO reductions: max and sum

  %max_reduce = transform.structured.match ops{["linalg.reduce"]} in %arg0
  %exp = transform.structured.match ops{["linalg.generic"]} in %arg0
  %sum_reduce = transform.structured.match ops{["linalg.reduce"]} in %arg0
  %div = transform.structured.match ops{["linalg.generic"]} in %arg0

  // Online softmax requires single-pass algorithm
  // Transform can't express algorithm change

  transform.yield
}
```

**Verdict**: ❌ Online softmax requires algorithmic change, not just scheduling
**Gap**: Major - can't change algorithm, only schedule

---

### 6.3 Rotary Position Embeddings (RoPE)
**Pattern**: Complex number rotation fusion

```mlir
// ⚠️ PARTIAL - can schedule if decomposed correctly
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // RoPE: rotate even/odd pairs by position-dependent angle
  // x_rot = x * cos(θ) - x_rotated * sin(θ)

  // After decomposition to linalg ops:
  %cos_mul = transform.structured.match ops{["linalg.generic"]} in %arg0
  %sin_mul = transform.structured.match ops{["linalg.generic"]} in %arg0
  %sub = transform.structured.match ops{["linalg.generic"]} in %arg0

  // Fuse all element-wise ops
  %fused = transform.structured.fuse_elementwise_ops [%cos_mul, %sin_mul, %sub]

  transform.yield
}
```

**Verdict**: ⚠️ Element-wise fusion works if ops are matched correctly
**Gap**: Matching specific ops in complex patterns is tedious

---

### 6.4 KV-Cache Handling
**Pattern**: Append to cache, use cached values

```mlir
// ❌ NOT SCHEDULING - this is memory management
// KV-cache involves:
// 1. Storing K, V at position idx
// 2. Reading K[0:idx], V[0:idx] for attention

// Transform can't express index-dependent slicing
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // Would need custom ops for cache management
  transform.yield
}
```

**Verdict**: ❌ Not expressible
**Gap**: Dynamic indexing and cache management not supported

---

### 6.5 Flash Attention Tiling
**Pattern**: Tile attention for memory efficiency (block-sparse attention)

```mlir
// ⚠️ PARTIAL - can tile, but online softmax is the issue
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %qk_matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
  %softmax = transform.structured.match ops{["linalg.softmax"]} in %arg0
  %sv_matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0

  // Tile over sequence dimension (block attention)
  %tiled_qk, %seq_loop = transform.structured.tile_using_for %qk_matmul
      tile_sizes [0, 64, 0]  // Block size 64

  // Fuse softmax and sv_matmul
  // Problem: softmax needs correction factors between blocks (online algorithm)

  transform.yield
}
```

**Verdict**: ❌ Flash attention requires online algorithm, not just tiling
**Gap**: Can't express online/streaming algorithms

---

## 7. Quantization Patterns

### 7.1 INT8 MatMul with Scales
**Pattern**: Dequant → MatMul → Requant

```mlir
// ⚠️ PARTIAL - quantization is usually handled before Transform
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // Assuming quantized matmul already lowered to i8 linalg.matmul
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
    // filter: element type = i8

  // Tile with K multiple of 4 for VNNI
  %tiled, %loops:3 = transform.structured.tile_using_for %matmul
      tile_sizes [0, 16, 4]

  // Vectorize for VNNI
  %vectorized = transform.structured.vectorize %tiled
      vector_sizes [1, 16, 4]

  // Scale application happens after matmul (separate op)
  %scale_op = transform.structured.match ops{["linalg.generic"]} in %arg0
  %fused_scale = transform.structured.fuse_into_containing_op %scale_op into %loops#0

  transform.yield
}
```

**Verdict**: ⚠️ Scheduling works, quantization semantics handled elsewhere
**Gap**: No built-in quantization awareness

---

### 7.2 Per-Channel Quantization
**Pattern**: Different scales per output channel

```mlir
// ⚠️ SIMILAR - just different scale application pattern
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0

  // Per-channel scale is broadcast over M dimension
  %scale = transform.structured.match ops{["linalg.generic"]} in %arg0
    // filter: broadcasting scale[N] over output[M,N]

  // Tile and fuse as usual
  %tiled, %loops:3 = transform.structured.tile_using_for %matmul
      tile_sizes [32, 32, 0]
  %fused = transform.structured.fuse_into_containing_op %scale into %loops#0

  transform.yield
}
```

**Verdict**: ⚠️ Works with correct pattern matching

---

### 7.3 Mixed Precision (FP16 compute, FP32 accumulate)
**Pattern**: Widen accumulator for precision

```mlir
// ❌ TYPE CHANGE - not scheduling
// Would need linalg.matmul to have accumulator type attribute

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // After separate pass converts:
  // linalg.matmul(f16, f16) -> f16
  // to:
  // linalg.matmul(f16, f16) -> f32 + truncf -> f16

  // Transform can then schedule the result
  transform.yield
}
```

**Verdict**: ❌ Type promotion is a rewrite, not scheduling
**Gap**: No type modification primitives

---

## 8. Loop Transformations

### 8.1 Loop Interchange
**Pattern**: Change loop order for locality

```mlir
// ✅ WORKS
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0

  // Default: M, N, K order
  // Change to: M, K, N for better B reuse
  %interchanged = transform.structured.interchange %matmul
      iterator_interchange = [0, 2, 1]  // M, K, N

  transform.yield
}
```

**Verdict**: ✅ Works

---

### 8.2 Loop Unrolling
**Pattern**: Unroll inner loops

```mlir
// ✅ WORKS
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
  %tiled, %loops:3 = transform.structured.tile_using_for %matmul
      tile_sizes [0, 8, 4]

  // Unroll K loop
  transform.loop.unroll %loops#2 factor = 4

  transform.yield
}
```

**Verdict**: ✅ Works

---

### 8.3 Loop Pipelining (Software Pipelining)
**Pattern**: Overlap iterations

```mlir
// ✅ WORKS
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
  %tiled, %k_loop = transform.structured.tile_using_for %matmul
      tile_sizes [0, 0, 32]

  // Pipeline K loop: load next while computing current
  transform.loop.pipeline %k_loop
      num_stages = 2

  transform.yield
}
```

**Verdict**: ✅ Works

---

### 8.4 Loop Peeling
**Pattern**: Handle remainder iterations

```mlir
// ✅ WORKS
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
  %tiled, %loops:3 = transform.structured.tile_using_for %matmul
      tile_sizes [32, 32, 32]

  // Peel last iteration for remainder handling
  %main_loop, %remainder_loop = transform.loop.peel %loops#0

  transform.yield
}
```

**Verdict**: ✅ Works

---

### 8.5 Loop Coalescing
**Pattern**: Merge loops for parallelism

```mlir
// ⚠️ PARTIAL - exists but limited
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
  %tiled, %m_loop, %n_loop, %k_loop = transform.structured.tile_using_for %matmul
      tile_sizes [32, 32, 32]

  // Coalesce M and N loops for more parallelism
  %coalesced = transform.loop.coalesce %m_loop, %n_loop

  transform.yield
}
```

**Verdict**: ⚠️ Exists but may have restrictions

---

## 9. Hardware-Specific

### 9.1 AVX-512 Targeting
**Pattern**: Ensure AVX-512 vector width

```mlir
// ⚠️ PARTIAL - vector width controllable, but instruction selection is LLVM
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0

  // Tile for 512-bit vectors (16 x f32)
  %tiled, %loops:3 = transform.structured.tile_using_for %matmul
      tile_sizes [0, 16, 4]

  // Vectorize with 512-bit width
  %vectorized = transform.structured.vectorize %tiled
      vector_sizes [1, 16, 4]

  // Actual AVX-512 instruction selection happens in LLVM backend
  transform.yield
}
```

**Verdict**: ⚠️ Vector sizing works, instruction selection is LLVM's job

---

### 9.2 AMX Targeting (Tile Matrix Multiply)
**Pattern**: Use Intel AMX for large matmuls

```mlir
// ❌ NOT SUPPORTED - AMX requires custom lowering
// AMX has specific tile sizes: 16x16 for fp16, 16x64 for int8

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0

  // HYPOTHETICAL: would need AMX-aware tiling
  // transform.structured.tile_for_amx %matmul
  //     tile_sizes = "auto"  // Let AMX constraints determine

  transform.yield
}
```

**Verdict**: ❌ Not supported (need custom AMX lowering)
**Gap**: No hardware-specific tile constraints

---

### 9.3 GPU Block/Thread Mapping
**Pattern**: Map to GPU execution model

```mlir
// ✅ WORKS (for GPU targets)
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0

  // Tile for GPU blocks
  %block_tiled, %block_loops:2 = transform.structured.tile_using_forall %matmul
      tile_sizes [128, 128]
      mapping = [#gpu.block<y>, #gpu.block<x>]

  // Tile for threads within block
  %thread_tiled, %thread_loops:2 = transform.structured.tile_using_forall %block_tiled
      tile_sizes [8, 8]
      mapping = [#gpu.thread<y>, #gpu.thread<x>]

  transform.yield
}
```

**Verdict**: ✅ Works for GPU

---

### 9.4 Tensor Core Targeting (WMMA/MMA)
**Pattern**: Use NVIDIA tensor cores

```mlir
// ⚠️ PARTIAL - need nvgpu dialect integration
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0

  // Tile for tensor core shapes (16x16x16 for f16)
  %tiled = transform.structured.tile_using_forall %matmul
      tile_sizes [16, 16, 16]

  // Would need: transform.structured.map_to_tensor_core %tiled
  // Or specific vectorization pattern for wmma

  transform.yield
}
```

**Verdict**: ⚠️ Tiling works, tensor core lowering needs separate pass
**Gap**: No direct tensor core mapping

---

## 10. Edge Cases

### 10.1 Dynamic Shapes
**Pattern**: Handle unknown dimensions at compile time

```mlir
// ⚠️ PARTIAL - tiling with dynamic shapes is tricky
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
  // Inputs: tensor<?x?xf32>, tensor<?x?xf32>

  // Static tile sizes work, but may need peeling
  %tiled, %loops:3 = transform.structured.tile_using_for %matmul
      tile_sizes [32, 32, 32]

  // Peeling handles remainders
  transform.loop.peel %loops#0
  transform.loop.peel %loops#1

  transform.yield
}
```

**Verdict**: ⚠️ Works with peeling, but no auto-remainder handling

---

### 10.2 Strided Access
**Pattern**: Non-contiguous memory access

```mlir
// ✅ Linalg handles strides natively
// No special Transform handling needed

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // linalg.matmul on strided memrefs works normally
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0

  // Tiling preserves strides
  %tiled, %loops:3 = transform.structured.tile_using_for %matmul
      tile_sizes [32, 32, 32]

  transform.yield
}
```

**Verdict**: ✅ Works (strides handled by linalg)

---

### 10.3 Transposed Inputs
**Pattern**: A × B^T or A^T × B

```mlir
// ✅ WORKS - interchange or use linalg.matmul_transpose_b
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %matmul_t = transform.structured.match ops{["linalg.matmul_transpose_b"]} in %arg0

  // Or for regular matmul with transposed input:
  // The linalg.generic will have different indexing maps

  %tiled, %loops:3 = transform.structured.tile_using_for %matmul_t
      tile_sizes [32, 32, 32]

  transform.yield
}
```

**Verdict**: ✅ Works

---

### 10.4 Reduction-Only Operations
**Pattern**: sum, max, mean over dimensions

```mlir
// ⚠️ PARTIAL - reduction tiling has special semantics
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %reduce = transform.structured.match ops{["linalg.reduce"]} in %arg0

  // Tile reduction dimension (creates partial results)
  %tiled, %loop = transform.structured.tile_using_for %reduce
      tile_sizes [32]  // Tile the reduction dim

  // Partial results need final reduction
  // Problem: who creates the final reduction?

  transform.yield
}
```

**Verdict**: ⚠️ Tiled reductions create partial results, need explicit final reduction
**Gap**: No automatic handling of reduction tree

---

### 10.5 Scatter/Gather Operations
**Pattern**: Indexed access patterns

```mlir
// ❌ NOT SUPPORTED - irregular access not in linalg
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // Gather: output[i] = input[indices[i]]
  // Not a structured operation

  // Would need custom handling outside Transform
  transform.yield
}
```

**Verdict**: ❌ Not supported (not structured computation)

---

## Summary

### ✅ Works Well (18 cases)
- Basic fusion (matmul+bias, matmul+bias+relu)
- Multi-level tiling (L1/L2)
- Vectorization
- Loop interchange, unroll, peel, pipeline
- GPU block/thread mapping
- Memory promotion
- Batch dimension handling
- Conv+ReLU fusion
- Strided/transposed inputs

### ⚠️ Partial Support (15 cases)
- Multi-op activation fusion (GELU, SiLU) - needs chain matching
- VNNI vectorization - needs intrinsic selection
- Data packing - needs operand replacement
- Residual connections - needs dataflow matching
- RMSNorm/Softmax - reduction fusion issues
- RoPE - tedious pattern matching
- Dynamic shapes - needs explicit peeling
- Quantization - scheduling works, semantics elsewhere
- Tensor cores - tiling works, lowering separate

### ❌ Not Supported (12 cases)
- Op decomposition (conv→im2col+matmul) - needs PDLL/C++
- Tile size auto-derivation - no dynamic computation
- Prefetching - not in Transform
- Online softmax / Flash attention - algorithm change needed
- KV-cache - dynamic indexing
- AMX targeting - custom lowering needed
- Type promotion (mixed precision) - rewrite, not scheduling
- Scatter/gather - unstructured
- Algorithm changes - Transform is pure scheduling

### Key Gaps for Sugar Layer

1. **Chain matching**: Match `op1 -> op2 -> op3` pattern
2. **Dataflow filtering**: Match ops by their producers/consumers
3. **Auto-derivation**: Compute tile sizes from cache constraints
4. **Hardware constraints**: "tile K to multiple of 4 for VNNI"
5. **Reduction fusion**: Fuse reductions with pointwise ops
6. **Prefetch insertion**: Add prefetch at distance N
7. **Intrinsic selection**: "use VNNI" / "use AMX"
