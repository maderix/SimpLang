# Lum SSA Compatibility Design

## The Problem

MLIR Transform Dialect is strictly SSA:

```mlir
// Transform Dialect - every op returns NEW handles
%mm = transform.structured.match ops{["linalg.matmul"]} in %arg0
%tiled, %loop_m, %loop_n, %loop_k = transform.structured.tile_using_for %mm
    tile_sizes [64, 64, 32]
// %mm is now CONSUMED - cannot be used again!
// Must use %tiled for the tiled operation

%vectorized = transform.structured.vectorize %tiled   // Use %tiled, not %mm
```

**Key rules:**
1. Each transform consumes its input handle
2. Returns new handle(s) for transformed op(s)
3. Using consumed handle is undefined behavior
4. Loop handles are separate from op handles

---

## Design Options

### Option A: Implicit Handle Tracking (Magic)

```lum
schedule Opt(linalg.matmul) {
    tile [64, 64, 32]         # mm implicitly updated to tiled version
    vec [16]                  # operates on "current" mm
}
```

**Pros:** Clean syntax, beginner-friendly
**Cons:** Confusing when you need both old and new handles, hard to debug

### Option B: Explicit SSA (Verbose)

```lum
schedule Opt(linalg.matmul) {
    mm1, m, n, k = tile mm [64, 64, 32]
    mm2 = vec mm1 [16]
}
```

**Pros:** Matches Transform Dialect exactly, clear data flow
**Cons:** Verbose, lots of temporary names

### Option C: Hybrid with Shadow Binding (Recommended)

```lum
schedule Opt(linalg.matmul) {
    # Default: handle is rebound (shadows previous)
    tile [64, 64, 32] => m, n, k     # mm is REBOUND to tiled op
    vec [16]                          # uses rebound mm

    # Explicit: keep both handles
    tile mm [64, 64, 32] => tiled, m, n, k   # mm unchanged, tiled is new
}
```

**Pros:** Clean for common case, explicit when needed
**Cons:** Need to understand rebinding semantics

---

## Recommended Design: Hybrid SSA

### Rule 1: Pattern variables are mutable bindings

```lum
pattern MatmulBias {
    mm: linalg.matmul      # mm is a binding, not a value
    bias: linalg.generic
    mm -> bias
}

schedule Opt(MatmulBias) {
    # mm refers to the matched matmul
    tile mm [64, 64, 32]
    # mm NOW refers to the tiled version (rebound)
}
```

### Rule 2: Transforms rebind by default

```lum
schedule Opt(linalg.matmul) {
    tile [64, 64, 32]     # implicit: target is 'mm' from pattern
                          # mm is rebound to tiled op

    fuse bias into m      # bias is rebound to fused op

    vec [16]              # operates on current mm (tiled version)
}
```

### Rule 3: Explicit binding preserves original

```lum
schedule Opt(linalg.matmul) {
    # Keep original handle, create new one
    tile mm [64, 64, 32] => tiled, m, n, k

    # Now both are valid:
    trace mm.shape        # Original (but consumed in IR!)
    trace tiled.shape     # Tiled version

    # ERROR: mm was consumed by tile!
    vec mm [16]           # COMPILE ERROR: mm consumed
}
```

### Rule 4: Loop handles are separate

```lum
schedule Opt(linalg.matmul) {
    tile [64, 64, 32] => m, n, k    # m, n, k are LOOP handles
                                     # mm is rebound to tiled OP handle

    # Use loop handles for loop transforms
    unroll k 4
    parallel m
    interchange [n, k, m]

    # Use op handle for op transforms
    vec [16]                         # operates on mm (tiled)
}
```

### Rule 5: Consumed handles are tracked

```lum
schedule Opt(linalg.matmul) {
    tile mm [64, 64, 32] => tiled, m, n, k

    # Compiler tracks: mm is CONSUMED, tiled is LIVE

    fuse bias into m      # OK: m is live loop handle
    vec tiled [16]        # OK: tiled is live op handle
    vec mm [16]           # ERROR: mm was consumed by tile
}
```

---

## Syntax Specification

### Implicit rebinding (common case)

```lum
# Transform target, rebind to result
tile [M, N, K]                    # target inferred from pattern, rebound
tile [M, N, K] => m, n, k         # + capture loop handles

fuse A into B                     # A rebound to fused op
vec [N]                           # target inferred, rebound
```

### Explicit binding (when needed)

```lum
# Explicit target, explicit result binding
tile mm [M, N, K] => tiled        # mm consumed, tiled is new op
tile mm [M, N, K] => tiled, m, n, k   # + loop handles

# Multiple results
fuse [a, b, c] into loop => fused     # all consumed, fused is result
```

### Handle types

```lum
# Op handles (point to operations)
mm: linalg.matmul                 # Op handle from pattern
tiled = tile mm [64, 64, 32]      # Op handle from transform

# Loop handles (point to loop nests)
m, n, k = tile mm [64, 64, 32]    # Loop handles
m: scf.for                        # Type is loop

# Value handles (point to SSA values) - rarely needed
result = mm.result                # Value handle
```

---

## Transform Dialect Mapping

### Lum with implicit rebinding:

```lum
schedule Opt(MatmulBias) {
    tile [64, 64, 32] => m, n, k
    fuse bias into m
    vec [16]
}
```

### Generated Transform Dialect:

```mlir
transform.named_sequence @Opt(%arg0: !transform.any_op) {
  // Pattern matching
  %mm_0 = transform.structured.match ops{["linalg.matmul"]} in %arg0
  %bias_0 = transform.structured.match ops{["linalg.generic"]} in %arg0

  // tile [64, 64, 32] => m, n, k
  // mm_0 consumed, mm_1 is new op handle
  %mm_1, %m, %n, %k = transform.structured.tile_using_for %mm_0
      tile_sizes [64, 64, 32]

  // fuse bias into m
  // bias_0 consumed, bias_1 is new op handle
  %bias_1 = transform.structured.fuse_into_containing_op %bias_0 into %m

  // vec [16]
  // mm_1 consumed, mm_2 is new op handle
  %mm_2 = transform.structured.vectorize %mm_1 vector_sizes [1, 16, 1]

  transform.yield
}
```

### Lum with explicit bindings:

```lum
schedule Opt(MatmulBias) {
    tile mm [64, 64, 32] => tiled, m, n, k
    fuse bias into m => fused_bias
    vec tiled [16] => vectorized
}
```

### Generated Transform Dialect:

```mlir
transform.named_sequence @Opt(%arg0: !transform.any_op) {
  %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0
  %bias = transform.structured.match ops{["linalg.generic"]} in %arg0

  %tiled, %m, %n, %k = transform.structured.tile_using_for %mm
      tile_sizes [64, 64, 32]

  %fused_bias = transform.structured.fuse_into_containing_op %bias into %m

  %vectorized = transform.structured.vectorize %tiled vector_sizes [1, 16, 1]

  transform.yield
}
```

---

## Static Analysis

The Lum compiler performs SSA analysis:

### 1. Liveness tracking

```lum
schedule Example(linalg.matmul) {
    tile [64, 64, 32] => m, n, k
    # State: mm=LIVE(tiled), m=LIVE, n=LIVE, k=LIVE

    unroll k 4
    # State: mm=LIVE(tiled), m=LIVE, n=LIVE, k=CONSUMED

    vec [16]
    # State: mm=CONSUMED, m=LIVE, n=LIVE, k=CONSUMED
}
```

### 2. Use-after-consume detection

```lum
schedule BadExample(linalg.matmul) {
    tile mm [64, 64, 32] => tiled, m, n, k
    vec mm [16]      # ERROR: mm consumed by tile on line 2
}
```

Error message:
```
error[E0001]: use of consumed handle 'mm'
  --> schedule.lum:3:9
   |
 2 |     tile mm [64, 64, 32] => tiled, m, n, k
   |          -- handle consumed here
 3 |     vec mm [16]
   |         ^^ cannot use consumed handle
   |
help: did you mean to use 'tiled'?
   |
 3 |     vec tiled [16]
   |         ~~~~~
```

### 3. Unused handle warning

```lum
schedule WastedWork(linalg.matmul) {
    tile [64, 64, 32] => m, n, k
    # mm rebound but never used again
    # m, n, k captured but never used
}
```

Warning:
```
warning[W0001]: unused loop handles
  --> schedule.lum:2:29
   |
 2 |     tile [64, 64, 32] => m, n, k
   |                             ^^^^^^ handles never used
   |
help: remove unused captures or use '_' to ignore
   |
 2 |     tile [64, 64, 32]
   |     or
 2 |     tile [64, 64, 32] => _, _, _
```

---

## Multi-result Operations

Some transforms produce multiple op handles:

### Split/decompose

```lum
schedule Decompose(linalg.conv_2d) {
    # Decompose produces multiple ops
    decompose conv => im2col, mm, reshape

    # Each is a separate handle
    tile mm [64, 64, 32]
    fuse im2col into mm.k_loop
}
```

### Match multiple

```lum
pattern MultiMatmul {
    mm1: linalg.matmul
    mm2: linalg.matmul
    mm1 -> mm2
}

schedule OptBoth(MultiMatmul) {
    # Both are separate handles
    tile mm1 [64, 64, 32]
    tile mm2 [64, 64, 32]

    # Or apply same transform to all
    for mm in [mm1, mm2] {
        tile mm [64, 64, 32]
    }
}
```

---

## Handle Scoping

### Block scope

```lum
schedule Scoped(linalg.matmul) {
    tile [64, 64, 32] => m, n, k

    if mm.M > 256 {
        # m, n, k visible here
        parallel m
    }
    # m, n, k still visible here
}
```

### Loop scope

```lum
schedule LoopScope(MultiMatmul) {
    for mm in [mm1, mm2] {
        tile mm [64, 64, 32] => m, n, k
        # m, n, k are LOCAL to this iteration
        parallel m
    }
    # m, n, k NOT visible here
}
```

---

## Summary

| Aspect | Design Choice |
|--------|---------------|
| Default behavior | Implicit rebinding (shadows) |
| Explicit binding | `=> name` syntax |
| Consumption tracking | Static analysis at compile time |
| Loop vs op handles | Separate types, clear distinction |
| Multi-result | Explicit binding required |
| Error handling | Clear messages with suggestions |

This design:
1. ✅ Matches Transform Dialect SSA semantics exactly
2. ✅ Provides clean syntax for common cases
3. ✅ Explicit when needed
4. ✅ Catches errors at compile time
5. ✅ Clear generated code

---

## Updated Syntax Examples

### Before (naive, SSA-unaware):
```lum
tile mm [64, 64, 32]
fuse bias into m
vec mm [16]          # What is mm here? Unclear!
```

### After (SSA-aware, hybrid):
```lum
# Option 1: Implicit rebinding (recommended for simple cases)
tile [64, 64, 32] => m, n, k     # mm rebound to tiled
fuse bias into m                  # bias rebound to fused
vec [16]                          # uses current mm (tiled, fused)

# Option 2: Explicit (when you need clarity)
tile mm [64, 64, 32] => tiled, m, n, k
fuse bias into m => fused_bias
vec tiled [16] => final

# Option 3: Chain syntax (future consideration)
mm |> tile [64, 64, 32] |> vec [16]    # Pipeline, each step rebinds
```

The key insight: **Lum handles are bindings, not values. Transforms update bindings by default, creating new SSA values under the hood.**
