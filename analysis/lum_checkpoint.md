# Lum DSL Implementation Checkpoint

**Last Updated**: 2025-12-27
**Branch**: `feature/lum-dsl`
**Commit**: `e2a7a4a`

---

## Overview

Lum is a domain-specific language for MLIR Transform Dialect generation. It enables declarative specification of optimization schedules for SimpLang kernels.

---

## Implementation Status

### Phase 1: MVP Core (COMPLETE)

| Component | Status | Notes |
|-----------|--------|-------|
| **Lexer** (`lexer.l`) | ✅ Done | All 34 keywords, operators, literals |
| **Parser** (`parser.y`) | ✅ Done | Patterns, schedules, transforms |
| **AST Nodes** | ✅ Done | base, pattern, schedule, transform |
| **Transform Emitter** | ✅ Done | Generates valid Transform Dialect |
| **CLI** (`main.cpp`) | ✅ Done | `--emit-transform`, `--dump-ast` |
| **Tests** | ✅ Done | 3 test .lum files |

#### MVP Transforms Implemented

| Transform | Lum Syntax | MLIR Output | Tested |
|-----------|------------|-------------|--------|
| `tile` | `tile [M,N,K] => m,n,k` | `tile_using_for` | ✅ |
| `fuse` | `fuse A into L` | `fuse_into_containing_op` | ✅ |
| `vec` | `vec [N]` | `vectorize` | ✅ |
| `check` | `check accuracy` | Comment (placeholder) | ✅ |
| `unroll` | `unroll L N` | `loop.unroll` | ✅ |
| `interchange` | `interchange [i,j,k]` | `structured.interchange` | ✅ |
| `parallel` | `parallel L` | `loop.coalesce` | ✅ |

#### End-to-End Verification

```
✅ lum generates valid Transform Dialect MLIR
✅ mlir-opt parses the output without errors
✅ mlir-opt --transform-interpreter applies transforms correctly
✅ Applied to SimpLang Phase1 MLIR successfully
✅ Verified tiled matmul produces correct result (128.0)
```

**Test Flow Verified:**
```
SimpLang (.sl) → Phase1 MLIR (linalg.matmul) → Lum Transform → Tiled MLIR (scf.for loops)
```

---

### Phase 2: SSA Handle Tracking (PARTIAL)

| Feature | Status | Priority |
|---------|--------|----------|
| Consumed handle detection | ✅ | High |
| Use-after-consume errors | ✅ | High |
| Implicit rebinding | ✅ | Medium |
| Handle type tracking (Op/Loop/Value) | ❌ | Medium |

**Design Doc**: `analysis/lum_ssa_design.md`

---

### Phase 3: Additional Transforms (PARTIAL)

| Transform | Lum Syntax | MLIR Equivalent | Status |
|-----------|------------|-----------------|--------|
| `unroll` | `unroll L N` | `loop.unroll` | ✅ |
| `interchange` | `interchange [i,j,k]` | `structured.interchange` | ✅ |
| `parallel` | `parallel L` | `loop.coalesce` | ✅ |
| `pack` | `pack A [dims]` | `structured.pack` | ❌ |
| `peel` | `peel L N` | `loop.peel` | ❌ |
| `promote` | `promote A` | `structured.promote` | ❌ |

---

### Phase 4: Fusion Variants (PARTIAL)

| Fusion Type | Lum Syntax | Use Case | Status |
|-------------|------------|----------|--------|
| `fuse_chain` | `fuse_chain [ops]` | Loop fusion | ✅ |
| `fuse_chain with` | `fuse_chain [ops] with op` | Kernel fusion | ✅ |
| `fuse_horizontal` | `fuse_horizontal [ops]` | Independent ops (QKV) | ❌ |
| `fuse_reduction` | `fuse_reduction R with [ops]` | Softmax, RMSNorm | ❌ |
| `fuse_elementwise` | `fuse_elementwise *` | Auto-fuse | ❌ |

**Chained arrow syntax**: `a -> b -> c -> d` is now supported (creates edges a→b, b→c, c→d)

---

### Phase 5: Control Flow & Constraints (PARTIAL)

| Feature | Lum Syntax | Status |
|---------|------------|--------|
| `let` bindings | `let x = expr` | ❌ |
| `if` conditionals | `if cond { }` | ❌ |
| `for` iteration | `for x in range { }` | ❌ |
| `derive` auto-tuning | `derive tile from [...]` | ❌ |
| Pattern dtype constraint | `mm: linalg.matmul where dtype=i8` | ✅ |

---

### Phase 6: Verification (NOT STARTED)

| Feature | Description | Status |
|---------|-------------|--------|
| `check accuracy` | Compare before/after results | ❌ (placeholder only) |
| `check perf` | Benchmark comparison | ❌ |
| `check memory` | Memory usage analysis | ❌ |
| Test input generation | Random/structured inputs | ❌ |

---

### Phase 7: Debug & Pipelines (NOT STARTED)

| Feature | Lum Syntax | Status |
|---------|------------|--------|
| `trace` | `trace mm` | ❌ |
| `break` | `break` | ❌ |
| `assert` | `assert cond` | ❌ |
| `snapshot` | `snapshot "name"` | ❌ |
| `diff` | `diff snap1 snap2` | ❌ |
| `pipeline` | `pipeline Name { }` | ❌ |
| `phase` | `phase Name { }` | ❌ |
| `gate` | `gate Name(cond)` | ❌ |

---

### Phase 8: SimpLang Integration (COMPLETE)

| Feature | Description | Status |
|---------|-------------|--------|
| `--schedule` flag | `simplang kernel.sl --schedule opt.lum` | ✅ Done |
| Disable internal tiling | Lum schedule takes precedence | ✅ Done |
| Workflow guidance | Prints lum → mlir-opt workflow | ✅ Done |
| Pipeline injection | Direct Transform Dialect interpreter | ✅ Done |
| Pre/post comparison | Benchmark with/without Lum | ✅ Done |

**Current workflow** (with `--schedule` flag):
```bash
# Direct integration - Lum transforms applied automatically in pipeline:
simplang kernel.sl --emit-mlir --schedule schedule.mlir -o kernel.o

# Or compile .lum to .mlir first:
lum opt.lum --emit-transform > schedule.mlir
simplang kernel.sl --emit-mlir --schedule schedule.mlir -o kernel.o
```

**Transform Dialect Integration Details:**
- Phase 1.5 in mlir_pipeline.cpp applies transforms after Simp→Linalg lowering
- Automatically parses transform module and finds `@__transform_main` entry point
- Registers Linalg and SCF transform extensions for full transform support
- Internal tiling disabled when Lum schedule is provided

**Benchmark Results (512x512 F32 matmul, AMD Zen4):**

| Configuration | Time (ms) | GFLOPS |
|---------------|-----------|--------|
| SimpLang Internal (hierarchical) | 4.31 | 62.25 |
| Lum tile [64,64,32] | 16.98 | 15.81 |
| Lum tile [32,32,32] + unroll | 5.98 | 44.85 |

**Analysis:**
- SimpLang internal pipeline achieves best performance with hierarchical 2-level tiling
- Lum provides flexibility for experimentation with custom schedules
- Lum transforms apply at Phase 1.5, followed by Phase 2 vectorization
- Best Lum performance (44.85 GFLOPS) is ~72% of internal (62.25 GFLOPS)

---

## File Inventory

### Implemented Files

```
src/lum/
├── CMakeLists.txt           # Build config with Flex/Bison
├── lexer.l                  # Flex lexer (169 lines)
├── parser.y                 # Bison parser (353 lines)
├── main.cpp                 # CLI entry point (155 lines)
├── ast/
│   ├── base.hpp             # Node base class, NodeKind enum
│   ├── pattern.hpp/cpp      # PatternDecl, OpPattern, DataflowEdge
│   ├── schedule.hpp/cpp     # ScheduleDecl, Program
│   └── transform.hpp/cpp    # TileTransform, FuseTransform, VecTransform, CheckTransform
└── codegen/
    ├── transform_emitter.hpp
    └── transform_emitter.cpp  # MLIR text generation (342 lines)

tests/lum/schedules/
├── basic_tile.lum           # Simple tiling test
├── matmul_bias.lum          # Pattern + tile + fuse
└── vnni_matmul.lum          # VNNI vectorization
```

### Total Lines: ~1,783

---

## Spec Compliance

**Reference**: `analysis/lum_spec_v0.2.md`

| Spec Section | Coverage | Notes |
|--------------|----------|-------|
| 1. Core Syntax | Partial | Basic structure, not all keywords |
| 2. SSA Semantics | ❌ | Not implemented |
| 3. Pattern Matching | Partial | Basic patterns, no constraints |
| 4. Scheduling (tile) | ✅ | Full |
| 4. Scheduling (fuse) | Partial | Basic only |
| 4. Scheduling (vec) | ✅ | Full |
| 4. Other transforms | ❌ | Not implemented |
| 5. Verification | ❌ | Placeholder only |
| 6. Debug | ❌ | Not implemented |
| 7. Pipelines | ❌ | Not implemented |

---

## Next Steps (Priority Order)

1. **SimpLang Integration**:
   - Add `--schedule` flag to inject Lum transforms into MLIR pipeline
   - Disable internal tiling when Lum schedule is provided
   - Enable performance comparison (before/after Lum)

2. **Entry Point Fix**: Rename generated sequence to `@__transform_main` or add CLI flag

3. **SSA Tracking**: Implement consumed handle detection and use-after-consume errors

4. **More Transforms**: `unroll`, `interchange`, `parallel`

5. **Fusion Variants**: `fuse_chain` for element-wise ops (GELU, Softmax)

6. **Constraints**: Pattern predicates (`mm.dtype == i8`, `mm.K % 4 == 0`)

7. **Verification**: Implement actual `check accuracy` with test input generation

---

## Usage Examples

### Generate Transform Dialect

```bash
cd build_mlir
./src/lum/lum ../tests/lum/schedules/basic_tile.lum --emit-transform
```

### Apply to MLIR

```bash
# 1. Generate SimpLang MLIR
./src/simplang kernel.sl --emit-mlir --dump-mlir-passes -o kernel.o

# 2. Generate Lum transform
./src/lum/lum schedule.lum --emit-transform > transform.mlir

# 3. Combine and apply
cat kernel.o_phase1_simp_lowering.mlir transform.mlir > combined.mlir
mlir-opt combined.mlir --transform-interpreter
```

---

## Known Issues

1. **Entry point naming**: Transform sequence must be named `@__transform_main` for mlir-opt auto-discovery (currently uses schedule name like `@TileMatmul`)
2. **Module nesting**: Combined SimpLang + Lum output has nested modules
3. **Handle collision**: Multiple tiles reuse `%loops` name (need unique naming per tile)
4. **SimpLang internal tiling**: SimpLang Phase 2 applies its own 16x16x16 tiling - need to disable or replace with Lum schedule
5. **No performance comparison yet**: Need to benchmark Lum-optimized vs SimpLang-internal tiling

---

## Test Commands

```bash
# Build
cd build_mlir && make lum

# Parse test
./src/lum/lum ../tests/lum/schedules/basic_tile.lum --dump-ast

# Generate MLIR
./src/lum/lum ../tests/lum/schedules/basic_tile.lum --emit-transform

# Verify with mlir-opt
./src/lum/lum ../tests/lum/schedules/basic_tile.lum --emit-transform | \
  mlir-opt 2>&1 | head -20
```
