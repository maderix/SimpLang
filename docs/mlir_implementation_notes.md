# MLIR Integration - Implementation Notes & Best Practices

**Document Version:** 1.0
**Date:** 2025-01-18
**Companion to:** mlir_integration_plan.md
**Status:** Implementation Guidance

---

## Critical Implementation Insights

These are practical, battle-tested recommendations from experienced MLIR developers to avoid common pitfalls and accelerate implementation.

---

## 1. Dialect Design: Keep It SSA-Pure

### Principle: First-Class MLIR Citizen

**DO:** Design `simp.array_*` operations to be SSA-pure (no hidden state)
```mlir
// GOOD: SSA-pure, affine passes can reason about this
%arr = simp.array_create %len : i64 -> !simp.array<f32>
%val = simp.array_get %arr[%idx] : !simp.array<f32> -> f32
%arr2 = simp.array_set %arr[%idx], %val : !simp.array<f32>, f32 -> !simp.array<f32>
```

**DON'T:** Create operations with hidden state
```mlir
// BAD: Mutation, affine/vector passes can't optimize
simp.array_set_inplace %arr[%idx], %val  // Modifies %arr in place
```

### Why This Matters

✅ Affine passes can analyze data dependencies
✅ Vector passes can reason about memory access patterns
✅ Standard MLIR optimization passes work automatically
✅ Makes the dialect composable with upstream MLIR ecosystem

### Implementation Checklist

- [ ] All operations return new SSA values (functional style)
- [ ] No implicit mutations or side effects
- [ ] Memory effects are explicitly modeled
- [ ] Operations have proper aliasing information

---

## 2. Lowering Strategy: Memref-First Approach

### Key Insight: Bias Toward Memref Early

**Recommended Lowering Path:**
```
SimpLang AST
    ↓
simp dialect (arrays, matrices)
    ↓
memref dialect (memory representation)  ← Do this EARLY
    ↓
linalg dialect (structured ops on memrefs)
    ↓
affine/scf (optimized loops)
    ↓
llvm dialect
```

**Not:**
```
simp → tensor → linalg → ...
```

### Rationale

⚠️ **MLIR's tensor path is still dynamic-shape heavy** - You'll get faster convergence with memref
✅ **Memref path is more stable** - Better tooling, fewer surprises
✅ **Add tensor support in Phase 3** - When you actually need it for SimpTensor

### Implementation Strategy

**Phase 1-2:**
```cpp
// Lower directly to memref
simp.array_create %len → memref.alloc(%len) : memref<?xf32>
simp.array_get %arr[%i] → memref.load %arr[%i] : memref<?xf32>
```

**Phase 3 (if needed for SimpTensor):**
```cpp
// Add tensor support for high-level optimizations
simp.tensor_create %shape → tensor.empty %shape : tensor<?x?xf32>
// Then bufferize to memref for lowering
```

---

## 3. SimpBLAS Integration: Pass Ordering Matters

### Critical: Register Pattern Late in Pipeline

**CORRECT Pipeline:**
```cpp
void buildSimpBLASPipeline(PassManager &pm) {
    // 1. High-level linalg optimizations FIRST
    pm.addPass(createLinalgFusionPass());
    pm.addPass(createLinalgTilingPass({32, 32, 32}));

    // 2. Bufferization
    pm.addPass(createLinalgBufferizePass());

    // 3. THEN lower to SimpBLAS (LATE!)
    pm.addPass(createLinalgToSimpBLASPass());  // ← Do this LATE

    // 4. Continue lowering
    pm.addPass(createConvertLinalgToAffineLoopsPass());
}
```

**WRONG Pipeline:**
```cpp
// DON'T do this - canonicalizes too early!
pm.addPass(createLinalgToSimpBLASPass());  // ← TOO EARLY
pm.addPass(createLinalgFusionPass());      // Nothing to fuse anymore!
```

### Why Order Matters

❌ **Early BLAS lowering freezes the graph** - No more fusion opportunities
✅ **Late BLAS lowering preserves optimization opportunities** - Fusion, tiling happen first
✅ **Let MLIR do its magic before committing to BLAS calls**

### Gated Pass Pipeline

Create a pass pipeline alias for easy tuning:

```cpp
// src/mlir/passes/pass_pipelines.cpp
void registerSimpOptPipeline() {
    PassPipelineRegistration<>(
        "simp-opt",
        "SimpLang optimization pipeline with optional BLAS lowering",
        [](OpPassManager &pm) {
            buildSimpBLASPipeline(pm);
        }
    );
}
```

**Usage:**
```bash
# Standard pipeline
mlir-opt input.mlir --simp-opt

# Custom tuning
mlir-opt input.mlir --linalg-fusion --linalg-tile="tile-sizes=64,64" --linalg-to-simpblas
```

---

## 4. Testing Strategy: Start Simple, Iterate

### Phase 1 Testing: Skip the Complexity

**DO in Phase 1:**
```bash
# Sanity check: Does it emit valid MLIR?
mlir-opt input.mlir --dump-pass-pipeline

# Does it verify?
mlir-opt input.mlir --verify-diagnostics

# Can it lower to LLVM?
mlir-opt input.mlir --convert-to-llvm | llvm-as
```

**DON'T in Phase 1:**
```bash
# Skip FileCheck until dialect stabilizes
# RUN: mlir-opt %s | FileCheck %s  ← Skip this early
```

### When to Add FileCheck

✅ **Phase 2+:** Once operations and lowering patterns are stable
✅ **For regression tests:** When you fix a bug
✅ **For optimization validation:** When verifying specific transforms

### Recommended Test Progression

**Week 1-3 (Phase 1):**
- Basic "does it compile" tests
- `mlir-opt --dump-pass-pipeline` sanity checks
- Manual verification of LLVM IR output

**Week 4-6 (Phase 2):**
- Add FileCheck for critical lowering paths
- Vectorization validation tests
- Performance regression detection

**Week 7+ (Phase 3):**
- Comprehensive FileCheck test suite
- Numerical accuracy validation
- Full CI integration

### Test Coverage Goal

**Implement 60-70% of test plan = CI-grade**
- Focus on critical paths first
- Add comprehensive coverage incrementally
- Don't let perfect be the enemy of good

---

## 5. Timeline Reality Check

### First 10 Sessions = 80% Infrastructure

**Reality of Phase 1:**

**Sessions 1-10:** Infrastructure hell
- CMake integration with MLIR
- TableGen wrangling
- Build system debugging
- "Why won't this link?!"
- Understanding MLIR's type system

**Sessions 11-16:** The fun begins
- Actual codegen working
- Seeing MLIR IR for the first time
- Optimizations start showing results

### Phase 1 Success Metric (Revised)

**Original:** "Basic MLIR integration, feature parity"
**Realistic:** "Get ANY valid MLIR IR out of parser"

Don't aim for perfection in Phase 1. Aim for:
- ✅ Parser produces valid MLIR module
- ✅ MLIR module lowers to LLVM IR
- ✅ LLVM IR compiles to object code
- ✅ Object code runs (correctness = bonus)

**Performance is Phase 2+ problem.**

### CMake Hell Survival Guide

You WILL encounter CMake issues. Budget time for:
- MLIR's complex build dependencies
- TableGen integration
- Proper library linking order
- Include path hell

**Pro Tip:** Copy CMakeLists.txt from MLIR Toy tutorial first, then modify incrementally.

---

## 6. Minimal Phase 1 Dialect

### Absolute Minimum to Get Started

**Start with these 5 operations:**

```tablegen
// simp.constant - Literal values
def Simp_ConstantOp : Simp_Op<"constant"> {
    let results = (outs AnyType:$result);
    let arguments = (ins AnyAttr:$value);
}

// simp.add - Binary arithmetic
def Simp_AddOp : Simp_Op<"add"> {
    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);
}

// simp.array_create - Array allocation
def Simp_ArrayCreateOp : Simp_Op<"array_create"> {
    let arguments = (ins I64:$len);
    let results = (outs Simp_ArrayType:$result);
}

// simp.array_get - Array load
def Simp_ArrayGetOp : Simp_Op<"array_get"> {
    let arguments = (ins Simp_ArrayType:$array, I64:$index);
    let results = (outs AnyType:$result);
}

// simp.matmul - Matrix multiply (placeholder for SimpBLAS)
def Simp_MatMulOp : Simp_Op<"matmul"> {
    let arguments = (ins Simp_ArrayType:$lhs, Simp_ArrayType:$rhs);
    let results = (outs Simp_ArrayType:$result);
}
```

### End-to-End Test Program

```simplang
// test_minimal.sl
fn kernel_main() {
    var x = 10.0;
    var y = 5.0;
    return x + y;
}
```

**Expected MLIR Output:**
```mlir
func.func @kernel_main() -> f64 {
    %c10 = simp.constant 10.0 : f64
    %c5 = simp.constant 5.0 : f64
    %result = simp.add %c10, %c5 : f64
    func.return %result : f64
}
```

### Iteration Plan

**Session 4-5:** Get this working ↑
**Session 6-7:** Add array operations
**Session 8-9:** Add matmul (just the operation, no lowering yet)
**Session 10+:** Expand dialect as needed

**Key Principle:** End-to-end codegen first, sophistication later.

---

## 7. Future Payoff: The Long Game

### Why This Investment Pays Off

Once `SimpLang → MLIR` exists, you unlock:

**1. SimpNN/SimpLang Fusion (Nearly Free)**
```mlir
// SimpLang arrays + SimpNN models in same IR
%model = simpnn.load "mobilenet.onnx"
%input = simp.array_create ...
%output = simpnn.infer %model, %input
```

**2. GPU Dialect Work (Trivial After MLIR)**
```mlir
// Same simp dialect, different lowering
simp.matmul → linalg.matmul → gpu.launch_func
```

**3. Custom AOT Kernels**
```mlir
// Plug in hand-optimized kernels
linalg.matmul → custom.optimized_gemm_avx512
```

**4. DSP/Accelerator Lowering**
```mlir
// Retarget to different hardware
simp → linalg → tile-to-dsp → custom_target
```

### The MLIR Advantage

Traditional compiler: Add GPU support = rewrite half the compiler
MLIR compiler: Add GPU support = add new lowering pass

**This is why MLIR is used by TensorFlow, PyTorch, ONNX.**

---

## 8. Implementation Checklist

### Phase 1 Kickoff

Before Session 1:
- [ ] Read MLIR Toy tutorial Parts 1-3
- [ ] Understand SSA vs imperative IR
- [ ] Familiarize with TableGen syntax

Session 1:
- [ ] Clone LLVM/MLIR
- [ ] Build MLIR successfully
- [ ] Run `mlir-opt --help` and explore built-in dialects

Sessions 2-3:
- [ ] Complete Toy tutorial Parts 4-7
- [ ] Study linalg dialect docs
- [ ] Draft minimal `simp` dialect spec

Sessions 4-7:
- [ ] Implement minimal dialect (5 operations)
- [ ] Get end-to-end codegen working
- [ ] Emit valid MLIR IR from SimpLang source

### Phase 1 Success Gate

At Session 11, you should have:
- ✅ Any .sl file → MLIR IR → LLVM IR → object code
- ✅ Basic arithmetic working
- ✅ Arrays compiling (even if slow)
- ✅ Build system stable

If YES to all → **Continue to Phase 2**
If NO → **Debug Phase 1 for 2-3 more sessions, then reassess**

---

## 9. Common Pitfalls to Avoid

### 1. Over-Engineering the Dialect

❌ **Don't:** Create 50 operations for every SimpLang feature
✅ **Do:** Start with 5-10 core operations, add as needed

### 2. Premature Optimization

❌ **Don't:** Try to beat LLVM's codegen in Phase 1
✅ **Do:** Get correctness first, optimize in Phase 2+

### 3. Fighting MLIR's Design

❌ **Don't:** Create imperative operations (mutations, side effects)
✅ **Do:** Embrace SSA and functional style

### 4. Skipping Toy Tutorial

❌ **Don't:** Jump straight to SimpLang integration
✅ **Do:** Actually complete the Toy tutorial - it's worth it

### 5. Complex Type System Early

❌ **Don't:** Implement full tensor type system in Phase 1
✅ **Do:** Use !simp.array<T> generic type, specialize later

---

## 10. Quick Reference Commands

### Essential MLIR Tools

```bash
# Verify MLIR module
mlir-opt input.mlir --verify-diagnostics

# Show available passes
mlir-opt --help | grep -A 5 "PASS OPTIONS"

# Dump pass pipeline
mlir-opt input.mlir --dump-pass-pipeline

# Lower to LLVM
mlir-opt input.mlir --convert-to-llvm

# Translate to LLVM IR
mlir-translate input.mlir --mlir-to-llvmir > output.ll

# Full pipeline test
mlir-opt input.mlir --simp-opt --convert-to-llvm | mlir-translate --mlir-to-llvmir | llc -filetype=obj
```

### Debugging Commands

```bash
# Print IR after each pass
mlir-opt input.mlir --simp-opt --print-ir-after-all

# Print IR before/after specific pass
mlir-opt input.mlir --print-ir-before=linalg-fusion --print-ir-after=linalg-fusion

# Show module structure
mlir-opt input.mlir --print-op-stats

# Verify at each step
mlir-opt input.mlir --simp-opt --verify-each
```

---

## Conclusion

The MLIR integration plan is technically sound. These implementation notes provide the **tactical, battle-tested details** to execute it successfully:

1. **Keep dialect SSA-pure** - First-class MLIR citizen
2. **Memref-first lowering** - Faster convergence than tensor path
3. **Late SimpBLAS lowering** - After fusion/tiling
4. **Start simple on testing** - 60-70% of test plan = CI-grade
5. **Expect infra work** - First 10 sessions = 80% build/CMake
6. **Minimal dialect first** - 5 operations, end-to-end, then iterate
7. **Future payoff is huge** - GPU/SimpNN/custom targets become trivial

**Green light to proceed.** The design is solid, the plan is realistic, and these notes give you the practical implementation wisdom to succeed.

**Kick off Phase 1 with: constant, add, array_create/get, matmul → valid MLIR module → iterate.**

Good luck! 🚀

---

## References

**Essential Reading Before Phase 1:**
- MLIR Toy Tutorial: https://mlir.llvm.org/docs/Tutorials/Toy/
- Linalg Dialect Rationale: https://mlir.llvm.org/docs/Rationale/RationaleLinalgDialect/
- Bufferization Guide: https://mlir.llvm.org/docs/Bufferization/

**Community Resources:**
- MLIR Discord: https://discord.gg/xS7Z362 (ask questions here!)
- MLIR Discourse: https://discourse.llvm.org/c/mlir/
- Example Projects: torch-mlir, IREE, ONNX-MLIR (steal patterns from these)

**When You Get Stuck:**
- Search MLIR test files: `llvm-project/mlir/test/` - examples of everything
- Search for patterns: `git grep -r "linalg.matmul" mlir/`
- Ask on Discord - MLIR community is very helpful
