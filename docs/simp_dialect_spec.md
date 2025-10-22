# Simp Dialect Specification

**Document Version:** 1.0
**Date:** 2025-10-19
**Status:** Draft - Session 2
**Purpose:** Define the custom MLIR dialect for SimpLang

---

## Table of Contents

1. [Overview](#overview)
2. [Design Principles](#design-principles)
3. [Type System](#type-system)
4. [Operations](#operations)
5. [Lowering Strategy](#lowering-strategy)
6. [TableGen Structure](#tablegen-structure)
7. [Pass Pipeline](#pass-pipeline)
8. [Key Questions Answered](#key-questions-answered)

---

## 1. Overview

The `simp` dialect is a custom MLIR dialect for SimpLang that represents high-level SimpLang semantics before progressive lowering to LLVM IR. It serves as the first MLIR representation after AST generation.

### Goals

- **Preserve High-Level Semantics**: Maintain array and matrix operation semantics for optimization
- **Enable MLIR Optimizations**: Leverage affine loop optimization, auto-vectorization, and fusion
- **SimpBLAS Integration**: Natural lowering path to SimpBLAS via linalg dialect
- **Progressive Lowering**: Support multi-level lowering: simp → memref → linalg → affine → llvm

### Non-Goals (Phase 1)

- Complex tensor operations (Phase 3)
- GPU support (Future)
- Dynamic shapes (Future)

---

## 2. Design Principles

### 2.1 SSA-Pure Operations (CRITICAL)

All `simp` dialect operations MUST be **SSA-pure** (single static assignment with no hidden state):

**✅ GOOD - Functional Style:**
```mlir
// Array set returns NEW array value
%arr2 = simp.array_set %arr1[%idx], %val : !simp.array<f64>
```

**❌ BAD - Mutation:**
```mlir
// In-place mutation breaks MLIR optimization
simp.array_set_inplace %arr[%idx], %val  // DON'T DO THIS!
```

**Why This Matters:**
- Affine passes can analyze data dependencies
- Vector passes can reason about memory access patterns
- Standard MLIR optimization passes work automatically
- Makes the dialect composable with upstream MLIR ecosystem

### 2.2 Memref-First Lowering

**Lowering Strategy:**
```
SimpLang AST
    ↓
simp dialect (arrays, matrices)
    ↓
memref dialect (memory representation)  ← Lower to memref EARLY
    ↓
linalg dialect (structured operations)
    ↓
affine/scf (optimized loops)
    ↓
llvm dialect
```

**Rationale:**
- MLIR's tensor path is still dynamic-shape heavy
- Memref path is more stable and better documented
- Faster convergence for SimpLang's use case
- Add tensor support later in Phase 3 if needed for SimpTensor

### 2.3 Minimal Phase 1 Scope

Start with **5 core operations** for end-to-end flow:

1. `simp.constant` - Literal values
2. `simp.add` - Binary arithmetic (representative operation)
3. `simp.array_create` - Array allocation
4. `simp.array_get` - Array load (functional)
5. `simp.matmul` - Matrix multiply (placeholder for Phase 3 SimpBLAS)

**Principle:** End-to-end first, sophistication later.

---

## 3. Type System

### 3.1 Core Types

#### `!simp.array<T>`

Generic array type for Phase 1.

**Syntax:**
```
simp-array-type ::= `!simp.array` `<` element-type `>`
```

**Examples:**
```mlir
!simp.array<f64>    // Array of f64
!simp.array<f32>    // Array of f32
!simp.array<i32>    // Array of i32
```

**Properties:**
- Element type: `T` (any MLIR type)
- Shape: Dynamic (runtime-determined)
- Memory: To be lowered to memref with alignment

**Future Extensions (Phase 3):**
```mlir
!simp.tensor<shape, T>  // Shaped tensor type
!simp.matrix<M, N, T>   // Typed matrix
```

### 3.2 Type Lowering

**simp → memref:**
```mlir
// Before
%arr = simp.array_create %len : i64 -> !simp.array<f64>

// After memref lowering
%arr = memref.alloc(%len) : memref<?xf64>  // Dynamic 1D memref
```

---

## 4. Operations

### 4.1 `simp.constant`

Materialize a constant value.

**Syntax:**
```mlir
simp.constant <value> : <type>
```

**Examples:**
```mlir
%c10 = simp.constant 10.0 : f64
%c5 = simp.constant 5 : i32
```

**TableGen Definition:**
```tablegen
def Simp_ConstantOp : Simp_Op<"constant", [NoSideEffect]> {
  let summary = "Constant value";
  let description = [{
    The "constant" operation creates an SSA value from a compile-time constant.
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";

  let hasFolder = 1;
}
```

**Lowering:** Direct to `arith.constant` or `llvm.mlir.constant`

### 4.2 `simp.add`

Binary addition operation (representative of all binary arithmetic).

**Syntax:**
```mlir
%result = simp.add %lhs, %rhs : <type>
```

**Examples:**
```mlir
%sum = simp.add %a, %b : f64
%total = simp.add %x, %y : i32
```

**TableGen Definition:**
```tablegen
def Simp_AddOp : Simp_Op<"add", [NoSideEffect, Commutative]> {
  let summary = "Addition operation";
  let description = [{
    Performs addition of two values of the same type.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";

  let hasCanonicalizer = 1;
}
```

**Canonicalization Patterns:**
- Fold constant operands: `constant(a) + constant(b) → constant(a+b)`
- Identity: `x + 0 → x`

**Lowering:** To `arith.addf` for floats, `arith.addi` for integers

**Note:** Similar operations for `simp.sub`, `simp.mul`, `simp.div` will be added incrementally in Phase 1.

### 4.3 `simp.array_create`

Allocate a new array with given size.

**Syntax:**
```mlir
%arr = simp.array_create %len : i64 -> !simp.array<T>
```

**Example:**
```mlir
%len = simp.constant 100 : i64
%arr = simp.array_create %len : i64 -> !simp.array<f64>
```

**TableGen Definition:**
```tablegen
def Simp_ArrayCreateOp : Simp_Op<"array_create", [NoSideEffect]> {
  let summary = "Create array with given size";
  let description = [{
    Allocates a new array with the specified number of elements.
    Returns a new SSA value representing the array.
  }];

  let arguments = (ins I64:$size);
  let results = (outs Simp_ArrayType:$result);

  let assemblyFormat = "$size attr-dict `:` type($result)";
}
```

**Lowering to Memref:**
```cpp
// Pattern: simp.array_create → memref.alloc
struct ArrayCreateToMemRefPattern : public ConversionPattern {
  LogicalResult matchAndRewrite(ArrayCreateOp op,
                                ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) {
    Value size = operands[0];
    Type elementType = op.getType().getElementType();

    // Create aligned allocation
    auto memrefType = MemRefType::get({ShapedType::kDynamicSize}, elementType);
    Value alloc = rewriter.create<memref::AllocOp>(
        op.getLoc(), memrefType, ValueRange{size});

    rewriter.replaceOp(op, alloc);
    return success();
  }
};
```

### 4.4 `simp.array_get`

Load element from array (functional - does not modify array).

**Syntax:**
```mlir
%val = simp.array_get %arr[%idx] : !simp.array<T>
```

**Example:**
```mlir
%idx = simp.constant 42 : i64
%val = simp.array_get %arr[%idx] : !simp.array<f64> -> f64
```

**TableGen Definition:**
```tablegen
def Simp_ArrayGetOp : Simp_Op<"array_get", [NoSideEffect]> {
  let summary = "Get element from array";
  let description = [{
    Loads an element from the array at the specified index.
    This is a functional operation - it does not modify the array.
  }];

  let arguments = (ins Simp_ArrayType:$array, I64:$index);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$array `[` $index `]` attr-dict `:` type($array) `->` type($result)";
}
```

**Lowering to Memref:**
```cpp
// Pattern: simp.array_get → memref.load
struct ArrayGetToMemRefPattern : public ConversionPattern {
  LogicalResult matchAndRewrite(ArrayGetOp op,
                                ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) {
    Value memref = operands[0];  // Already converted to memref
    Value index = operands[1];

    Value loaded = rewriter.create<memref::LoadOp>(
        op.getLoc(), memref, ValueRange{index});

    rewriter.replaceOp(op, loaded);
    return success();
  }
};
```

**Note on `simp.array_set`:**

Phase 1 may not need `array_set` if we can use memref stores directly after lowering. This simplifies the SSA-pure requirement. To be determined based on implementation needs.

### 4.5 `simp.matmul`

Matrix multiplication operation (placeholder for SimpBLAS integration in Phase 3).

**Syntax:**
```mlir
%C = simp.matmul %A, %B : (!simp.array<f64>, !simp.array<f64>) -> !simp.array<f64>
```

**Example:**
```mlir
// A is M×K, B is K×N, C is M×N (dimensions runtime-known)
%C = simp.matmul %A, %B : (!simp.array<f64>, !simp.array<f64>) -> !simp.array<f64>
```

**TableGen Definition:**
```tablegen
def Simp_MatMulOp : Simp_Op<"matmul", [NoSideEffect]> {
  let summary = "Matrix multiplication";
  let description = [{
    Performs matrix multiplication C = A @ B.
    Phase 1: Placeholder operation
    Phase 3: Lowers to linalg.matmul → sb_gemm_f32
  }];

  let arguments = (ins Simp_ArrayType:$lhs, Simp_ArrayType:$rhs);
  let results = (outs Simp_ArrayType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, $result)";
}
```

**Phase 1 Lowering:** Direct to naive loops

**Phase 3 Lowering Path:**
```
simp.matmul
  → linalg.matmul
  → sb_gemm_f32 (via pattern matching)
```

**Phase 3 SimpBLAS Integration (Critical Component):**
```cpp
// Pattern: linalg.matmul → sb_gemm_f32
struct LinalgMatmulToSimpBLASPattern : public OpRewritePattern<linalg::MatmulOp> {
  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const {
    // Extract matrices
    Value A = op.getInputs()[0];
    Value B = op.getInputs()[1];
    Value C = op.getOutputs()[0];

    // Extract dimensions (M, N, K)
    Value M = rewriter.create<memref::DimOp>(loc, A, 0);
    Value N = rewriter.create<memref::DimOp>(loc, B, 1);
    Value K = rewriter.create<memref::DimOp>(loc, A, 1);

    // Convert to i32 (SimpBLAS signature)
    M = rewriter.create<arith::IndexCastOp>(loc, i32Type, M);
    N = rewriter.create<arith::IndexCastOp>(loc, i32Type, N);
    K = rewriter.create<arith::IndexCastOp>(loc, i32Type, K);

    // Leading dimensions
    Value lda = K, ldb = N, ldc = N;

    // Create sb_gemm_f32 call
    auto funcOp = lookupOrCreateSimpBLASFunc(rewriter, module);
    rewriter.create<func::CallOp>(loc, funcOp,
                                  {M, N, K, A, lda, B, ldb, C, ldc});

    rewriter.eraseOp(op);
    return success();
  }
};
```

---

## 5. Lowering Strategy

### 5.1 Multi-Level Progressive Lowering

```
┌─────────────────┐
│  SimpLang AST   │
└────────┬────────┘
         │ mlir_codegen.cpp
         ▼
┌─────────────────┐
│  simp dialect   │  High-level: arrays, matmul
└────────┬────────┘
         │ LowerSimpToMemRefPass
         ▼
┌─────────────────┐
│ memref dialect  │  Memory representation (aligned)
└────────┬────────┘
         │ (Phase 3) LowerSimpToLinalgPass
         ▼
┌─────────────────┐
│ linalg dialect  │  Structured operations (matmul)
└────────┬────────┘
         │ LinalgToSimpBLASPass (LATE!)
         ▼
┌─────────────────┐
│ SimpBLAS calls  │  sb_gemm_f32(...)
└────────┬────────┘
         │ Existing passes
         ▼
┌─────────────────┐
│ affine/scf      │  Loop optimization
└────────┬────────┘
         │ AffineToStd, VectorizePass
         ▼
┌─────────────────┐
│ vector dialect  │  Auto-vectorization
└────────┬────────┘
         │ ConvertToLLVM passes
         ▼
┌─────────────────┐
│  LLVM dialect   │
└────────┬────────┘
         │ translateModuleToLLVMIR
         ▼
┌─────────────────┐
│    LLVM IR      │
└─────────────────┘
```

### 5.2 Pass Ordering (CRITICAL)

**Correct Order:**
```cpp
void buildSimpOptPipeline(PassManager &pm) {
  // 1. High-level simp optimizations
  pm.addPass(createCanonicalizerPass());

  // 2. Lower simp → memref
  pm.addPass(createLowerSimpToMemRefPass());

  // 3. (Phase 3) Lower to linalg for matmul
  pm.addPass(createLowerSimpToLinalgPass());

  // 4. Linalg optimizations BEFORE SimpBLAS
  pm.addPass(createLinalgFusionPass());
  pm.addPass(createLinalgTilingPass({32, 32, 32}));

  // 5. LATE SimpBLAS lowering (after fusion!)
  pm.addPass(createLinalgToSimpBLASPass());  // ← LATE!

  // 6. Continue lowering
  pm.addPass(createConvertLinalgToAffineLoopsPass());
  pm.addPass(createAffineVectorizePass());
  pm.addPass(createLowerAffinePass());

  // 7. Final LLVM conversion
  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(createConvertMemRefToLLVMPass());
  pm.addPass(createConvertFuncToLLVMPass());
}
```

**Why Late SimpBLAS Lowering:**
- ❌ Early lowering freezes the graph - no fusion opportunities
- ✅ Late lowering preserves optimization opportunities
- ✅ Let MLIR perform fusion, tiling BEFORE committing to BLAS

### 5.3 Conversion Targets

**Phase 1: Simp → Memref:**
```cpp
ConversionTarget target(getContext());
target.addLegalDialect<memref::MemRefDialect,
                       arith::ArithDialect,
                       func::FuncDialect>();
target.addIllegalDialect<SimpDialect>();
```

**Phase 3: Linalg → SimpBLAS:**
```cpp
ConversionTarget target(getContext());
target.addLegalDialect<func::FuncDialect,
                       memref::MemRefDialect,
                       arith::ArithDialect>();
target.addDynamicallyLegalOp<linalg::MatmulOp>([](linalg::MatmulOp op) {
  // Legal only if already converted to BLAS call
  return false;  // Force conversion
});
```

---

## 6. TableGen Structure

### 6.1 File Organization

```
include/mlir/
├── simp_dialect.hpp        // Dialect class declaration
├── simp_ops.hpp            // Operation class declarations (generated)
└── simp_types.hpp          // Type class declarations

include/mlir/Dialects/Simp/
├── SimpBase.td             // Dialect definition
├── SimpOps.td              // Operation definitions
└── SimpTypes.td            // Type definitions

src/mlir/
├── simp_dialect.cpp        // Dialect implementation
├── simp_ops.cpp            // Operation implementations (generated + custom)
└── simp_types.cpp          // Type implementations
```

### 6.2 Dialect Definition (SimpBase.td)

```tablegen
//===- SimpBase.td - Simp dialect definition ---------------*- tablegen -*-===//

#ifndef SIMP_BASE
#define SIMP_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Simp Dialect Definition
//===----------------------------------------------------------------------===//

def Simp_Dialect : Dialect {
  let name = "simp";
  let summary = "SimpLang high-level operations dialect";
  let description = [{
    The Simp dialect represents high-level SimpLang semantics before
    progressive lowering to LLVM IR. It preserves array and matrix
    operation semantics for optimization.
  }];

  let cppNamespace = "::mlir::simp";

  let useDefaultTypePrinterParser = 0;  // Custom type parsing
  let hasConstantMaterializer = 1;       // For constant folding
}

//===----------------------------------------------------------------------===//
// Base Simp Operation Definition
//===----------------------------------------------------------------------===//

class Simp_Op<string mnemonic, list<Trait> traits = []> :
    Op<Simp_Dialect, mnemonic, traits>;

#endif // SIMP_BASE
```

### 6.3 Type Definitions (SimpTypes.td)

```tablegen
//===- SimpTypes.td - Simp dialect types -------------------*- tablegen -*-===//

#ifndef SIMP_TYPES
#define SIMP_TYPES

include "SimpBase.td"

//===----------------------------------------------------------------------===//
// Simp Type Definitions
//===----------------------------------------------------------------------===//

def Simp_ArrayType : DialectType<Simp_Dialect,
    CPred<"$_self.isa<::mlir::simp::ArrayType>()">,
    "Simp array type"> {
  let description = [{
    Array type representing a 1D dynamically-sized array of elements.
    Syntax: !simp.array<element-type>
  }];
}

// Type constraint for operations
def Simp_AnyArray : Type<Simp_ArrayType.predicate, "any simp array">;

#endif // SIMP_TYPES
```

### 6.4 Operations File (SimpOps.td)

```tablegen
//===- SimpOps.td - Simp dialect operations ----------------*- tablegen -*-===//

#ifndef SIMP_OPS
#define SIMP_OPS

include "SimpBase.td"
include "SimpTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Simp Operations
//===----------------------------------------------------------------------===//

def Simp_ConstantOp : Simp_Op<"constant", [NoSideEffect]> {
  let summary = "Constant value";
  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
  let hasFolder = 1;
}

def Simp_AddOp : Simp_Op<"add", [NoSideEffect, Commutative]> {
  let summary = "Addition operation";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let hasCanonicalizer = 1;
}

def Simp_ArrayCreateOp : Simp_Op<"array_create", [NoSideEffect]> {
  let summary = "Create array with given size";
  let arguments = (ins I64:$size);
  let results = (outs Simp_ArrayType:$result);
  let assemblyFormat = "$size attr-dict `:` type($result)";
}

def Simp_ArrayGetOp : Simp_Op<"array_get", [NoSideEffect]> {
  let summary = "Get element from array";
  let arguments = (ins Simp_ArrayType:$array, I64:$index);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$array `[` $index `]` attr-dict `:` type($array) `->` type($result)";
}

def Simp_MatMulOp : Simp_Op<"matmul", [NoSideEffect]> {
  let summary = "Matrix multiplication";
  let arguments = (ins Simp_ArrayType:$lhs, Simp_ArrayType:$rhs);
  let results = (outs Simp_ArrayType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, $result)";
}

#endif // SIMP_OPS
```

---

## 7. Pass Pipeline

### 7.1 Phase 1 Pipeline (Sessions 1-16)

**Goal:** Basic MLIR integration with feature parity

```cpp
// src/mlir/passes/simp_passes.cpp

std::unique_ptr<Pass> createLowerSimpToMemRefPass();

void registerSimpPasses() {
  PassRegistration<LowerSimpToMemRefPass>();
}

void buildPhase1Pipeline(PassManager &pm) {
  // Canonicalization
  pm.addPass(createCanonicalizerPass());

  // Lower simp → memref
  pm.addPass(createLowerSimpToMemRefPass());

  // Standard conversions
  pm.addPass(createConvertMemRefToLLVMPass());
  pm.addPass(createConvertFuncToLLVMPass());
}
```

### 7.2 Phase 2 Pipeline (Sessions 17-28)

**Goal:** Array optimization and auto-vectorization

```cpp
void buildPhase2Pipeline(PassManager &pm) {
  // High-level optimizations
  pm.addPass(createCanonicalizerPass());

  // Lower to memref
  pm.addPass(createLowerSimpToMemRefPass());

  // Affine optimization
  pm.addPass(createLowerToAffinePass());
  pm.addPass(createAffineLoopFusionPass());
  pm.addPass(createAffineLoopTilePass({32, 32}));

  // Vectorization
  pm.addPass(createAffineVectorizePass());

  // Final lowering
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(createConvertMemRefToLLVMPass());
  pm.addPass(createConvertFuncToLLVMPass());
}
```

### 7.3 Phase 3 Pipeline (Sessions 29-52)

**Goal:** Tensor operations and SimpBLAS integration

```cpp
void buildPhase3Pipeline(PassManager &pm) {
  // High-level optimizations
  pm.addPass(createCanonicalizerPass());

  // Lower simp → memref, linalg
  pm.addPass(createLowerSimpToMemRefPass());
  pm.addPass(createLowerSimpToLinalgPass());  // For matmul

  // Linalg optimizations (BEFORE BLAS!)
  pm.addPass(createLinalgFusionPass());
  pm.addPass(createLinalgTilingPass({32, 32, 32}));
  pm.addPass(createLinalgBufferizePass());

  // LATE SimpBLAS lowering
  pm.addPass(createLinalgToSimpBLASPass());  // ← CRITICAL

  // Continue with affine/vector
  pm.addPass(createConvertLinalgToAffineLoopsPass());
  pm.addPass(createAffineVectorizePass());
  pm.addPass(createLowerAffinePass());

  // Final LLVM lowering
  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(createConvertMemRefToLLVMPass());
  pm.addPass(createConvertFuncToLLVMPass());
}
```

---

## 8. Key Questions Answered

### Question 1: How do we implement pattern matching for optimizations?

**Answer:** Using two approaches from Toy tutorial:

**A. Imperative C++ Pattern Matching:**
```cpp
struct SimplifyRedundantOp : public OpRewritePattern<MyOp> {
  LogicalResult matchAndRewrite(MyOp op, PatternRewriter &rewriter) const {
    // Match condition
    if (!shouldOptimize(op))
      return failure();

    // Perform rewrite
    rewriter.replaceOp(op, newValue);
    return success();
  }
};

// Register with canonicalizer
void MyOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                       MLIRContext *context) {
  results.add<SimplifyRedundantOp>(context);
}
```

**B. Declarative DRR (TableGen):**
```tablegen
// Pattern: constant fold addition
def FoldConstantAdd : Pat<
  (Simp_AddOp (Simp_ConstantOp $a), (Simp_ConstantOp $b)),
  (Simp_ConstantOp (AddValues $a, $b))
>;

// With constraints
def SimplifyIdentity : Pat<
  (Simp_AddOp $x, (Simp_ConstantOp $zero)),
  (replaceWithValue $x),
  [(IsZero $zero)]  // Constraint
>;
```

**For SimpLang:**
- Use DRR for simple patterns (constant folding, identities)
- Use C++ for complex patterns (SimpBLAS lowering)

---

### Question 2: What's the exact process for dialect lowering (simp → memref)?

**Answer:** Using MLIR's `DialectConversion` framework with 3 components:

**Step 1: Define Conversion Target**
```cpp
void LowerSimpToMemRefPass::runOnOperation() {
  ConversionTarget target(getContext());

  // memref, arith, func are legal
  target.addLegalDialect<memref::MemRefDialect,
                         arith::ArithDialect,
                         func::FuncDialect>();

  // simp dialect is illegal (must be converted)
  target.addIllegalDialect<SimpDialect>();

  ...
}
```

**Step 2: Define Conversion Patterns**
```cpp
// Pattern uses ConversionPattern (not OpRewritePattern!)
struct ArrayCreateToMemRefPattern : public ConversionPattern {
  ArrayCreateToMemRefPattern(MLIRContext *ctx)
      : ConversionPattern(ArrayCreateOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op,
      ArrayRef<Value> operands,  // Already remapped to new types!
      ConversionPatternRewriter &rewriter) const override {

    auto arrayCreateOp = cast<ArrayCreateOp>(op);
    Value size = operands[0];  // Remapped operand

    // Create memref allocation
    Type elementType = arrayCreateOp.getType().getElementType();
    auto memrefType = MemRefType::get({ShapedType::kDynamicSize}, elementType);

    Value alloc = rewriter.create<memref::AllocOp>(
        op->getLoc(), memrefType, ValueRange{size});

    rewriter.replaceOp(op, alloc);
    return success();
  }
};
```

**Step 3: Collect Patterns and Apply Conversion**
```cpp
void LowerSimpToMemRefPass::runOnOperation() {
  ...

  // Collect patterns
  RewritePatternSet patterns(&getContext());
  patterns.add<ArrayCreateToMemRefPattern,
               ArrayGetToMemRefPattern,
               /* ... more patterns */>();

  // Apply partial conversion (some ops may remain)
  if (failed(applyPartialConversion(getOperation(), target, patterns)))
    signalPassFailure();
}
```

**Key Insight:**
- `ConversionPattern` receives **remapped operands** (already converted types)
- This is critical for type conversions (simp.array → memref)

---

### Question 3: How do we define dialect interfaces for optimization passes?

**Answer:** Using Operation Interfaces defined in ODS (TableGen):

**Step 1: Define Interface in ODS**
```tablegen
// include/mlir/Dialects/Simp/SimpInterfaces.td

def SimpShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let description = [{
    Interface for operations that support shape inference.
    Used by shape propagation pass.
  }];

  let methods = [
    InterfaceMethod<
      "Infer and set output shape for this operation",
      "void", "inferShapes"
    >
  ];
}
```

**Step 2: Add Interface to Operations**
```tablegen
// In SimpOps.td
def Simp_AddOp : Simp_Op<"add", [
    NoSideEffect,
    Commutative,
    DeclareOpInterfaceMethods<SimpShapeInferenceOpInterface>  // Add interface
  ]> {
  ...
}
```

**Step 3: Implement Interface Methods in C++**
```cpp
// src/mlir/simp_ops.cpp

void AddOp::inferShapes() {
  // Infer result shape from operands
  Type lhsType = getLhs().getType();
  getResult().setType(lhsType);
}
```

**Step 4: Use Interface in Pass**
```cpp
// Shape inference pass
void ShapeInferencePass::runOnOperation() {
  FuncOp function = getOperation();

  // Build worklist
  for (Operation &op : function.getOps()) {
    if (ShapeInference shapeOp = dyn_cast<ShapeInference>(&op)) {
      // Operation implements interface
      shapeOp.inferShapes();
    }
  }
}
```

**For SimpLang, we'll define interfaces for:**
- Shape inference (Phase 2)
- Cost model (Phase 3 - for SimpBLAS heuristics)
- Memory layout (Phase 3 - for optimization)

---

### Question 4: What's the structure of a complete .td file for operations?

**Answer:** Based on Toy tutorial and Linalg dialect:

**Complete Example - SimpOps.td:**
```tablegen
//===- SimpOps.td - Simp dialect operations ----------------*- tablegen -*-===//

// 1. Include directives
#ifndef SIMP_OPS
#define SIMP_OPS

include "SimpBase.td"
include "SimpTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// 2. Operation Definitions
//===----------------------------------------------------------------------===//

// Template for operations
def Simp_BinaryOp<string mnemonic> : Simp_Op<mnemonic, [
    NoSideEffect,        // Trait: operation has no side effects
    Commutative,         // Trait: operands can be swapped
    SameOperandsAndResultType  // Constraint: types must match
  ]> {

  let summary = "Binary operation: " # mnemonic;

  let description = [{
    Performs binary operation on two operands of the same type.
  }];

  // Arguments (inputs)
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);

  // Results (outputs)
  let results = (outs AnyType:$result);

  // Assembly format (how it prints/parses)
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";

  // Optional: enable canonicalization
  let hasCanonicalizer = 1;

  // Optional: enable constant folding
  let hasFolder = 1;

  // Optional: custom C++ code
  let extraClassDeclaration = [{
    // Helper methods can be added here
    Type getElementType() { return getResult().getType(); }
  }];
}

// Instantiate specific operations
def Simp_AddOp : Simp_BinaryOp<"add">;
def Simp_SubOp : Simp_BinaryOp<"sub">;
def Simp_MulOp : Simp_BinaryOp<"mul">;
def Simp_DivOp : Simp_BinaryOp<"div">;

// Complex operation with custom builder
def Simp_ArrayCreateOp : Simp_Op<"array_create", [NoSideEffect]> {
  let summary = "Create array with given size";

  let arguments = (ins I64:$size);
  let results = (outs Simp_ArrayType:$result);

  let assemblyFormat = "$size attr-dict `:` type($result)";

  // Custom builder for convenience
  let builders = [
    OpBuilder<(ins "Value":$size, "Type":$elementType), [{
      Type arrayType = ArrayType::get($_builder.getContext(), elementType);
      build($_builder, $_state, arrayType, size);
    }]>
  ];

  // Verification logic
  let verifier = [{ return ::verify(*this); }];
}

#endif // SIMP_OPS
```

**Key Components:**
1. **Include directives** - Import base definitions and interfaces
2. **Operation traits** - NoSideEffect, Commutative, etc.
3. **Arguments/Results** - `(ins ...)` and `(outs ...)`
4. **Assembly format** - How operation prints/parses
5. **Flags** - hasCanonicalizer, hasFolder, hasVerifier
6. **Builders** - Custom construction methods
7. **Extra declarations** - Helper methods in C++

---

### Question 5: How do we integrate with LLVM dialect in the final step?

**Answer:** Using standard MLIR conversion passes with `TypeConverter`:

**Step 1: Set up Type Converter**
```cpp
void LowerToLLVMPass::runOnOperation() {
  // Type converter handles type translation
  LLVMTypeConverter typeConverter(&getContext());

  // Custom type conversions if needed
  typeConverter.addConversion([](SimpArrayType type) {
    // Convert !simp.array<T> to LLVM struct
    // (This should be done in memref conversion, not here)
    return nullptr;  // Fall back to default
  });

  ...
}
```

**Step 2: Define Conversion Target**
```cpp
  ConversionTarget target(getContext());

  // Only LLVM dialect is legal
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<ModuleOp>();  // Module is always legal
```

**Step 3: Reuse Standard Patterns**
```cpp
  RewritePatternSet patterns(&getContext());

  // Populate with standard conversions (transitive lowering!)
  populateAffineToStdConversionPatterns(patterns);
  populateArithmeticToLLVMConversionPatterns(typeConverter, patterns);
  populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  populateVectorToLLVMConversionPatterns(typeConverter, patterns);
```

**Step 4: Apply Full Conversion**
```cpp
  // Full conversion - all ops must be legal
  if (failed(applyFullConversion(getOperation(), target, patterns)))
    signalPassFailure();
```

**Step 5: Export to LLVM IR**
```cpp
// In main compiler driver (src/main.cpp)

// After MLIR passes complete:
llvm::LLVMContext llvmContext;
auto llvmModule = translateModuleToLLVMIR(mlirModule, llvmContext);

if (!llvmModule) {
  llvm::errs() << "Failed to emit LLVM IR\n";
  return -1;
}

// Optional: Run LLVM optimization passes
auto optPipeline = makeOptimizingTransformer(
    /*optLevel=*/3, /*sizeLevel=*/0, /*targetMachine=*/nullptr);
optPipeline(llvmModule.get());

// Emit object code
// (Integrate with existing SimpLang compilation pipeline)
```

**Key Insights:**
- **Transitive Lowering**: Don't need to emit LLVM dialect directly
- **Reuse Standard Patterns**: MLIR provides conversions for standard dialects
- **TypeConverter**: Handles block argument type conversion automatically
- **Full Conversion**: Ensures all operations are lowered

**For SimpLang:**
- Phase 1: Rely entirely on standard conversion patterns
- Phase 2/3: May need custom patterns for specific optimizations
- Final step: Export to LLVM IR via `translateModuleToLLVMIR`

---

## 9. Example End-to-End Flow

### SimpLang Input
```simplang
fn kernel_main() {
  var arr = make(Array, 100);
  arr[0] = 42.0;
  var x = arr[0];
  return x + 10.0;
}
```

### Stage 1: Simp Dialect (After MLIRCodeGen)
```mlir
func.func @kernel_main() -> f64 {
  %c100 = simp.constant 100 : i64
  %c0 = simp.constant 0 : i64
  %c42 = simp.constant 42.0 : f64
  %c10 = simp.constant 10.0 : f64

  %arr = simp.array_create %c100 : i64 -> !simp.array<f64>
  // Note: array_set might be direct memref store after lowering
  %x = simp.array_get %arr[%c0] : !simp.array<f64> -> f64
  %result = simp.add %x, %c10 : f64

  func.return %result : f64
}
```

### Stage 2: After Memref Lowering
```mlir
func.func @kernel_main() -> f64 {
  %c100 = arith.constant 100 : index
  %c0 = arith.constant 0 : index
  %c42 = arith.constant 42.0 : f64
  %c10 = arith.constant 10.0 : f64

  %arr = memref.alloc(%c100) : memref<?xf64>
  memref.store %c42, %arr[%c0] : memref<?xf64>
  %x = memref.load %arr[%c0] : memref<?xf64>
  %result = arith.addf %x, %c10 : f64

  memref.dealloc %arr : memref<?xf64>
  func.return %result : f64
}
```

### Stage 3: After LLVM Dialect Conversion
```mlir
llvm.func @kernel_main() -> f64 {
  %c100 = llvm.mlir.constant(100 : i64) : i64
  %c42 = llvm.mlir.constant(42.0 : f64) : f64
  %c10 = llvm.mlir.constant(10.0 : f64) : f64

  %ptr = llvm.call @malloc(%c100) : (i64) -> !llvm.ptr<f64>
  llvm.store %c42, %ptr : !llvm.ptr<f64>
  %x = llvm.load %ptr : !llvm.ptr<f64>
  %result = llvm.fadd %x, %c10 : f64

  llvm.call @free(%ptr) : (!llvm.ptr<f64>) -> ()
  llvm.return %result : f64
}
```

### Stage 4: LLVM IR (Final)
```llvm
define double @kernel_main() {
  %1 = call i8* @malloc(i64 800)
  %2 = bitcast i8* %1 to double*
  store double 42.0, double* %2
  %3 = load double, double* %2
  %4 = fadd double %3, 10.0
  call void @free(i8* %1)
  ret double %4
}
```

---

## 10. Next Steps

### Session 3: Dialect Implementation
1. Implement `SimpDialect` class
2. Implement `ArrayType` with TypeStorage
3. Set up CMake integration with MLIR
4. Register dialect and types

### Session 4-7: Operations & Lowering
1. Generate operation classes from TableGen
2. Implement lowering passes (simp → memref)
3. Write unit tests with mlir-opt
4. Test end-to-end compilation

### Success Criteria (End of Phase 1)
- ✅ `test_arithmetic.sl` compiles via MLIR
- ✅ Output matches LLVM IR path
- ✅ Performance >= 95% of baseline
- ✅ All unit tests pass

---

## References

**MLIR Documentation:**
- Toy Tutorial: https://mlir.llvm.org/docs/Tutorials/Toy/
- Dialect Conversion: https://mlir.llvm.org/docs/DialectConversion/
- Operation Definition: https://mlir.llvm.org/docs/OpDefinitions/
- Declarative Rewrites: https://mlir.llvm.org/docs/DeclarativeRewrites/

**Linalg Dialect:**
- Linalg Docs: https://mlir.llvm.org/docs/Dialects/Linalg/
- LinalgOps.td: `llvm-project/mlir/include/mlir/Dialect/Linalg/IR/`

**SimpLang Project:**
- Implementation Notes: `docs/mlir_implementation_notes.md`
- Integration Plan: `docs/mlir_integration_plan.md`

---

**Document Status:** Draft - Ready for Session 3
**Next Update:** After dialect implementation (Session 4)
