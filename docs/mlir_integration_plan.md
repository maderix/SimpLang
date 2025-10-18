# MLIR Integration for SimpLang - Comprehensive Analysis

**Document Version:** 1.1 (AI-Assisted Timeline)
**Date:** 2025-01-18
**Branch:** `feature/mlir-integration`
**Status:** Planning Phase
**Development Approach:** AI-Assisted Pair Programming (3-4 hour focused sessions)

---

## Executive Summary

**Recommendation: MLIR integration is HIGHLY FEASIBLE and STRATEGICALLY VALUABLE** for SimpLang's future, particularly given its deep learning and SIMD optimization focus. This is a **major architectural change** that can be accomplished with AI-assisted development.

**Key Benefits:**
- Better tensor/matrix operation optimization (critical for MobileNet POC)
- More powerful loop vectorization and polyhedral optimization
- Seamless SimpBLAS integration with linalg dialect
- Easier GPU backend integration (future CUDA/OpenCL support)
- Industry-standard IR used by TensorFlow, PyTorch, ONNX
- Multi-level abstraction preserving high-level semantics

**Integration Complexity:** Medium-High (existing LLVM knowledge transfers well)

**Risk Level:** Medium (Manageable with proper planning)

**Timeline with AI Assistance:** 9-13 weeks (36-52 focused 3-4 hour sessions)

---

## Part 1: MLIR Overview & Strategic Fit

### 1.1 What is MLIR?

MLIR (Multi-Level Intermediate Representation) is a compiler infrastructure developed by Chris Lattner (creator of LLVM) at Google in 2018, now part of LLVM. It addresses limitations in traditional single-level IRs like LLVM IR.

**Core Concepts:**
- **Dialects**: Extensible operation sets for different abstraction levels
- **Progressive Lowering**: High-level → Mid-level → LLVM IR → Machine Code
- **Unified Infrastructure**: Multiple IRs coexist in single compilation unit

**Official Documentation:** https://mlir.llvm.org/

### 1.2 Why MLIR for SimpLang?

**Perfect Alignment with Project Goals:**

1. **Deep Learning Focus**: Built-in tensor operations, used by TensorFlow/PyTorch
2. **Array/Matrix Operations**: Linalg dialect perfectly matches array-based approach
3. **SimpBLAS Integration**: Natural fit with linalg → BLAS lowering
4. **Auto-Vectorization**: Superior loop vectorization for array operations
5. **Future GPU Support**: Native GPU dialects (GPU, NVGPU, AMDGPU)
6. **Tensor Library**: SimpTensor would benefit from tensor dialect
7. **MobileNet POC**: Linalg dialect designed for ML workload optimization

**Industry Adoption:**
- TensorFlow XLA compiler
- PyTorch (torch-mlir)
- ONNX (ONNX-MLIR)
- Xilinx AI Engine compiler
- IREE (Intermediate Representation Execution Environment)

### 1.3 MLIR vs LLVM IR

| Aspect | LLVM IR | MLIR |
|--------|---------|------|
| **Abstraction Level** | Single (low-level) | Multiple (high to low) |
| **Extensibility** | Limited | Highly extensible (dialects) |
| **Domain-Specific** | Generic | Supports domain-specific ops |
| **Optimization** | Function-level | Multi-level progressive |
| **Tensor Operations** | Manual lowering | Built-in tensor/linalg |
| **GPU Support** | Limited | First-class GPU dialects |

---

## Part 2: Feasibility Analysis

### 2.1 Technical Feasibility: ✅ HIGH

**Strengths of Current Codebase:**

✅ **Clean Modular AST**: Recent refactoring into `ast/expr`, `ast/stmt`, `ast/type` provides excellent foundation
- `include/ast/expr/`: Expression nodes (literal, variable, operator, call, array, etc.)
- `include/ast/stmt/`: Statement nodes (declaration, control flow, function, etc.)
- `include/ast/type/`: Type system (type_info, simd_types)

✅ **Well-Defined Compilation Pipeline** (src/main.cpp:82-105):
```cpp
yyparse() → programBlock (AST) → context.generateCode() → LLVM IR → Object Code
```

✅ **LLVM Integration Experience**: Already using LLVM 14+
- Target machine configuration (src/codegen.cpp:18-73)
- Optimization passes (src/main.cpp:143-154)
- Code generation pipeline

✅ **SimpBLAS Integration** (src/codegen.cpp:751-811):
```cpp
void generateGemmCall(M, N, K, A, B, C) → sb_gemm_f32(...)
```

✅ **Array-Based Architecture**: No deprecated slice types
- Arrays are first-class citizens
- Clean separation from deprecated SSE/AVX slices

✅ **Good Test Infrastructure**:
- `run_tests.sh`: Comprehensive test runner
- Unit tests for arithmetic, loops, functions
- Performance benchmarks

**Integration Points Identified:**

1. **AST → MLIR Dialect** (new layer between AST and LLVM IR)
2. **MLIR → LLVM IR** (existing MLIR infrastructure provides this)
3. **Array Operations → Linalg Dialect → SimpBLAS** (pattern matching)
4. **Fallback Path**: Keep current LLVM IR generation as option

### 2.2 Resource Requirements

**Development Time Estimate (AI-Assisted Development):**

| Phase | Sessions | Calendar Time | Key Deliverables |
|-------|----------|---------------|------------------|
| Phase 1: Foundation | 12-16 sessions | 3-4 weeks | Basic MLIR integration, feature parity |
| Phase 2: Arrays & Vectorization | 8-12 sessions | 2-3 weeks | Array ops, auto-vectorization |
| Phase 3: Tensor & SimpBLAS | 16-24 sessions | 4-6 weeks | Linalg integration, BLAS lowering |
| **Total** | **36-52 sessions** | **9-13 weeks** | **2-3 months** |

**Note:** Assumes 3-4 hour focused AI pair-programming sessions, 3-4 sessions per week. AI significantly accelerates:
- Boilerplate code generation (dialects, operations, passes)
- Test code generation
- Documentation writing
- Research and pattern finding

**Critical path items that still require time:**
- MLIR learning curve (concepts, TableGen, etc.)
- Integration debugging and testing
- Performance tuning (iterative process)
- Architectural decision-making

**Infrastructure Dependencies:**
- LLVM/MLIR 14+ (already have LLVM 14 ✅)
- Additional ~200MB build artifacts
- No new external dependencies
- CMake 3.20+ (already have ✅)

**Build System Changes:**
- Add MLIR CMake configuration
- New compiler flag: `-DUSE_MLIR=ON` (default OFF for gradual rollout)
- Parallel build paths (LLVM IR and MLIR coexist)

### 2.3 Integration Complexity Analysis

**Complexity Breakdown:**

| Component | Complexity | Rationale |
|-----------|------------|-----------|
| AST → MLIR Lowering | **Medium** | Similar to current AST → LLVM IR |
| Dialect Design | **Medium** | Well-documented patterns, good examples |
| Array → Linalg Lowering | **Low-Medium** | Well-established patterns in MLIR community |
| SimpBLAS Integration | **Low** | MLIR → BLAS is common pattern |
| Optimization Passes | **Low-Medium** | Reuse built-in MLIR passes |
| Testing | **Medium** | Need new test infrastructure, but clear requirements |
| Performance Tuning | **High** | Iterative process, requires experimentation |

**Overall Complexity:** Medium-High (manageable with phased approach)

### 2.4 Risk Assessment

**Identified Risks:**

| Risk | Severity | Probability | Mitigation Strategy |
|------|----------|-------------|---------------------|
| **Learning Curve** | Medium | High | Excellent documentation, active community, Toy tutorial |
| **Build Time** | Low-Medium | Medium | Phased integration, optional flag, incremental builds |
| **Regression Risk** | Medium-High | Medium | Parallel LLVM path, comprehensive test suite, gradual rollout |
| **Maintenance Burden** | Medium | Low | Better abstractions offset complexity, long-term benefit |
| **Performance Regression** | Medium | Low | Continuous benchmarking, performance gates in CI |
| **API Churn** | Low | Low | MLIR is stable (part of LLVM since 2019) |

**Overall Risk Level:** MEDIUM (Manageable with proper planning and phased rollout)

**Risk Mitigation Plan:**
1. Keep LLVM IR path as fallback
2. Comprehensive test coverage at each phase
3. Performance benchmarks run automatically
4. Community engagement (MLIR Discord, forums)
5. Incremental rollout (opt-in initially)

### 2.5 Alternative Approaches Considered

**Option A: Continue with LLVM IR Only**
- ✅ No learning curve
- ✅ Existing implementation works
- ❌ Limited optimization for tensor operations
- ❌ Harder GPU backend integration
- ❌ Manual vectorization optimizations

**Option B: Full MLIR Migration**
- ✅ Maximum optimization potential
- ✅ Better long-term architecture
- ❌ High risk (all-or-nothing)
- ❌ Long development time
- ❌ Difficult rollback

**Option C: Hybrid Approach (RECOMMENDED)**
- ✅ Gradual migration
- ✅ Fallback to LLVM IR
- ✅ Focus on high-value use cases (tensors, arrays)
- ✅ Lower risk
- ✅ Proven in industry (TensorFlow uses hybrid)

**Decision: Proceed with Option C (Hybrid Approach)**

---

## Part 3: Detailed Design Plan

### 3.1 Architecture Overview

**Current Pipeline:**
```
.sl Source Code
    ↓
Lexer (src/lexer.l) → Tokens
    ↓
Parser (src/parser.y) → AST
    ↓
CodeGen (src/codegen.cpp) → LLVM IR
    ↓
LLVM Optimizations (src/main.cpp:143-154)
    ↓
Object Code (.o/.so)
    ↓
SimpBLAS Runtime Calls (sb_gemm_f32, etc.)
```

**Proposed MLIR Pipeline:**
```
.sl Source Code
    ↓
Lexer (src/lexer.l) → Tokens
    ↓
Parser (src/parser.y) → AST
    ↓
MLIR CodeGen (NEW: src/mlir/mlir_codegen.cpp)
    ↓
simp Dialect (HIGH-LEVEL: arrays, matrices, tensors)
    ↓
Linalg/Tensor/Memref Dialects (MID-LEVEL: structured operations)
    ↓
Pattern Matching: linalg.matmul → SimpBLAS Calls
    ↓
Affine Dialect (LOOP OPTIMIZATION: polyhedral)
    ↓
Vector Dialect (AUTO-VECTORIZATION: SSE/AVX/AVX-512)
    ↓
LLVM Dialect
    ↓
LLVM IR
    ↓
Object Code + SimpBLAS Calls
```

**Key Advantage:** Multi-level optimization with progressive lowering

### 3.2 MLIR Dialects Strategy

**Dialects to Use:**

1. **`simp` (Custom)**: SimpLang-specific high-level operations
   - Purpose: First lowering target from AST
   - Operations: array_create, array_get/set, matmul, tensor ops
   - Types: !simp.array<T>, !simp.tensor<T>

2. **`linalg`**: Linear algebra operations
   - Purpose: Matrix/tensor operations before BLAS
   - Operations: matmul, conv, transpose, etc.
   - **Key**: Maps to SimpBLAS via custom lowering pass

3. **`tensor` / `memref`**: Array storage
   - Purpose: Memory representation
   - tensor: Immutable SSA values (functional)
   - memref: Mutable memory references (imperative)

4. **`affine`**: Polyhedral loop optimization
   - Purpose: Loop-level optimizations
   - Operations: affine.for, affine.if, affine.load/store
   - Enables: Loop fusion, tiling, interchange

5. **`vector`**: SIMD operations
   - Purpose: Auto-vectorization target
   - Operations: vector.load, vector.store, vector.fma
   - Targets: SSE, AVX, AVX-512

6. **`scf`**: Structured control flow
   - Purpose: General control flow
   - Operations: scf.for, scf.if, scf.while

7. **`arith` / `math`**: Arithmetic operations
   - Purpose: Standard math operations
   - Operations: arith.addf, math.sqrt, etc.

8. **`func`**: Function definitions
   - Purpose: Function abstraction
   - Operations: func.func, func.call, func.return

9. **`llvm`**: LLVM dialect
   - Purpose: Final lowering target
   - Maps directly to LLVM IR

### 3.3 Phased Integration Strategy

#### Phase 1: Foundation (12-16 sessions, 3-4 weeks)

**Goal:** Basic MLIR integration with feature parity

**Session Breakdown:**
- Sessions 1-3: MLIR build setup, study Toy tutorial, initial dialect design
- Sessions 4-7: Implement `simp` dialect and basic AST lowering (AI generates boilerplate)
- Sessions 8-11: Implement MLIR → LLVM IR lowering
- Sessions 12-16: Testing, validation, bug fixing

**Deliverables:**

1. **Build System**
   - CMake configuration for MLIR
   - `-DUSE_MLIR=ON` flag
   - Parallel build of LLVM and MLIR paths

2. **`simp` Dialect Implementation**
   ```cpp
   // include/mlir/simp_dialect.hpp
   class SimpDialect : public mlir::Dialect {
       // Dialect registration
   };

   // Operations:
   // - simp.array_create
   // - simp.array_get / simp.array_set
   // - simp.constant
   // - simp.add / simp.sub / simp.mul / simp.div
   ```

3. **AST → MLIR Lowering**
   ```cpp
   // src/mlir/mlir_codegen.cpp
   class MLIRCodeGen {
       mlir::Value* lowerExpr(ExprAST* expr);
       mlir::Value* lowerStmt(StmtAST* stmt);
       mlir::FuncOp lowerFunction(FunctionDeclAST* func);
   };
   ```

4. **MLIR → LLVM Lowering**
   - Use standard MLIR conversion passes
   - Generate LLVM IR compatible with existing runtime

5. **Tests Passing**
   - test_arithmetic.sl
   - test_loop.sl
   - test_return.sl
   - Basic array operations

**Success Criteria:**
- ✅ All basic tests pass
- ✅ Numerical accuracy within 1e-10
- ✅ Compilation time < 2x LLVM path
- ✅ Runtime performance >= 95%

#### Phase 2: Array Operations & Vectorization (8-12 sessions, 2-3 weeks)

**Goal:** Leverage MLIR for array optimization and auto-vectorization

**Session Breakdown:**
- Sessions 1-3: Array operations → memref/tensor dialects (AI helps with lowering patterns)
- Sessions 4-6: Affine optimization passes and tuning
- Sessions 7-9: Vector dialect and auto-vectorization
- Sessions 10-12: Performance testing and optimization (if needed)

**Deliverables:**

1. **Array Lowering**
   ```mlir
   // simp.array_create → memref.alloc
   %arr = simp.array_create %len : i64 -> !simp.array<f32>
   ⇓
   %arr = memref.alloc(%len) : memref<?xf32>
   ```

2. **Affine Loop Optimization**
   ```mlir
   // Enable polyhedral optimizations
   scf.for %i → affine.for %i

   // Apply passes:
   // - affine-loop-fusion
   // - affine-loop-tile
   // - affine-loop-unroll
   ```

3. **Auto-Vectorization**
   ```mlir
   // affine-super-vectorize pass
   affine.for %i → vector operations

   // Example:
   affine.for %i = 0 to %N {
       %v = affine.load %A[%i]
       %r = arith.addf %v, %c
       affine.store %r, %B[%i]
   }
   ⇓
   vector.load %A[%i : %i+8] → <8xf32>
   vector.add → <8xf32>
   vector.store → %B[%i : %i+8]
   ```

4. **Optimization Pipeline**
   ```cpp
   // src/mlir/mlir_pipeline.cpp
   void buildArrayOptimizationPipeline(PassManager &pm) {
       pm.addPass(createLowerSimpToMemRefPass());
       pm.addPass(createAffineLoopFusionPass());
       pm.addPass(createAffineLoopTilePass({32, 32}));
       pm.addPass(createAffineVectorizePass(
           {.vectorSize = 8, .fastestVarying = 1}));
       pm.addPass(createLowerAffinePass());
   }
   ```

**Success Criteria:**
- ✅ All array tests pass
- ✅ Auto-vectorization generates vector instructions
- ✅ Performance on array code >= 100% of LLVM path
- ✅ Memory layout optimizations measurable

#### Phase 3: Tensor Operations & SimpBLAS Integration (16-24 sessions, 4-6 weeks)

**Goal:** Enable advanced ML optimizations with SimpBLAS

**Session Breakdown:**
- Sessions 1-3: Linalg dialect integration
- Sessions 4-7: Linalg → SimpBLAS lowering (CRITICAL - needs careful implementation)
- Sessions 8-11: SimpTensor integration and testing
- Sessions 12-16: MobileNet POC optimization (iterative tuning)
- Sessions 17-20: Performance tuning and comprehensive benchmarking
- Sessions 21-24: Documentation, final polish, and migration guide

**Deliverables:**

1. **Matrix Operation Lowering**
   ```mlir
   // simp.matmul → linalg.matmul
   %C = simp.matmul %A, %B : (!simp.array<f32>, !simp.array<f32>)
   ⇓
   %C = linalg.matmul ins(%A, %B : memref<?x?xf32>, memref<?x?xf32>)
                       outs(%C : memref<?x?xf32>)
   ```

2. **Linalg → SimpBLAS Lowering (CRITICAL COMPONENT)**
   ```cpp
   // src/mlir/lowering/linalg_to_simpblas.cpp

   class LinalgMatmulToSimpBLASPattern
       : public OpRewritePattern<linalg::MatmulOp> {
   public:
       LogicalResult matchAndRewrite(
           linalg::MatmulOp op, PatternRewriter &rewriter) const {

           // Extract matrices and dimensions
           auto A = op.getInputs()[0];
           auto B = op.getInputs()[1];
           auto C = op.getOutputs()[0];

           auto M = rewriter.create<memref::DimOp>(loc, A, 0);
           auto N = rewriter.create<memref::DimOp>(loc, B, 1);
           auto K = rewriter.create<memref::DimOp>(loc, A, 1);

           // Create call to sb_gemm_f32
           auto funcOp = lookupFunction("sb_gemm_f32");
           SmallVector<Value> args = {
               M, N, K,           // Dimensions
               A, K,              // Matrix A, lda
               B, N,              // Matrix B, ldb
               C, N               // Matrix C, ldc
           };

           rewriter.create<func::CallOp>(loc, funcOp, args);
           rewriter.eraseOp(op);

           return success();
       }
   };
   ```

3. **Operation Fusion**
   ```mlir
   // Before fusion:
   %T1 = linalg.matmul %A, %B
   %T2 = linalg.matmul %T1, %C

   // After fusion (reduces memory traffic):
   %T2 = linalg.matmul(%A @ %B), %C  // Fused computation
   ```

4. **SimpTensor Integration**
   - Lower SimpTensor operations to linalg dialect
   - Enable tensor dialect optimizations
   - Map to optimized SimpBLAS kernels

5. **Optimization Pipeline**
   ```cpp
   void buildTensorOptimizationPipeline(PassManager &pm) {
       // High-level linalg optimizations
       pm.addPass(createLinalgFusionPass());
       pm.addPass(createLinalgTilingPass({32, 32, 32}));
       pm.addPass(createLinalgBufferizePass());

       // CRITICAL: Lower to SimpBLAS
       pm.addPass(createLinalgToSimpBLASPass());

       // Continue with affine/vector optimizations
       pm.addPass(createConvertLinalgToAffineLoopsPass());
       pm.addPass(createAffineVectorizePass());

       // Final lowering
       pm.addPass(createConvertVectorToLLVMPass());
       pm.addPass(createConvertMemRefToLLVMPass());
       pm.addPass(createConvertFuncToLLVMPass());
   }
   ```

**Success Criteria:**
- ✅ Matrix multiplication uses SimpBLAS
- ✅ Tensor operations 10-30% faster
- ✅ MobileNet POC shows improvement
- ✅ Memory usage reasonable
- ✅ All regression tests pass

### 3.4 Custom `simp` Dialect Design

**Dialect Definition:**

```cpp
// include/mlir/simp_dialect.hpp

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace simp {

class SimpDialect : public Dialect {
public:
    explicit SimpDialect(MLIRContext *context);

    static StringRef getDialectNamespace() { return "simp"; }

    // Type parsing and printing
    Type parseType(DialectAsmParser &parser) const override;
    void printType(Type type, DialectAsmPrinter &os) const override;
};

} // namespace simp
} // namespace mlir
```

**Type System:**

```cpp
// Types
!simp.array<T>       // Array type: simp.array<f32>, simp.array<i32>
!simp.tensor<T>      // Tensor type: simp.tensor<f32>
!simp.shape          // Shape descriptor

// Type definitions
class ArrayType : public Type::TypeBase<ArrayType, Type, ArrayTypeStorage> {
    Type getElementType() const;
};

class TensorType : public Type::TypeBase<TensorType, Type, TensorTypeStorage> {
    Type getElementType() const;
    ArrayRef<int64_t> getShape() const;
};
```

**Operations:**

```cpp
// src/mlir/simp_ops.td (TableGen definition)

def Simp_ArrayCreateOp : Simp_Op<"array_create"> {
    let summary = "Create array with given size";
    let arguments = (ins I64:$size);
    let results = (outs Simp_ArrayType:$result);
    let assemblyFormat = "$size attr-dict `:` type($result)";
}

def Simp_ArrayGetOp : Simp_Op<"array_get"> {
    let summary = "Get element from array";
    let arguments = (ins Simp_ArrayType:$array, I64:$index);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$array `[` $index `]` attr-dict `:` type($array)";
}

def Simp_ArraySetOp : Simp_Op<"array_set"> {
    let summary = "Set element in array";
    let arguments = (ins Simp_ArrayType:$array, I64:$index, AnyType:$value);
    let assemblyFormat = "$array `[` $index `]` `,` $value attr-dict";
}

def Simp_MatMulOp : Simp_Op<"matmul"> {
    let summary = "Matrix multiplication";
    let arguments = (ins Simp_ArrayType:$lhs, Simp_ArrayType:$rhs);
    let results = (outs Simp_ArrayType:$result);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}
```

**Example MLIR Code:**

```mlir
// SimpLang source:
// var arr = make(Array, 100);
// arr[0] = 42.0;
// return arr[0];

func.func @kernel_main() -> f64 {
    %c100 = arith.constant 100 : i64
    %c0 = arith.constant 0 : i64
    %c42 = arith.constant 42.0 : f64

    %arr = simp.array_create %c100 : i64 -> !simp.array<f64>
    simp.array_set %arr[%c0], %c42 : !simp.array<f64>, f64
    %val = simp.array_get %arr[%c0] : !simp.array<f64> -> f64

    func.return %val : f64
}
```

### 3.5 File Structure

**New Directory Structure:**

```
src/mlir/
├── mlir_codegen.cpp              # Main MLIR code generator
├── mlir_pipeline.cpp             # Optimization pipeline configuration
├── simp_dialect.cpp              # Custom dialect implementation
├── simp_ops.cpp                  # Operation implementations
├── simp_types.cpp                # Type implementations
├── lowering/
│   ├── lower_simp_to_memref.cpp  # Array → memref
│   ├── lower_simp_to_linalg.cpp  # Matrix → linalg
│   ├── linalg_to_simpblas.cpp    # Linalg → SimpBLAS (CRITICAL)
│   └── lower_to_llvm.cpp         # Final LLVM lowering
└── passes/
    ├── array_optimization.cpp    # Array-specific optimizations
    └── simpblas_integration.cpp  # BLAS integration pass

include/mlir/
├── mlir_codegen.hpp
├── simp_dialect.hpp
├── simp_ops.hpp
├── simp_types.hpp
└── passes.hpp

tests/mlir/
├── unit/
│   ├── test_simp_ops.mlir        # Dialect operation tests
│   ├── test_array_lowering.mlir  # Lowering pass tests
│   └── test_simpblas.mlir        # SimpBLAS integration tests
├── integration/
│   ├── test_array_basic.sl       # Basic array tests
│   ├── test_matrix_multiply.sl   # Matrix operations
│   └── test_mobilenet_layer.sl   # ML workload
└── benchmarks/
    ├── bench_array_ops.sl
    ├── bench_gemm.sl
    └── bench_vectorization.sl
```

**Modified Files:**

```
src/main.cpp
  + Add --use-mlir command-line flag
  + Conditional compilation path selection

CMakeLists.txt
  + Add MLIR dependencies
  + Add option(USE_MLIR "Use MLIR backend" OFF)
  + Add mlir/ subdirectory

build.sh
  + Add MLIR configuration support
```

### 3.6 CMake Integration

**Root CMakeLists.txt changes:**

```cmake
# Add MLIR option
option(USE_MLIR "Enable MLIR backend" OFF)

if(USE_MLIR)
    # Find MLIR
    find_package(MLIR REQUIRED CONFIG)

    message(STATUS "Using MLIR from: ${MLIR_DIR}")
    message(STATUS "MLIR include dirs: ${MLIR_INCLUDE_DIRS}")

    # Add MLIR includes and libraries
    include_directories(${MLIR_INCLUDE_DIRS})
    link_directories(${MLIR_LIBRARY_DIRS})

    # Define preprocessor flag
    add_definitions(-DUSE_MLIR)

    # Add MLIR subdirectory
    add_subdirectory(src/mlir)
endif()
```

**src/mlir/CMakeLists.txt (new file):**

```cmake
# MLIR CodeGen library
add_library(simplang_mlir
    mlir_codegen.cpp
    mlir_pipeline.cpp
    simp_dialect.cpp
    simp_ops.cpp
    simp_types.cpp
    lowering/lower_simp_to_memref.cpp
    lowering/lower_simp_to_linalg.cpp
    lowering/linalg_to_simpblas.cpp
    lowering/lower_to_llvm.cpp
    passes/array_optimization.cpp
    passes/simpblas_integration.cpp
)

# MLIR libraries to link
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(simplang_mlir
    PUBLIC
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRTransforms
        MLIRSupport
)

target_include_directories(simplang_mlir
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include/mlir
        ${MLIR_INCLUDE_DIRS}
)
```

---

## Part 4: Comprehensive Testing Plan

### 4.1 Testing Strategy

**Three-Tier Testing Approach:**

1. **Unit Tests**: Individual dialect operations and passes
2. **Integration Tests**: End-to-end compilation of .sl files
3. **Performance Tests**: Benchmarking and regression detection

**Testing Tools:**
- `mlir-opt`: Pass testing and IR validation
- `FileCheck`: Pattern matching in IR
- Google Test: C++ unit tests
- Custom benchmarking framework

### 4.2 Unit Testing

#### 4.2.1 Dialect Operation Tests

**Test File: tests/mlir/unit/test_simp_ops.mlir**

```mlir
// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: func @test_array_create
func.func @test_array_create() -> !simp.array<f32> {
    %c100 = arith.constant 100 : i64
    // CHECK: simp.array_create
    %arr = simp.array_create %c100 : i64 -> !simp.array<f32>
    return %arr : !simp.array<f32>
}

// CHECK-LABEL: func @test_array_access
func.func @test_array_access(%arg0: !simp.array<f32>) -> f32 {
    %c0 = arith.constant 0 : i64
    // CHECK: simp.array_get
    %val = simp.array_get %arg0[%c0] : !simp.array<f32> -> f32
    return %val : f32
}

// CHECK-LABEL: func @test_matmul
func.func @test_matmul(%arg0: !simp.array<f32>, %arg1: !simp.array<f32>) {
    // CHECK: simp.matmul
    %result = simp.matmul %arg0, %arg1 : (!simp.array<f32>, !simp.array<f32>) -> !simp.array<f32>
    return
}
```

#### 4.2.2 Lowering Pass Tests

**Test File: tests/mlir/unit/test_array_lowering.mlir**

```mlir
// RUN: mlir-opt %s --lower-simp-to-memref | FileCheck %s

// CHECK-LABEL: func @test_lower_array_create
func.func @test_lower_array_create() -> memref<?xf32> {
    %c100 = arith.constant 100 : i64
    %arr = simp.array_create %c100 : i64 -> !simp.array<f32>
    // CHECK: memref.alloc
    // CHECK-SAME: memref<?xf32>
    %0 = builtin.unrealized_conversion_cast %arr : !simp.array<f32> to memref<?xf32>
    return %0 : memref<?xf32>
}

// CHECK-LABEL: func @test_lower_array_get
func.func @test_lower_array_get(%arg0: !simp.array<f32>) -> f32 {
    %c0 = arith.constant 0 : i64
    %val = simp.array_get %arg0[%c0] : !simp.array<f32> -> f32
    // CHECK: memref.load
    // CHECK-SAME: memref<?xf32>
    return %val : f32
}
```

**Test File: tests/mlir/unit/test_simpblas.mlir**

```mlir
// RUN: mlir-opt %s --linalg-to-simpblas | FileCheck %s

// CHECK-LABEL: func @test_matmul_to_simpblas
func.func @test_matmul_to_simpblas(
    %A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {

    // Input: linalg.matmul
    linalg.matmul ins(%A, %B : memref<?x?xf32>, memref<?x?xf32>)
                   outs(%C : memref<?x?xf32>)

    // CHECK: func.call @sb_gemm_f32
    // CHECK-SAME: memref<?x?xf32>

    return
}
```

#### 4.2.3 Vectorization Tests

**Test File: tests/mlir/unit/test_vectorization.mlir**

```mlir
// RUN: mlir-opt %s --affine-super-vectorize="virtual-vector-size=8" | FileCheck %s

// CHECK-LABEL: func @test_array_loop_vectorization
func.func @test_array_loop_vectorization(%A: memref<?xf32>, %B: memref<?xf32>, %N: index) {
    affine.for %i = 0 to %N {
        %a = affine.load %A[%i] : memref<?xf32>
        %b = affine.load %B[%i] : memref<?xf32>
        %c = arith.addf %a, %b : f32
        affine.store %c, %A[%i] : memref<?xf32>
    }

    // CHECK: vector.load
    // CHECK: vector.load
    // CHECK: arith.addf {{.*}} : vector<8xf32>
    // CHECK: vector.store

    return
}
```

### 4.3 Integration Testing

#### 4.3.1 Array Operation Tests

**Test File: tests/mlir/integration/test_array_basic.sl**

```simplang
fn kernel_main() {
    // Test array creation
    var arr = make(Array, 10);

    // Test array assignment
    arr[0] = 42.0;
    arr[1] = 3.14;

    // Test array access
    var sum = arr[0] + arr[1];

    return sum;  // Expected: 45.14
}
```

**Test File: tests/mlir/integration/test_array_loops.sl**

```simplang
fn kernel_main() {
    var arr = make(Array, 100);
    var sum = 0.0;
    var i = 0.0;

    // Initialize array
    while (i < 100.0) {
        arr[i] = i;
        i = i + 1.0;
    }

    // Sum array (should auto-vectorize)
    i = 0.0;
    while (i < 100.0) {
        sum = sum + arr[i];
        i = i + 1.0;
    }

    return sum;  // Expected: 4950.0
}
```

#### 4.3.2 Matrix Operation Tests

**Test File: tests/mlir/integration/test_matrix_multiply.sl**

```simplang
fn kernel_main() {
    // Create matrices
    var A = make(Array, 100);  // 10x10 matrix (flattened)
    var B = make(Array, 100);
    var C = make(Array, 100);

    // Initialize A and B
    var i = 0.0;
    while (i < 100.0) {
        A[i] = i;
        B[i] = i;
        i = i + 1.0;
    }

    // Matrix multiply (should generate sb_gemm_f32 call)
    matmul(A, B, C, 10, 10, 10);

    return C[0];
}
```

**Test File: tests/mlir/integration/test_simpblas_integration.sl**

```simplang
// Verify SimpBLAS integration
fn kernel_main() {
    var M = 32.0;
    var N = 32.0;
    var K = 32.0;

    var A = make(Array, 1024);  // 32x32
    var B = make(Array, 1024);  // 32x32
    var C = make(Array, 1024);  // 32x32

    // Initialize (omitted for brevity)

    // This should lower to:
    // linalg.matmul → sb_gemm_f32(32, 32, 32, A, 32, B, 32, C, 32)
    matmul(A, B, C, M, N, K);

    return C[0];
}
```

#### 4.3.3 Test Runner

**Script: tests/mlir/run_integration_tests.sh**

```bash
#!/bin/bash
set -e

SIMPLANG_BIN="./build/src/simplang"
TEST_DIR="tests/mlir/integration"

echo "Running MLIR integration tests..."

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

passed=0
failed=0

for test_file in $TEST_DIR/*.sl; do
    test_name=$(basename "$test_file" .sl)
    echo -n "Testing $test_name... "

    # Compile with MLIR backend
    if $SIMPLANG_BIN --use-mlir "$test_file" -o "/tmp/$test_name.o" 2>/tmp/error.log; then
        # Run the compiled kernel
        if ./build/tests/simplang_runner "/tmp/$test_name.o" > /tmp/output.log 2>&1; then
            # Compare output with expected (if expected file exists)
            if [ -f "$TEST_DIR/$test_name.expected" ]; then
                if diff -q /tmp/output.log "$TEST_DIR/$test_name.expected" > /dev/null; then
                    echo -e "${GREEN}PASS${NC}"
                    ((passed++))
                else
                    echo -e "${RED}FAIL${NC} (output mismatch)"
                    ((failed++))
                fi
            else
                echo -e "${GREEN}PASS${NC}"
                ((passed++))
            fi
        else
            echo -e "${RED}FAIL${NC} (runtime error)"
            cat /tmp/output.log
            ((failed++))
        fi
    else
        echo -e "${RED}FAIL${NC} (compilation error)"
        cat /tmp/error.log
        ((failed++))
    fi
done

echo ""
echo "Results: $passed passed, $failed failed"
exit $failed
```

### 4.4 Performance Benchmarking

#### 4.4.1 Benchmark Suite

**Benchmark Categories:**

1. **Compilation Time**: MLIR vs LLVM IR path
2. **Array Operations**: Vectorization effectiveness
3. **Matrix Operations**: SimpBLAS integration performance
4. **End-to-End**: Real-world workloads

#### 4.4.2 Array Performance Benchmarks

**Benchmark: benchmarks/mlir/bench_array_vectorization.sl**

```simplang
// Benchmark auto-vectorization quality
fn kernel_main() {
    var N = 1000000.0;  // 1M elements
    var A = make(Array, N);
    var B = make(Array, N);
    var C = make(Array, N);

    var i = 0.0;

    // Initialize
    while (i < N) {
        A[i] = i;
        B[i] = i * 2.0;
        i = i + 1.0;
    }

    // Vector addition (should auto-vectorize)
    i = 0.0;
    while (i < N) {
        C[i] = A[i] + B[i];
        i = i + 1.0;
    }

    return C[999999];
}
```

**Runner: benchmarks/mlir/run_array_benchmark.sh**

```bash
#!/bin/bash

echo "Array Vectorization Benchmark"
echo "=============================="

# Compile with LLVM path
./build/src/simplang bench_array_vectorization.sl -o /tmp/bench_llvm.o
# Compile with MLIR path
./build/src/simplang --use-mlir bench_array_vectorization.sl -o /tmp/bench_mlir.o

echo "Running LLVM version..."
time_llvm=$(./benchmark_runner /tmp/bench_llvm.o --iterations 100)

echo "Running MLIR version..."
time_mlir=$(./benchmark_runner /tmp/bench_mlir.o --iterations 100)

speedup=$(echo "scale=2; $time_llvm / $time_mlir" | bc)

echo ""
echo "Results:"
echo "  LLVM: $time_llvm ms"
echo "  MLIR: $time_mlir ms"
echo "  Speedup: ${speedup}x"
```

#### 4.4.3 Matrix Performance Benchmarks

**Benchmark: benchmarks/mlir/bench_gemm.sl**

```simplang
// Benchmark matrix multiplication with varying sizes
fn bench_gemm(var M, var N, var K) {
    var size_A = M * K;
    var size_B = K * N;
    var size_C = M * N;

    var A = make(Array, size_A);
    var B = make(Array, size_B);
    var C = make(Array, size_C);

    // Initialize (omitted)

    // Matrix multiply (should use SimpBLAS)
    matmul(A, B, C, M, N, K);

    return C[0];
}

fn kernel_main() {
    // Test different sizes
    bench_gemm(64, 64, 64);     // Small
    bench_gemm(256, 256, 256);  // Medium
    bench_gemm(1024, 1024, 1024); // Large

    return 0.0;
}
```

**Runner: benchmarks/mlir/run_gemm_benchmark.sh**

```bash
#!/bin/bash

echo "Matrix Multiplication Benchmark"
echo "================================"

sizes=(64 128 256 512 1024)

echo "Size,LLVM(ms),MLIR(ms),Speedup"

for size in "${sizes[@]}"; do
    # Compile both versions
    ./build/src/simplang bench_gemm.sl -o /tmp/bench_llvm.o
    ./build/src/simplang --use-mlir bench_gemm.sl -o /tmp/bench_mlir.o

    # Run benchmarks
    time_llvm=$(./benchmark_runner /tmp/bench_llvm.o --size $size)
    time_mlir=$(./benchmark_runner /tmp/bench_mlir.o --size $size)

    speedup=$(echo "scale=2; $time_llvm / $time_mlir" | bc)

    echo "$size,$time_llvm,$time_mlir,$speedup"
done
```

#### 4.4.4 Performance Regression Detection

**CI Script: .github/workflows/mlir_performance.yml**

```yaml
name: MLIR Performance Regression

on:
  pull_request:
    paths:
      - 'src/mlir/**'
      - 'tests/mlir/**'
      - 'benchmarks/mlir/**'

jobs:
  performance-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Build with MLIR
        run: |
          cmake -B build -DUSE_MLIR=ON
          cmake --build build

      - name: Run performance benchmarks
        run: |
          cd benchmarks/mlir
          ./run_all_benchmarks.sh > results.csv

      - name: Check for regressions
        run: |
          python3 scripts/check_performance_regression.py \
            --baseline baseline_results.csv \
            --current results.csv \
            --threshold 0.05  # 5% regression threshold

      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: benchmarks/mlir/results.csv
```

**Regression Checker: scripts/check_performance_regression.py**

```python
#!/usr/bin/env python3
import csv
import sys
import argparse

def check_regression(baseline_file, current_file, threshold):
    # Read baseline
    baseline = {}
    with open(baseline_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            baseline[row['benchmark']] = float(row['time_ms'])

    # Read current results
    regressions = []
    with open(current_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['benchmark']
            current_time = float(row['time_ms'])
            baseline_time = baseline.get(name)

            if baseline_time:
                change = (current_time - baseline_time) / baseline_time
                if change > threshold:
                    regressions.append({
                        'name': name,
                        'baseline': baseline_time,
                        'current': current_time,
                        'change': change * 100
                    })

    # Report
    if regressions:
        print("⚠️  Performance Regressions Detected:")
        for r in regressions:
            print(f"  {r['name']}: {r['baseline']:.2f}ms → {r['current']:.2f}ms "
                  f"({r['change']:+.1f}%)")
        sys.exit(1)
    else:
        print("✅ No performance regressions detected")
        sys.exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--current', required=True)
    parser.add_argument('--threshold', type=float, default=0.05)
    args = parser.parse_args()

    check_regression(args.baseline, args.current, args.threshold)
```

### 4.5 Validation Testing

#### 4.5.1 Numerical Accuracy

**Test: tests/mlir/validation/numerical_accuracy_test.cpp**

```cpp
#include <gtest/gtest.h>
#include "mlir_codegen.hpp"
#include "codegen.hpp"

class NumericalAccuracyTest : public ::testing::Test {
protected:
    double runWithLLVMPath(const std::string& source) {
        CodeGenContext ctx;
        // Compile and run with LLVM path
        // ...
        return result;
    }

    double runWithMLIRPath(const std::string& source) {
        MLIRCodeGenContext ctx;
        // Compile and run with MLIR path
        // ...
        return result;
    }
};

TEST_F(NumericalAccuracyTest, BasicArithmetic) {
    std::string source = R"(
        fn kernel_main() {
            var x = 10.5;
            var y = 3.2;
            return x * y + 5.0;
        }
    )";

    double llvm_result = runWithLLVMPath(source);
    double mlir_result = runWithMLIRPath(source);

    EXPECT_NEAR(llvm_result, mlir_result, 1e-10);
}

TEST_F(NumericalAccuracyTest, ArrayOperations) {
    std::string source = R"(
        fn kernel_main() {
            var arr = make(Array, 10);
            var i = 0.0;
            var sum = 0.0;

            while (i < 10.0) {
                arr[i] = i * i;
                i = i + 1.0;
            }

            i = 0.0;
            while (i < 10.0) {
                sum = sum + arr[i];
                i = i + 1.0;
            }

            return sum;
        }
    )";

    double llvm_result = runWithLLVMPath(source);
    double mlir_result = runWithMLIRPath(source);

    EXPECT_NEAR(llvm_result, mlir_result, 1e-10);
    EXPECT_DOUBLE_EQ(mlir_result, 285.0);  // 0²+1²+2²+...+9² = 285
}
```

#### 4.5.2 SimpBLAS Integration Validation

**Test: tests/mlir/validation/simpblas_validation_test.cpp**

```cpp
#include <gtest/gtest.h>
#include "mlir_codegen.hpp"

TEST(SimpBLASValidation, MatMulGeneratesBLASCall) {
    std::string source = R"(
        fn kernel_main() {
            var A = make(Array, 64);  // 8x8
            var B = make(Array, 64);
            var C = make(Array, 64);

            matmul(A, B, C, 8, 8, 8);

            return C[0];
        }
    )";

    MLIRCodeGenContext ctx;
    auto mlir = ctx.generateMLIR(source);

    // Verify that sb_gemm_f32 call is generated
    std::string mlir_str = mlir.toString();
    EXPECT_TRUE(mlir_str.find("sb_gemm_f32") != std::string::npos);

    // Verify correct parameters
    EXPECT_TRUE(mlir_str.find("i32 8") != std::string::npos); // M
    EXPECT_TRUE(mlir_str.find("i32 8") != std::string::npos); // N
    EXPECT_TRUE(mlir_str.find("i32 8") != std::string::npos); // K
}

TEST(SimpBLASValidation, MatMulNumericalCorrectness) {
    // Small matrix multiply: verify MLIR → BLAS gives correct result

    std::string source = R"(
        fn kernel_main() {
            var A = make(Array, 4);  // 2x2 matrix
            var B = make(Array, 4);
            var C = make(Array, 4);

            // Initialize: A = [1 2; 3 4], B = [5 6; 7 8]
            A[0] = 1.0; A[1] = 2.0; A[2] = 3.0; A[3] = 4.0;
            B[0] = 5.0; B[1] = 6.0; B[2] = 7.0; B[3] = 8.0;

            matmul(A, B, C, 2, 2, 2);

            // Expected C = [19 22; 43 50]
            return C[0] + C[1] + C[2] + C[3];
        }
    )";

    MLIRCodeGenContext ctx;
    double result = ctx.compileAndRun(source);

    // Expected: 19 + 22 + 43 + 50 = 134
    EXPECT_DOUBLE_EQ(result, 134.0);
}
```

#### 4.5.3 Memory Safety Validation

**Test: tests/mlir/validation/memory_safety_test.cpp**

```cpp
#include <gtest/gtest.h>

TEST(MemorySafety, NoMemoryLeaks) {
    // Run with AddressSanitizer or Valgrind
    std::string source = R"(
        fn kernel_main() {
            var arr1 = make(Array, 1000);
            var arr2 = make(Array, 2000);
            var arr3 = make(Array, 3000);

            // Use arrays
            arr1[0] = 42.0;
            arr2[100] = 3.14;
            arr3[500] = 2.71;

            return arr1[0] + arr2[100] + arr3[500];
        }
    )";

    // Compile and run multiple times
    for (int i = 0; i < 100; i++) {
        MLIRCodeGenContext ctx;
        ctx.compileAndRun(source);
    }

    // AddressSanitizer will detect leaks
    // This test passes if no leaks are detected
}

TEST(MemorySafety, ArrayAlignment) {
    // Verify arrays are properly aligned for SIMD
    std::string source = R"(
        fn kernel_main() {
            var arr = make(Array, 1000);
            return arr[0];
        }
    )";

    MLIRCodeGenContext ctx;
    auto mlir = ctx.generateMLIR(source);

    // Check alignment attributes in generated MLIR
    std::string mlir_str = mlir.toString();
    // Should have alignment attribute for memref.alloc
    EXPECT_TRUE(mlir_str.find("alignment = 32") != std::string::npos ||
                mlir_str.find("alignment = 64") != std::string::npos);
}
```

### 4.6 Test Metrics & Success Criteria

#### Phase 1: Foundation (6-8 weeks)

**Success Criteria:**
- ✅ All dialect operations have unit tests
- ✅ All basic .sl tests pass (arithmetic, loops, functions)
- ✅ Numerical accuracy within 1e-10 tolerance
- ✅ Compilation time < 2x LLVM IR path
- ✅ Runtime performance >= 95% of LLVM IR path
- ✅ No memory leaks detected
- ✅ Zero test failures in CI

**Test Coverage:**
- Unit tests: 50+ test cases
- Integration tests: 10+ .sl files
- Code coverage: >= 80% of MLIR code

#### Phase 2: Arrays & Vectorization (4-6 weeks)

**Success Criteria:**
- ✅ All array operation tests pass
- ✅ Auto-vectorization generates vector instructions
- ✅ Performance on array operations >= 100% of LLVM path
- ✅ Cache utilization improvements measurable
- ✅ Vectorization tests show >= 2x speedup on large arrays
- ✅ Memory layout optimizations verified

**Test Coverage:**
- Array tests: 20+ test cases
- Vectorization tests: 15+ test cases
- Benchmarks: 5+ array operation benchmarks

#### Phase 3: Tensor & SimpBLAS (8-12 weeks)

**Success Criteria:**
- ✅ Matrix operations automatically generate SimpBLAS calls
- ✅ linalg.matmul correctly lowers to sb_gemm_f32
- ✅ Tensor operations 10-30% faster than baseline
- ✅ MobileNet POC shows measurable performance improvement
- ✅ All SimpTensor integration tests pass
- ✅ Memory usage remains reasonable (< 10% increase)
- ✅ All regression tests pass (0 failures)

**Test Coverage:**
- Matrix operation tests: 25+ test cases
- SimpBLAS integration tests: 15+ test cases
- Tensor operation tests: 20+ test cases
- End-to-end benchmarks: 10+ workloads

---

## Part 5: Implementation Roadmap (AI-Assisted Development)

### 5.1 Timeline Overview

**Total Effort:** 36-52 sessions (9-13 weeks at 3-4 sessions/week)

**Session Format:** 3-4 hour focused AI pair-programming sessions
- AI generates boilerplate and test code
- Human provides architecture decisions and validation
- Iterative development with fast feedback loops

### 5.2 Phase 1: Foundation (Sessions 1-16, ~3-4 weeks)

#### Sessions 1-3: Environment Setup & Learning

**Objectives:**
- Set up MLIR build environment
- Study MLIR fundamentals
- Design `simp` dialect

**Session 1: MLIR Build & Initial Learning (3-4 hours)**
- Clone LLVM/MLIR monorepo
- Build LLVM/MLIR from source (AI helps with build script)
- Start MLIR Toy tutorial (complete Part 1-2)
- Verify mlir-opt works

**Session 2: Deep Dive MLIR Concepts (3-4 hours)**
- Complete MLIR Toy tutorial (Part 3-7)
- Study linalg → BLAS lowering examples
- Review dialect conversion framework
- Research TensorFlow/PyTorch MLIR integrations

**Session 3: Dialect Design (3-4 hours)**
- Draft `simp` dialect specification (AI assists with docs)
- Design operation set based on SimpLang AST
- Design type system (!simp.array, !simp.tensor)
- Create initial TableGen definitions (AI generates boilerplate)

**Deliverables:**
- MLIR builds successfully ✅
- `simp` dialect specification document ✅
- Basic TableGen definitions drafted ✅

#### Sessions 4-7: Dialect Implementation & AST Lowering

**Session 4: Dialect Implementation (3-4 hours)**
- Implement SimpDialect class (AI generates from template)
- Register operations and types
- Implement operation verifiers
- Set up CMake integration for MLIR

**Session 5-6: AST → MLIR Lowering (6-8 hours total)**
- Session 5:
  - Implement MLIRCodeGen class structure
  - Lower basic expressions (DoubleExprAST, IntegerExprAST, BinaryOpAST)
  - Lower variable declarations and assignments
  - AI generates boilerplate lowering code

- Session 6:
  - Lower control flow (IfStmtAST, WhileStmtAST)
  - Lower function declarations
  - Lower array operations (array_create, array_get/set)
  - AI generates pattern matching code

**Session 7: Unit Testing (3-4 hours)**
- Create test infrastructure (AI generates test templates)
- Test dialect operations with mlir-opt
- Test AST lowering for basic programs
- Fix bugs found during testing

**Deliverables:**
- `simp` dialect fully implemented ✅
- Basic AST → MLIR working ✅
- Unit tests passing ✅

#### Sessions 8-11: LLVM Lowering & Integration

**Session 8-9: Lowering Pipeline (6-8 hours total)**
- Session 8:
  - Create pass manager infrastructure
  - Configure lowering passes (simp → std → llvm)
  - AI generates pass registration code
  - Implement basic simp → standard dialect lowering

- Session 9:
  - Implement standard → LLVM dialect lowering
  - Handle type conversions (memref → LLVM pointers)
  - Test lowering pipeline with mlir-opt

**Session 10: Pipeline Integration (3-4 hours)**
- Add --use-mlir flag to src/main.cpp
- Conditional compilation path selection
- Error handling and reporting
- AI helps with command-line arg parsing

**Session 11: End-to-End Testing (3-4 hours)**
- Compile test_arithmetic.sl via MLIR
- Compile test_loop.sl via MLIR
- Compare outputs with LLVM IR path
- Fix integration bugs

**Deliverables:**
- End-to-end compilation working ✅
- test_arithmetic.sl and test_loop.sl pass ✅

#### Sessions 12-16: Testing, Validation & Bug Fixing

**Session 12-13: Test Suite Creation (6-8 hours total)**
- AI generates comprehensive test suite
- Create integration tests for all .sl files
- Set up benchmark infrastructure
- Configure CI integration

**Session 14: Performance Baseline (3-4 hours)**
- Measure compilation time (MLIR vs LLVM)
- Measure runtime performance
- Document baselines
- Create performance regression tests

**Session 15-16: Bug Fixing & Polish (6-8 hours total)**
- Address all failing tests
- Fix performance issues
- Improve error messages
- Code cleanup and documentation

**Deliverables:**
- All Phase 1 tests passing ✅
- Performance baseline documented ✅
- CI pipeline working ✅

**Phase 1 Milestone:** Basic MLIR integration complete ✅ (~3-4 weeks)

---

### 5.3 Phase 2: Arrays & Vectorization (Sessions 17-28, ~2-3 weeks)

#### Sessions 17-19: Array Operations (9-12 hours total)

**Session 17-18: Array Lowering**
- Implement lower_simp_to_memref.cpp (AI generates pattern matching code)
- simp.array_create → memref.alloc with alignment
- simp.array_get/set → memref.load/store
- Handle multi-dimensional arrays

**Session 19: Memory Management & Testing**
- Implement aligned allocation
- Memory deallocation and lifetime analysis
- Test array operations (AI generates tests)
- Memory safety validation

**Deliverables:**
- Array operations lowered to memref ✅
- test_array_basic.sl works ✅
- Memory safety verified ✅

#### Sessions 20-22: Affine Optimization (9-12 hours total)

**Session 20-21: Affine Lowering**
- Lower scf loops to affine.for
- Convert memory accesses to affine.load/store
- Analyze loop bounds and dependencies
- AI helps with polyhedral analysis code

**Session 22: Optimization Pass Configuration**
- Enable loop fusion, tiling, invariant code motion
- Tune pass parameters for best performance
- Test optimizations with mlir-opt
- Benchmark improvements

**Deliverables:**
- Affine optimizations working ✅
- Loop fusion verified ✅
- Performance measurable improvement ✅

#### Sessions 23-25: Vectorization (9-12 hours total)

**Session 23-24: Vectorization Implementation**
- Configure affine-super-vectorize pass
- Set vector sizes (SSE: 4, AVX: 8, AVX-512: 16)
- Lower to vector dialect operations
- AI generates vectorization test cases

**Session 25: Testing & Tuning**
- Verify SIMD instructions in generated code
- Benchmark vectorized vs scalar performance
- Tune for different array sizes
- Iterate on vectorization parameters

**Deliverables:**
- Auto-vectorization working ✅
- Performance >= 2x on large arrays ✅
- Vector instructions verified ✅

#### Sessions 26-28: Phase 2 Polish (Optional, 9-12 hours total)

**Session 26-28: Performance Optimization**
- Profile array operations and identify bottlenecks
- Optimize memory access patterns
- Fine-tune vectorization heuristics
- Comprehensive benchmarking

**Phase 2 Milestone:** Array optimization complete ✅ (~2-3 weeks)

---

### 5.4 Phase 3: Tensor & SimpBLAS Integration (Sessions 29-52, ~4-6 weeks)

#### Sessions 29-31: Linalg Integration (9-12 hours total)

**Session 29-30: Linalg Lowering**
- Implement lower_simp_to_linalg.cpp (AI generates patterns)
- simp.matmul → linalg.matmul
- Handle tensor/memref conversion
- AI helps with linalg operation builders

**Session 31: Linalg Optimization Configuration**
- Enable fusion, tiling passes
- Configure tile sizes for cache efficiency
- Test linalg optimizations
- Verify operation fusion

**Deliverables:**
- Matrix operations lower to linalg ✅
- Linalg optimizations working ✅
- test_matrix_multiply.sl works ✅

#### Sessions 32-38: SimpBLAS Integration (CRITICAL - 18-24 hours total)

**Session 32-34: SimpBLAS Lowering Pass**
- Implement LinalgMatmulToSimpBLASPattern (AI generates template)
- Extract matrix dimensions (M, N, K)
- Handle pointer extraction from memref
- Test pattern matching with mlir-opt

**Session 35-36: BLAS Function Integration**
- Declare sb_gemm_f32 in MLIR module
- Handle parameter type conversions
- Implement leading dimension calculations
- Test BLAS call generation

**Session 37: Numerical Validation**
- Create correctness tests (AI generates test matrices)
- Verify results match reference implementation
- Test with various matrix sizes
- Debug numerical issues if any

**Session 38: Performance Benchmarking**
- Benchmark BLAS-accelerated matmul
- Compare vs naive implementation
- Measure speedup across sizes
- Document performance characteristics

**Deliverables:**
- linalg.matmul → sb_gemm_f32 working ✅
- Numerical correctness verified ✅
- Performance >= SimpBLAS baseline ✅

#### Sessions 39-42: SimpTensor Integration (12-16 hours total)

**Session 39-40: Tensor Operation Lowering**
- Lower SimpTensor ops to linalg/tensor dialects
- Enable tensor-level optimizations
- Map operations to SimpBLAS where applicable
- AI generates lowering patterns

**Session 41: Integration Testing**
- Create SimpTensor integration tests (AI generates)
- Verify numerical correctness
- Test edge cases and error handling

**Session 42: Performance Validation**
- Benchmark tensor operations
- Measure performance improvements
- Compare against baseline

**Deliverables:**
- SimpTensor operations optimized ✅
- Integration tests passing ✅
- Performance improvement measured ✅

#### Sessions 43-48: MobileNet POC Optimization (18-24 hours total)

**Session 43-45: Layer-by-Layer Optimization**
- Optimize convolution layers with linalg
- Optimize fully-connected layers
- Enable cross-layer operation fusion
- AI helps identify fusion opportunities

**Session 46-47: End-to-End Tuning**
- Tune tile sizes for each layer
- Optimize vectorization parameters
- Optimize memory layout and access patterns
- Iterative performance tuning

**Session 48: Comprehensive Benchmarking**
- Measure full inference time
- Compare to baseline implementation
- Profile and identify remaining bottlenecks
- Document performance gains

**Deliverables:**
- MobileNet POC 10-30% faster ✅
- Benchmark results documented ✅
- Optimization report complete ✅

#### Sessions 49-52: Finalization & Documentation (12-16 hours total)

**Session 49: Final Performance Tuning**
- Profile all benchmarks
- Optimize critical paths
- Tune pass ordering for best results
- Validate no regressions

**Session 50-51: Documentation (AI-assisted)**
- AI generates user guide from code
- Document `simp` dialect operations
- Create migration guide (LLVM → MLIR)
- Write integration examples

**Session 52: Code Review Preparation**
- Clean up code, remove debug statements
- Add comprehensive comments
- Prepare PR description
- Final testing sweep

**Deliverables:**
- All tests passing ✅
- Documentation complete ✅
- Ready for code review ✅

**Phase 3 Milestone:** MLIR integration complete ✅ (~4-6 weeks)

---

### 5.5 Critical Path Analysis

**Critical Path (Session-Based):**
1. MLIR Build Setup (Sessions 1-3)
2. Dialect Implementation (Sessions 4-7)
3. End-to-End Pipeline (Sessions 8-11)
4. SimpBLAS Integration (Sessions 32-38) ← **MOST CRITICAL**
5. MobileNet Optimization (Sessions 43-48)

**Dependencies:**
- SimpBLAS integration depends on Linalg integration (Session 29-31)
- Vectorization depends on Affine optimization (Session 20-22)
- MobileNet optimization depends on SimpBLAS integration (Session 32-38)

**AI Acceleration Points:**
- ✨ Boilerplate code generation (dialects, operations, passes) - **3-5x faster**
- ✨ Test code generation - **5-10x faster**
- ✨ Documentation writing - **10x faster**
- ✨ Pattern matching code - **2-3x faster**

**Still Human-Critical:**
- ⚠️ MLIR learning and understanding
- ⚠️ Architectural decisions
- ⚠️ Performance tuning and iteration
- ⚠️ Integration debugging

---

## Part 6: SimpBLAS Integration Deep Dive

### 6.1 Current SimpBLAS Architecture

**From src/codegen.cpp:751-811:**

```cpp
void CodeGenContext::initializeSimpBLASFunctions() {
    // SimpBLAS initialization: int sb_init(void)
    llvm::FunctionType* initType = llvm::FunctionType::get(
        llvm::Type::getInt32Ty(context), {}, false);
    sbInitFunc = llvm::Function::Create(
        initType, llvm::Function::ExternalLinkage, "sb_init", module.get());

    // SimpBLAS GEMM: void sb_gemm_f32(int M, int N, int K,
    //                                  float* A, int lda,
    //                                  float* B, int ldb,
    //                                  float* C, int ldc)
    std::vector<llvm::Type*> gemmArgs = {
        llvm::Type::getInt32Ty(context),     // M
        llvm::Type::getInt32Ty(context),     // N
        llvm::Type::getInt32Ty(context),     // K
        llvm::Type::getFloatPtrTy(context),  // A
        llvm::Type::getInt32Ty(context),     // lda
        llvm::Type::getFloatPtrTy(context),  // B
        llvm::Type::getInt32Ty(context),     // ldb
        llvm::Type::getFloatPtrTy(context),  // C
        llvm::Type::getInt32Ty(context)      // ldc
    };
    llvm::FunctionType* gemmType = llvm::FunctionType::get(
        llvm::Type::getVoidTy(context), gemmArgs, false);
    sbGemmFunc = llvm::Function::Create(
        gemmType, llvm::Function::ExternalLinkage, "sb_gemm_f32", module.get());
}

void CodeGenContext::generateGemmCall(llvm::Value* M, llvm::Value* N, llvm::Value* K,
                                     llvm::Value* A, llvm::Value* B, llvm::Value* C) {
    // Cast arrays if needed
    if (A->getType() != llvm::Type::getFloatPtrTy(context)) {
        A = builder.CreateBitCast(A, llvm::Type::getFloatPtrTy(context));
    }
    // ... similar for B and C

    // Leading dimensions (assume row-major, packed)
    llvm::Value* lda = N;
    llvm::Value* ldb = K;
    llvm::Value* ldc = N;

    // Create GEMM call
    std::vector<llvm::Value*> gemmArgs = {M, N, K, A, lda, B, ldb, C, ldc};
    builder.CreateCall(sbGemmFunc, gemmArgs);
}
```

### 6.2 MLIR Linalg → SimpBLAS Strategy

**Concept:** Pattern matching in MLIR to replace linalg.matmul with calls to SimpBLAS

**Advantages:**
1. ✅ Preserve high-level semantics during optimization
2. ✅ Enable operation fusion before BLAS lowering
3. ✅ Automatic tiling and layout optimization
4. ✅ Cleaner abstraction than direct LLVM IR generation

### 6.3 Implementation Details

#### 6.3.1 Linalg Operation

**Input MLIR (from SimpLang matmul):**

```mlir
func.func @kernel_main() -> f32 {
    %c32 = arith.constant 32 : index

    // Allocate matrices (32x32)
    %A = memref.alloc(%c32, %c32) : memref<?x?xf32>
    %B = memref.alloc(%c32, %c32) : memref<?x?xf32>
    %C = memref.alloc(%c32, %c32) : memref<?x?xf32>

    // Initialize A and B (omitted)

    // Matrix multiply
    linalg.matmul ins(%A, %B : memref<?x?xf32>, memref<?x?xf32>)
                   outs(%C : memref<?x?xf32>)

    %result = memref.load %C[%c0, %c0] : memref<?x?xf32>
    return %result : f32
}
```

#### 6.3.2 Pattern Matching Pass

**Implementation: src/mlir/lowering/linalg_to_simpblas.cpp**

```cpp
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace simp {

// Pattern to convert linalg.matmul to sb_gemm_f32 call
struct LinalgMatmulToSimpBLASPattern
    : public OpRewritePattern<linalg::MatmulOp> {

    using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
        Location loc = op.getLoc();

        // Get input and output matrices
        Value A = op.getInputs()[0];
        Value B = op.getInputs()[1];
        Value C = op.getOutputs()[0];

        // Extract matrix dimensions
        // For memref<?x?xf32>, we need to get runtime dimensions
        auto aType = A.getType().cast<MemRefType>();
        auto bType = B.getType().cast<MemRefType>();

        Value M, N, K;
        if (aType.isDynamicDim(0)) {
            // Runtime dimension
            M = rewriter.create<memref::DimOp>(loc, A, 0);
        } else {
            // Static dimension
            M = rewriter.create<arith::ConstantIndexOp>(
                loc, aType.getDimSize(0));
        }

        if (bType.isDynamicDim(1)) {
            N = rewriter.create<memref::DimOp>(loc, B, 1);
        } else {
            N = rewriter.create<arith::ConstantIndexOp>(
                loc, bType.getDimSize(1));
        }

        if (aType.isDynamicDim(1)) {
            K = rewriter.create<memref::DimOp>(loc, A, 1);
        } else {
            K = rewriter.create<arith::ConstantIndexOp>(
                loc, aType.getDimSize(1));
        }

        // Convert index to i32 (SimpBLAS uses i32)
        auto i32Type = rewriter.getI32Type();
        M = rewriter.create<arith::IndexCastOp>(loc, i32Type, M);
        N = rewriter.create<arith::IndexCastOp>(loc, i32Type, N);
        K = rewriter.create<arith::IndexCastOp>(loc, i32Type, K);

        // Leading dimensions (assume row-major, packed)
        Value lda = K;  // A is MxK, so lda = K
        Value ldb = N;  // B is KxN, so ldb = N
        Value ldc = N;  // C is MxN, so ldc = N

        // Get base pointers (extract from memref)
        auto ptrType = LLVM::LLVMPointerType::get(
            rewriter.getContext());

        Value A_ptr = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
            loc, A);
        A_ptr = rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, A_ptr);

        Value B_ptr = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
            loc, B);
        B_ptr = rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, B_ptr);

        Value C_ptr = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
            loc, C);
        C_ptr = rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, C_ptr);

        // Lookup or create sb_gemm_f32 function declaration
        auto module = op->getParentOfType<ModuleOp>();
        auto funcOp = module.lookupSymbol<func::FuncOp>("sb_gemm_f32");

        if (!funcOp) {
            // Create function declaration
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(module.getBody());

            // Function type: (i32, i32, i32, ptr, i32, ptr, i32, ptr, i32) -> void
            auto funcType = rewriter.getFunctionType(
                {i32Type, i32Type, i32Type,
                 ptrType, i32Type,
                 ptrType, i32Type,
                 ptrType, i32Type},
                {});

            funcOp = rewriter.create<func::FuncOp>(
                loc, "sb_gemm_f32", funcType);
            funcOp.setPrivate();
        }

        // Create function call
        SmallVector<Value> callArgs = {
            M, N, K,
            A_ptr, lda,
            B_ptr, ldb,
            C_ptr, ldc
        };

        rewriter.create<func::CallOp>(
            loc, funcOp, callArgs);

        // Erase the original linalg.matmul operation
        rewriter.eraseOp(op);

        return success();
    }
};

// Conversion pass
struct LinalgToSimpBLASPass
    : public PassWrapper<LinalgToSimpBLASPass, OperationPass<ModuleOp>> {

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<linalg::LinalgDialect,
                       func::FuncDialect,
                       memref::MemRefDialect,
                       arith::ArithDialect,
                       LLVM::LLVMDialect>();
    }

    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<LinalgMatmulToSimpBLASPattern>(&getContext());

        if (failed(applyPatternsAndFoldGreedily(
                getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }

    StringRef getArgument() const final {
        return "linalg-to-simpblas";
    }

    StringRef getDescription() const final {
        return "Lower linalg.matmul to SimpBLAS calls";
    }
};

// Pass registration
std::unique_ptr<Pass> createLinalgToSimpBLASPass() {
    return std::make_unique<LinalgToSimpBLASPass>();
}

} // namespace simp
} // namespace mlir
```

#### 6.3.3 Output MLIR (after lowering)

```mlir
func.func @kernel_main() -> f32 {
    %c32 = arith.constant 32 : i32
    %c0 = arith.constant 0 : index

    // Allocate matrices
    %A = memref.alloc(%c32, %c32) : memref<?x?xf32>
    %B = memref.alloc(%c32, %c32) : memref<?x?xf32>
    %C = memref.alloc(%c32, %c32) : memref<?x?xf32>

    // Extract pointers
    %A_ptr = memref.extract_aligned_pointer_as_index %A : memref<?x?xf32> -> index
    %B_ptr = memref.extract_aligned_pointer_as_index %B : memref<?x?xf32> -> index
    %C_ptr = memref.extract_aligned_pointer_as_index %C : memref<?x?xf32> -> index

    // Call SimpBLAS
    // sb_gemm_f32(M=32, N=32, K=32, A, lda=32, B, ldb=32, C, ldc=32)
    func.call @sb_gemm_f32(%c32, %c32, %c32,
                          %A_ptr, %c32,
                          %B_ptr, %c32,
                          %C_ptr, %c32) : (i32, i32, i32, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, i32) -> ()

    %result = memref.load %C[%c0, %c0] : memref<?x?xf32>
    return %result : f32
}

// External declaration
func.func private @sb_gemm_f32(i32, i32, i32, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, i32)
```

### 6.4 Benefits of MLIR Approach

**Compared to Direct LLVM IR Generation:**

1. **Operation Fusion**
   ```mlir
   // Before SimpBLAS lowering, can fuse operations:
   %T1 = linalg.matmul %A, %B
   %T2 = linalg.add %T1, %bias  // Fuse with matmul

   // Lower to single optimized kernel
   ```

2. **Automatic Tiling**
   ```mlir
   // linalg-tile pass can tile before BLAS lowering
   // Enables better cache utilization
   ```

3. **Layout Optimization**
   ```mlir
   // Can transform data layouts at high level
   // Convert row-major ↔ column-major before BLAS call
   ```

4. **Pattern Matching**
   ```mlir
   // Recognize matrix chain: (A @ B) @ C
   // Optimize multiplication order
   ```

5. **Hardware Specialization**
   ```mlir
   // Can generate different code for different targets
   // AVX-512: larger tiles
   // ARM NEON: different BLAS variant
   ```

### 6.5 Testing Strategy for SimpBLAS Integration

**Critical Tests:**

1. **Correctness Test**
   ```cpp
   TEST(SimpBLAS, MatmulCorrectness) {
       // 2x2 matrix multiply
       // A = [1 2; 3 4], B = [5 6; 7 8]
       // Expected C = [19 22; 43 50]

       auto result = compileAndRun("test_matmul.sl");
       EXPECT_ARRAY_EQ(result, {19, 22, 43, 50});
   }
   ```

2. **Code Generation Test**
   ```cpp
   TEST(SimpBLAS, GeneratesBLASCall) {
       auto mlir = compileToMLIR("matmul.sl");
       EXPECT_TRUE(contains(mlir, "func.call @sb_gemm_f32"));
   }
   ```

3. **Performance Test**
   ```cpp
   TEST(SimpBLAS, PerformanceBenchmark) {
       // Large matrix (1024x1024)
       auto blas_time = benchmarkSimpBLAS();
       auto naive_time = benchmarkNaive();

       // SimpBLAS should be significantly faster
       EXPECT_LT(blas_time, naive_time * 0.1);  // 10x faster
   }
   ```

### 6.6 Future Extensions

**Potential Enhancements:**

1. **Additional BLAS Operations**
   - GEMV (matrix-vector multiply)
   - AXPY (vector addition)
   - DOT (dot product)

2. **Advanced Patterns**
   - Fused multiply-add: C = αAB + βC
   - Batched matrix multiply

3. **Automatic Tuning**
   - Auto-select tile sizes based on matrix size
   - Cache-aware optimization

4. **Multi-Level Codegen**
   ```mlir
   Large matrices → SimpBLAS
   Small matrices → inline vectorized code
   ```

---

## Part 7: Recommendations & Next Steps

### 7.1 Final Recommendation

**Proceed with MLIR Integration: YES ✅**

**Rationale:**
1. ✅ **Strategic Alignment**: Perfect fit for deep learning focus
2. ✅ **Technical Feasibility**: High, existing LLVM knowledge transfers
3. ✅ **Risk**: Medium and manageable with phased approach
4. ✅ **ROI**: 10-30% performance improvement on tensor operations
5. ✅ **Future-Proof**: Industry standard, enables GPU backends
6. ✅ **SimpBLAS Synergy**: Natural integration with linalg dialect

**Key Success Factors:**
- Phased rollout (keep LLVM IR path as fallback)
- Comprehensive testing at each phase
- Focus on high-value use cases (arrays, matrices, tensors)
- Community engagement (MLIR Discord, forums)

### 7.2 Implementation Strategy

**Recommended Approach: Hybrid (Option C)**

```
Phase 1 (Weeks 1-8):   Foundation - Basic operations via MLIR
Phase 2 (Weeks 9-14):  Arrays - Vectorization and optimization
Phase 3 (Weeks 15-24): Tensors - SimpBLAS and MobileNet optimization
```

**Rollout Plan:**
1. **Weeks 1-8**: Opt-in MLIR path (`--use-mlir` flag)
2. **Weeks 9-14**: Gradual expansion, more tests
3. **Weeks 15-24**: Production-ready, consider default
4. **Post-24**: Deprecate LLVM-only path (optional)

### 7.3 Resource Allocation

**Team Requirements:**
- 1 developer full-time (experienced with LLVM/MLIR)
- 0.5 developer part-time (testing and validation)
- Access to MLIR community (Discord, forums)

**Infrastructure:**
- CI/CD with MLIR build support
- Performance benchmarking infrastructure
- Documentation platform

### 7.4 Success Metrics

**Quantitative Metrics:**

| Metric | Target | Measurement |
|--------|--------|-------------|
| Compilation Time | < 2x LLVM path | Benchmarks |
| Runtime Performance | >= 100% baseline | Benchmarks |
| Tensor Operations | 10-30% faster | MobileNet POC |
| Code Coverage | >= 80% | gcov/lcov |
| Test Pass Rate | 100% | CI |
| Memory Overhead | < 10% | Profiling |

**Qualitative Metrics:**
- Code maintainability improved
- Better abstraction for optimizations
- Easier GPU backend integration
- Community adoption

### 7.5 Risk Management

**Mitigation Strategies:**

1. **Learning Curve**
   - Complete Toy tutorial
   - Study existing projects (TensorFlow, PyTorch)
   - Engage with MLIR community early

2. **Performance Regression**
   - Continuous benchmarking in CI
   - Performance gates (auto-fail if > 5% slower)
   - Profiling tools integration

3. **Maintenance Burden**
   - Comprehensive documentation
   - Clean code structure
   - Regular refactoring

4. **API Changes**
   - Pin MLIR version initially
   - Track MLIR release notes
   - Plan migration for major MLIR updates

### 7.6 Alternative Approaches

**If MLIR Integration is Deferred:**

1. **Enhance Current LLVM Path**
   - Improve auto-vectorization hints
   - Better SimpBLAS integration
   - Manual optimization passes

2. **Specialized Tensor Compiler**
   - Build custom tensor optimization
   - Limited scope but faster development
   - Less general but works for current needs

3. **External Compiler Integration**
   - Use XLA, TVM, or other tensor compilers
   - Delegate tensor operations
   - Integration complexity

**None of these match MLIR's benefits long-term**

### 7.7 Next Steps (AI-Assisted Approach)

**Session 1: Environment Setup & Initial Learning (3-4 hours)**

1. **MLIR Build (AI-assisted)**
   ```bash
   # Clone LLVM/MLIR (AI provides script)
   git clone https://github.com/llvm/llvm-project.git

   # Build MLIR (AI generates optimized build config)
   cd llvm-project
   mkdir build && cd build
   cmake -G Ninja ../llvm \
       -DLLVM_ENABLE_PROJECTS=mlir \
       -DLLVM_TARGETS_TO_BUILD="X86" \
       -DCMAKE_BUILD_TYPE=Release \
       -DLLVM_ENABLE_ASSERTIONS=ON
   ninja
   ```

2. **Initial Learning**
   - Start MLIR Toy tutorial (Parts 1-2)
   - Verify mlir-opt works with examples
   - Explore MLIR IR syntax

**Sessions 2-3: Deep Learning & Design (6-8 hours)**

- Complete Toy tutorial with AI assistance
- AI helps extract patterns from linalg → BLAS examples
- Draft `simp` dialect spec (AI generates documentation template)
- Create TableGen definitions (AI generates boilerplate)

**Decision Points:**

- **After Session 11 (3-4 weeks):** Review Phase 1 progress
  - ✅ Basic compilation working?
  - ✅ Performance acceptable (>= 95% baseline)?
  - ✅ Continue to Phase 2 or adjust approach?

- **After Session 28 (6-7 weeks):** Review Phase 2 progress
  - ✅ Vectorization working?
  - ✅ Performance improvements measurable?
  - ✅ Continue to Phase 3 SimpBLAS integration?

- **After Session 52 (9-13 weeks):** Final review
  - ✅ All goals met?
  - ✅ Ready for production use?
  - ✅ Merge to main or keep as experimental?

### 7.8 Long-Term Vision

**MLIR as Foundation (12-24 months):**

1. **GPU Backend** (Post Phase 3)
   - Add GPU dialect lowering
   - CUDA/ROCm/OpenCL code generation
   - Unified CPU/GPU compilation

2. **Advanced Optimizations**
   - Polyhedral optimization for nested loops
   - Automatic kernel fusion
   - Memory layout optimization

3. **Language Extensions**
   - Custom operators via MLIR
   - User-defined dialects
   - Meta-programming support

4. **Ecosystem Integration**
   - Export ONNX models
   - Import PyTorch/TensorFlow
   - Interop with ML frameworks

**MLIR enables a modern, extensible compiler architecture for the next decade of SimpLang development.**

---

## Conclusion

MLIR integration represents a **strategic investment** in SimpLang's future. With **AI-assisted development**, this can be accomplished in **9-13 weeks** (36-52 focused 3-4 hour sessions) instead of traditional 4-6 month timelines.

### Why AI Makes This Feasible

✅ **Boilerplate Acceleration**: AI generates dialect operations, TableGen definitions, and lowering patterns **3-5x faster**
✅ **Test Generation**: Comprehensive test suites written **5-10x faster** with AI assistance
✅ **Documentation**: AI generates docs from code **10x faster**
✅ **Pattern Matching**: AI helps find and adapt existing MLIR patterns **2-3x faster**

### Expected Benefits

✅ **Performance**: 10-30% improvement on tensor operations
✅ **Capability**: Superior auto-vectorization and optimization
✅ **Future-Proofing**: Industry-standard IR, GPU-ready
✅ **Maintainability**: Cleaner abstractions, better tooling
✅ **Alignment**: Perfect fit for deep learning focus

### Realistic Timeline

- **Phase 1 (3-4 weeks):** Basic MLIR integration, feature parity
- **Phase 2 (2-3 weeks):** Array optimization, auto-vectorization
- **Phase 3 (4-6 weeks):** Tensor operations, SimpBLAS integration, MobileNet POC
- **Total: 9-13 weeks** of focused AI pair-programming sessions

**The array-first, SimpBLAS-integrated architecture makes this integration cleaner and more valuable than originally anticipated.**

**Recommendation: PROCEED with AI-assisted phased MLIR integration starting Session 1.**

**Success Metric:** Continuous progress validation after every phase with clear go/no-go decision points.

---

## References

### MLIR Resources
- Official Documentation: https://mlir.llvm.org/
- Toy Tutorial: https://mlir.llvm.org/docs/Tutorials/Toy/
- Linalg Dialect: https://mlir.llvm.org/docs/Dialects/Linalg/
- Dialect Conversion: https://mlir.llvm.org/docs/DialectConversion/

### Research Papers
- "MLIR: A Compiler Infrastructure for the End of Moore's Law" (Lattner et al., 2020)
- "Compiler Support for Sparse Tensor Computations in MLIR" (2022)

### Related Projects
- TensorFlow XLA: https://www.tensorflow.org/xla
- torch-mlir: https://github.com/llvm/torch-mlir
- ONNX-MLIR: https://github.com/onnx/onnx-mlir
- IREE: https://github.com/openxla/iree

### Community
- MLIR Discourse: https://discourse.llvm.org/c/mlir/
- MLIR Discord: https://discord.gg/xS7Z362
- Weekly Meetings: https://mlir.llvm.org/getting_started/Community/

---

**Document Status:** Draft v1.0
**Author:** Claude Code
**Date:** 2025-01-18
**Review Status:** Pending team review
**Next Update:** After Phase 1 completion
