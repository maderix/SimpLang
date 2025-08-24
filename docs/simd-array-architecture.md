# SIMD Array Architecture Design

## Overview

This document outlines the design for extending SimpLang's existing array interface with SIMD (Single Instruction, Multiple Data) capabilities. The design leverages our current high-performance array implementation while adding plugin-based SIMD backend support.

## Plugin-Based SIMD Backend Architecture

### Frontend Syntax (Unified)
```simplang
// SIMD-aligned array declaration
var data = array<f32, align=simd>([1024]);      // Auto-detect best SIMD
var avx_data = array<f32, simd=avx2>([1024]);   // Force AVX2
var neon_data = array<f32, simd=neon>([1024]);  // Force ARM NEON

// Vector slice operations (backend-agnostic)
data[0:8] = data[8:16] + data[16:24];     // Vectorized ops
var result = dot(data[0:256], data[256:512]);

// Auto-vectorization hints
#[vectorize(width=auto)]  // Let backend decide
#[vectorize(width=8)]     // Force 8-wide vectors
fn matrix_mul(a: array<f32>, b: array<f32>) -> array<f32> { ... }
```

### Plugin Architecture
```
SimpLang Core
├── Frontend Parser (unified syntax)
├── SIMD Backend Manager
│   ├── Backend Detection (runtime/compile-time)
│   ├── Plugin Loader
│   └── LLVM IR Generation
└── Plugins/
    ├── libsimd_avx512.so    (Intel AVX-512)
    ├── libsimd_avx2.so      (Intel AVX2)  
    ├── libsimd_sse42.so     (Intel SSE4.2)
    ├── libsimd_neon.so      (ARM NEON)
    ├── libsimd_sve.so       (ARM SVE)
    └── libsimd_hexagon.so   (Qualcomm Hexagon)
```

### Backend Plugin Interface
```cpp
// include/simd_backend.hpp
class SIMDBackend {
public:
    virtual ~SIMDBackend() = default;
    
    // Capability queries
    virtual bool supportsTarget() const = 0;
    virtual int getVectorWidth(Type type) const = 0;  // f32->8, f64->4 for AVX
    virtual int getAlignment() const = 0;             // 32 for AVX, 16 for SSE
    
    // LLVM IR generation
    virtual llvm::Value* createVectorLoad(llvm::IRBuilder<>&, llvm::Value* ptr, int width) = 0;
    virtual llvm::Value* createVectorStore(llvm::IRBuilder<>&, llvm::Value* val, llvm::Value* ptr) = 0;
    virtual llvm::Value* createVectorAdd(llvm::IRBuilder<>&, llvm::Value* a, llvm::Value* b) = 0;
    virtual llvm::Value* createVectorMul(llvm::IRBuilder<>&, llvm::Value* a, llvm::Value* b) = 0;
    virtual llvm::Value* createVectorFMA(llvm::IRBuilder<>&, llvm::Value* a, llvm::Value* b, llvm::Value* c) = 0;
    
    // Advanced operations
    virtual llvm::Value* createDotProduct(llvm::IRBuilder<>&, llvm::Value* a, llvm::Value* b, int len) = 0;
    virtual llvm::Value* createMatMul(llvm::IRBuilder<>&, llvm::Value* a, llvm::Value* b, int m, int n, int k) = 0;
};

// Plugin entry point
extern "C" SIMDBackend* createBackend();
```

### Example Plugin Implementation (AVX)
```cpp
// plugins/avx2_backend.cpp
class AVX2Backend : public SIMDBackend {
public:
    bool supportsTarget() const override {
        return __builtin_cpu_supports("avx2");
    }
    
    int getVectorWidth(Type type) const override {
        return type == Type::F32 ? 8 : 4;  // 8 floats, 4 doubles
    }
    
    llvm::Value* createVectorAdd(llvm::IRBuilder<>& builder, 
                                llvm::Value* a, llvm::Value* b) override {
        auto* intrinsic = llvm::Intrinsic::getDeclaration(
            builder.GetInsertBlock()->getModule(), 
            llvm::Intrinsic::x86_avx_add_ps_256);
        return builder.CreateCall(intrinsic, {a, b});
    }
    
    // Optimized dot product using FMA
    llvm::Value* createDotProduct(llvm::IRBuilder<>& builder,
                                 llvm::Value* a, llvm::Value* b, int len) override {
        // Use AVX2 FMA instructions for optimal performance
        auto* fma_intrinsic = llvm::Intrinsic::getDeclaration(
            builder.GetInsertBlock()->getModule(),
            llvm::Intrinsic::x86_fma_vfmadd_ps_256);
        // ... implementation
    }
};

extern "C" SIMDBackend* createBackend() { 
    return new AVX2Backend(); 
}
```

## SIMD Extension Design - Leveraging Existing Array Interface

### 1. Extend TypeInfo for SIMD Arrays
```cpp
// include/ast.hpp - Natural extension
class ArrayTypeInfo : public TypeInfo {
public:
    std::unique_ptr<TypeInfo> elementType;
    std::vector<size_t> staticDimensions;  // Existing
    
    // SIMD Extensions (new)
    enum class SIMDHint {
        None,     // Regular array
        Auto,     // Auto-detect best SIMD  
        SSE,      // Force SSE
        AVX,      // Force AVX
        AVX512,   // Force AVX-512
        NEON,     // Force ARM NEON
        SVE       // Force ARM SVE
    };
    
    SIMDHint simdHint;
    int alignment;        // 16=SSE, 32=AVX, 64=AVX512
    bool vectorizable;    // Can use vector operations
    
    ArrayTypeInfo(std::unique_ptr<TypeInfo> elemType, 
                  std::vector<size_t> dimensions,
                  SIMDHint simd = SIMDHint::None)
        : TypeInfo(TypeKind::Array), 
          elementType(std::move(elemType)),
          staticDimensions(std::move(dimensions)),
          simdHint(simd) {
        
        vectorizable = (simd != SIMDHint::None) && 
                      (elementType->isFloat() || elementType->isInteger());
        alignment = getSIMDAlignment(simd);
    }
};
```

### 2. Parser Extension - Natural Syntax
```cpp
// src/parser.y - Extend existing array syntax
array_type : TARRAY '<' type '>' '[' dimension_list ']'                 { /* existing */ }
           | TARRAY '<' type ',' SIMD '=' simd_hint '>' '[' dimension_list ']'  { /* new */ }
           ;

simd_hint : TAUTO     { $$ = SIMDHint::Auto; }
          | TSSE      { $$ = SIMDHint::SSE; }
          | TAVX      { $$ = SIMDHint::AVX; }
          | TAVX512   { $$ = SIMDHint::AVX512; }
          | TNEON     { $$ = SIMDHint::NEON; }
          ;
```

**Results in natural syntax:**
```simplang
// Existing syntax (unchanged)
var arr = array<f32>([1024]);

// Extended syntax (new)
var simd_arr = array<f32, simd=auto>([1024]);    // Auto-detect
var avx_arr = array<f32, simd=avx>([1024]);      // Force AVX
var neon_arr = array<f32, simd=neon>([1024]);    // Force NEON
```

### 3. ArrayCreateExprAST - Enhanced, Not Replaced
```cpp
// src/ast.cpp - Extend existing codeGen
llvm::Value* ArrayCreateExprAST::codeGen(CodeGenContext& context) {
    auto* arrayTypeInfo = static_cast<ArrayTypeInfo*>(elementType.get());
    
    if (arrayTypeInfo->vectorizable) {
        // NEW: SIMD-aligned allocation
        auto backend = context.getSIMDBackend(arrayTypeInfo->simdHint);
        if (backend) {
            return createSIMDAlignedArray(context, backend);
        }
    }
    
    // EXISTING: Fallback to regular array (unchanged)
    return createRegularArray(context);
}
```

### 4. ArrayAccessExprAST - Vector Slice Support
```cpp
// Enhanced array access for vector slices
llvm::Value* ArrayAccessExprAST::codeGen(CodeGenContext& context) {
    if (isVectorSlice()) {  // NEW: arr[0:8] syntax detection
        auto backend = context.getSIMDBackend();
        if (backend) {
            return backend->createVectorSlice(context.getBuilder(), 
                                            array->codeGen(context),
                                            indices);
        }
    }
    
    // EXISTING: Single element access (unchanged)
    return createElementAccess(context);
}
```

### 5. New Vector Slice Syntax
```simplang
// Existing (unchanged)
var element = arr[i];        // Single element
var elem2d = matrix[i, j];   // 2D access

// NEW: Vector slices
var slice = arr[0:8];        // 8 elements starting at 0
var slice2d = matrix[i, 0:4]; // Vector slice from 2D array
var strided = arr[0:16:2];   // Every 2nd element (0,2,4...14)

// NEW: SIMD operations on slices
arr[0:8] = arr[8:16] + arr[16:24];  // Vector addition
var dot = sum(arr[0:256] * arr[256:512]); // Dot product
```

### 6. Backend Manager Integration
```cpp
// src/codegen.cpp - Backend detection
class CodeGenContext {
    std::map<ArrayTypeInfo::SIMDHint, std::unique_ptr<SIMDBackend>> backends;
    
public:
    SIMDBackend* getSIMDBackend(ArrayTypeInfo::SIMDHint hint = ArrayTypeInfo::SIMDHint::Auto) {
        if (hint == ArrayTypeInfo::SIMDHint::Auto) {
            // Runtime detection - pick best available
            if (backends[ArrayTypeInfo::SIMDHint::AVX512]) return backends[ArrayTypeInfo::SIMDHint::AVX512].get();
            if (backends[ArrayTypeInfo::SIMDHint::AVX]) return backends[ArrayTypeInfo::SIMDHint::AVX].get();
            return backends[ArrayTypeInfo::SIMDHint::SSE].get();
        }
        return backends[hint].get();
    }
};
```

### CodeGen Integration
```cpp
// src/codegen.cpp - Modified ArrayExprAST
llvm::Value* ArrayExprAST::codeGen(CodeGenContext& context) {
    auto backend = context.getSIMDBackend();  // Auto-detected or forced
    
    if (backend && isVectorizable()) {
        // Generate vectorized code
        int width = backend->getVectorWidth(elementType);
        return backend->createVectorizedOperation(context.getBuilder(), ...);
    } else {
        // Fallback to scalar code
        return generateScalarCode(context);
    }
}
```

## Benefits of this Approach

✅ **Backward compatible** - all existing array code unchanged  
✅ **Natural progression** - extends familiar syntax  
✅ **Reuses existing infrastructure** - ArrayCreateExprAST, ArrayAccessExprAST  
✅ **Gradual migration** - can mix SIMD and regular arrays  
✅ **Performance path** - opt-in SIMD when needed  
✅ **Single syntax** for all platforms
✅ **Optimal performance** per target  
✅ **Easy extensibility** - new backends via plugins
✅ **Graceful degradation** - falls back to scalar
✅ **Runtime detection** - picks best available SIMD
✅ **Future-proof** - easy to add new instruction sets

## Migration Example

```simplang
// Phase 1: Regular arrays (existing)
var data = array<f32>([1024]);
for i in 0..1024 { data[i] = i * 2.0; }

// Phase 2: SIMD arrays (new, same operations)  
var simd_data = array<f32, simd=auto>([1024]);
for i in 0..1024 { simd_data[i] = i * 2.0; }  // Auto-vectorized!

// Phase 3: Explicit vector ops (new)
simd_data[0:8] = simd_data[8:16] * 2.0;  // 8-wide SIMD multiply
```

## Future: SimpTensor Integration

This SIMD array foundation will enable high-performance tensor operations:

```simplang
// SimpTensor built on SIMD arrays
var tensor = Tensor<f32, simd=avx>([batch, height, width]);
var result = tensor.matmul(weights);  // Uses AVX-optimized GEMM
```

This design leverages our excellent existing array foundation while naturally extending it for SIMD performance! The interface remains intuitive and the implementation can reuse most of the current codeGen infrastructure.