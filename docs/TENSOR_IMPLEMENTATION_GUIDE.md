# SimpLang MLIR Tensor Implementation Guide

**Date**: October 21, 2024
**Status**: Planning Phase Complete
**Target**: NHWC Layout Support + Native Tensor Dialect

---

## Table of Contents

1. [Overview](#overview)
2. [Research Summary](#research-summary)
3. [Current State Analysis](#current-state-analysis)
4. [Design Decisions](#design-decisions)
5. [Implementation Plan](#implementation-plan)
6. [Technical Details](#technical-details)
7. [Testing Strategy](#testing-strategy)
8. [Performance Targets](#performance-targets)

---

## Overview

This guide documents the implementation of tensor abstractions with NHWC layout support for the SimpLang MLIR backend. The implementation follows a **hybrid progressive approach** combining quick wins with long-term architectural improvements.

### Goals

1. **Enable 4D tensor operations** for deep learning workloads
2. **Support NHWC layout** (optimal for GPU Tensor Cores)
3. **Provide clean tensor abstractions** in SimpLang syntax
4. **Enable tensor optimizations** (fusion, tiling, layout transformation)
5. **Maintain performance** comparable to hand-written MLIR

### Non-Goals

- Full tensor shape inference (use explicit dimensions)
- Dynamic shape support beyond current capabilities
- Automatic layout selection (user-specified layouts)
- GPU code generation (CPU-only initially)

---

## Research Summary

### MLIR Tensor vs MemRef

**Key Findings from Research:**

| Aspect | Tensor | MemRef |
|--------|--------|--------|
| **Abstraction Level** | High-level mathematical | Low-level memory buffer |
| **Mutability** | Immutable (SSA values) | Mutable memory |
| **Layout** | Abstract, can be inferred | Explicit layout/strides |
| **Optimizations** | Fusion, tiling easier | Direct memory control |
| **Use Case** | Initial representation | Final lowering stage |

**Recommendation**: Use tensors for high-level ops, bufferize to memref for execution.

### NHWC Layout Advantages

**Why NHWC for Deep Learning:**

1. **GPU Tensor Cores require NHWC** for optimal performance (NVIDIA)
2. **Coalesced memory access** - channels at same spatial position are contiguous
3. **Industry standard** - TensorFlow default, PyTorch supports
4. **Performance gain**: 1.5-3x faster than NCHW on modern GPUs

**NHWC Affine Map Representation:**
```mlir
#nhwc = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// Logical indexing: [N, C, H, W]
// Physical memory:   [N, H, W, C]
```

### MLIR Bufferization

**One-Shot Bufferize** (recommended for MLIR 14.0):
- Analyzes entire function for in-place opportunities
- Minimizes allocations through SSA analysis
- Preserves layout information from tensor to memref
- Handles function boundaries correctly

**Key Pass**: `-one-shot-bufferize="bufferize-function-boundaries"`

---

## Current State Analysis

### ✅ What Works (Session 10)

**Type System:**
- `SimpTensorType` fully defined in TableGen but **never instantiated**
- `ArrayTypeInfo` has `std::vector<int> dimensions` for multi-dim support
- SIMD hints and alignment fields exist but unused

**MLIR Integration:**
- 1D arrays working: `!simp.array<T>` → `memref<?xT>`
- MatMul proves the pattern: uses `memref.reinterpret_cast` for 2D views
- Solid pipeline: Simp → MemRef+Linalg → SCF → LLVM

**Current MatMul Pattern** (proves multi-dim handling):
```cpp
// 1D array → 2D memref view with explicit strides
lhs2D = memref.reinterpret_cast %lhs : memref<?xf32>
    to memref<?x?xf32>
    sizes: [%m, %k]
    strides: [%k, 1]  // Row-major: [K, 1]
```

### ❌ What's Missing

**Multi-Dimensional Support:**
- Multi-dim indices ignored in array access/store lowering
- Only first dimension used in array creation
- No shape inference or static shape propagation

**Layout Abstractions:**
- Hardcoded row-major only
- No affine maps for layout specification
- No NHWC/NCHW representation

**Tensor Infrastructure:**
- No tensor dialect usage
- No bufferization passes
- No tensor optimizations (fusion, tiling for tensors)

### 🔍 Key Insight from MatMul

The matmul implementation provides the **proven pattern** for tensor operations:

**Pattern: 1D Array → Multi-Dim View**
```
1. Accept 1D arrays from SimpLang (simple syntax)
2. Use memref.reinterpret_cast to create multi-dim views
3. Pass explicit dimensions + strides
4. Call linalg operations on views
```

**This pattern can be extended to tensors with affine maps for layouts.**

---

## Design Decisions

### Strategic Choice: Hybrid Progressive Approach

**Phase 1: Quick Win (Path A)**
- Extend memref.reinterpret_cast pattern from matmul
- Add affine maps for NHWC layout
- Works with current infrastructure
- **Timeline**: 1 session (4-5 hours)
- **Deliverable**: Working NHWC convolution

**Phase 2: Proper Infrastructure (Path B)**
- Activate native MLIR tensor dialect
- Add bufferization infrastructure
- Enable tensor optimizations
- **Timeline**: 3-4 sessions (9-13 hours)
- **Deliverable**: Production-quality tensor ops

**Rationale**:
- Path A gets NHWC functional quickly for experimentation
- Path B provides long-term scalability and clean abstractions
- Progressive approach allows iterative refinement

### Layout Strategy

**Supported Layouts (Priority Order):**

1. **NHWC** (Channels Last) - Primary target
   - Affine map: `(d0, d1, d2, d3) -> (d0, d2, d3, d1)`
   - Strides: `[H*W*C, W*C, C, 1]`
   - Use case: Convolution operations

2. **NCHW** (Channels First) - Secondary
   - Affine map: `(d0, d1, d2, d3) -> (d0, d1, d2, d3)` (identity)
   - Strides: `[C*H*W, H*W, W, 1]`
   - Use case: Some CPU operations, PyTorch compatibility

3. **Row-Major** (Existing) - Fallback
   - 2D: Strides `[N, 1]`
   - Use case: Matrix operations

### Type System Extension

**New Type: TensorTypeInfo**
```cpp
class TensorTypeInfo : public TypeInfo {
    std::unique_ptr<TypeInfo> elementType;  // f32, f64, etc.
    std::vector<int64_t> shape;             // [N, H, W, C] or [-1, -1, -1, -1]
    TensorLayout layout;                    // NHWC, NCHW, RowMajor, Custom
    std::string customLayoutMap;            // Affine map string for Custom layout

    // Methods
    bool hasStaticShape() const;
    bool isDynamic(size_t dim) const;
    std::string toString() const;           // "tensor<NxHxWxCxf32, NHWC>"
};
```

---

## Implementation Plan

### Session 11: NHWC Support via MemRef + Affine Maps (Path A)

**Duration**: 4-5 hours
**Objective**: Add 4D tensor support with NHWC layout using memref.reinterpret_cast

#### Step 1: Extend AST Multi-Dimensional Support (1 hour)

**Files to Modify:**
- `src/parser.y` - Parse 4D tensor syntax
- `src/mlir/mlir_codegen.cpp` - Use all dimensions from ArrayTypeInfo

**New Syntax:**
```cpp
// SimpLang code
var input = tensor<f32>([1, 56, 56, 64]);  // NHWC: [N, H, W, C]
input[0, 10, 10, 5] = 1.0;                 // Multi-dim indexing
```

**Implementation:**
```cpp
// mlir_codegen.cpp - lowerArrayCreate
SmallVector<Value> dims;
for (int64_t dim : arrayType->dimensions) {
    dims.push_back(builder.create<arith::ConstantIndexOp>(loc, dim));
}
// Total size = N * H * W * C
Value totalSize = dims[0];
for (size_t i = 1; i < dims.size(); i++) {
    totalSize = builder.create<arith::MulIOp>(loc, totalSize, dims[i]);
}
// Allocate 1D array of total size
Value array = builder.create<ArrayCreateOp>(loc, totalSize, elemType);
```

**Multi-Dim Indexing:**
```cpp
// input[n, h, w, c] → flattened index: n*H*W*C + h*W*C + w*C + c
Value computeFlattenedIndex(ArrayRef<Value> indices, ArrayRef<Value> dims) {
    // For NHWC: idx = n*(H*W*C) + h*(W*C) + w*C + c
    Value idx = indices[0];
    Value stride = dims[1];
    for (size_t i = 2; i < dims.size(); i++) {
        stride = builder.create<arith::MulIOp>(loc, stride, dims[i]);
    }
    idx = builder.create<arith::MulIOp>(loc, idx, stride);

    // Add remaining dimensions
    // ... (full implementation in code)

    return idx;
}
```

#### Step 2: Add Layout Enumeration (30 min)

**Files to Create:**
- `include/ast/type/tensor_layout.hpp`

**Content:**
```cpp
#ifndef AST_TYPE_TENSOR_LAYOUT_HPP
#define AST_TYPE_TENSOR_LAYOUT_HPP

#include <string>

enum class TensorLayout {
    RowMajor,   // Default C-style (for 2D matrices)
    NHWC,       // (N, H, W, C) - Channels last
    NCHW,       // (N, C, H, W) - Channels first
    Custom      // User-defined affine map
};

inline std::string tensorLayoutToString(TensorLayout layout) {
    switch (layout) {
        case TensorLayout::RowMajor: return "row_major";
        case TensorLayout::NHWC:     return "nhwc";
        case TensorLayout::NCHW:     return "nchw";
        case TensorLayout::Custom:   return "custom";
        default:                     return "unknown";
    }
}

#endif // AST_TYPE_TENSOR_LAYOUT_HPP
```

**Extend TypeInfo:**
```cpp
// type_info.hpp
class TensorTypeInfo : public TypeInfo {
public:
    std::unique_ptr<TypeInfo> elementType;
    std::vector<int64_t> shape;     // [N, H, W, C] or [-1, ...] for dynamic
    TensorLayout layout;
    std::string customLayoutMap;    // For TensorLayout::Custom

    TensorTypeInfo(std::unique_ptr<TypeInfo> elemType,
                   std::vector<int64_t> shp,
                   TensorLayout lay = TensorLayout::RowMajor)
        : TypeInfo(TypeKind::Array),  // Reuse Array kind for now
          elementType(std::move(elemType)),
          shape(std::move(shp)),
          layout(lay) {}

    bool hasStaticShape() const {
        return std::all_of(shape.begin(), shape.end(),
                          [](int64_t dim) { return dim != -1; });
    }

    std::string toString() const override;
};
```

#### Step 3: Add MLIR Affine Map Helpers (1 hour)

**Files to Modify:**
- `src/mlir/mlir_codegen.cpp`

**New Helper Functions:**
```cpp
#include "mlir/IR/AffineMap.h"

namespace {

/// Create affine map for NHWC layout
/// Logical indices: (N, C, H, W)
/// Physical memory: (N, H, W, C)
mlir::AffineMap createNHWCAffineMap(mlir::MLIRContext* ctx) {
    using namespace mlir;
    auto d0 = getAffineDimExpr(0, ctx);  // N
    auto d1 = getAffineDimExpr(1, ctx);  // C
    auto d2 = getAffineDimExpr(2, ctx);  // H
    auto d3 = getAffineDimExpr(3, ctx);  // W

    // Permute: NCHW → NHWC
    return AffineMap::get(4, 0, {d0, d2, d3, d1}, ctx);
}

/// Create affine map for NCHW layout (identity)
mlir::AffineMap createNCHWAffineMap(mlir::MLIRContext* ctx) {
    return mlir::AffineMap::getMultiDimIdentityMap(4, ctx);
}

/// Create affine map for given layout
mlir::AffineMap createAffineMapForLayout(TensorLayout layout,
                                          unsigned rank,
                                          mlir::MLIRContext* ctx) {
    switch (layout) {
        case TensorLayout::NHWC:
            if (rank == 4) return createNHWCAffineMap(ctx);
            break;
        case TensorLayout::NCHW:
            if (rank == 4) return createNCHWAffineMap(ctx);
            break;
        case TensorLayout::RowMajor:
            return mlir::AffineMap::getMultiDimIdentityMap(rank, ctx);
        case TensorLayout::Custom:
            // TODO: Parse custom affine map
            break;
    }
    // Default: identity map
    return mlir::AffineMap::getMultiDimIdentityMap(rank, ctx);
}

/// Compute strides for NHWC layout
/// Memory layout: [N][H][W][C]
/// Strides: [H*W*C, W*C, C, 1]
SmallVector<Value> computeNHWCStrides(Value H, Value W, Value C,
                                       OpBuilder& builder, Location loc) {
    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value WC = builder.create<arith::MulIOp>(loc, W, C);
    Value HWC = builder.create<arith::MulIOp>(loc, H, WC);

    return {HWC, WC, C, one};
}

/// Compute strides for NCHW layout
/// Memory layout: [N][C][H][W]
/// Strides: [C*H*W, H*W, W, 1]
SmallVector<Value> computeNCHWStrides(Value C, Value H, Value W,
                                       OpBuilder& builder, Location loc) {
    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value HW = builder.create<arith::MulIOp>(loc, H, W);
    Value CHW = builder.create<arith::MulIOp>(loc, C, HW);

    return {CHW, HW, W, one};
}

/// Create memref type with layout
MemRefType createMemRefWithLayout(ArrayRef<int64_t> shape,
                                   Type elemType,
                                   TensorLayout layout,
                                   MLIRContext* ctx) {
    AffineMap layoutMap = createAffineMapForLayout(layout, shape.size(), ctx);
    return MemRefType::get(shape, elemType, layoutMap);
}

} // anonymous namespace
```

#### Step 4: Implement Conv2D Operation (1.5 hours)

**Files to Modify:**
- `include/mlir/Dialects/Simp/SimpOps.td`
- `src/mlir/lowering/ConvertSimpToMemRef.cpp`

**SimpOps.td - Add Conv2D:**
```tablegen
def Simp_Conv2DOp : Simp_Op<"conv2d", [Pure]> {
    let summary = "2D Convolution operation with NHWC layout";
    let description = [{
        Performs 2D convolution on 4D tensors in NHWC layout.

        Input:  [N, H, W, C_in]  (batch, height, width, in_channels)
        Filter: [kH, kW, C_in, C_out] (kernel_h, kernel_w, in_channels, out_channels)
        Output: [N, H', W', C_out] (pre-allocated by caller)

        Strides and padding determine output spatial dimensions:
        H' = (H + 2*pad_h - kH) / stride_h + 1
        W' = (W + 2*pad_w - kW) / stride_w + 1

        Example:
        ```mlir
        %result = simp.conv2d %input, %filter, %output,
                              %N, %H, %W, %C_in,
                              %kH, %kW, %C_out,
                              %stride_h, %stride_w
                  : !simp.array<f32>
        ```
    }];

    let arguments = (ins
        Simp_ArrayType:$input,      // NHWC: [N, H, W, C_in] as 1D
        Simp_ArrayType:$filter,     // HWCF: [kH, kW, C_in, C_out] as 1D
        Simp_ArrayType:$output,     // NHWC: [N, H', W', C_out] as 1D (pre-allocated)
        I64:$N, I64:$H, I64:$W, I64:$C_in,      // Input dimensions
        I64:$kH, I64:$kW, I64:$C_out,           // Filter dimensions
        I64:$stride_h, I64:$stride_w,           // Strides
        I64:$pad_h, I64:$pad_w                  // Padding
    );

    let results = (outs Simp_ArrayType:$result);

    let assemblyFormat = [{
        $input `,` $filter `,` $output `,`
        $N `,` $H `,` $W `,` $C_in `,`
        $kH `,` $kW `,` $C_out `,`
        $stride_h `,` $stride_w `,`
        $pad_h `,` $pad_w
        attr-dict `:` type($input)
    }];
}
```

**ConvertSimpToMemRef.cpp - Conv2D Lowering:**
```cpp
struct ConvertConv2DOpLowering : public OpConversionPattern<Conv2DOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(Conv2DOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter& rewriter) const override {
        auto loc = op.getLoc();
        auto ctx = rewriter.getContext();

        // Get dimensions and convert to index type
        Value N = castToIndex(op.N(), rewriter, loc);
        Value H = castToIndex(op.H(), rewriter, loc);
        Value W = castToIndex(op.W(), rewriter, loc);
        Value C_in = castToIndex(op.C_in(), rewriter, loc);
        Value kH = castToIndex(op.kH(), rewriter, loc);
        Value kW = castToIndex(op.kW(), rewriter, loc);
        Value C_out = castToIndex(op.C_out(), rewriter, loc);

        // Get element type
        auto arrayType = op.input().getType().cast<ArrayType>();
        Type elemType = arrayType.getElementType();

        // Create NHWC affine map
        auto nhwcMap = createNHWCAffineMap(ctx);

        // Create 4D memref types with NHWC layout
        // Input: memref<NxHxWxC_in xf32, #nhwc>
        auto inputType = MemRefType::get({-1, -1, -1, -1}, elemType, nhwcMap);

        // Filter: HWCF layout (height, width, channels_in, channels_out)
        // This is the natural layout for NHWC convolution
        auto filterType = MemRefType::get({-1, -1, -1, -1}, elemType);

        // Compute output spatial dimensions
        // H_out = (H + 2*pad_h - kH) / stride_h + 1
        // W_out = (W + 2*pad_w - kW) / stride_w + 1
        Value stride_h = castToIndex(op.stride_h(), rewriter, loc);
        Value stride_w = castToIndex(op.stride_w(), rewriter, loc);
        Value pad_h = castToIndex(op.pad_h(), rewriter, loc);
        Value pad_w = castToIndex(op.pad_w(), rewriter, loc);

        Value two = rewriter.create<arith::ConstantIndexOp>(loc, 2);
        Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

        Value H_padded = rewriter.create<arith::MulIOp>(loc, two, pad_h);
        H_padded = rewriter.create<arith::AddIOp>(loc, H, H_padded);
        H_padded = rewriter.create<arith::SubIOp>(loc, H_padded, kH);
        Value H_out = rewriter.create<arith::DivUIOp>(loc, H_padded, stride_h);
        H_out = rewriter.create<arith::AddIOp>(loc, H_out, one);

        Value W_padded = rewriter.create<arith::MulIOp>(loc, two, pad_w);
        W_padded = rewriter.create<arith::AddIOp>(loc, W, W_padded);
        W_padded = rewriter.create<arith::SubIOp>(loc, W_padded, kW);
        Value W_out = rewriter.create<arith::DivUIOp>(loc, W_padded, stride_w);
        W_out = rewriter.create<arith::AddIOp>(loc, W_out, one);

        auto outputType = MemRefType::get({-1, -1, -1, -1}, elemType, nhwcMap);

        // Reinterpret 1D arrays as 4D memrefs
        Value constZero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

        // Input: NHWC strides [H*W*C_in, W*C_in, C_in, 1]
        auto inputStrides = computeNHWCStrides(H, W, C_in, rewriter, loc);
        Value input4D = rewriter.create<memref::ReinterpretCastOp>(
            loc, inputType, adaptor.input(),
            constZero,
            ValueRange{N, H, W, C_in},
            inputStrides
        );

        // Filter: HWCF strides [kW*C_in*C_out, C_in*C_out, C_out, 1]
        Value kWC_in = rewriter.create<arith::MulIOp>(loc, kW, C_in);
        Value kWC_inC_out = rewriter.create<arith::MulIOp>(loc, kWC_in, C_out);
        Value C_inC_out = rewriter.create<arith::MulIOp>(loc, C_in, C_out);
        SmallVector<Value> filterStrides = {kWC_inC_out, C_inC_out, C_out, one};

        Value filter4D = rewriter.create<memref::ReinterpretCastOp>(
            loc, filterType, adaptor.filter(),
            constZero,
            ValueRange{kH, kW, C_in, C_out},
            filterStrides
        );

        // Output: NHWC strides [H_out*W_out*C_out, W_out*C_out, C_out, 1]
        auto outputStrides = computeNHWCStrides(H_out, W_out, C_out, rewriter, loc);
        Value output4D = rewriter.create<memref::ReinterpretCastOp>(
            loc, outputType, adaptor.output(),
            constZero,
            ValueRange{N, H_out, W_out, C_out},
            outputStrides
        );

        // Call linalg.conv_2d_nhwc_hwcf
        // This is the NHWC-specific convolution operation
        rewriter.create<linalg::Conv2DNhwcHwcfOp>(
            loc,
            ValueRange{input4D, filter4D},  // inputs
            ValueRange{output4D},            // outputs
            ArrayAttr{}  // No additional attributes for now
        );

        // Result is the output array (mutation semantics after lowering)
        rewriter.replaceOp(op, adaptor.output());

        return success();
    }

private:
    Value castToIndex(Value val, OpBuilder& builder, Location loc) const {
        if (val.getType().isa<IndexType>()) return val;
        return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), val);
    }
};
```

#### Step 5: Testing and Validation (1 hour)

**Files to Create:**
- `tests/mlir/integration/test_conv2d_nhwc.sl`
- `tests/mlir/integration/conv2d_host.cpp`

**test_conv2d_nhwc.sl:**
```c
// 2D Convolution with NHWC layout
// Input: [1, 28, 28, 3] - 1 image, 28x28 pixels, 3 channels (RGB)
// Filter: [3, 3, 3, 32] - 3x3 kernel, 3 input channels, 32 output channels
// Output: [1, 26, 26, 32] - 1 image, 26x26 pixels, 32 channels (no padding, stride 1)

fn conv2d_nhwc_test(
    f32[] input,   // Flattened NHWC: [1*28*28*3 = 2352 elements]
    f32[] filter,  // Flattened HWCF: [3*3*3*32 = 864 elements]
    f32[] output   // Flattened NHWC: [1*26*26*32 = 21632 elements]
) -> f32 {
    // Dimensions
    var N = 1;
    var H = 28;
    var W = 28;
    var C_in = 3;
    var kH = 3;
    var kW = 3;
    var C_out = 32;
    var stride_h = 1;
    var stride_w = 1;
    var pad_h = 0;
    var pad_w = 0;

    // Perform convolution
    conv2d(input, filter, output,
           N, H, W, C_in,
           kH, kW, C_out,
           stride_h, stride_w,
           pad_h, pad_w);

    // Return first output element
    return output[0];
}
```

**conv2d_host.cpp:**
```cpp
#include <iostream>
#include <chrono>
#include <dlfcn.h>
#include <vector>
#include <random>

// MemRef descriptor structure (5 args per memref)
typedef float (*Conv2DKernelFunc)(
    float*, float*, int64_t, int64_t, int64_t,  // input memref
    float*, float*, int64_t, int64_t, int64_t,  // filter memref
    float*, float*, int64_t, int64_t, int64_t   // output memref
);

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <kernel.so>" << std::endl;
        return 1;
    }

    // Load compiled kernel
    void* handle = dlopen(argv[1], RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load kernel: " << dlerror() << std::endl;
        return 1;
    }

    auto kernel = (Conv2DKernelFunc)dlsym(handle, "conv2d_nhwc_test");
    if (!kernel) {
        std::cerr << "Failed to find function: " << dlerror() << std::endl;
        return 1;
    }

    // Initialize data
    const int N = 1, H = 28, W = 28, C_in = 3;
    const int kH = 3, kW = 3, C_out = 32;
    const int H_out = 26, W_out = 26;  // With padding=0, stride=1

    const int input_size = N * H * W * C_in;           // 2352
    const int filter_size = kH * kW * C_in * C_out;    // 864
    const int output_size = N * H_out * W_out * C_out; // 21632

    std::vector<float> input(input_size);
    std::vector<float> filter(filter_size);
    std::vector<float> output(output_size, 0.0f);

    // Random initialization
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& val : input)  val = dist(gen);
    for (auto& val : filter) val = dist(gen);

    // Call kernel (memref calling convention: 5 args per memref)
    auto start = std::chrono::high_resolution_clock::now();

    float result = kernel(
        input.data(), input.data(), 0, input_size, 1,    // input memref
        filter.data(), filter.data(), 0, filter_size, 1, // filter memref
        output.data(), output.data(), 0, output_size, 1  // output memref
    );

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Compute GFLOPS
    // FLOPs for convolution: 2 * N * H_out * W_out * C_out * kH * kW * C_in
    int64_t flops = 2LL * N * H_out * W_out * C_out * kH * kW * C_in;
    double gflops = (flops / 1e9) / (duration.count() / 1e6);

    std::cout << "Conv2D NHWC Performance:" << std::endl;
    std::cout << "  Input:  [" << N << ", " << H << ", " << W << ", " << C_in << "]" << std::endl;
    std::cout << "  Filter: [" << kH << ", " << kW << ", " << C_in << ", " << C_out << "]" << std::endl;
    std::cout << "  Output: [" << N << ", " << H_out << ", " << W_out << ", " << C_out << "]" << std::endl;
    std::cout << "  Time:   " << duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "  GFLOPS: " << gflops << std::endl;
    std::cout << "  First output element: " << result << std::endl;

    dlclose(handle);
    return 0;
}
```

**Build and Test Commands:**
```bash
# Compile kernel
./build_mlir/src/simplang tests/mlir/integration/test_conv2d_nhwc.sl \
    --emit-mlir --enable-tiling -o conv2d_kernel.o

# Create shared library
gcc -shared -fPIC conv2d_kernel.o -o conv2d_kernel.so

# Compile host
g++ -O3 tests/mlir/integration/conv2d_host.cpp -o conv2d_host -ldl

# Run test
./conv2d_host conv2d_kernel.so
```

### Session 11 Deliverables Summary

**✅ What We'll Have After Session 11:**
1. 4D tensor syntax in SimpLang
2. Multi-dimensional indexing support
3. NHWC layout representation via affine maps
4. Working Conv2D operation
5. Stride computation for NHWC
6. Test infrastructure for tensor ops

**📊 Success Criteria:**
- Conv2D compiles without errors
- Generated MLIR shows correct NHWC affine maps
- Performance is reasonable (will optimize in Path B)
- Test produces correct output

---

### Sessions 12-14: Native Tensor Dialect Integration (Path B)

**Duration**: 9-13 hours across 3-4 sessions
**Objective**: Replace memref.reinterpret_cast with proper tensor dialect

#### Session 12: Activate SimpTensorType (3-4 hours)

**Step 1: Enable Tensor Type in Codegen**

**Files to Modify:**
- `src/mlir/mlir_codegen.cpp`

**Changes:**
```cpp
Type MLIRCodeGenContext::convertType(const std::string& simpType) {
    if (simpType.find("tensor<") == 0) {
        // Parse: "tensor<f32, [1, 28, 28, 3]>"
        auto [elemType, shape] = parseTensorType(simpType);

        // Create SimpTensorType (already defined in TableGen!)
        return SimpTensorType::get(context, shape, elemType);
        // Result: !simp.tensor<1x28x28x3xf32>
    }

    // Fallback to array for backward compatibility
    if (simpType.find("array<") == 0) {
        return ArrayType::get(context, parseElementType(simpType));
    }
}
```

**Step 2: Implement Tensor Operations**

**Files to Modify:**
- `include/mlir/Dialects/Simp/SimpOps.td`

**New Operations:**
```tablegen
def Simp_TensorCreateOp : Simp_Op<"tensor.create", [Pure]> {
    let summary = "Create a tensor with given dimensions";
    let arguments = (ins Variadic<Index>:$dimensions);
    let results = (outs Simp_SimpTensorType:$result);
}

def Simp_TensorLoadOp : Simp_Op<"tensor.load", [Pure]> {
    let summary = "Load element from tensor";
    let arguments = (ins
        Simp_SimpTensorType:$tensor,
        Variadic<I64>:$indices
    );
    let results = (outs AnyType:$result);
}

def Simp_TensorStoreOp : Simp_Op<"tensor.store", [Pure]> {
    let summary = "Store element to tensor (SSA-pure, returns new tensor)";
    let arguments = (ins
        Simp_SimpTensorType:$tensor,
        Variadic<I64>:$indices,
        AnyType:$value
    );
    let results = (outs Simp_SimpTensorType:$result);
}
```

**Step 3: Lower to MLIR Tensor Dialect**

**Files to Create:**
- `src/mlir/lowering/ConvertSimpTensorToMLIRTensor.cpp`

**Lowering Patterns:**
```cpp
// simp.tensor.create → tensor.empty
struct ConvertTensorCreateLowering : public OpConversionPattern<TensorCreateOp> {
    LogicalResult matchAndRewrite(TensorCreateOp op, ...) {
        auto tensorType = op.getResult().getType().cast<SimpTensorType>();

        // Create MLIR RankedTensorType
        auto mlirTensorType = RankedTensorType::get(
            tensorType.getShape(),
            tensorType.getElementType()
        );

        // Generate: tensor.empty() : tensor<1x28x28x3xf32>
        Value emptyTensor = rewriter.create<tensor::EmptyOp>(
            loc, mlirTensorType.getShape(), mlirTensorType.getElementType()
        );

        rewriter.replaceOp(op, emptyTensor);
        return success();
    }
};

// simp.tensor.load → tensor.extract
struct ConvertTensorLoadLowering : public OpConversionPattern<TensorLoadOp> {
    LogicalResult matchAndRewrite(TensorLoadOp op, ...) {
        // tensor.extract %tensor[%i0, %i1, %i2, %i3] : tensor<...>
        rewriter.replaceOpWithNewOp<tensor::ExtractOp>(
            op, op.tensor(), op.indices()
        );
        return success();
    }
};

// simp.tensor.store → tensor.insert
struct ConvertTensorStoreLowering : public OpConversionPattern<TensorStoreOp> {
    LogicalResult matchAndRewrite(TensorStoreOp op, ...) {
        // tensor.insert %value into %tensor[%i0, %i1, %i2, %i3]
        rewriter.replaceOpWithNewOp<tensor::InsertOp>(
            op, op.value(), op.tensor(), op.indices()
        );
        return success();
    }
};
```

#### Session 13: Tensor Optimization Passes (3-4 hours)

**Step 1: Add Fusion Pass**

**Files to Modify:**
- `src/mlir/mlir_pipeline.cpp`

**New Phase:**
```cpp
bool MLIRCompilationPipeline::runTensorOptimizationPasses() {
    PassManager pm(module.getContext());

    // Fuse element-wise operations
    pm.addNestedPass<func::FuncOp>(createLinalgElementwiseOpFusionPass());

    // Tile linalg operations on tensors
    if (enableTiling) {
        pm.addNestedPass<func::FuncOp>(
            createLinalgTilingPass({1, 8, 8, 32})  // Tile NHWC
        );
    }

    // Canonicalize
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    return succeeded(pm.run(module));
}
```

**Step 2: Add Layout Propagation**

**Goal**: Minimize transpose operations

```cpp
pm.addPass(createDataLayoutPropagationPass());
```

#### Session 14: Bufferization Infrastructure (3-5 hours)

**Step 1: Integrate One-Shot Bufferize**

**Files to Modify:**
- `src/mlir/mlir_pipeline.cpp`
- `src/mlir/CMakeLists.txt` (link bufferization libraries)

**Implementation:**
```cpp
bool MLIRCompilationPipeline::runBufferizationPass() {
    PassManager pm(module.getContext());

    // Configure bufferization options
    bufferization::OneShotBufferizationOptions options;
    options.bufferizeFunctionBoundaries = true;
    options.setFunctionBoundaryTypeConversion(
        bufferization::LayoutMapOption::IdentityLayoutMap
    );

    // Unknown type conversion: use fully dynamic layouts
    options.unknownTypeConverterFn =
        [](Value value, Attribute memorySpace,
           const bufferization::BufferizationOptions& options) {
        return bufferization::getMemRefTypeWithFullyDynamicLayout(
            value.getType().cast<TensorType>(), memorySpace
        );
    };

    // Set buffer alignment (32 bytes for AVX)
    options.bufferAlignment = 32;

    // Add one-shot bufferize pass
    pm.addPass(bufferization::createOneShotBufferizePass(options));

    // Buffer deallocation
    pm.addPass(bufferization::createBufferDeallocationPass());

    // Convert bufferization dialect to memref
    pm.addPass(createBufferizationToMemRefPass());

    if (failed(pm.run(module))) {
        llvm::errs() << "Bufferization pass failed\n";
        return false;
    }

    return true;
}
```

**Step 2: Layout Preservation**

**Challenge**: Ensure NHWC layout survives bufferization

**Solution**: Attach layout as encoding attribute

```cpp
// When creating tensor type, attach layout encoding
auto nhwcAttr = AffineMapAttr::get(createNHWCAffineMap(ctx));
auto tensorType = RankedTensorType::get(
    shape, elemType,
    /*encoding=*/nhwcAttr
);

// During bufferization:
// tensor<1x28x28x3xf32, #nhwc> → memref<1x28x28x3xf32, #nhwc>
```

**Step 3: Updated Pipeline**

```cpp
bool MLIRCompilationPipeline::runPasses() {
    // Phase 1: Simp → Tensor + Linalg
    if (!runSimpToTensorLowering()) return false;

    // Phase 2: Tensor optimizations (fusion, tiling)
    if (!runTensorOptimizationPasses()) return false;

    // Phase 3: Bufferization (tensor → memref with layouts)
    if (!runBufferizationPass()) return false;

    // Phase 4: Linalg → SCF loops
    if (!runLinalgToLoopsLowering()) return false;

    // Phase 5: → LLVM dialect
    if (!runToLLVMDialectLowering()) return false;

    return true;
}
```

---

## Technical Details

### NHWC Stride Computation

**Formula for NHWC [N, H, W, C]:**
```
Linear index = n*(H*W*C) + h*(W*C) + w*C + c
Strides = [H*W*C, W*C, C, 1]
```

**Example: [1, 28, 28, 3]**
```
Strides = [28*28*3, 28*3, 3, 1]
        = [2352, 84, 3, 1]
```

**Verification:**
- Element [0, 10, 15, 2] → 0*2352 + 10*84 + 15*3 + 2 = 887 ✓

### Affine Map Semantics

**NHWC Affine Map:**
```mlir
#nhwc = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
```

**Interpretation:**
- Logical indexing: `tensor[N, C, H, W]` (user code)
- Physical memory: `memory[N, H, W, C]` (layout)
- Permutation: `(0, 1, 2, 3) → (0, 2, 3, 1)`

**Usage in MemRef:**
```mlir
memref<1x64x28x28xf32, #nhwc>
// Logical: [N=1, C=64, H=28, W=28]
// Memory:  [N=1, H=28, W=28, C=64]
```

### Linalg Conv2D Operations

**MLIR provides multiple conv2d variants:**

1. **`linalg.conv_2d_nhwc_hwcf`** - NHWC input, HWCF filter
   - Input: `memref<?x?x?x?xf32>` as [N, H, W, C]
   - Filter: `memref<?x?x?x?xf32>` as [kH, kW, C_in, C_out]
   - Output: `memref<?x?x?x?xf32>` as [N, H', W', C_out]

2. **`linalg.conv_2d_nchw_fchw`** - NCHW input, FCHW filter
   - Input: `memref<?x?x?x?xf32>` as [N, C, H, W]
   - Filter: `memref<?x?x?x?xf32>` as [C_out, C_in, kH, kW]
   - Output: `memref<?x?x?x?xf32>` as [N, C_out, H', W']

**We use variant 1 for NHWC support.**

### MemRef Calling Convention

**For 1D memref:**
```
5 arguments: (allocated_ptr, aligned_ptr, offset, size, stride)
```

**For 4D memref:**
```
11 arguments: (allocated_ptr, aligned_ptr, offset,
               size_0, size_1, size_2, size_3,
               stride_0, stride_1, stride_2, stride_3)
```

**Example C++ call:**
```cpp
kernel(
    ptr, ptr, 0,          // base memref info
    N, H, W, C,           // sizes
    H*W*C, W*C, C, 1      // strides
);
```

---

## Testing Strategy

### Unit Tests

**Test Hierarchy:**

1. **Type Tests** (`test_tensor_types.cpp`)
   - TensorTypeInfo creation
   - Shape parsing
   - Layout enumeration

2. **Operation Tests** (`test_tensor_ops.mlir`)
   - Tensor creation
   - Multi-dim indexing
   - Layout affine maps

3. **Lowering Tests** (`test_tensor_lowering.mlir`)
   - Simp → Tensor dialect
   - Tensor → MemRef bufferization
   - Layout preservation

### Integration Tests

**Test 1: Simple 4D Tensor**
```c
// test_tensor_4d.sl
var t = tensor<f32>([2, 3, 4, 5]);
t[1, 2, 3, 4] = 42.0;
return t[1, 2, 3, 4];  // Should return 42.0
```

**Test 2: Conv2D NHWC**
```c
// test_conv2d_nhwc.sl (from Session 11)
conv2d(input, filter, output, ...);
```

**Test 3: Tensor Fusion**
```c
// test_tensor_fusion.sl
var x = conv2d(...);
var y = relu(x);      // Should fuse into conv
return y[0];
```

### Performance Benchmarks

**Benchmark 1: Conv2D Layout Comparison**
```bash
# NHWC layout
./conv2d_host conv2d_nhwc.so
# Expected: ~X GFLOPS

# NCHW layout
./conv2d_host conv2d_nchw.so
# Expected: NHWC should be faster on GPU-like architectures
```

**Benchmark 2: Fusion Impact**
```bash
# Without fusion
./bench_conv_relu_separate
# Result: 2 allocations, higher memory traffic

# With fusion
./bench_conv_relu_fused
# Result: 1 allocation, lower memory traffic
# Expected speedup: 1.2-1.5x
```

### Correctness Verification

**Method 1: Compare with NumPy**
```python
# Generate test data
import numpy as np
input = np.random.randn(1, 28, 28, 3).astype(np.float32)
filter = np.random.randn(3, 3, 3, 32).astype(np.float32)

# NumPy reference
from scipy.ndimage import convolve
output_ref = convolve(input, filter, mode='valid')

# Save to file, load in C++, compare with SimpLang output
```

**Method 2: Analytical Tests**
```c
// Use simple patterns (all 1s, identity filters)
// where output is mathematically predictable
```

---

## Performance Targets

### Session 11 Targets (Path A)

**Conv2D Performance:**
- **Input**: [1, 28, 28, 3]
- **Filter**: [3, 3, 3, 32]
- **Target**: > 1 GFLOPS (baseline, unoptimized)

**Acceptance Criteria:**
- Compiles without errors
- Produces correct results
- Performance within 5x of hand-written C++ (acceptable for initial implementation)

### Sessions 12-14 Targets (Path B)

**Conv2D Performance (with optimizations):**
- **Target**: > 5 GFLOPS (with tiling + vectorization)
- **Fusion**: 1.2-1.5x speedup over separate ops

**Memory Efficiency:**
- Fusion reduces allocations by ~50%
- In-place bufferization minimizes memory overhead

**Comparison Baseline:**
```cpp
// Hand-written C++ NHWC conv2d
// Expected: ~10-15 GFLOPS on modern CPU
// SimpLang target: Within 2-3x of hand-written
```

---

## File Modifications Checklist

### Session 11 Files

**New Files:**
- [ ] `include/ast/type/tensor_layout.hpp`
- [ ] `tests/mlir/integration/test_conv2d_nhwc.sl`
- [ ] `tests/mlir/integration/conv2d_host.cpp`

**Modified Files:**
- [ ] `include/ast/type/type_info.hpp` - Add TensorTypeInfo
- [ ] `src/ast/type/type_info.cpp` - Implement TensorTypeInfo methods
- [ ] `src/parser.y` - Parse tensor<T>([...]) syntax
- [ ] `src/mlir/mlir_codegen.cpp` - Multi-dim support, affine map helpers
- [ ] `include/mlir/Dialects/Simp/SimpOps.td` - Add Conv2DOp
- [ ] `src/mlir/lowering/ConvertSimpToMemRef.cpp` - Conv2D lowering
- [ ] `src/mlir/CMakeLists.txt` - Link any new libraries
- [ ] `tests/CMakeLists.txt` - Add conv2d test targets

### Sessions 12-14 Files

**New Files:**
- [ ] `src/mlir/lowering/ConvertSimpTensorToMLIRTensor.cpp`
- [ ] `docs/tensor_architecture.md`

**Modified Files:**
- [ ] `src/mlir/mlir_pipeline.cpp` - Add tensor optimization & bufferization phases
- [ ] `src/mlir/mlir_codegen.cpp` - Activate SimpTensorType
- [ ] `include/mlir/Dialects/Simp/SimpOps.td` - Add tensor ops
- [ ] `src/mlir/CMakeLists.txt` - Link bufferization libraries

---

## Dependencies

### MLIR Libraries (Already Linked)

- `MLIRIR` - Core IR
- `MLIRLinalg` - Linalg operations
- `MLIRMemRef` - MemRef dialect
- `MLIRArithmetic` - Arithmetic ops

### New Libraries Needed (Sessions 12-14)

- `MLIRTensor` - Tensor dialect
- `MLIRBufferization` - Bufferization infrastructure
- `MLIRBufferizationTransforms` - Bufferization passes

**CMakeLists.txt Addition:**
```cmake
target_link_libraries(simplang
    # ... existing libraries ...
    MLIRTensor
    MLIRBufferization
    MLIRBufferizationTransforms
)
```

---

## Known Limitations & Future Work

### Current Limitations

1. **Static Shapes Only**: Dynamic shapes require tensor shape inference
2. **CPU Only**: No GPU code generation (would need GPU dialect)
3. **Limited Layouts**: Only NHWC, NCHW, row-major
4. **Manual Layout Selection**: No automatic layout optimization

### Future Enhancements

1. **Dynamic Shape Support**
   - Add shape inference pass
   - Support `-1` dimensions in tensor types
   - Runtime shape propagation

2. **GPU Backend**
   - Lower to GPU dialect
   - Add CUDA/ROCm code generation
   - Tensor Core utilization on NVIDIA GPUs

3. **More Layouts**
   - Tiled layouts (e.g., 16x16 tiles)
   - Packed layouts for matrix ops
   - Custom user-defined layouts

4. **Auto-Layout Selection**
   - Profile-guided layout optimization
   - Cost model for layout transformations
   - Automatic transpose insertion/removal

5. **More Operations**
   - Pooling (max, avg)
   - Batch normalization
   - Activation functions (ReLU, sigmoid, tanh)
   - Elementwise operations (add, mul, etc.)

---

## References

### MLIR Documentation

- **Toy Tutorial Chapter 6**: Lowering to LLVM (bufferization pattern)
- **Linalg Dialect**: https://mlir.llvm.org/docs/Dialects/Linalg/
- **Bufferization**: https://mlir.llvm.org/docs/Bufferization/
- **Affine Dialect**: https://mlir.llvm.org/docs/Dialects/Affine/

### SimpLang Codebase References

- **MatMul Implementation**: `src/mlir/lowering/ConvertSimpToMemRef.cpp:380-450`
- **Type System**: `include/ast/type/type_info.hpp`
- **MLIR Pipeline**: `src/mlir/mlir_pipeline.cpp`
- **SimpTensorType**: `include/mlir/Dialects/Simp/SimpTypes.td:20-40`

### Performance Resources

- **NHWC vs NCHW**: NVIDIA Deep Learning Performance Guide
- **Tensor Cores**: NVIDIA Ampere Architecture Whitepaper
- **MLIR Performance**: "MLIR: A Compiler Infrastructure for the End of Moore's Law"

---

## Changelog

**October 21, 2024**:
- Initial document creation
- Research phase complete
- Implementation plan finalized
- Hybrid progressive approach approved

---

**Next Action**: Begin Session 11 implementation of Path A (NHWC via MemRef + Affine Maps)
