//===- ConvertSimpToMemRef.cpp - Lower Simp to MemRef dialect ------------===//
//
// Part of the SimpLang Project
//
// This file implements the conversion pass from Simp dialect to MemRef and
// Arith dialects. This is Phase 1 of the progressive lowering strategy.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/simp_dialect.hpp"
#include "mlir/simp_ops.hpp"
#include "mlir/simp_types.hpp"
#include "llvm/ADT/Optional.h"

using namespace mlir;
using namespace mlir::simp;

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

/// Convert Simp types to MemRef types
class SimpTypeConverter : public TypeConverter {
public:
  SimpTypeConverter() {
    // Keep standard types as-is
    addConversion([](Type type) { return type; });

    // Convert !simp.array<T> to memref<?xT>
    addConversion([](ArrayType type) -> llvm::Optional<Type> {
      // Use -1 for dynamic dimension in MLIR 14
      return MemRefType::get({-1}, type.getElementType());
    });

    // Add argument materialization: handles function arguments
    addArgumentMaterialization([](OpBuilder &builder, Type resultType,
                                   ValueRange inputs, Location loc) -> Value {
      // Materialize memref from simp.array for function arguments
      // Just create an unrealized cast that later passes will clean up
      if (inputs.size() == 1)
        return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
      return nullptr;
    });
  }
};

//===----------------------------------------------------------------------===//
// Operation Conversion Patterns
//===----------------------------------------------------------------------===//

/// Convert FuncOp signature (convert !simp.array args to memref args)
struct FuncOpSignatureConversion : public OpConversionPattern<FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(FuncOp funcOp, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    // Convert function signature
    TypeConverter::SignatureConversion signatureConv(funcOp.getNumArguments());
    for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
      Type argType = funcOp.getArgument(i).getType();
      Type convertedType = getTypeConverter()->convertType(argType);
      signatureConv.addInputs(i, convertedType);
    }

    // Convert result types
    SmallVector<Type, 1> resultTypes;
    if (failed(getTypeConverter()->convertTypes(funcOp.getType().getResults(), resultTypes)))
      return failure();

    // Create new function type
    auto newFuncType = FunctionType::get(
        getContext(),
        signatureConv.getConvertedTypes(),
        resultTypes);

    // Update function signature in-place
    rewriter.updateRootInPlace(funcOp, [&] {
      funcOp.setType(newFuncType);
      for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
        funcOp.getArgument(i).setType(signatureConv.getConvertedTypes()[i]);
      }
    });

    return success();
  }
};

/// Convert simp.constant to arith.constant
struct ConstantOpLowering : public OpConversionPattern<simp::ConstantOp> {
  using OpConversionPattern<simp::ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      simp::ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // Replace with arith.constant (use value() method from TableGen)
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.value());
    return success();
  }
};

/// Convert simp.array_create to memref.alloc
struct ArrayCreateOpLowering : public OpConversionPattern<simp::ArrayCreateOp> {
  using OpConversionPattern<simp::ArrayCreateOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      simp::ArrayCreateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // Get the array type
    auto arrayType = op.getType().cast<ArrayType>();

    // Convert to memref<?xT> (-1 means dynamic dimension)
    auto memrefType = MemRefType::get({-1}, arrayType.getElementType());

    // Get the size operand
    Value size = adaptor.getOperands()[0];

    // memref.alloc requires 'index' type, so convert i64 -> index if needed
    if (!size.getType().isIndex()) {
      size = rewriter.create<arith::IndexCastOp>(
          op.getLoc(), rewriter.getIndexType(), size);
    }

    // Create memref.alloc with dynamic size
    rewriter.replaceOpWithNewOp<memref::AllocOp>(op, memrefType, ValueRange{size});

    return success();
  }
};

/// Convert simp.array_get to memref.load
struct ArrayGetOpLowering : public OpConversionPattern<simp::ArrayGetOp> {
  using OpConversionPattern<simp::ArrayGetOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      simp::ArrayGetOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // adaptor.getOperands() returns [array, index]
    auto operands = adaptor.getOperands();
    Value memref = operands[0];  // array (now memref)
    Value index = operands[1];   // index

    // memref.load requires 'index' type, so convert i64 -> index if needed
    if (!index.getType().isIndex()) {
      index = rewriter.create<arith::IndexCastOp>(
          op.getLoc(), rewriter.getIndexType(), index);
    }

    // Replace with memref.load
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, memref, ValueRange{index});

    return success();
  }
};

/// Convert simp.array_set to memref.store
/// Note: This changes semantics from SSA-pure to mutation
struct ArraySetOpLowering : public OpConversionPattern<simp::ArraySetOp> {
  using OpConversionPattern<simp::ArraySetOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      simp::ArraySetOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // adaptor.getOperands() returns [array, index, value]
    auto operands = adaptor.getOperands();
    Value memref = operands[0];  // array (now memref)
    Value index = operands[1];   // index
    Value value = operands[2];   // value to store

    // memref.store requires 'index' type, so convert i64 -> index if needed
    if (!index.getType().isIndex()) {
      index = rewriter.create<arith::IndexCastOp>(
          op.getLoc(), rewriter.getIndexType(), index);
    }

    // Create memref.store (mutates the memref)
    rewriter.create<memref::StoreOp>(op.getLoc(), value, memref, ValueRange{index});

    // Array set returns the "updated" array, but in memref semantics,
    // we just return the same memref (since it's mutated in-place)
    rewriter.replaceOp(op, memref);

    return success();
  }
};

/// Convert simp.add to arith.addf
struct AddOpLowering : public OpConversionPattern<simp::AddOp> {
  using OpConversionPattern<simp::AddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      simp::AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // Check if operating on floats or integers
    auto resultType = op.getResult().getType();
    auto operands = adaptor.getOperands();

    if (resultType.isa<FloatType>()) {
      rewriter.replaceOpWithNewOp<arith::AddFOp>(op, operands[0], operands[1]);
    } else if (resultType.isa<IntegerType>()) {
      rewriter.replaceOpWithNewOp<arith::AddIOp>(op, operands[0], operands[1]);
    } else {
      return failure();
    }

    return success();
  }
};

/// Convert simp.sub to arith.subf
struct SubOpLowering : public OpConversionPattern<simp::SubOp> {
  using OpConversionPattern<simp::SubOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      simp::SubOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto resultType = op.getResult().getType();
    auto operands = adaptor.getOperands();

    if (resultType.isa<FloatType>()) {
      rewriter.replaceOpWithNewOp<arith::SubFOp>(op, operands[0], operands[1]);
    } else if (resultType.isa<IntegerType>()) {
      rewriter.replaceOpWithNewOp<arith::SubIOp>(op, operands[0], operands[1]);
    } else {
      return failure();
    }

    return success();
  }
};

/// Convert simp.mul to arith.mulf
struct MulOpLowering : public OpConversionPattern<simp::MulOp> {
  using OpConversionPattern<simp::MulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      simp::MulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto resultType = op.getResult().getType();
    auto operands = adaptor.getOperands();

    if (resultType.isa<FloatType>()) {
      rewriter.replaceOpWithNewOp<arith::MulFOp>(op, operands[0], operands[1]);
    } else if (resultType.isa<IntegerType>()) {
      rewriter.replaceOpWithNewOp<arith::MulIOp>(op, operands[0], operands[1]);
    } else {
      return failure();
    }

    return success();
  }
};

/// Convert simp.div to arith.divf
struct DivOpLowering : public OpConversionPattern<simp::DivOp> {
  using OpConversionPattern<simp::DivOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      simp::DivOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto resultType = op.getResult().getType();
    auto operands = adaptor.getOperands();

    if (resultType.isa<FloatType>()) {
      rewriter.replaceOpWithNewOp<arith::DivFOp>(op, operands[0], operands[1]);
    } else if (resultType.isa<IntegerType>()) {
      // Use signed integer division
      rewriter.replaceOpWithNewOp<arith::DivSIOp>(op, operands[0], operands[1]);
    } else {
      return failure();
    }

    return success();
  }
};

/// Convert simp.mod to arith.remf/remsi
struct ModOpLowering : public OpConversionPattern<simp::ModOp> {
  using OpConversionPattern<simp::ModOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      simp::ModOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto resultType = op.getResult().getType();
    auto operands = adaptor.getOperands();

    if (resultType.isa<FloatType>()) {
      rewriter.replaceOpWithNewOp<arith::RemFOp>(op, operands[0], operands[1]);
    } else if (resultType.isa<IntegerType>()) {
      // Use signed integer remainder
      rewriter.replaceOpWithNewOp<arith::RemSIOp>(op, operands[0], operands[1]);
    } else {
      return failure();
    }

    return success();
  }
};

/// Convert simp.neg to arith.negf (floats) or 0 - x (integers)
struct NegOpLowering : public OpConversionPattern<simp::NegOp> {
  using OpConversionPattern<simp::NegOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      simp::NegOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto resultType = op.getResult().getType();
    Value operand = adaptor.getOperands()[0];

    if (resultType.isa<FloatType>()) {
      // Float negation: use arith.negf
      rewriter.replaceOpWithNewOp<arith::NegFOp>(op, operand);
    } else if (resultType.isa<IntegerType>()) {
      // Integer negation: compute 0 - x
      auto zero = rewriter.create<arith::ConstantIntOp>(
          op.getLoc(), 0, resultType.cast<IntegerType>());
      rewriter.replaceOpWithNewOp<arith::SubIOp>(op, zero, operand);
    } else {
      return failure();
    }

    return success();
  }
};

/// Convert simp.matmul to linalg.matmul
/// Operands are already converted to memrefs by the type converter
struct MatMulOpLowering : public OpConversionPattern<simp::MatMulOp> {
  using OpConversionPattern<simp::MatMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      simp::MatMulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();

    // Get converted operands (now memref<?xT> after type conversion)
    auto operands = adaptor.getOperands();
    Value lhs = operands[0];     // A: memref<?xT>
    Value rhs = operands[1];     // B: memref<?xT>
    Value output = operands[2];  // C: memref<?xT> (pre-allocated by caller)
    Value m = operands[3];       // Rows of A
    Value k = operands[4];       // Cols of A / Rows of B
    Value n = operands[5];       // Cols of B

    // Get element type
    auto lhsMemRefType = lhs.getType().cast<MemRefType>();
    Type elemType = lhsMemRefType.getElementType();

    // Convert dimensions to index type
    if (!m.getType().isIndex()) {
      m = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), m);
    }
    if (!k.getType().isIndex()) {
      k = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), k);
    }
    if (!n.getType().isIndex()) {
      n = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), n);
    }

    // Create 2D memref types for matrices
    auto matrixAType = MemRefType::get({-1, -1}, elemType);  // MxK
    auto matrixBType = MemRefType::get({-1, -1}, elemType);  // KxN
    auto matrixCType = MemRefType::get({-1, -1}, elemType);  // MxN

    // Create stride constants
    Value zeroOffset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value oneStride = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Reinterpret 1D arrays as 2D matrices
    auto lhs2D = rewriter.create<memref::ReinterpretCastOp>(
        loc, matrixAType, lhs, zeroOffset,
        ValueRange{m, k}, ValueRange{k, oneStride});

    auto rhs2D = rewriter.create<memref::ReinterpretCastOp>(
        loc, matrixBType, rhs, zeroOffset,
        ValueRange{k, n}, ValueRange{n, oneStride});

    // Use the pre-allocated output buffer (host-kernel model: no allocations in kernel)
    // Reinterpret output as 2D matrix
    auto output2D = rewriter.create<memref::ReinterpretCastOp>(
        loc, matrixCType, output, zeroOffset,
        ValueRange{m, n}, ValueRange{n, oneStride});

    // NOTE: linalg.matmul performs C += A * B (accumulation, not overwrite)
    // The caller is responsible for initializing C to zero if needed
    // This avoids redundant zero-initialization on every call

    // Create linalg.matmul: C += A * B (accumulates into pre-allocated output)
    rewriter.create<linalg::MatmulOp>(
        loc,
        ValueRange{lhs2D.getResult(), rhs2D.getResult()},
        ValueRange{output2D.getResult()});

    // Return the output buffer (same buffer that was passed in)
    rewriter.replaceOp(op, output);

    return success();
  }
};

/// Convert simp.conv2d to nested loops with memref operations
/// Generates optimized loop nest for 2D convolution with NHWC layout
struct Conv2DOpLowering : public OpConversionPattern<simp::Conv2DOp> {
  using OpConversionPattern<simp::Conv2DOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      simp::Conv2DOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();

    // Get converted operands (now memref<?xT> after type conversion)
    auto operands = adaptor.getOperands();
    Value input = operands[0];      // Input: NHWC as memref<?xT>
    Value weights = operands[1];    // Weights: [OC,KH,KW,IC] as memref<?xT>
    Value bias = operands[2];       // Bias: [OC] as memref<?xT>
    Value output = operands[3];     // Output: NHWC as memref<?xT> (pre-allocated)
    Value batch = operands[4];      // N
    Value in_h = operands[5];       // Input height
    Value in_w = operands[6];       // Input width
    Value in_c = operands[7];       // Input channels
    Value out_c = operands[8];      // Output channels
    Value kernel_h = operands[9];   // Kernel height
    Value kernel_w = operands[10];  // Kernel width
    Value stride_h = operands[11];  // Vertical stride
    Value stride_w = operands[12];  // Horizontal stride
    Value pad_h = operands[13];     // Vertical padding
    Value pad_w = operands[14];     // Horizontal padding

    // Get element type
    auto inputMemRefType = input.getType().cast<MemRefType>();
    Type elemType = inputMemRefType.getElementType();

    // Helper to convert i64 -> index
    auto toIndex = [&](Value v) -> Value {
      if (!v.getType().isIndex()) {
        return rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), v);
      }
      return v;
    };

    // Convert all dimensions to index type
    Value batchIdx = toIndex(batch);
    Value inHIdx = toIndex(in_h);
    Value inWIdx = toIndex(in_w);
    Value inCIdx = toIndex(in_c);
    Value outCIdx = toIndex(out_c);
    Value kernelHIdx = toIndex(kernel_h);
    Value kernelWIdx = toIndex(kernel_w);
    Value strideHIdx = toIndex(stride_h);
    Value strideWIdx = toIndex(stride_w);
    Value padHIdx = toIndex(pad_h);
    Value padWIdx = toIndex(pad_w);

    // Constants
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Compute output dimensions: out_h = (in_h + 2*pad_h - kernel_h) / stride_h + 1
    Value twoPadH = rewriter.create<arith::MulIOp>(loc, padHIdx,
        rewriter.create<arith::ConstantIndexOp>(loc, 2));
    Value tempH = rewriter.create<arith::AddIOp>(loc, inHIdx, twoPadH);
    tempH = rewriter.create<arith::SubIOp>(loc, tempH, kernelHIdx);
    Value outHIdx = rewriter.create<arith::DivSIOp>(loc, tempH, strideHIdx);
    outHIdx = rewriter.create<arith::AddIOp>(loc, outHIdx, c1);

    Value twoPadW = rewriter.create<arith::MulIOp>(loc, padWIdx,
        rewriter.create<arith::ConstantIndexOp>(loc, 2));
    Value tempW = rewriter.create<arith::AddIOp>(loc, inWIdx, twoPadW);
    tempW = rewriter.create<arith::SubIOp>(loc, tempW, kernelWIdx);
    Value outWIdx = rewriter.create<arith::DivSIOp>(loc, tempW, strideWIdx);
    outWIdx = rewriter.create<arith::AddIOp>(loc, outWIdx, c1);

    // Generate nested loops: for b in [0, batch)
    auto batchLoop = rewriter.create<scf::ForOp>(loc, c0, batchIdx, c1);
    rewriter.setInsertionPointToStart(batchLoop.getBody());
    Value b = batchLoop.getInductionVar();

    // for oh in [0, out_h)
    auto ohLoop = rewriter.create<scf::ForOp>(loc, c0, outHIdx, c1);
    rewriter.setInsertionPointToStart(ohLoop.getBody());
    Value oh = ohLoop.getInductionVar();

    // for ow in [0, out_w)
    auto owLoop = rewriter.create<scf::ForOp>(loc, c0, outWIdx, c1);
    rewriter.setInsertionPointToStart(owLoop.getBody());
    Value ow = owLoop.getInductionVar();

    // for oc in [0, out_c)
    auto ocLoop = rewriter.create<scf::ForOp>(loc, c0, outCIdx, c1);
    rewriter.setInsertionPointToStart(ocLoop.getBody());
    Value oc = ocLoop.getInductionVar();

    // Initialize accumulator with bias[oc]
    Value biasVal = rewriter.create<memref::LoadOp>(loc, bias, ValueRange{oc});

    // Create inner reduction loops with iter_args for accumulation
    // for kh in [0, kernel_h)
    auto khLoop = rewriter.create<scf::ForOp>(loc, c0, kernelHIdx, c1, ValueRange{biasVal});
    rewriter.setInsertionPointToStart(khLoop.getBody());
    Value kh = khLoop.getInductionVar();
    Value accKh = khLoop.getRegionIterArgs()[0];

    // for kw in [0, kernel_w)
    auto kwLoop = rewriter.create<scf::ForOp>(loc, c0, kernelWIdx, c1, ValueRange{accKh});
    rewriter.setInsertionPointToStart(kwLoop.getBody());
    Value kw = kwLoop.getInductionVar();
    Value accKw = kwLoop.getRegionIterArgs()[0];

    // for ic in [0, in_c)
    auto icLoop = rewriter.create<scf::ForOp>(loc, c0, inCIdx, c1, ValueRange{accKw});
    rewriter.setInsertionPointToStart(icLoop.getBody());
    Value ic = icLoop.getInductionVar();
    Value acc = icLoop.getRegionIterArgs()[0];

    // Compute input coordinates: ih = oh * stride_h + kh - pad_h
    Value ihTemp = rewriter.create<arith::MulIOp>(loc, oh, strideHIdx);
    ihTemp = rewriter.create<arith::AddIOp>(loc, ihTemp, kh);
    Value ih = rewriter.create<arith::SubIOp>(loc, ihTemp, padHIdx);

    // iw = ow * stride_w + kw - pad_w
    Value iwTemp = rewriter.create<arith::MulIOp>(loc, ow, strideWIdx);
    iwTemp = rewriter.create<arith::AddIOp>(loc, iwTemp, kw);
    Value iw = rewriter.create<arith::SubIOp>(loc, iwTemp, padWIdx);

    // Bounds check for padding (if ih < 0 || ih >= in_h || iw < 0 || iw >= in_w, skip)
    Value ihValid = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, ih, c0);
    Value ihInBounds = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, ih, inHIdx);
    Value iwValid = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, iw, c0);
    Value iwInBounds = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, iw, inWIdx);

    Value hValid = rewriter.create<arith::AndIOp>(loc, ihValid, ihInBounds);
    Value wValid = rewriter.create<arith::AndIOp>(loc, iwValid, iwInBounds);
    Value isValid = rewriter.create<arith::AndIOp>(loc, hValid, wValid);

    // Conditional computation: only load and compute if in bounds
    auto ifOp = rewriter.create<scf::IfOp>(loc, elemType, isValid, /*withElseRegion=*/true);

    // Then region: load input and weight, compute product
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());

    // Flatten input index: input[b, ih, iw, ic] -> input[b*in_h*in_w*in_c + ih*in_w*in_c + iw*in_c + ic]
    Value inputIdx = rewriter.create<arith::MulIOp>(loc, b, inHIdx);
    inputIdx = rewriter.create<arith::MulIOp>(loc, inputIdx, inWIdx);
    inputIdx = rewriter.create<arith::MulIOp>(loc, inputIdx, inCIdx);

    Value ihOffset = rewriter.create<arith::MulIOp>(loc, ih, inWIdx);
    ihOffset = rewriter.create<arith::MulIOp>(loc, ihOffset, inCIdx);
    inputIdx = rewriter.create<arith::AddIOp>(loc, inputIdx, ihOffset);

    Value iwOffset = rewriter.create<arith::MulIOp>(loc, iw, inCIdx);
    inputIdx = rewriter.create<arith::AddIOp>(loc, inputIdx, iwOffset);
    inputIdx = rewriter.create<arith::AddIOp>(loc, inputIdx, ic);

    Value inputVal = rewriter.create<memref::LoadOp>(loc, input, ValueRange{inputIdx});

    // Flatten weight index: weights[oc, kh, kw, ic] -> weights[oc*kh*kw*ic + kh*kw*ic + kw*ic + ic]
    Value weightIdx = rewriter.create<arith::MulIOp>(loc, oc, kernelHIdx);
    weightIdx = rewriter.create<arith::MulIOp>(loc, weightIdx, kernelWIdx);
    weightIdx = rewriter.create<arith::MulIOp>(loc, weightIdx, inCIdx);

    Value khOffset = rewriter.create<arith::MulIOp>(loc, kh, kernelWIdx);
    khOffset = rewriter.create<arith::MulIOp>(loc, khOffset, inCIdx);
    weightIdx = rewriter.create<arith::AddIOp>(loc, weightIdx, khOffset);

    Value kwOffset = rewriter.create<arith::MulIOp>(loc, kw, inCIdx);
    weightIdx = rewriter.create<arith::AddIOp>(loc, weightIdx, kwOffset);
    weightIdx = rewriter.create<arith::AddIOp>(loc, weightIdx, ic);

    Value weightVal = rewriter.create<memref::LoadOp>(loc, weights, ValueRange{weightIdx});

    // Compute product and add to accumulator
    Value product;
    if (elemType.isa<FloatType>()) {
      product = rewriter.create<arith::MulFOp>(loc, inputVal, weightVal);
    } else {
      product = rewriter.create<arith::MulIOp>(loc, inputVal, weightVal);
    }

    Value newAcc;
    if (elemType.isa<FloatType>()) {
      newAcc = rewriter.create<arith::AddFOp>(loc, acc, product);
    } else {
      newAcc = rewriter.create<arith::AddIOp>(loc, acc, product);
    }
    rewriter.create<scf::YieldOp>(loc, newAcc);

    // Else region: return accumulator unchanged
    rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
    rewriter.create<scf::YieldOp>(loc, acc);

    // Yield accumulator from ic loop
    rewriter.setInsertionPointAfter(ifOp);
    rewriter.create<scf::YieldOp>(loc, ifOp.getResult(0));

    // Yield from kw loop
    rewriter.setInsertionPointAfter(icLoop);
    rewriter.create<scf::YieldOp>(loc, icLoop.getResult(0));

    // Yield from kh loop
    rewriter.setInsertionPointAfter(kwLoop);
    rewriter.create<scf::YieldOp>(loc, kwLoop.getResult(0));

    // Store final result to output[b, oh, ow, oc]
    rewriter.setInsertionPointAfter(khLoop);
    Value finalAcc = khLoop.getResult(0);

    // Flatten output index: output[b, oh, ow, oc] -> output[b*out_h*out_w*out_c + oh*out_w*out_c + ow*out_c + oc]
    Value outputIdx = rewriter.create<arith::MulIOp>(loc, b, outHIdx);
    outputIdx = rewriter.create<arith::MulIOp>(loc, outputIdx, outWIdx);
    outputIdx = rewriter.create<arith::MulIOp>(loc, outputIdx, outCIdx);

    Value ohOffset = rewriter.create<arith::MulIOp>(loc, oh, outWIdx);
    ohOffset = rewriter.create<arith::MulIOp>(loc, ohOffset, outCIdx);
    outputIdx = rewriter.create<arith::AddIOp>(loc, outputIdx, ohOffset);

    Value owOffset = rewriter.create<arith::MulIOp>(loc, ow, outCIdx);
    outputIdx = rewriter.create<arith::AddIOp>(loc, outputIdx, owOffset);
    outputIdx = rewriter.create<arith::AddIOp>(loc, outputIdx, oc);

    rewriter.create<memref::StoreOp>(loc, finalAcc, output, ValueRange{outputIdx});

    // Return to module level and replace original op with output buffer
    rewriter.setInsertionPointAfter(batchLoop);
    rewriter.replaceOp(op, output);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertSimpToMemRefPass
    : public PassWrapper<ConvertSimpToMemRefPass, OperationPass<ModuleOp>> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
    registry.insert<arith::ArithmeticDialect>();
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();

    // Set up type converter
    SimpTypeConverter typeConverter;

    // Set up conversion target
    ConversionTarget target(getContext());
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<arith::ArithmeticDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalOp<ModuleOp>();

    // FuncOp is legal only if its signature has been converted (no simp types)
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      return typeConverter.isSignatureLegal(op.getType());
    });

    // Allow unrealized_conversion_cast (will be cleaned up by later passes)
    target.addLegalOp<UnrealizedConversionCastOp>();

    // SCF operations are legal only if their operands/results use legal types
    target.addDynamicallyLegalDialect<scf::SCFDialect>([&](Operation *op) {
      // Check if all operands and results have legal types
      return typeConverter.isLegal(op);
    });

    // Mark simp dialect as illegal (must be converted)
    target.addIllegalDialect<SimpDialect>();

    // Populate conversion patterns
    RewritePatternSet patterns(&getContext());
    patterns.add<
        FuncOpSignatureConversion,
        ConstantOpLowering,
        ArrayCreateOpLowering,
        ArrayGetOpLowering,
        ArraySetOpLowering,
        AddOpLowering,
        SubOpLowering,
        MulOpLowering,
        DivOpLowering,
        ModOpLowering,
        NegOpLowering,
        MatMulOpLowering,
        Conv2DOpLowering
    >(typeConverter, &getContext());

    // Add SCF structural type conversions to handle scf.while, scf.if, etc. with type changes
    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns, target);

    // Apply conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

namespace mlir {
namespace simp {

/// Create the Simp to MemRef lowering pass
std::unique_ptr<Pass> createConvertSimpToMemRefPass() {
  return std::make_unique<ConvertSimpToMemRefPass>();
}

} // namespace simp
} // namespace mlir
