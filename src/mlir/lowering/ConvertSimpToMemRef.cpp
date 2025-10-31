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
#include "mlir/Dialect/Math/IR/Math.h"
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
#include <optional>

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

    // Convert !simp.tensor<shape x T> to memref<shape x T>
    addConversion([](simp::SimpTensorType type) -> llvm::Optional<Type> {
      return MemRefType::get(type.getShape(), type.getElementType());
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

    // Add source materialization: converts memref back to simp.array when needed
    // This is crucial for loop-carried values that cross function boundaries
    addSourceMaterialization([](OpBuilder &builder, Type resultType,
                                ValueRange inputs, Location loc) -> Value {
      if (inputs.size() == 1)
        return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
      return nullptr;
    });

    // Add target materialization: converts simp.array to memref for results
    // This handles conversions in return values and yields
    addTargetMaterialization([](OpBuilder &builder, Type resultType,
                                ValueRange inputs, Location loc) -> Value {
      if (inputs.size() == 1)
        return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
      return nullptr;
    });
  }
};

//===----------------------------------------------------------------------===//
// Type Promotion Helpers
//===----------------------------------------------------------------------===//

/// Helper function to promote value to target type using standard MLIR conversion ops
static Value promoteType(Value val, Type targetType, OpBuilder& builder, Location loc) {
  Type srcType = val.getType();
  if (srcType == targetType) return val;

  // Float to float (extend or truncate)
  if (srcType.isa<FloatType>() && targetType.isa<FloatType>()) {
    auto srcFloat = srcType.cast<FloatType>();
    auto targetFloat = targetType.cast<FloatType>();
    if (targetFloat.getWidth() > srcFloat.getWidth()) {
      return builder.create<arith::ExtFOp>(loc, targetType, val);
    } else if (targetFloat.getWidth() < srcFloat.getWidth()) {
      return builder.create<arith::TruncFOp>(loc, targetType, val);
    }
  }

  // Int to int (extend or truncate)
  if (srcType.isa<IntegerType>() && targetType.isa<IntegerType>()) {
    auto srcInt = srcType.cast<IntegerType>();
    auto targetInt = targetType.cast<IntegerType>();
    if (srcInt.getWidth() == 1) return val;  // Don't convert bool
    if (targetInt.getWidth() > srcInt.getWidth()) {
      return builder.create<arith::ExtSIOp>(loc, targetType, val);
    } else if (targetInt.getWidth() < srcInt.getWidth()) {
      return builder.create<arith::TruncIOp>(loc, targetType, val);
    }
  }

  // Int to float (C++ style: always promote int to float in mixed arithmetic)
  if (srcType.isa<IntegerType>() && targetType.isa<FloatType>()) {
    return builder.create<arith::SIToFPOp>(loc, targetType, val);
  }

  // Float to int (rarely needed, but support it)
  if (srcType.isa<FloatType>() && targetType.isa<IntegerType>()) {
    return builder.create<arith::FPToSIOp>(loc, targetType, val);
  }

  return val;  // No conversion possible
}

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

    // Ensure value type matches memref element type using type promotion
    auto memrefType = memref.getType().cast<MemRefType>();
    Type expectedType = memrefType.getElementType();
    if (value.getType() != expectedType) {
      value = promoteType(value, expectedType, rewriter, op.getLoc());
    }

    // Create memref.store (mutates the memref)
    rewriter.create<memref::StoreOp>(op.getLoc(), value, memref, ValueRange{index});

    // Array set returns the "updated" array, but in memref semantics,
    // we just return the same memref (since it's mutated in-place)
    rewriter.replaceOp(op, memref);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Tensor Lowering Patterns
//===----------------------------------------------------------------------===//

/// Convert simp.tensor_create to memref.alloc
struct TensorCreateOpLowering : public OpConversionPattern<simp::TensorCreateOp> {
  using OpConversionPattern<simp::TensorCreateOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      simp::TensorCreateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // Get the tensor type
    auto tensorType = op.getType().cast<simp::SimpTensorType>();

    // Convert to memref<shape x T> with static dimensions
    auto memrefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());

    // Create memref.alloc with static shape
    rewriter.replaceOpWithNewOp<memref::AllocOp>(op, memrefType);

    return success();
  }
};

/// Convert simp.tensor_get to memref.load
struct TensorGetOpLowering : public OpConversionPattern<simp::TensorGetOp> {
  using OpConversionPattern<simp::TensorGetOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      simp::TensorGetOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // adaptor.getOperands() returns [tensor, index0, index1, ...]
    auto operands = adaptor.getOperands();
    Value memref = operands[0];  // tensor (now memref)

    // Collect indices and convert to index type if needed
    SmallVector<Value, 4> indices;
    for (size_t i = 1; i < operands.size(); ++i) {
      Value index = operands[i];
      if (!index.getType().isIndex()) {
        index = rewriter.create<arith::IndexCastOp>(
            op.getLoc(), rewriter.getIndexType(), index);
      }
      indices.push_back(index);
    }

    // Replace with memref.load
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, memref, indices);

    return success();
  }
};

/// Convert simp.tensor_set to memref.store
/// Note: This changes semantics from SSA-pure to mutation
struct TensorSetOpLowering : public OpConversionPattern<simp::TensorSetOp> {
  using OpConversionPattern<simp::TensorSetOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      simp::TensorSetOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // adaptor.getOperands() returns [tensor, index0, index1, ..., value]
    auto operands = adaptor.getOperands();
    Value memref = operands[0];  // tensor (now memref)
    Value value = operands[operands.size() - 1];  // last operand is value

    // Collect indices (all operands except first and last)
    SmallVector<Value, 4> indices;
    for (size_t i = 1; i < operands.size() - 1; ++i) {
      Value index = operands[i];
      if (!index.getType().isIndex()) {
        index = rewriter.create<arith::IndexCastOp>(
            op.getLoc(), rewriter.getIndexType(), index);
      }
      indices.push_back(index);
    }

    // Ensure value type matches memref element type using type promotion
    auto memrefType = memref.getType().cast<MemRefType>();
    Type expectedType = memrefType.getElementType();
    if (value.getType() != expectedType) {
      value = promoteType(value, expectedType, rewriter, op.getLoc());
    }

    // Create memref.store (mutates the memref)
    rewriter.create<memref::StoreOp>(op.getLoc(), value, memref, indices);

    // Tensor set returns the "updated" tensor, but in memref semantics,
    // we just return the same memref (since it's mutated in-place)
    rewriter.replaceOp(op, memref);

    return success();
  }
};

/// Convert tensor element-wise binary ops (add, mul, sub, div) to loops
template<typename SimpOp, typename ArithOp>
struct TensorBinaryOpLowering : public OpConversionPattern<SimpOp> {
  using OpConversionPattern<SimpOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<SimpOp>::OpAdaptor;

  LogicalResult matchAndRewrite(
      SimpOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto operands = adaptor.getOperands();
    Value lhs = operands[0];
    Value rhs = operands[1];

    // Get memref type
    auto memrefType = lhs.getType().cast<MemRefType>();
    auto shape = memrefType.getShape();

    // Allocate result memref
    Value result = rewriter.create<memref::AllocOp>(loc, memrefType);

    // Build nested loops for each dimension
    SmallVector<Value, 4> lowerBounds, upperBounds, steps;
    for (int64_t dim : shape) {
      lowerBounds.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
      upperBounds.push_back(rewriter.create<arith::ConstantIndexOp>(loc, dim));
      steps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
    }

    // Create nested scf.for loops (using the void-returning buildLoopNest)
    scf::buildLoopNest(
        rewriter, loc, lowerBounds, upperBounds, steps,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          // Load from lhs and rhs
          Value lhsVal = builder.create<memref::LoadOp>(loc, lhs, ivs);
          Value rhsVal = builder.create<memref::LoadOp>(loc, rhs, ivs);

          // Perform operation
          Value resultVal = builder.create<ArithOp>(loc, lhsVal, rhsVal);

          // Store to result
          builder.create<memref::StoreOp>(loc, resultVal, result, ivs);
        });

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Convert tensor element-wise unary ops (relu, sigmoid, tanh) to loops
template<typename SimpOp>
struct TensorUnaryOpLowering : public OpConversionPattern<SimpOp> {
  using OpConversionPattern<SimpOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<SimpOp>::OpAdaptor;

  enum class UnaryOpKind { ReLU, Sigmoid, Tanh };
  UnaryOpKind kind;

  TensorUnaryOpLowering(TypeConverter &converter, MLIRContext *context, UnaryOpKind k)
      : OpConversionPattern<SimpOp>(converter, context), kind(k) {}

  LogicalResult matchAndRewrite(
      SimpOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    Value input = adaptor.getOperands()[0];

    // Get memref type
    auto memrefType = input.getType().cast<MemRefType>();
    auto shape = memrefType.getShape();
    auto elemType = memrefType.getElementType();

    // Allocate result memref
    Value result = rewriter.create<memref::AllocOp>(loc, memrefType);

    // Build nested loops
    SmallVector<Value, 4> lowerBounds, upperBounds, steps;
    for (int64_t dim : shape) {
      lowerBounds.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
      upperBounds.push_back(rewriter.create<arith::ConstantIndexOp>(loc, dim));
      steps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
    }

    // Create nested scf.for loops (using the void-returning buildLoopNest)
    scf::buildLoopNest(
        rewriter, loc, lowerBounds, upperBounds, steps,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          Value inputVal = builder.create<memref::LoadOp>(loc, input, ivs);
          Value resultVal;

          switch (kind) {
            case UnaryOpKind::ReLU: {
              // ReLU: max(0, x)
              Value zero = builder.create<arith::ConstantOp>(
                  loc, elemType, builder.getZeroAttr(elemType));
              Value cmp = builder.create<arith::CmpFOp>(
                  loc, arith::CmpFPredicate::OGT, inputVal, zero);
              resultVal = builder.create<SelectOp>(loc, cmp, inputVal, zero);
              break;
            }
            case UnaryOpKind::Sigmoid: {
              // sigmoid(x) = 1 / (1 + exp(-x))
              Value negX = builder.create<arith::NegFOp>(loc, inputVal);
              Value expNegX = builder.create<math::ExpOp>(loc, negX);
              Value one = builder.create<arith::ConstantOp>(
                  loc, elemType, builder.getFloatAttr(elemType, 1.0));
              Value denom = builder.create<arith::AddFOp>(loc, one, expNegX);
              resultVal = builder.create<arith::DivFOp>(loc, one, denom);
              break;
            }
            case UnaryOpKind::Tanh: {
              // Use math.tanh operation
              resultVal = builder.create<math::TanhOp>(loc, inputVal);
              break;
            }
          }

          builder.create<memref::StoreOp>(loc, resultVal, result, ivs);
        });

    rewriter.replaceOp(op, result);
    return success();
  }
};

// Type aliases for specific operations
using TensorAddOpLowering = TensorBinaryOpLowering<simp::TensorAddOp, arith::AddFOp>;
using TensorMulOpLowering = TensorBinaryOpLowering<simp::TensorMulOp, arith::MulFOp>;
using TensorSubOpLowering = TensorBinaryOpLowering<simp::TensorSubOp, arith::SubFOp>;
using TensorDivOpLowering = TensorBinaryOpLowering<simp::TensorDivOp, arith::DivFOp>;

//===----------------------------------------------------------------------===//
// Tensor Reduction Operations Lowering
//===----------------------------------------------------------------------===//

/// Convert tensor.sum to loops with accumulator (supports full and axis reductions)
struct TensorSumOpLowering : public OpConversionPattern<simp::TensorSumOp> {
  using OpConversionPattern<simp::TensorSumOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      simp::TensorSumOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    Value input = adaptor.getOperands()[0];
    auto memrefType = input.getType().cast<MemRefType>();
    auto shape = memrefType.getShape();
    Type elemType = memrefType.getElementType();
    int64_t rank = shape.size();

    // Initialize zero value
    Value zero;
    if (elemType.isa<FloatType>()) {
      zero = rewriter.create<arith::ConstantOp>(loc, elemType,
                 rewriter.getFloatAttr(elemType, 0.0));
    } else if (elemType.isa<IntegerType>()) {
      zero = rewriter.create<arith::ConstantOp>(loc, elemType,
                 rewriter.getIntegerAttr(elemType, 0));
    } else {
      return op.emitError("Unsupported element type for tensor_sum");
    }

    // CASE 1: Full reduction (no axis specified)
    if (!op.axis()) {
      // Allocate scalar accumulator on stack
      auto accumulatorType = MemRefType::get({}, elemType);
      Value accumulator = rewriter.create<memref::AllocaOp>(loc, accumulatorType);
      rewriter.create<memref::StoreOp>(loc, zero, accumulator, ValueRange{});

      // Build nested loops over all dimensions
      SmallVector<Value, 4> lowerBounds, upperBounds, steps;
      for (int64_t dim : shape) {
        lowerBounds.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
        upperBounds.push_back(rewriter.create<arith::ConstantIndexOp>(loc, dim));
        steps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
      }

      scf::buildLoopNest(rewriter, loc, lowerBounds, upperBounds, steps,
          [&](OpBuilder &builder, Location loc, ValueRange ivs) {
            Value elem = builder.create<memref::LoadOp>(loc, input, ivs);
            Value currentSum = builder.create<memref::LoadOp>(loc, accumulator, ValueRange{});
            Value newSum;
            if (elemType.isa<FloatType>()) {
              newSum = builder.create<arith::AddFOp>(loc, currentSum, elem);
            } else {
              newSum = builder.create<arith::AddIOp>(loc, currentSum, elem);
            }
            builder.create<memref::StoreOp>(loc, newSum, accumulator, ValueRange{});
          });

      Value result = rewriter.create<memref::LoadOp>(loc, accumulator, ValueRange{});
      rewriter.replaceOp(op, result);
      return success();
    }

    // CASE 2: Axis reduction
    // Extract axis value from operand (handle both arith.constant and simp.constant)
    Value axisOperand = op.axis();
    int64_t axis;

    if (auto arithConstOp = axisOperand.getDefiningOp<arith::ConstantOp>()) {
      auto axisIntAttr = arithConstOp.getValue().dyn_cast<IntegerAttr>();
      if (!axisIntAttr) {
        return op.emitError("Axis must be an integer");
      }
      axis = axisIntAttr.getInt();
    } else if (auto simpConstOp = axisOperand.getDefiningOp<simp::ConstantOp>()) {
      auto axisIntAttr = simpConstOp.value().dyn_cast<IntegerAttr>();
      if (!axisIntAttr) {
        return op.emitError("Axis must be an integer");
      }
      axis = axisIntAttr.getInt();
    } else {
      return op.emitError("Axis must be a constant integer");
    }

    // Handle negative axis
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) {
      return op.emitError("Axis out of bounds");
    }

    // Compute result shape (remove axis dimension)
    SmallVector<int64_t, 4> resultShape;
    for (int64_t i = 0; i < rank; i++) {
      if (i != axis) {
        resultShape.push_back(shape[i]);
      }
    }

    // Allocate result tensor (use heap allocation for non-scalar results)
    auto resultType = MemRefType::get(resultShape, elemType);
    Value result = rewriter.create<memref::AllocOp>(loc, resultType);

    // Initialize result tensor to zero
    SmallVector<Value, 4> outerLBs, outerUBs, outerSteps;
    for (int64_t dim : resultShape) {
      outerLBs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
      outerUBs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, dim));
      outerSteps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
    }

    scf::buildLoopNest(rewriter, loc, outerLBs, outerUBs, outerSteps,
        [&](OpBuilder &builder, Location loc, ValueRange outerIvs) {
          builder.create<memref::StoreOp>(loc, zero, result, outerIvs);
        });

    // Build reduction loops
    scf::buildLoopNest(rewriter, loc, outerLBs, outerUBs, outerSteps,
        [&](OpBuilder &builder, Location loc, ValueRange outerIvs) {
          // Inner loop over reduction axis
          Value lowerBound = builder.create<arith::ConstantIndexOp>(loc, 0);
          Value upperBound = builder.create<arith::ConstantIndexOp>(loc, shape[axis]);
          Value step = builder.create<arith::ConstantIndexOp>(loc, 1);

          builder.create<scf::ForOp>(loc, lowerBound, upperBound, step, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value iv, ValueRange args) {
                // Construct full input indices
                SmallVector<Value, 4> inputIndices;
                int64_t outerIdx = 0;
                for (int64_t i = 0; i < rank; i++) {
                  if (i == axis) {
                    inputIndices.push_back(iv);
                  } else {
                    inputIndices.push_back(outerIvs[outerIdx++]);
                  }
                }

                // Load input element and current sum
                Value elem = builder.create<memref::LoadOp>(loc, input, inputIndices);
                Value currentSum = builder.create<memref::LoadOp>(loc, result, outerIvs);

                // Accumulate
                Value newSum;
                if (elemType.isa<FloatType>()) {
                  newSum = builder.create<arith::AddFOp>(loc, currentSum, elem);
                } else {
                  newSum = builder.create<arith::AddIOp>(loc, currentSum, elem);
                }

                builder.create<memref::StoreOp>(loc, newSum, result, outerIvs);
                builder.create<scf::YieldOp>(loc);
              });
        });

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Convert tensor.mean to loops with accumulator + division (supports full and axis reductions)
struct TensorMeanOpLowering : public OpConversionPattern<simp::TensorMeanOp> {
  using OpConversionPattern<simp::TensorMeanOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      simp::TensorMeanOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    Value input = adaptor.getOperands()[0];
    auto memrefType = input.getType().cast<MemRefType>();
    auto shape = memrefType.getShape();
    Type elemType = memrefType.getElementType();
    int64_t rank = shape.size();

    if (!elemType.isa<FloatType>()) {
      return op.emitError("Mean only supported for float types");
    }

    Value zero = rewriter.create<arith::ConstantOp>(loc, elemType,
                   rewriter.getFloatAttr(elemType, 0.0));

    // CASE 1: Full reduction
    if (!op.axis()) {
      int64_t totalElements = 1;
      for (int64_t dim : shape) {
        totalElements *= dim;
      }

      auto accumulatorType = MemRefType::get({}, elemType);
      Value accumulator = rewriter.create<memref::AllocaOp>(loc, accumulatorType);
      rewriter.create<memref::StoreOp>(loc, zero, accumulator, ValueRange{});

      SmallVector<Value, 4> lowerBounds, upperBounds, steps;
      for (int64_t dim : shape) {
        lowerBounds.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
        upperBounds.push_back(rewriter.create<arith::ConstantIndexOp>(loc, dim));
        steps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
      }

      scf::buildLoopNest(rewriter, loc, lowerBounds, upperBounds, steps,
          [&](OpBuilder &builder, Location loc, ValueRange ivs) {
            Value elem = builder.create<memref::LoadOp>(loc, input, ivs);
            Value currentSum = builder.create<memref::LoadOp>(loc, accumulator, ValueRange{});
            Value newSum = builder.create<arith::AddFOp>(loc, currentSum, elem);
            builder.create<memref::StoreOp>(loc, newSum, accumulator, ValueRange{});
          });

      Value sum = rewriter.create<memref::LoadOp>(loc, accumulator, ValueRange{});
      Value count = rewriter.create<arith::ConstantOp>(loc, elemType,
                       rewriter.getFloatAttr(elemType, (double)totalElements));
      Value mean = rewriter.create<arith::DivFOp>(loc, sum, count);

      rewriter.replaceOp(op, mean);
      return success();
    }

    // CASE 2: Axis reduction
    Value axisOperand = op.axis();
    int64_t axis;
    if (auto arithConstOp = axisOperand.getDefiningOp<arith::ConstantOp>()) {
      auto axisIntAttr = arithConstOp.getValue().dyn_cast<IntegerAttr>();
      if (!axisIntAttr) {
        return op.emitError("Axis must be an integer");
      }
      axis = axisIntAttr.getInt();
    } else if (auto simpConstOp = axisOperand.getDefiningOp<simp::ConstantOp>()) {
      auto axisIntAttr = simpConstOp.value().dyn_cast<IntegerAttr>();
      if (!axisIntAttr) {
        return op.emitError("Axis must be an integer");
      }
      axis = axisIntAttr.getInt();
    } else {
      return op.emitError("Axis must be a constant integer");
    }
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) {
      return op.emitError("Axis out of bounds");
    }

    // Compute result shape and reduction count
    SmallVector<int64_t, 4> resultShape;
    for (int64_t i = 0; i < rank; i++) {
      if (i != axis) {
        resultShape.push_back(shape[i]);
      }
    }
    int64_t reductionCount = shape[axis];

    // Allocate result tensor and initialize to zero (use heap allocation for non-scalar results)
    auto resultType = MemRefType::get(resultShape, elemType);
    Value result = rewriter.create<memref::AllocOp>(loc, resultType);

    SmallVector<Value, 4> outerLBs, outerUBs, outerSteps;
    for (int64_t dim : resultShape) {
      outerLBs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
      outerUBs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, dim));
      outerSteps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
    }

    scf::buildLoopNest(rewriter, loc, outerLBs, outerUBs, outerSteps,
        [&](OpBuilder &builder, Location loc, ValueRange outerIvs) {
          builder.create<memref::StoreOp>(loc, zero, result, outerIvs);
        });

    // Build reduction loops (sum first)
    scf::buildLoopNest(rewriter, loc, outerLBs, outerUBs, outerSteps,
        [&](OpBuilder &builder, Location loc, ValueRange outerIvs) {
          Value lowerBound = builder.create<arith::ConstantIndexOp>(loc, 0);
          Value upperBound = builder.create<arith::ConstantIndexOp>(loc, shape[axis]);
          Value step = builder.create<arith::ConstantIndexOp>(loc, 1);

          builder.create<scf::ForOp>(loc, lowerBound, upperBound, step, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value iv, ValueRange args) {
                SmallVector<Value, 4> inputIndices;
                int64_t outerIdx = 0;
                for (int64_t i = 0; i < rank; i++) {
                  if (i == axis) {
                    inputIndices.push_back(iv);
                  } else {
                    inputIndices.push_back(outerIvs[outerIdx++]);
                  }
                }

                Value elem = builder.create<memref::LoadOp>(loc, input, inputIndices);
                Value currentSum = builder.create<memref::LoadOp>(loc, result, outerIvs);
                Value newSum = builder.create<arith::AddFOp>(loc, currentSum, elem);
                builder.create<memref::StoreOp>(loc, newSum, result, outerIvs);
                builder.create<scf::YieldOp>(loc);
              });
        });

    // Divide by count to get mean
    Value count = rewriter.create<arith::ConstantOp>(loc, elemType,
                     rewriter.getFloatAttr(elemType, (double)reductionCount));

    scf::buildLoopNest(rewriter, loc, outerLBs, outerUBs, outerSteps,
        [&](OpBuilder &builder, Location loc, ValueRange outerIvs) {
          Value sum = builder.create<memref::LoadOp>(loc, result, outerIvs);
          Value mean = builder.create<arith::DivFOp>(loc, sum, count);
          builder.create<memref::StoreOp>(loc, mean, result, outerIvs);
        });

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Convert tensor.max to loops with max accumulator (supports full and axis reductions)
struct TensorMaxOpLowering : public OpConversionPattern<simp::TensorMaxOp> {
  using OpConversionPattern<simp::TensorMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      simp::TensorMaxOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    Value input = adaptor.getOperands()[0];
    auto memrefType = input.getType().cast<MemRefType>();
    auto shape = memrefType.getShape();
    Type elemType = memrefType.getElementType();
    int64_t rank = shape.size();

    // Initialize to negative infinity (for floats) or minimum value (for ints)
    Value initialValue;
    if (auto floatType = elemType.dyn_cast<FloatType>()) {
      initialValue = rewriter.create<arith::ConstantOp>(loc, elemType,
          rewriter.getFloatAttr(elemType,
              -std::numeric_limits<double>::infinity()));
    } else if (auto intType = elemType.dyn_cast<IntegerType>()) {
      int64_t minVal = intType.isUnsigned() ? 0 :
          -(1LL << (intType.getWidth() - 1));
      initialValue = rewriter.create<arith::ConstantOp>(loc, elemType,
          rewriter.getIntegerAttr(elemType, minVal));
    } else {
      return op.emitError("Unsupported element type for tensor_max");
    }

    // CASE 1: Full reduction
    if (!op.axis()) {
      auto accumulatorType = MemRefType::get({}, elemType);
      Value accumulator = rewriter.create<memref::AllocaOp>(loc, accumulatorType);
      rewriter.create<memref::StoreOp>(loc, initialValue, accumulator, ValueRange{});

      SmallVector<Value, 4> lowerBounds, upperBounds, steps;
      for (int64_t dim : shape) {
        lowerBounds.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
        upperBounds.push_back(rewriter.create<arith::ConstantIndexOp>(loc, dim));
        steps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
      }

      scf::buildLoopNest(rewriter, loc, lowerBounds, upperBounds, steps,
          [&](OpBuilder &builder, Location loc, ValueRange ivs) {
            Value elem = builder.create<memref::LoadOp>(loc, input, ivs);
            Value currentMax = builder.create<memref::LoadOp>(loc, accumulator, ValueRange{});
            Value newMax;
            if (elemType.isa<FloatType>()) {
              newMax = builder.create<arith::MaxFOp>(loc, currentMax, elem);
            } else {
              auto intType = elemType.cast<IntegerType>();
              if (intType.isUnsigned()) {
                newMax = builder.create<arith::MaxUIOp>(loc, currentMax, elem);
              } else {
                newMax = builder.create<arith::MaxSIOp>(loc, currentMax, elem);
              }
            }
            builder.create<memref::StoreOp>(loc, newMax, accumulator, ValueRange{});
          });

      Value result = rewriter.create<memref::LoadOp>(loc, accumulator, ValueRange{});
      rewriter.replaceOp(op, result);
      return success();
    }

    // CASE 2: Axis reduction
    Value axisOperand = op.axis();
    int64_t axis;
    if (auto arithConstOp = axisOperand.getDefiningOp<arith::ConstantOp>()) {
      auto axisIntAttr = arithConstOp.getValue().dyn_cast<IntegerAttr>();
      if (!axisIntAttr) {
        return op.emitError("Axis must be an integer");
      }
      axis = axisIntAttr.getInt();
    } else if (auto simpConstOp = axisOperand.getDefiningOp<simp::ConstantOp>()) {
      auto axisIntAttr = simpConstOp.value().dyn_cast<IntegerAttr>();
      if (!axisIntAttr) {
        return op.emitError("Axis must be an integer");
      }
      axis = axisIntAttr.getInt();
    } else {
      return op.emitError("Axis must be a constant integer");
    }
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) {
      return op.emitError("Axis out of bounds");
    }

    SmallVector<int64_t, 4> resultShape;
    for (int64_t i = 0; i < rank; i++) {
      if (i != axis) {
        resultShape.push_back(shape[i]);
      }
    }

    auto resultType = MemRefType::get(resultShape, elemType);
    Value result = rewriter.create<memref::AllocOp>(loc, resultType);

    SmallVector<Value, 4> outerLBs, outerUBs, outerSteps;
    for (int64_t dim : resultShape) {
      outerLBs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
      outerUBs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, dim));
      outerSteps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
    }

    // Initialize result tensor to minimum value
    scf::buildLoopNest(rewriter, loc, outerLBs, outerUBs, outerSteps,
        [&](OpBuilder &builder, Location loc, ValueRange outerIvs) {
          builder.create<memref::StoreOp>(loc, initialValue, result, outerIvs);
        });

    // Build reduction loops
    scf::buildLoopNest(rewriter, loc, outerLBs, outerUBs, outerSteps,
        [&](OpBuilder &builder, Location loc, ValueRange outerIvs) {
          Value lowerBound = builder.create<arith::ConstantIndexOp>(loc, 0);
          Value upperBound = builder.create<arith::ConstantIndexOp>(loc, shape[axis]);
          Value step = builder.create<arith::ConstantIndexOp>(loc, 1);

          builder.create<scf::ForOp>(loc, lowerBound, upperBound, step, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value iv, ValueRange args) {
                SmallVector<Value, 4> inputIndices;
                int64_t outerIdx = 0;
                for (int64_t i = 0; i < rank; i++) {
                  if (i == axis) {
                    inputIndices.push_back(iv);
                  } else {
                    inputIndices.push_back(outerIvs[outerIdx++]);
                  }
                }

                Value elem = builder.create<memref::LoadOp>(loc, input, inputIndices);
                Value currentMax = builder.create<memref::LoadOp>(loc, result, outerIvs);

                Value newMax;
                if (elemType.isa<FloatType>()) {
                  newMax = builder.create<arith::MaxFOp>(loc, currentMax, elem);
                } else {
                  auto intType = elemType.cast<IntegerType>();
                  if (intType.isUnsigned()) {
                    newMax = builder.create<arith::MaxUIOp>(loc, currentMax, elem);
                  } else {
                    newMax = builder.create<arith::MaxSIOp>(loc, currentMax, elem);
                  }
                }

                builder.create<memref::StoreOp>(loc, newMax, result, outerIvs);
                builder.create<scf::YieldOp>(loc);
              });
        });

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Convert tensor.min to loops with min accumulator (supports full and axis reductions)
struct TensorMinOpLowering : public OpConversionPattern<simp::TensorMinOp> {
  using OpConversionPattern<simp::TensorMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      simp::TensorMinOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    Value input = adaptor.getOperands()[0];
    auto memrefType = input.getType().cast<MemRefType>();
    auto shape = memrefType.getShape();
    Type elemType = memrefType.getElementType();
    int64_t rank = shape.size();

    // Initialize to positive infinity (for floats) or maximum value (for ints)
    Value initialValue;
    if (auto floatType = elemType.dyn_cast<FloatType>()) {
      initialValue = rewriter.create<arith::ConstantOp>(loc, elemType,
          rewriter.getFloatAttr(elemType,
              std::numeric_limits<double>::infinity()));
    } else if (auto intType = elemType.dyn_cast<IntegerType>()) {
      int64_t maxVal = intType.isUnsigned() ?
          ((1ULL << intType.getWidth()) - 1) :
          ((1LL << (intType.getWidth() - 1)) - 1);
      initialValue = rewriter.create<arith::ConstantOp>(loc, elemType,
          rewriter.getIntegerAttr(elemType, maxVal));
    } else {
      return op.emitError("Unsupported element type for tensor_min");
    }

    // CASE 1: Full reduction
    if (!op.axis()) {
      auto accumulatorType = MemRefType::get({}, elemType);
      Value accumulator = rewriter.create<memref::AllocaOp>(loc, accumulatorType);
      rewriter.create<memref::StoreOp>(loc, initialValue, accumulator, ValueRange{});

      SmallVector<Value, 4> lowerBounds, upperBounds, steps;
      for (int64_t dim : shape) {
        lowerBounds.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
        upperBounds.push_back(rewriter.create<arith::ConstantIndexOp>(loc, dim));
        steps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
      }

      scf::buildLoopNest(rewriter, loc, lowerBounds, upperBounds, steps,
          [&](OpBuilder &builder, Location loc, ValueRange ivs) {
            Value elem = builder.create<memref::LoadOp>(loc, input, ivs);
            Value currentMin = builder.create<memref::LoadOp>(loc, accumulator, ValueRange{});
            Value newMin;
            if (elemType.isa<FloatType>()) {
              newMin = builder.create<arith::MinFOp>(loc, currentMin, elem);
            } else {
              auto intType = elemType.cast<IntegerType>();
              if (intType.isUnsigned()) {
                newMin = builder.create<arith::MinUIOp>(loc, currentMin, elem);
              } else {
                newMin = builder.create<arith::MinSIOp>(loc, currentMin, elem);
              }
            }
            builder.create<memref::StoreOp>(loc, newMin, accumulator, ValueRange{});
          });

      Value result = rewriter.create<memref::LoadOp>(loc, accumulator, ValueRange{});
      rewriter.replaceOp(op, result);
      return success();
    }

    // CASE 2: Axis reduction
    Value axisOperand = op.axis();
    int64_t axis;
    if (auto arithConstOp = axisOperand.getDefiningOp<arith::ConstantOp>()) {
      auto axisIntAttr = arithConstOp.getValue().dyn_cast<IntegerAttr>();
      if (!axisIntAttr) {
        return op.emitError("Axis must be an integer");
      }
      axis = axisIntAttr.getInt();
    } else if (auto simpConstOp = axisOperand.getDefiningOp<simp::ConstantOp>()) {
      auto axisIntAttr = simpConstOp.value().dyn_cast<IntegerAttr>();
      if (!axisIntAttr) {
        return op.emitError("Axis must be an integer");
      }
      axis = axisIntAttr.getInt();
    } else {
      return op.emitError("Axis must be a constant integer");
    }
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) {
      return op.emitError("Axis out of bounds");
    }

    SmallVector<int64_t, 4> resultShape;
    for (int64_t i = 0; i < rank; i++) {
      if (i != axis) {
        resultShape.push_back(shape[i]);
      }
    }

    auto resultType = MemRefType::get(resultShape, elemType);
    Value result = rewriter.create<memref::AllocOp>(loc, resultType);

    SmallVector<Value, 4> outerLBs, outerUBs, outerSteps;
    for (int64_t dim : resultShape) {
      outerLBs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
      outerUBs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, dim));
      outerSteps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
    }

    // Initialize result tensor to maximum value
    scf::buildLoopNest(rewriter, loc, outerLBs, outerUBs, outerSteps,
        [&](OpBuilder &builder, Location loc, ValueRange outerIvs) {
          builder.create<memref::StoreOp>(loc, initialValue, result, outerIvs);
        });

    // Build reduction loops
    scf::buildLoopNest(rewriter, loc, outerLBs, outerUBs, outerSteps,
        [&](OpBuilder &builder, Location loc, ValueRange outerIvs) {
          Value lowerBound = builder.create<arith::ConstantIndexOp>(loc, 0);
          Value upperBound = builder.create<arith::ConstantIndexOp>(loc, shape[axis]);
          Value step = builder.create<arith::ConstantIndexOp>(loc, 1);

          builder.create<scf::ForOp>(loc, lowerBound, upperBound, step, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value iv, ValueRange args) {
                SmallVector<Value, 4> inputIndices;
                int64_t outerIdx = 0;
                for (int64_t i = 0; i < rank; i++) {
                  if (i == axis) {
                    inputIndices.push_back(iv);
                  } else {
                    inputIndices.push_back(outerIvs[outerIdx++]);
                  }
                }

                Value elem = builder.create<memref::LoadOp>(loc, input, inputIndices);
                Value currentMin = builder.create<memref::LoadOp>(loc, result, outerIvs);

                Value newMin;
                if (elemType.isa<FloatType>()) {
                  newMin = builder.create<arith::MinFOp>(loc, currentMin, elem);
                } else {
                  auto intType = elemType.cast<IntegerType>();
                  if (intType.isUnsigned()) {
                    newMin = builder.create<arith::MinUIOp>(loc, currentMin, elem);
                  } else {
                    newMin = builder.create<arith::MinSIOp>(loc, currentMin, elem);
                  }
                }

                builder.create<memref::StoreOp>(loc, newMin, result, outerIvs);
                builder.create<scf::YieldOp>(loc);
              });
        });

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Convert tensor.argmax to loops tracking both max value and index (supports full and axis reductions)
struct TensorArgmaxOpLowering : public OpConversionPattern<simp::TensorArgmaxOp> {
  using OpConversionPattern<simp::TensorArgmaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      simp::TensorArgmaxOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    Value input = adaptor.getOperands()[0];
    auto memrefType = input.getType().cast<MemRefType>();
    auto shape = memrefType.getShape();
    Type elemType = memrefType.getElementType();
    int64_t rank = shape.size();

    // Initialize to negative infinity / min int
    Value initialValue;
    if (auto floatType = elemType.dyn_cast<FloatType>()) {
      initialValue = rewriter.create<arith::ConstantOp>(loc, elemType,
          rewriter.getFloatAttr(elemType,
              -std::numeric_limits<double>::infinity()));
    } else if (auto intType = elemType.dyn_cast<IntegerType>()) {
      int64_t minVal = intType.isUnsigned() ? 0 :
          -(1LL << (intType.getWidth() - 1));
      initialValue = rewriter.create<arith::ConstantOp>(loc, elemType,
          rewriter.getIntegerAttr(elemType, minVal));
    } else {
      return op.emitError("Unsupported element type for tensor_argmax");
    }

    Value zeroIndex = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(0));

    // CASE 1: Full reduction (return linear index)
    if (!op.axis()) {
      auto maxValueType = MemRefType::get({}, elemType);
      auto maxIndexType = MemRefType::get({}, rewriter.getI64Type());
      Value maxValueAccum = rewriter.create<memref::AllocaOp>(loc, maxValueType);
      Value maxIndexAccum = rewriter.create<memref::AllocaOp>(loc, maxIndexType);

      rewriter.create<memref::StoreOp>(loc, initialValue, maxValueAccum, ValueRange{});
      rewriter.create<memref::StoreOp>(loc, zeroIndex, maxIndexAccum, ValueRange{});

      auto linearIndexType = MemRefType::get({}, rewriter.getI64Type());
      Value linearIndexAccum = rewriter.create<memref::AllocaOp>(loc, linearIndexType);
      rewriter.create<memref::StoreOp>(loc, zeroIndex, linearIndexAccum, ValueRange{});

      SmallVector<Value, 4> lowerBounds, upperBounds, steps;
      for (int64_t dim : shape) {
        lowerBounds.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
        upperBounds.push_back(rewriter.create<arith::ConstantIndexOp>(loc, dim));
        steps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
      }

      scf::buildLoopNest(rewriter, loc, lowerBounds, upperBounds, steps,
          [&](OpBuilder &builder, Location loc, ValueRange ivs) {
            Value elem = builder.create<memref::LoadOp>(loc, input, ivs);
            Value currentMax = builder.create<memref::LoadOp>(loc, maxValueAccum, ValueRange{});
            Value currentLinearIdx = builder.create<memref::LoadOp>(loc, linearIndexAccum, ValueRange{});

            Value isGreater;
            if (elemType.isa<FloatType>()) {
              isGreater = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, elem, currentMax);
            } else {
              auto intType = elemType.cast<IntegerType>();
              if (intType.isUnsigned()) {
                isGreater = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, elem, currentMax);
              } else {
                isGreater = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, elem, currentMax);
              }
            }

            Value newMaxValue = builder.create<SelectOp>(loc, isGreater, elem, currentMax);
            Value newMaxIndex = builder.create<SelectOp>(loc, isGreater, currentLinearIdx,
                builder.create<memref::LoadOp>(loc, maxIndexAccum, ValueRange{}));

            builder.create<memref::StoreOp>(loc, newMaxValue, maxValueAccum, ValueRange{});
            builder.create<memref::StoreOp>(loc, newMaxIndex, maxIndexAccum, ValueRange{});

            Value one = builder.create<arith::ConstantOp>(loc, builder.getI64Type(),
                builder.getI64IntegerAttr(1));
            Value nextLinearIdx = builder.create<arith::AddIOp>(loc, currentLinearIdx, one);
            builder.create<memref::StoreOp>(loc, nextLinearIdx, linearIndexAccum, ValueRange{});
          });

      Value resultIndex = rewriter.create<memref::LoadOp>(loc, maxIndexAccum, ValueRange{});
      rewriter.replaceOp(op, resultIndex);
      return success();
    }

    // CASE 2: Axis reduction (return tensor of indices along reduction axis)
    Value axisOperand = op.axis();
    int64_t axis;
    if (auto arithConstOp = axisOperand.getDefiningOp<arith::ConstantOp>()) {
      auto axisIntAttr = arithConstOp.getValue().dyn_cast<IntegerAttr>();
      if (!axisIntAttr) {
        return op.emitError("Axis must be an integer");
      }
      axis = axisIntAttr.getInt();
    } else if (auto simpConstOp = axisOperand.getDefiningOp<simp::ConstantOp>()) {
      auto axisIntAttr = simpConstOp.value().dyn_cast<IntegerAttr>();
      if (!axisIntAttr) {
        return op.emitError("Axis must be an integer");
      }
      axis = axisIntAttr.getInt();
    } else {
      return op.emitError("Axis must be a constant integer");
    }
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) {
      return op.emitError("Axis out of bounds");
    }

    SmallVector<int64_t, 4> resultShape;
    for (int64_t i = 0; i < rank; i++) {
      if (i != axis) {
        resultShape.push_back(shape[i]);
      }
    }

    // Allocate result tensor (indices) and temp tensor (max values)
    auto resultType = MemRefType::get(resultShape, rewriter.getI64Type());
    auto maxValuesType = MemRefType::get(resultShape, elemType);
    Value result = rewriter.create<memref::AllocOp>(loc, resultType);
    Value maxValues = rewriter.create<memref::AllocOp>(loc, maxValuesType);

    SmallVector<Value, 4> outerLBs, outerUBs, outerSteps;
    for (int64_t dim : resultShape) {
      outerLBs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
      outerUBs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, dim));
      outerSteps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
    }

    // Initialize result indices to 0 and max values to -inf
    scf::buildLoopNest(rewriter, loc, outerLBs, outerUBs, outerSteps,
        [&](OpBuilder &builder, Location loc, ValueRange outerIvs) {
          builder.create<memref::StoreOp>(loc, zeroIndex, result, outerIvs);
          builder.create<memref::StoreOp>(loc, initialValue, maxValues, outerIvs);
        });

    // Build reduction loops
    scf::buildLoopNest(rewriter, loc, outerLBs, outerUBs, outerSteps,
        [&](OpBuilder &builder, Location loc, ValueRange outerIvs) {
          Value lowerBound = builder.create<arith::ConstantIndexOp>(loc, 0);
          Value upperBound = builder.create<arith::ConstantIndexOp>(loc, shape[axis]);
          Value step = builder.create<arith::ConstantIndexOp>(loc, 1);

          builder.create<scf::ForOp>(loc, lowerBound, upperBound, step, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value iv, ValueRange args) {
                SmallVector<Value, 4> inputIndices;
                int64_t outerIdx = 0;
                for (int64_t i = 0; i < rank; i++) {
                  if (i == axis) {
                    inputIndices.push_back(iv);
                  } else {
                    inputIndices.push_back(outerIvs[outerIdx++]);
                  }
                }

                Value elem = builder.create<memref::LoadOp>(loc, input, inputIndices);
                Value currentMax = builder.create<memref::LoadOp>(loc, maxValues, outerIvs);

                // Compare: is elem > currentMax?
                Value isGreater;
                if (elemType.isa<FloatType>()) {
                  isGreater = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, elem, currentMax);
                } else {
                  auto intType = elemType.cast<IntegerType>();
                  if (intType.isUnsigned()) {
                    isGreater = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, elem, currentMax);
                  } else {
                    isGreater = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, elem, currentMax);
                  }
                }

                // Update max value and index if greater
                Value newMaxValue = builder.create<SelectOp>(loc, isGreater, elem, currentMax);
                builder.create<memref::StoreOp>(loc, newMaxValue, maxValues, outerIvs);

                // Convert reduction axis index from index to i64
                Value ivI64 = builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), iv);
                Value currentIdx = builder.create<memref::LoadOp>(loc, result, outerIvs);
                Value newIdx = builder.create<SelectOp>(loc, isGreater, ivI64, currentIdx);
                builder.create<memref::StoreOp>(loc, newIdx, result, outerIvs);

                builder.create<scf::YieldOp>(loc);
              });
        });

    rewriter.replaceOp(op, result);
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

/// Convert CallOp to handle type conversions
struct CallOpLowering : public OpConversionPattern<CallOp> {
  using OpConversionPattern<CallOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CallOp callOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // Convert result types
    SmallVector<Type, 1> resultTypes;
    if (failed(getTypeConverter()->convertTypes(callOp.getResultTypes(), resultTypes)))
      return failure();

    // Create new call with converted operands and result types
    rewriter.replaceOpWithNewOp<CallOp>(
        callOp,
        callOp.getCallee(),
        resultTypes,
        adaptor.getOperands());

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
    Value lhs = operands[0];          // A: memref<?xT>
    Value rhs = operands[1];          // B: memref<?xT>
    Value output = operands[2];       // C: memref<?xT> (pre-allocated by caller)
    Value m = operands[3];            // Rows of A
    Value k = operands[4];            // Cols of A / Rows of B
    Value n = operands[5];            // Cols of B
    Value lhs_offset = operands[6];   // Offset into lhs array
    Value rhs_offset = operands[7];   // Offset into rhs array
    Value output_offset = operands[8]; // Offset into output array

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

    // Convert offsets to index type
    if (!lhs_offset.getType().isIndex()) {
      lhs_offset = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), lhs_offset);
    }
    if (!rhs_offset.getType().isIndex()) {
      rhs_offset = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), rhs_offset);
    }
    if (!output_offset.getType().isIndex()) {
      output_offset = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), output_offset);
    }

    // Try to extract constant dimensions for static shapes (enables vectorization)
    std::optional<int64_t> mConst, kConst, nConst;

    // Check if m is constant
    if (auto constOp = m.getDefiningOp<arith::ConstantIndexOp>()) {
      mConst = constOp.value();
    } else if (auto indexCast = m.getDefiningOp<arith::IndexCastOp>()) {
      if (auto constInt = indexCast.getIn().getDefiningOp<arith::ConstantIntOp>()) {
        mConst = constInt.value();
      }
    }

    // Check if k is constant
    if (auto constOp = k.getDefiningOp<arith::ConstantIndexOp>()) {
      kConst = constOp.value();
    } else if (auto indexCast = k.getDefiningOp<arith::IndexCastOp>()) {
      if (auto constInt = indexCast.getIn().getDefiningOp<arith::ConstantIntOp>()) {
        kConst = constInt.value();
      }
    }

    // Check if n is constant
    if (auto constOp = n.getDefiningOp<arith::ConstantIndexOp>()) {
      nConst = constOp.value();
    } else if (auto indexCast = n.getDefiningOp<arith::IndexCastOp>()) {
      if (auto constInt = indexCast.getIn().getDefiningOp<arith::ConstantIntOp>()) {
        nConst = constInt.value();
      }
    }

    // Create 2D memref types for matrices
    // Use static shapes if all dimensions are constant (enables MLIR vectorization)
    MemRefType matrixAType, matrixBType, matrixCType;

    if (mConst && kConst && nConst) {
      // All dimensions are constant - use static shapes for vectorization!
      matrixAType = MemRefType::get({*mConst, *kConst}, elemType);  // MxK
      matrixBType = MemRefType::get({*kConst, *nConst}, elemType);  // KxN
      matrixCType = MemRefType::get({*mConst, *nConst}, elemType);  // MxN
    } else {
      // Dynamic shapes (fallback for runtime dimensions)
      matrixAType = MemRefType::get({-1, -1}, elemType);  // MxK
      matrixBType = MemRefType::get({-1, -1}, elemType);  // KxN
      matrixCType = MemRefType::get({-1, -1}, elemType);  // MxN
    }

    // Create stride constants
    Value oneStride = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Reinterpret 1D arrays as 2D matrices using provided offsets
    auto lhs2D = rewriter.create<memref::ReinterpretCastOp>(
        loc, matrixAType, lhs, lhs_offset,
        ValueRange{m, k}, ValueRange{k, oneStride});

    auto rhs2D = rewriter.create<memref::ReinterpretCastOp>(
        loc, matrixBType, rhs, rhs_offset,
        ValueRange{k, n}, ValueRange{n, oneStride});

    // Use the pre-allocated output buffer (host-kernel model: no allocations in kernel)
    // Reinterpret output as 2D matrix
    auto output2D = rewriter.create<memref::ReinterpretCastOp>(
        loc, matrixCType, output, output_offset,
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
// LLM/Transformer Operations
//===----------------------------------------------------------------------===//

// RMSNorm: output = (input / sqrt(mean(input^2) + eps)) * weight
struct RMSNormOpLowering : public OpConversionPattern<simp::RMSNormOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      simp::RMSNormOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto input = adaptor.input();
    auto weight = adaptor.weight();
    auto output = adaptor.output();
    auto size = adaptor.size();
    auto epsilon = adaptor.epsilon();
    auto weight_offset = adaptor.weight_offset();

    // Extract element type
    auto inputType = input.getType().dyn_cast<MemRefType>();
    auto elemType = inputType.getElementType();

    // Constants
    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto sizeIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), size);
    auto zeroFloat = rewriter.create<arith::ConstantOp>(loc, elemType, rewriter.getFloatAttr(elemType, 0.0));

    // Cast epsilon to element type if needed
    Value epsFloat = epsilon;
    if (epsilon.getType() != elemType) {
      if (epsilon.getType().cast<FloatType>().getWidth() > elemType.cast<FloatType>().getWidth()) {
        epsFloat = rewriter.create<arith::TruncFOp>(loc, elemType, epsilon);
      } else {
        epsFloat = rewriter.create<arith::ExtFOp>(loc, elemType, epsilon);
      }
    }

    // Step 1: Compute sum of squares
    auto sumSqAlloc = rewriter.create<memref::AllocaOp>(loc, MemRefType::get({}, elemType));
    rewriter.create<memref::StoreOp>(loc, zeroFloat, sumSqAlloc, ValueRange{});

    auto sumLoop = rewriter.create<scf::ForOp>(loc, c0, sizeIdx, c1);
    rewriter.setInsertionPointToStart(sumLoop.getBody());

    auto i = sumLoop.getInductionVar();
    auto val = rewriter.create<memref::LoadOp>(loc, input, ValueRange{i});
    auto sq = rewriter.create<arith::MulFOp>(loc, val, val);
    auto currentSum = rewriter.create<memref::LoadOp>(loc, sumSqAlloc, ValueRange());
    auto newSum = rewriter.create<arith::AddFOp>(loc, currentSum, sq);
    rewriter.create<memref::StoreOp>(loc, newSum, sumSqAlloc, ValueRange());

    rewriter.setInsertionPointAfter(sumLoop);

    // Step 2: Compute RMS = sqrt(mean + epsilon)
    auto sumSq = rewriter.create<memref::LoadOp>(loc, sumSqAlloc, ValueRange());
    auto sizeFloat = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), sizeIdx);
    auto sizeFP = rewriter.create<arith::SIToFPOp>(loc, elemType, sizeFloat);
    auto mean = rewriter.create<arith::DivFOp>(loc, sumSq, sizeFP);
    auto meanEps = rewriter.create<arith::AddFOp>(loc, mean, epsFloat);
    auto rms = rewriter.create<math::SqrtOp>(loc, meanEps);

    // Step 3: Normalize and scale
    auto normLoop = rewriter.create<scf::ForOp>(loc, c0, sizeIdx, c1);
    rewriter.setInsertionPointToStart(normLoop.getBody());

    auto normI = normLoop.getInductionVar();
    auto normVal = rewriter.create<memref::LoadOp>(loc, input, ValueRange{normI});

    // Add weight_offset to index for layer-specific weights
    auto offsetIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), weight_offset);
    auto weightIdx = rewriter.create<arith::AddIOp>(loc, offsetIdx, normI);
    auto w = rewriter.create<memref::LoadOp>(loc, weight, ValueRange{weightIdx});

    auto norm = rewriter.create<arith::DivFOp>(loc, normVal, rms);
    auto scaled = rewriter.create<arith::MulFOp>(loc, norm, w);
    rewriter.create<memref::StoreOp>(loc, scaled, output, ValueRange{normI});

    rewriter.setInsertionPointAfter(normLoop);
    rewriter.replaceOp(op, output);
    return success();
  }
};

// Softmax: output = exp(input - max) / sum(exp(input - max))
struct SoftmaxOpLowering : public OpConversionPattern<simp::SoftmaxOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      simp::SoftmaxOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto input = adaptor.input();
    auto output = adaptor.output();
    auto size = adaptor.size();
    auto input_offset = adaptor.input_offset();
    auto output_offset = adaptor.output_offset();

    auto inputType = input.getType().dyn_cast<MemRefType>();
    auto elemType = inputType.getElementType();

    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto sizeIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), size);

    // Convert offsets to index type
    auto inputOffsetIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), input_offset);
    auto outputOffsetIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), output_offset);

    // Step 1: Find max value
    auto firstIdx = rewriter.create<arith::AddIOp>(loc, c0, inputOffsetIdx);
    auto firstVal = rewriter.create<memref::LoadOp>(loc, input, ValueRange{firstIdx});
    auto maxAlloc = rewriter.create<memref::AllocaOp>(loc, MemRefType::get({}, elemType));
    rewriter.create<memref::StoreOp>(loc, firstVal, maxAlloc, ValueRange());

    auto maxLoop = rewriter.create<scf::ForOp>(loc, c1, sizeIdx, c1);
    rewriter.setInsertionPointToStart(maxLoop.getBody());

    auto i = maxLoop.getInductionVar();
    auto inputIdx = rewriter.create<arith::AddIOp>(loc, i, inputOffsetIdx);
    auto val = rewriter.create<memref::LoadOp>(loc, input, ValueRange{inputIdx});
    auto currentMax = rewriter.create<memref::LoadOp>(loc, maxAlloc, ValueRange());
    // Use compare + select instead of MaxFOp
    auto cmp = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, val, currentMax);
    auto newMax = rewriter.create<mlir::SelectOp>(loc, cmp, val, currentMax);
    rewriter.create<memref::StoreOp>(loc, newMax, maxAlloc, ValueRange());

    rewriter.setInsertionPointAfter(maxLoop);
    auto maxVal = rewriter.create<memref::LoadOp>(loc, maxAlloc, ValueRange());

    // Step 2: Compute exp(input - max) and sum
    auto zeroFloat = rewriter.create<arith::ConstantOp>(loc, elemType, rewriter.getFloatAttr(elemType, 0.0));
    auto sumAlloc = rewriter.create<memref::AllocaOp>(loc, MemRefType::get({}, elemType));
    rewriter.create<memref::StoreOp>(loc, zeroFloat, sumAlloc, ValueRange());

    auto expLoop = rewriter.create<scf::ForOp>(loc, c0, sizeIdx, c1);
    rewriter.setInsertionPointToStart(expLoop.getBody());

    auto expI = expLoop.getInductionVar();
    auto expInputIdx = rewriter.create<arith::AddIOp>(loc, expI, inputOffsetIdx);
    auto expOutputIdx = rewriter.create<arith::AddIOp>(loc, expI, outputOffsetIdx);
    auto expInputVal = rewriter.create<memref::LoadOp>(loc, input, ValueRange{expInputIdx});
    auto shifted = rewriter.create<arith::SubFOp>(loc, expInputVal, maxVal);
    auto expVal = rewriter.create<math::ExpOp>(loc, shifted);
    rewriter.create<memref::StoreOp>(loc, expVal, output, ValueRange{expOutputIdx});

    auto currentSum = rewriter.create<memref::LoadOp>(loc, sumAlloc, ValueRange());
    auto newSum = rewriter.create<arith::AddFOp>(loc, currentSum, expVal);
    rewriter.create<memref::StoreOp>(loc, newSum, sumAlloc, ValueRange());

    rewriter.setInsertionPointAfter(expLoop);

    // Step 3: Normalize by sum
    auto sumExp = rewriter.create<memref::LoadOp>(loc, sumAlloc, ValueRange());
    auto normLoop = rewriter.create<scf::ForOp>(loc, c0, sizeIdx, c1);
    rewriter.setInsertionPointToStart(normLoop.getBody());

    auto normI = normLoop.getInductionVar();
    auto normOutputIdx = rewriter.create<arith::AddIOp>(loc, normI, outputOffsetIdx);
    auto normExpVal = rewriter.create<memref::LoadOp>(loc, output, ValueRange{normOutputIdx});
    auto prob = rewriter.create<arith::DivFOp>(loc, normExpVal, sumExp);
    rewriter.create<memref::StoreOp>(loc, prob, output, ValueRange{normOutputIdx});

    rewriter.setInsertionPointAfter(normLoop);
    rewriter.replaceOp(op, output);
    return success();
  }
};

// SiLU: output = x / (1 + exp(-x))
struct SiLUOpLowering : public OpConversionPattern<simp::SiLUOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      simp::SiLUOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto input = adaptor.input();
    auto output = adaptor.output();
    auto size = adaptor.size();

    auto inputType = input.getType().dyn_cast<MemRefType>();
    auto elemType = inputType.getElementType();

    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto sizeIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), size);
    auto oneFloat = rewriter.create<arith::ConstantOp>(loc, elemType, rewriter.getFloatAttr(elemType, 1.0));

    auto siluLoop = rewriter.create<scf::ForOp>(loc, c0, sizeIdx, c1);
    rewriter.setInsertionPointToStart(siluLoop.getBody());

    auto i = siluLoop.getInductionVar();
    auto x = rewriter.create<memref::LoadOp>(loc, input, ValueRange{i});
    auto negX = rewriter.create<arith::NegFOp>(loc, x);
    auto expNegX = rewriter.create<math::ExpOp>(loc, negX);
    auto denom = rewriter.create<arith::AddFOp>(loc, oneFloat, expNegX);
    auto result = rewriter.create<arith::DivFOp>(loc, x, denom);
    rewriter.create<memref::StoreOp>(loc, result, output, ValueRange{i});

    rewriter.setInsertionPointAfter(siluLoop);
    rewriter.replaceOp(op, output);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Quantization Operations
//===----------------------------------------------------------------------===//

// DequantW4: Dequantize single 4-bit weight value
struct DequantW4OpLowering : public OpConversionPattern<simp::DequantW4Op> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      simp::DequantW4Op op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto qweights = adaptor.qweights();
    auto scales = adaptor.scales();
    auto zeros = adaptor.zeros();
    auto idx = adaptor.idx();
    auto group_size = adaptor.group_size();

    auto i8Type = rewriter.getIntegerType(8);
    auto f32Type = rewriter.getF32Type();

    // g = idx / group_size
    auto idxCast = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), idx);
    auto groupSizeCast = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), group_size);
    auto gIdx = rewriter.create<arith::DivUIOp>(loc, idxCast, groupSizeCast);

    // Load scale and zero for this group
    auto scale = rewriter.create<memref::LoadOp>(loc, scales, ValueRange{gIdx});
    auto zero = rewriter.create<memref::LoadOp>(loc, zeros, ValueRange{gIdx});

    // byte_idx = idx / 2
    auto c2Idx = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    auto byteIdx = rewriter.create<arith::DivUIOp>(loc, idxCast, c2Idx);

    // Load packed byte
    auto qbyte_i8 = rewriter.create<memref::LoadOp>(loc, qweights, ValueRange{byteIdx});

    // Convert i8 to i32 for bitwise operations
    auto qbyte = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI32Type(), qbyte_i8);

    // Extract 4-bit value based on idx % 2
    auto c2I64 = rewriter.create<arith::ConstantOp>(loc, idx.getType(), rewriter.getIntegerAttr(idx.getType(), 2));
    auto idxMod2 = rewriter.create<arith::RemUIOp>(loc, idx, c2I64);
    auto c0I64 = rewriter.create<arith::ConstantOp>(loc, idx.getType(), rewriter.getIntegerAttr(idx.getType(), 0));
    auto isEven = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, idxMod2, c0I64);

    // if (idx % 2 == 0): qval = qbyte & 15
    // else: qval = (qbyte >> 4) & 15
    auto c4I32 = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(4));
    auto c15I32 = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(15));

    auto qbyteShifted = rewriter.create<arith::ShRUIOp>(loc, qbyte, c4I32);
    auto qvalEven = rewriter.create<arith::AndIOp>(loc, qbyte, c15I32);
    auto qvalOdd = rewriter.create<arith::AndIOp>(loc, qbyteShifted, c15I32);
    auto qval_i32 = rewriter.create<SelectOp>(loc, isEven, qvalEven, qvalOdd);

    // Convert qval to float
    auto qval_f = rewriter.create<arith::UIToFPOp>(loc, f32Type, qval_i32);

    // result = qval_f * scale + zero
    auto scaled = rewriter.create<arith::MulFOp>(loc, qval_f, scale);
    auto result = rewriter.create<arith::AddFOp>(loc, scaled, zero);

    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};

/// Reshape: Change tensor shape (optimized with linalg.generic for vectorization)
struct TensorReshapeOpLowering : public OpConversionPattern<simp::TensorReshapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      simp::TensorReshapeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto input = adaptor.input();
    auto inputType = input.getType().cast<MemRefType>();
    auto resultType = getTypeConverter()->convertType(op.getType()).cast<MemRefType>();

    // Compute total number of elements
    int64_t totalElements = 1;
    for (auto dim : inputType.getShape()) {
      totalElements *= dim;
    }

    // Allocate result tensor
    Value result = rewriter.create<memref::AllocOp>(loc, resultType);

    // Use linalg.generic with flat 1D iteration for vectorization
    // Collapse both input and output to 1D views
    SmallVector<int64_t, 1> flatShape = {totalElements};
    auto flatType = MemRefType::get(flatShape, inputType.getElementType());

    Value inputFlat = rewriter.create<memref::ReinterpretCastOp>(
        loc, flatType, input, 0, flatShape, SmallVector<int64_t, 1>{1});
    Value resultFlat = rewriter.create<memref::ReinterpretCastOp>(
        loc, flatType, result, 0, flatShape, SmallVector<int64_t, 1>{1});

    // Create linalg.generic that copies flatInput[i] -> flatResult[i]
    // This will vectorize efficiently
    SmallVector<mlir::AffineMap, 2> indexingMaps = {
        mlir::AffineMap::getMultiDimIdentityMap(1, rewriter.getContext()),  // input
        mlir::AffineMap::getMultiDimIdentityMap(1, rewriter.getContext())   // output
    };
    SmallVector<mlir::StringRef, 1> iteratorTypes = {mlir::getParallelIteratorTypeName()};

    rewriter.create<mlir::linalg::GenericOp>(
        loc,
        TypeRange{},  // No results (writes to output arg)
        ValueRange{inputFlat},  // inputs
        ValueRange{resultFlat}, // outputs
        indexingMaps,
        iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          // Body: output[i] = input[i]
          b.create<mlir::linalg::YieldOp>(loc, args[0]);
        });

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Transpose: Permute tensor dimensions (loop-based)
struct TensorTransposeOpLowering : public OpConversionPattern<simp::TensorTransposeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      simp::TensorTransposeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto input = adaptor.input();
    auto inputType = input.getType().cast<MemRefType>();
    auto resultType = getTypeConverter()->convertType(op.getType()).cast<MemRefType>();

    // Extract permutation from operands (first operand is tensor, rest are indices)
    auto allOperands = adaptor.getOperands();
    int64_t rank = allOperands.size() - 1;  // Subtract 1 for the tensor operand

    // For 2D default transpose (no permutation args), use [1, 0]
    SmallVector<int64_t, 4> permutation;
    if (rank == 0 && inputType.getRank() == 2) {
      permutation = {1, 0};
      rank = 2;
    } else {
      // Extract constant permutation values (skip first operand which is the tensor)
      for (size_t i = 1; i < allOperands.size(); i++) {
        Value indexVal = allOperands[i];
        if (auto constOp = indexVal.getDefiningOp<arith::ConstantOp>()) {
          if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>()) {
            permutation.push_back(intAttr.getInt());
          }
        } else if (auto simpConstOp = indexVal.getDefiningOp<simp::ConstantOp>()) {
          if (auto intAttr = simpConstOp.value().dyn_cast<IntegerAttr>()) {
            permutation.push_back(intAttr.getInt());
          }
        }
      }
    }

    auto inputShape = inputType.getShape();
    auto resultShape = resultType.getShape();

    // Allocate result tensor
    Value result = rewriter.create<memref::AllocOp>(loc, resultType);

    // Build nested loops for transpose
    SmallVector<Value, 4> lbs, ubs, steps;
    for (int64_t i = 0; i < rank; i++) {
      lbs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
      ubs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, resultShape[i]));
      steps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
    }

    scf::buildLoopNest(rewriter, loc, lbs, ubs, steps,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          // Compute input indices by permuting output indices
          SmallVector<Value, 4> inputIndices(rank);
          for (int64_t i = 0; i < rank; i++) {
            inputIndices[permutation[i]] = ivs[i];
          }

          // Load from input, store to result
          Value elem = builder.create<memref::LoadOp>(loc, input, inputIndices);
          builder.create<memref::StoreOp>(loc, elem, result, ivs);
        });

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Slice: Extract sub-tensor (copy-based with SCF loops)
struct TensorSliceOpLowering : public OpConversionPattern<simp::TensorSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      simp::TensorSliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto input = adaptor.input();
    auto inputType = input.getType().cast<MemRefType>();
    auto resultType = getTypeConverter()->convertType(op.getType()).cast<MemRefType>();

    auto inputShape = inputType.getShape();
    auto resultShape = resultType.getShape();
    int64_t rank = inputShape.size();

    // Extract slice indices (start, end pairs)
    // First operand is tensor, rest are indices (start0, end0, start1, end1, ...)
    auto allOperands = adaptor.getOperands();
    size_t numIndices = allOperands.size() - 1;  // Subtract 1 for the tensor operand
    if (numIndices != static_cast<size_t>(rank * 2)) {
      return op.emitError("Slice requires 2 indices per dimension (start, end)");
    }

    SmallVector<Value, 4> starts, ends;
    for (int64_t i = 0; i < rank; i++) {
      starts.push_back(allOperands[1 + i * 2]);  // Skip first operand (tensor)
      ends.push_back(allOperands[1 + i * 2 + 1]);
    }

    // Allocate result tensor
    Value result = rewriter.create<memref::AllocOp>(loc, resultType);

    // Build nested loops to copy slice
    SmallVector<Value, 4> lbs, ubs, steps;
    for (int64_t i = 0; i < rank; i++) {
      lbs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
      ubs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, resultShape[i]));
      steps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
    }

    scf::buildLoopNest(rewriter, loc, lbs, ubs, steps,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          // Compute input indices: input[start + iv]
          SmallVector<Value, 4> inputIndices;
          for (int64_t i = 0; i < rank; i++) {
            Value startIdx = starts[i];
            if (!startIdx.getType().isIndex()) {
              startIdx = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), startIdx);
            }
            Value inputIdx = builder.create<arith::AddIOp>(loc, startIdx, ivs[i]);
            inputIndices.push_back(inputIdx);
          }

          // Load from input, store to result
          Value elem = builder.create<memref::LoadOp>(loc, input, inputIndices);
          builder.create<memref::StoreOp>(loc, elem, result, ivs);
        });

    rewriter.replaceOp(op, result);
    return success();
  }
};

// MatMulQuant: Quantized matrix multiplication with tile-based dequantization
// Strategy: Keep weights in W4 format, dequantize tiles on-the-fly for vectorization
// For W[M×K] @ input[K] = output[M], process in tiles of size TILE×K
struct MatMulQuantOpLowering : public OpConversionPattern<simp::MatMulQuantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      simp::MatMulQuantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto qweights = adaptor.qweights();
    auto scales = adaptor.scales();
    auto zeros = adaptor.zeros();
    auto input = adaptor.input();
    auto output = adaptor.output();
    auto rows = adaptor.rows();  // M
    auto cols = adaptor.cols();  // K
    auto group_size = adaptor.group_size();
    auto offset = adaptor.offset();

    auto f32Type = rewriter.getF32Type();
    auto i32Type = rewriter.getIntegerType(32);

    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto rowsIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), rows);
    auto colsIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), cols);

    // Constants for nibble extraction
    Value c4i32 = rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(4));
    Value c15i32 = rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(15));

    // Tile size for dequantization (8 rows at a time)
    constexpr int64_t TILE_SIZE = 8;
    auto tileSize = rewriter.create<arith::ConstantIndexOp>(loc, TILE_SIZE);

    // Try to extract constant dimensions for static tile shapes
    std::optional<int64_t> colsConst;
    if (auto constOp = cols.getDefiningOp<arith::ConstantIntOp>()) {
      colsConst = constOp.value();
    }

    // TILE-BASED VECTORIZATION: Process 16 weights at a time
    // Strategy: Dequantize 16 weights → temp buffer, then vectorized dot product

    Value zeroF32 = rewriter.create<arith::ConstantOp>(loc, f32Type, rewriter.getF32FloatAttr(0.0f));
    Value offsetIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), offset);
    Value groupSizeCast = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), group_size);
    Value c16 = rewriter.create<arith::ConstantIndexOp>(loc, 16);
    Value c8 = rewriter.create<arith::ConstantIndexOp>(loc, 8);

    // Allocate tile buffer for 16 dequantized weights
    auto tileType = MemRefType::get({16}, f32Type);
    Value weightTile = rewriter.create<memref::AllocaOp>(loc, tileType);

    // Row loop: for i in [0, rows)
    auto rowLoop = rewriter.create<scf::ForOp>(loc, c0, rowsIdx, c1);
    rewriter.setInsertionPointToStart(rowLoop.getBody());
    auto i = rowLoop.getInductionVar();
    auto rowOffset = rewriter.create<arith::MulIOp>(loc, i, colsIdx);

    // Initialize output[i] = 0
    rewriter.create<memref::StoreOp>(loc, zeroF32, output, ValueRange{i});

    // Group loop: for each group of GS weights (128 weights per group)
    auto numGroups = rewriter.create<arith::DivUIOp>(loc, colsIdx, groupSizeCast);
    auto groupLoop = rewriter.create<scf::ForOp>(loc, c0, numGroups, c1);
    rewriter.setInsertionPointToStart(groupLoop.getBody());
    auto g = groupLoop.getInductionVar();
    auto gStart = rewriter.create<arith::MulIOp>(loc, g, groupSizeCast);

    // Load scale/zero once per group
    auto rowGroupBase = rewriter.create<arith::DivUIOp>(loc, rowOffset, groupSizeCast);
    auto groupIdx = rewriter.create<arith::AddIOp>(loc, rowGroupBase, g);
    auto scale = rewriter.create<memref::LoadOp>(loc, scales, ValueRange{groupIdx});
    auto zero = rewriter.create<memref::LoadOp>(loc, zeros, ValueRange{groupIdx});

    // Tile loop: process 16 weights at a time within group (128/16 = 8 tiles)
    auto numTiles = rewriter.create<arith::ConstantIndexOp>(loc, 8);  // 128/16 = 8
    auto tileLoop = rewriter.create<scf::ForOp>(
        loc, c0, numTiles, c1,
        ValueRange{zeroF32},  // Accumulator
        [&](OpBuilder &b, Location loc, Value t, ValueRange iterArgs) {
          Value acc = iterArgs[0];
          auto tileStart = b.create<arith::MulIOp>(loc, t, c16);
          auto kStart = b.create<arith::AddIOp>(loc, gStart, tileStart);

          // Dequantize 16 weights into tile buffer
          // Load 8 bytes (16 nibbles) and unpack
          for (int j = 0; j < 16; j++) {
            auto jIdx = b.create<arith::ConstantIndexOp>(loc, j);
            auto kGlobal = b.create<arith::AddIOp>(loc, kStart, jIdx);

            // Weight index in qweights array
            auto wIdx = b.create<arith::AddIOp>(loc, rowOffset, kGlobal);
            wIdx = b.create<arith::AddIOp>(loc, offsetIdx, wIdx);

            // Extract nibble
            auto c1I64 = b.create<arith::ConstantOp>(loc, wIdx.getType(), b.getIntegerAttr(wIdx.getType(), 1));
            auto qByteIdx = b.create<arith::ShRUIOp>(loc, wIdx, c1I64);
            auto qByte_i8 = b.create<memref::LoadOp>(loc, qweights, ValueRange{qByteIdx});
            auto qByte = b.create<arith::ExtUIOp>(loc, i32Type, qByte_i8);

            auto idxAnd1 = b.create<arith::AndIOp>(loc, wIdx, c1I64);
            auto c0Idx = b.create<arith::ConstantOp>(loc, wIdx.getType(), b.getIntegerAttr(wIdx.getType(), 0));
            auto isEven = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, idxAnd1, c0Idx);

            auto lowNibble = b.create<arith::AndIOp>(loc, qByte, c15i32);
            auto qbyteShifted = b.create<arith::ShRUIOp>(loc, qByte, c4i32);
            auto highNibble = b.create<arith::AndIOp>(loc, qbyteShifted, c15i32);
            auto nibble = b.create<SelectOp>(loc, isEven, lowNibble, highNibble);

            // Dequantize: weight = nibble * scale + zero
            auto nibbleF = b.create<arith::UIToFPOp>(loc, f32Type, nibble);
            auto scaled = b.create<arith::MulFOp>(loc, nibbleF, scale);
            auto weightF = b.create<arith::AddFOp>(loc, scaled, zero);

            // Store in tile buffer
            b.create<memref::StoreOp>(loc, weightF, weightTile, ValueRange{jIdx});
          }

          // Simple loop for dot product
          auto dotLoop = b.create<scf::ForOp>(
              loc, c0, c16, c1,
              ValueRange{acc},
              [&](OpBuilder &b2, Location loc, Value j, ValueRange iterArgs2) {
                Value dotAcc = iterArgs2[0];
                auto kGlobal = b2.create<arith::AddIOp>(loc, kStart, j);

                // Bounds check
                auto kValid = b2.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, kGlobal, colsIdx);
                auto ifOp = b2.create<scf::IfOp>(
                    loc, TypeRange{f32Type}, kValid,
                    [&](OpBuilder &b3, Location loc) {
                      auto w = b3.create<memref::LoadOp>(loc, weightTile, ValueRange{j});
                      auto x = b3.create<memref::LoadOp>(loc, input, ValueRange{kGlobal});
                      auto prod = b3.create<arith::MulFOp>(loc, w, x);
                      auto newAcc = b3.create<arith::AddFOp>(loc, dotAcc, prod);
                      b3.create<scf::YieldOp>(loc, ValueRange{newAcc});
                    },
                    [&](OpBuilder &b3, Location loc) {
                      b3.create<scf::YieldOp>(loc, ValueRange{dotAcc});
                    });

                b2.create<scf::YieldOp>(loc, ValueRange{ifOp.getResult(0)});
              });

          b.create<scf::YieldOp>(loc, ValueRange{dotLoop.getResult(0)});
        });

    // Add tile result to output[i]
    auto currentOut = rewriter.create<memref::LoadOp>(loc, output, ValueRange{i});
    auto tileResult = tileLoop.getResult(0);
    auto newOut = rewriter.create<arith::AddFOp>(loc, currentOut, tileResult);
    rewriter.create<memref::StoreOp>(loc, newOut, output, ValueRange{i});

    rewriter.setInsertionPointAfter(groupLoop);
    rewriter.setInsertionPointAfter(rowLoop);

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
    registry.insert<math::MathDialect>();
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
    target.addLegalDialect<math::MathDialect>();
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalOp<ModuleOp>();

    // FuncOp is legal only if its signature has been converted (no simp types)
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      return typeConverter.isSignatureLegal(op.getType());
    });

    // CallOp is legal only if its operands and results have legal types
    target.addDynamicallyLegalOp<CallOp>([&](CallOp op) {
      return typeConverter.isLegal(op);
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
        TensorCreateOpLowering,
        TensorGetOpLowering,
        TensorSetOpLowering,
        TensorAddOpLowering,
        TensorMulOpLowering,
        TensorSubOpLowering,
        TensorDivOpLowering,
        TensorSumOpLowering,
        TensorMeanOpLowering,
        TensorMaxOpLowering,
        TensorMinOpLowering,
        TensorArgmaxOpLowering,
        AddOpLowering,
        SubOpLowering,
        MulOpLowering,
        DivOpLowering,
        ModOpLowering,
        NegOpLowering,
        CallOpLowering,
        MatMulOpLowering,
        Conv2DOpLowering,
        RMSNormOpLowering,
        SoftmaxOpLowering,
        SiLUOpLowering,
        DequantW4OpLowering,
        MatMulQuantOpLowering,
        TensorReshapeOpLowering,
        TensorTransposeOpLowering,
        TensorSliceOpLowering
    >(typeConverter, &getContext());

    // Add tensor unary operations with their specific kinds
    patterns.add<TensorUnaryOpLowering<simp::TensorReluOp>>(
        typeConverter, &getContext(), TensorUnaryOpLowering<simp::TensorReluOp>::UnaryOpKind::ReLU);
    patterns.add<TensorUnaryOpLowering<simp::TensorSigmoidOp>>(
        typeConverter, &getContext(), TensorUnaryOpLowering<simp::TensorSigmoidOp>::UnaryOpKind::Sigmoid);
    patterns.add<TensorUnaryOpLowering<simp::TensorTanhOp>>(
        typeConverter, &getContext(), TensorUnaryOpLowering<simp::TensorTanhOp>::UnaryOpKind::Tanh);

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
