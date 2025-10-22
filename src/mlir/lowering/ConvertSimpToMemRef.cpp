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
        MatMulOpLowering
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
