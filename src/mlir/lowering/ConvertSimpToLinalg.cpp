//===- ConvertSimpToLinalg.cpp - Lower Simp MatMul to Linalg -------------===//
//
// Part of the SimpLang Project
//
// This file implements the conversion pass from Simp dialect matmul operations
// to Linalg dialect operations. This enables high-level linear algebra
// optimizations and progressive lowering to loops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/simp_dialect.hpp"
#include "mlir/simp_ops.hpp"
#include "mlir/simp_types.hpp"

using namespace mlir;
using namespace mlir::simp;

//===----------------------------------------------------------------------===//
// MatMul Lowering Pattern
//===----------------------------------------------------------------------===//

/// Convert simp.matmul to linalg.matmul
///
/// NOTE: This pass runs AFTER ConvertSimpToMemRef, so arrays are already memrefs.
///
/// This pattern lowers:
///   %C = simp.matmul %A, %B, %m, %k, %n : (memref<?xf32>, memref<?xf32>, ...) -> memref<?xf32>
///
/// To:
///   %A_2d = memref.reinterpret_cast %A to memref<?x?xf32> [%m, %k]
///   %B_2d = memref.reinterpret_cast %B to memref<?x?xf32> [%k, %n]
///   %C_size = arith.muli %m, %n
///   %C = memref.alloc(%C_size) : memref<?xf32>
///   %C_2d = memref.reinterpret_cast %C to memref<?x?xf32> [%m, %n]
///   %zero = arith.constant 0.0 : f32
///   linalg.fill(%zero, %C_2d)
///   linalg.matmul ins(%A_2d, %B_2d : memref<?x?xf32>, memref<?x?xf32>)
///                 outs(%C_2d : memref<?x?xf32>)
///
struct MatMulOpLowering : public OpRewritePattern<simp::MatMulOp> {
  using OpRewritePattern<simp::MatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(simp::MatMulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get operands (now memrefs after SimpToMemRef pass)
    Value lhs = op.lhs();     // A: MxK matrix as memref<?xT>
    Value rhs = op.rhs();     // B: KxN matrix as memref<?xT>
    Value m = op.m();         // Rows of A
    Value k = op.k();         // Cols of A / Rows of B
    Value n = op.n();         // Cols of B

    // Get the element type from the memref
    auto lhsMemRefType = lhs.getType().dyn_cast<MemRefType>();
    if (!lhsMemRefType) {
      return failure();
    }
    Type elemType = lhsMemRefType.getElementType();

    // Ensure dimensions are index type (required for memref operations)
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

    // Reinterpret 1D arrays as 2D matrices
    // A: memref<?xT> -> memref<?x?xT> with shape [m, k]
    // Offset is 0 (constant value)
    Value zeroOffset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value oneStride = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    auto lhs2D = rewriter.create<memref::ReinterpretCastOp>(
        loc, matrixAType, lhs,
        /*offset=*/zeroOffset,
        /*sizes=*/ValueRange{m, k},
        /*strides=*/ValueRange{k, oneStride});

    // B: memref<?xT> -> memref<?x?xT> with shape [k, n]
    auto rhs2D = rewriter.create<memref::ReinterpretCastOp>(
        loc, matrixBType, rhs,
        /*offset=*/zeroOffset,
        /*sizes=*/ValueRange{k, n},
        /*strides=*/ValueRange{n, oneStride});

    // Allocate result matrix C (MxN as 1D array)
    Value resultSize = rewriter.create<arith::MulIOp>(loc, m, n);
    auto result1DType = MemRefType::get({-1}, elemType);
    Value result = rewriter.create<memref::AllocOp>(loc, result1DType, ValueRange{resultSize});

    // Reinterpret result as 2D matrix
    // C: memref<?xT> -> memref<?x?xT> with shape [m, n]
    auto result2D = rewriter.create<memref::ReinterpretCastOp>(
        loc, matrixCType, result,
        /*offset=*/zeroOffset,
        /*sizes=*/ValueRange{m, n},
        /*strides=*/ValueRange{n, oneStride});

    // Initialize result matrix to zero using linalg.fill
    Value zero;
    if (elemType.isa<FloatType>()) {
      // Create a properly typed float attribute
      auto floatType = elemType.cast<FloatType>();
      Attribute zeroAttr = rewriter.getFloatAttr(floatType, 0.0);
      zero = rewriter.create<arith::ConstantOp>(loc, floatType, zeroAttr);
    } else if (elemType.isa<IntegerType>()) {
      zero = rewriter.create<arith::ConstantIntOp>(loc, 0, elemType);
    } else {
      return failure();
    }

    rewriter.create<linalg::FillOp>(loc, zero, result2D.getResult());

    // Create linalg.matmul operation
    // linalg.matmul computes: C = A × B (C += A × B semantically)
    rewriter.create<linalg::MatmulOp>(
        loc,
        /*inputs=*/ValueRange{lhs2D.getResult(), rhs2D.getResult()},
        /*outputs=*/ValueRange{result2D.getResult()});

    // Replace the simp.matmul op with the 1D result memref
    rewriter.replaceOp(op, result);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertSimpToLinalgPass
    : public PassWrapper<ConvertSimpToLinalgPass, OperationPass<ModuleOp>> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<arith::ArithmeticDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();

    // Set up conversion target
    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<arith::ArithmeticDialect>();
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalDialect<SimpDialect>();
    target.addLegalOp<ModuleOp, FuncOp>();

    // Only mark simp.matmul as illegal
    target.addIllegalOp<simp::MatMulOp>();

    // Populate conversion patterns
    RewritePatternSet patterns(&getContext());
    patterns.add<MatMulOpLowering>(&getContext());

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

/// Create the Simp to Linalg lowering pass
std::unique_ptr<Pass> createConvertSimpToLinalgPass() {
  return std::make_unique<ConvertSimpToLinalgPass>();
}

} // namespace simp
} // namespace mlir
