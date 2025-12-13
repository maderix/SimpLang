//===- SpecializeShapes.cpp - Convert Dynamic Memrefs to Static ----------===//
//
// Part of the SimpLang Project
//
// This pass converts dynamic memrefs (memref<?x?xf32>) to static shapes
// (memref<NxMxf32>) when the dimensions are known, enabling affine
// optimizations that require static bounds.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <optional>

using namespace mlir;

namespace {

/// Pattern to specialize memref.reinterpret_cast with static sizes
struct SpecializeReinterpretCastPattern
    : public OpRewritePattern<memref::ReinterpretCastOp> {
  using OpRewritePattern<memref::ReinterpretCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::ReinterpretCastOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get the result type
    auto resultType = mlir::dyn_cast<MemRefType>(op.getType());
    if (!resultType || !resultType.hasRank())
      return failure();

    // Check if we have dynamic shapes
    if (!resultType.getNumDynamicDims())
      return failure(); // Already static

    // Try to extract constant sizes from the size operands
    SmallVector<int64_t, 4> staticSizes;
    for (Value sizeOp : op.getSizes()) {
      // Try to get constant value
      if (auto constOp = sizeOp.getDefiningOp<arith::ConstantIndexOp>()) {
        // getValue() returns TypedAttr in LLVM 21, cast to IntegerAttr to get int
        auto intAttr = mlir::cast<IntegerAttr>(constOp.getValue());
        staticSizes.push_back(intAttr.getInt());
      } else if (auto castOp = sizeOp.getDefiningOp<arith::IndexCastOp>()) {
        // Check if it's a cast from a constant
        Value castInput = castOp->getOperand(0);
        if (auto constI64 = castInput.getDefiningOp<arith::ConstantIntOp>()) {
          staticSizes.push_back(constI64.value());
        } else {
          // Can't determine static size
          return failure();
        }
      } else {
        // Can't determine static size
        return failure();
      }
    }

    // Create new static memref type
    auto staticType = MemRefType::get(
        staticSizes,
        resultType.getElementType(),
        resultType.getLayout(),
        resultType.getMemorySpace());

    // Create new reinterpret_cast with static result type
    auto newOp = rewriter.create<memref::ReinterpretCastOp>(
        loc,
        staticType,
        op.getSource(),
        op.getOffsets(),
        op.getSizes(),
        op.getStrides(),
        op.getStaticOffsets(),
        op.getStaticSizes(),
        op.getStaticStrides());

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

/// Pass that specializes dynamic memref shapes to static shapes
struct SpecializeShapesPass
    : public PassWrapper<SpecializeShapesPass, OperationPass<func::FuncOp>> {


  StringRef getArgument() const override {
    return "specialize-shapes";
  }

  StringRef getDescription() const override {
    return "Specialize dynamic memref shapes to static shapes when known";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<SpecializeReinterpretCastPattern>(context);

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace simp {

std::unique_ptr<Pass> createSpecializeShapesPass() {
  return std::make_unique<SpecializeShapesPass>();
}

/// Register the pass for command-line usage
void registerSpecializeShapesPass() {
  PassRegistration<SpecializeShapesPass>();
}

} // namespace simp
} // namespace mlir
