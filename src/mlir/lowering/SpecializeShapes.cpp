//===- SpecializeShapes.cpp - Convert Dynamic Memrefs to Static ----------===//
//
// Part of the SimpLang Project
//
// This pass converts dynamic memrefs (memref<?x?xf32>) to static shapes
// (memref<NxMxf32>) when the dimensions are known, enabling affine
// optimizations that require static bounds.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/Optional.h"

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
    auto resultType = op.getType().dyn_cast<MemRefType>();
    if (!resultType || !resultType.hasRank())
      return failure();

    // Check if we have dynamic shapes
    if (!resultType.getNumDynamicDims())
      return failure(); // Already static

    // Try to extract constant sizes from the size operands
    SmallVector<int64_t, 4> staticSizes;
    for (Value sizeOp : op.sizes()) {
      // Try to get constant value
      if (auto constOp = sizeOp.getDefiningOp<arith::ConstantIndexOp>()) {
        staticSizes.push_back(constOp.value());
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
        op.source(),
        op.offsets(),
        op.sizes(),
        op.strides(),
        op.static_offsets(),
        op.static_sizes(),
        op.static_strides());

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

/// Pass that specializes dynamic memref shapes to static shapes
struct SpecializeShapesPass
    : public PassWrapper<SpecializeShapesPass, OperationPass<FuncOp>> {
  void runOnOperation() override {
    FuncOp func = getOperation();
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

} // namespace simp
} // namespace mlir
