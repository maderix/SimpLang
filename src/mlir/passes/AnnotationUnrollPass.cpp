//===- AnnotationUnrollPass.cpp - Annotation-aware loop unrolling ---------===//
//
// Part of the SimpLang Project
//
// This pass reads @unroll(N) annotations from function attributes and applies
// loop unrolling with the specified factor.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "annotation-unroll"

using namespace mlir;

namespace {

/// Annotation-aware loop unrolling pass
/// Reads simp.unroll_factor from function or loop attributes
class AnnotationUnrollPass
    : public PassWrapper<AnnotationUnrollPass, OperationPass<func::FuncOp>> {
public:
  StringRef getArgument() const override { return "simp-annotation-unroll"; }
  StringRef getDescription() const override {
    return "Unroll loops based on @unroll annotation";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Read unroll factor from function attribute (default: 1 = no unroll)
    int64_t unrollFactor = 1;
    if (auto factorAttr = func->getAttrOfType<IntegerAttr>("simp.unroll_factor")) {
      unrollFactor = factorAttr.getInt();
    }

    if (unrollFactor <= 1) {
      LLVM_DEBUG(llvm::dbgs() << "No unroll annotation for " << func.getName() << "\n");
      return;
    }

    llvm::outs() << "[Unroll] Applying unroll factor=" << unrollFactor
                 << " to " << func.getName() << "\n";

    // Collect loops to unroll (can't modify while walking)
    SmallVector<scf::ForOp, 8> loopsToUnroll;
    func.walk([&](scf::ForOp forOp) {
      // Check for per-loop override
      int64_t loopFactor = unrollFactor;
      if (auto loopAttr = forOp->getAttrOfType<IntegerAttr>("simp.unroll_factor")) {
        loopFactor = loopAttr.getInt();
      }

      if (loopFactor > 1) {
        // Only unroll innermost loops (no nested scf.for inside)
        bool hasNestedLoop = false;
        forOp.walk([&](scf::ForOp inner) {
          if (inner != forOp) hasNestedLoop = true;
        });

        if (!hasNestedLoop) {
          loopsToUnroll.push_back(forOp);
        }
      }
    });

    // Unroll collected loops
    for (scf::ForOp forOp : loopsToUnroll) {
      int64_t factor = unrollFactor;
      if (auto loopAttr = forOp->getAttrOfType<IntegerAttr>("simp.unroll_factor")) {
        factor = loopAttr.getInt();
      }

      if (failed(unrollLoop(forOp, factor))) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to unroll loop\n");
      }
    }
  }

private:
  /// Manually unroll a loop by the given factor
  /// This duplicates the loop body and adjusts the step
  LogicalResult unrollLoop(scf::ForOp forOp, int64_t factor) {
    if (factor <= 1) return success();

    OpBuilder builder(forOp);
    Location loc = forOp.getLoc();

    // Get loop bounds
    Value lb = forOp.getLowerBound();
    Value ub = forOp.getUpperBound();
    Value step = forOp.getStep();

    // Try to get constant step
    auto stepOp = step.getDefiningOp<arith::ConstantIndexOp>();
    if (!stepOp) {
      LLVM_DEBUG(llvm::dbgs() << "Cannot unroll: non-constant step\n");
      return failure();
    }
    int64_t stepVal = stepOp.value();

    // Create new step = old_step * factor
    int64_t newStepVal = stepVal * factor;
    Value newStep = builder.create<arith::ConstantIndexOp>(loc, newStepVal);

    // Create unrolled loop
    auto unrolledLoop = builder.create<scf::ForOp>(
        loc, lb, ub, newStep, forOp.getInitArgs(),
        [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
          IRMapping mapper;

          // Map original IV and iter args
          mapper.map(forOp.getInductionVar(), iv);
          for (auto [oldArg, newArg] : llvm::zip(
                   forOp.getRegionIterArgs(), iterArgs)) {
            mapper.map(oldArg, newArg);
          }

          // Clone body for each unroll iteration
          SmallVector<Value> currentIterArgs(iterArgs.begin(), iterArgs.end());
          Value currentIV = iv;

          for (int64_t i = 0; i < factor; ++i) {
            if (i > 0) {
              // Update IV for this iteration: iv + i * step
              currentIV = b.create<arith::AddIOp>(
                  loc, iv,
                  b.create<arith::ConstantIndexOp>(loc, i * stepVal));
              mapper.map(forOp.getInductionVar(), currentIV);

              // Update iter args mapping
              for (auto [oldArg, newArg] : llvm::zip(
                       forOp.getRegionIterArgs(), currentIterArgs)) {
                mapper.map(oldArg, newArg);
              }
            }

            // Clone operations (except terminator)
            for (Operation &op : forOp.getBody()->without_terminator()) {
              b.clone(op, mapper);
            }

            // Get yielded values for next iteration
            auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
            currentIterArgs.clear();
            for (Value v : yield.getOperands()) {
              currentIterArgs.push_back(mapper.lookupOrDefault(v));
            }
          }

          // Yield final values
          b.create<scf::YieldOp>(loc, currentIterArgs);
        });

    // Replace uses and erase old loop
    forOp.replaceAllUsesWith(unrolledLoop.getResults());
    forOp.erase();

    LLVM_DEBUG(llvm::dbgs() << "Unrolled loop by factor " << factor << "\n");
    return success();
  }
};

} // namespace

namespace mlir {
namespace simp {

std::unique_ptr<Pass> createAnnotationUnrollPass() {
  return std::make_unique<AnnotationUnrollPass>();
}

} // namespace simp
} // namespace mlir
