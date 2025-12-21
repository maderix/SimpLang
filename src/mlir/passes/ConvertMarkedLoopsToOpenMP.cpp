//===- ConvertMarkedLoopsToOpenMP.cpp - Late-stage OpenMP conversion ------===//
//
// Part of the SimpLang Project
//
// This pass runs AFTER buffer management to convert loops marked with
// simp.parallel_loop attribute to OpenMP parallel constructs.
//
// Running after buffer management ensures that deallocation logic and other
// buffer-related operations don't end up inside omp.wsloop, which requires
// exactly one nested operation (omp.loop_nest).
//
// The pass looks for scf.for loops with simp.parallel_loop attribute and
// converts them to omp.parallel + omp.wsloop + omp.loop_nest structure.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "convert-marked-loops-to-openmp"

using namespace mlir;

namespace {

class ConvertMarkedLoopsToOpenMPPass
    : public PassWrapper<ConvertMarkedLoopsToOpenMPPass, OperationPass<func::FuncOp>> {
public:
  StringRef getArgument() const override { return "simp-convert-marked-loops-to-openmp"; }
  StringRef getDescription() const override {
    return "Convert loops marked with simp.parallel_loop to OpenMP (late-stage)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<omp::OpenMPDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Only process functions marked for OpenMP
    if (!func->hasAttr("simp.needs_openmp"))
      return;

    IRRewriter rewriter(&getContext());

    // Find the outermost marked loop (dimension 0)
    // For matmul, this is the M tile loop
    scf::ForOp outerLoop = nullptr;
    scf::ForOp innerLoop = nullptr;  // N tile loop if 2D parallel

    func.walk([&](scf::ForOp loop) {
      if (loop->hasAttr("simp.parallel_loop")) {
        auto dim = loop->getAttrOfType<IntegerAttr>("simp.parallel_dim");
        if (dim && dim.getInt() == 0) {
          outerLoop = loop;
        } else if (dim && dim.getInt() == 1) {
          innerLoop = loop;
        }
      }
    });

    if (!outerLoop) {
      llvm::errs() << "[OpenMP] No marked parallel loops found in " << func.getName() << "\n";
      return;
    }

    Location loc = outerLoop.getLoc();

    // Determine if this is 1D or 2D parallelization
    bool is2D = (innerLoop != nullptr);

    llvm::errs() << "[OpenMP] Converting " << (is2D ? "2D" : "1D")
                 << " parallel loops in " << func.getName() << "\n";

    // Insert before the outer loop
    rewriter.setInsertionPoint(outerLoop);

    // Create omp.parallel region
    auto parallelOp = rewriter.create<omp::ParallelOp>(loc);

    // Create the parallel region body
    Block *parallelBlock = rewriter.createBlock(&parallelOp.getRegion());
    rewriter.setInsertionPointToStart(parallelBlock);

    // Create omp.wsloop (worksharing loop wrapper)
    auto wsloopOp = rewriter.create<omp::WsloopOp>(loc);

    // Create wsloop region - MUST contain exactly one op (loop_nest)
    Block *wsloopBlock = rewriter.createBlock(&wsloopOp.getRegion());
    rewriter.setInsertionPointToStart(wsloopBlock);

    if (is2D) {
      // 2D parallel: M and N tile loops
      SmallVector<Value> lbs = {outerLoop.getLowerBound(), innerLoop.getLowerBound()};
      SmallVector<Value> ubs = {outerLoop.getUpperBound(), innerLoop.getUpperBound()};
      SmallVector<Value> steps = {outerLoop.getStep(), innerLoop.getStep()};

      auto loopNestOp = rewriter.create<omp::LoopNestOp>(loc, lbs, ubs, steps);

      // Create loop_nest region with 2 induction variables
      SmallVector<Type> ivTypes = {rewriter.getIndexType(), rewriter.getIndexType()};
      SmallVector<Location> ivLocs = {loc, loc};
      Block *loopNestBlock = rewriter.createBlock(
          &loopNestOp.getRegion(), {}, ivTypes, ivLocs);
      Value mIV = loopNestBlock->getArgument(0);
      Value nIV = loopNestBlock->getArgument(1);
      rewriter.setInsertionPointToStart(loopNestBlock);

      // Clone the inner loop's body (contains K loop and matmul ops)
      Block &innerBody = innerLoop.getRegion().front();
      IRMapping mapping;
      mapping.map(outerLoop.getInductionVar(), mIV);
      mapping.map(innerLoop.getInductionVar(), nIV);

      for (Operation &bodyOp : innerBody.without_terminator()) {
        rewriter.clone(bodyOp, mapping);
      }

      // Add omp.yield at end of loop_nest
      rewriter.create<omp::YieldOp>(loc);

    } else {
      // 1D parallel: just the outer loop
      auto loopNestOp = rewriter.create<omp::LoopNestOp>(
          loc,
          ValueRange{outerLoop.getLowerBound()},
          ValueRange{outerLoop.getUpperBound()},
          ValueRange{outerLoop.getStep()});

      Block *loopNestBlock = rewriter.createBlock(
          &loopNestOp.getRegion(), {}, {rewriter.getIndexType()}, {loc});
      Value iv = loopNestBlock->getArgument(0);
      rewriter.setInsertionPointToStart(loopNestBlock);

      Block &forBody = outerLoop.getRegion().front();
      IRMapping mapping;
      mapping.map(forBody.getArgument(0), iv);

      for (Operation &bodyOp : forBody.without_terminator()) {
        rewriter.clone(bodyOp, mapping);
      }

      rewriter.create<omp::YieldOp>(loc);
    }

    // Add omp.terminator at end of parallel region
    rewriter.setInsertionPointToEnd(parallelBlock);
    rewriter.create<omp::TerminatorOp>(loc);

    // Erase the original loop(s)
    rewriter.eraseOp(outerLoop);

    // Update function attributes
    func->removeAttr("simp.needs_openmp");
    func->setAttr("simp.uses_openmp", rewriter.getUnitAttr());

    llvm::errs() << "[OpenMP] Successfully converted to OpenMP in " << func.getName() << "\n";
  }
};

} // namespace

namespace mlir {
namespace simp {

std::unique_ptr<Pass> createConvertMarkedLoopsToOpenMPPass() {
  return std::make_unique<ConvertMarkedLoopsToOpenMPPass>();
}

void registerConvertMarkedLoopsToOpenMPPass() {
  PassRegistration<ConvertMarkedLoopsToOpenMPPass>();
}

} // namespace simp
} // namespace mlir
