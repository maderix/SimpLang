//===- InsertPrefetch.cpp - Insert prefetch operations for tiled loops ----===//
//
// Part of the SimpLang Project
//
// This pass inserts memref.prefetch operations into tiled loops to overlap
// memory access latency with computation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::arith;

namespace {

/// Insert prefetch operations into loops to hide memory latency
///
/// This pass walks scf.for loops and inserts memref.prefetch operations
/// for next-iteration loads to overlap memory access with computation.
///
class InsertPrefetchPass : public PassWrapper<InsertPrefetchPass, OperationPass<mlir::FuncOp>> {
public:
  StringRef getArgument() const override {
    return "simp-insert-prefetch";
  }

  StringRef getDescription() const override {
    return "Insert prefetch operations into tiled loops";
  }

  void runOnOperation() override {
    mlir::FuncOp func = getOperation();
    OpBuilder builder(func.getContext());

    // Walk all scf.for loops
    func.walk([&](scf::ForOp forOp) {
      Block *body = forOp.getBody();
      if (body->empty()) return;

      builder.setInsertionPointToStart(body);
      Location loc = forOp.getLoc();
      Value iv = forOp.getInductionVar();
      Value step = forOp.getOperand(2);
      Value lowerBound = forOp.getOperand(0);
      Value upperBound = forOp.getOperand(1);

      // Check if this is a nested loop (has inner scf.for)
      bool hasNestedLoop = false;
      scf::ForOp innerForOp;
      body->walk([&](scf::ForOp innerLoop) {
        if (innerLoop != forOp) {
          hasNestedLoop = true;
          innerForOp = innerLoop;
        }
      });

      // Outer loop: aggressive prefetch with high locality (keep in cache)
      // Inner loop: normal prefetch for next iteration
      unsigned localityHint = hasNestedLoop ? 3 : 2;  // L3 for outer, L2 for inner

      // Find memref.load operations (scalar loads)
      SmallVector<memref::LoadOp, 4> loads;
      body->walk([&](memref::LoadOp loadOp) {
        loads.push_back(loadOp);
      });

      for (auto loadOp : loads) {
        auto indices = loadOp.getIndices();
        if (indices.empty()) continue;

        // If first index is the loop IV, prefetch next iteration
        if (indices[0] == iv) {
          SmallVector<Value, 4> prefetchIndices(indices.begin(), indices.end());
          Value nextIV = builder.create<AddIOp>(loc, iv, step);
          prefetchIndices[0] = nextIV;

          builder.create<memref::PrefetchOp>(
              loc, loadOp.getMemRef(), prefetchIndices,
              /*isWrite=*/false, /*localityHint=*/localityHint, /*isDataCache=*/true);
        }
      }

      // Find vector.transfer_read operations (vectorized loads)
      SmallVector<vector::TransferReadOp, 4> vectorReads;
      body->walk([&](vector::TransferReadOp readOp) {
        vectorReads.push_back(readOp);
      });

      for (auto readOp : vectorReads) {
        // Check if any index uses the loop IV - prefetch subview memref
        // For matmul: vector.transfer_read reads from memref.subview results
        // We prefetch the base memref using the outer loop IV
        Value source = readOp.source();

        // If source is a subview, get its indices
        if (auto subviewOp = source.getDefiningOp<memref::SubViewOp>()) {
          auto subviewOffsets = subviewOp.getMixedOffsets();
          if (subviewOffsets.size() >= 2) {
            // Get the base memref type to check dimensionality
            auto baseMemref = subviewOp.source();
            auto baseType = baseMemref.getType().dyn_cast<MemRefType>();

            // Only apply row prefetching to 2D matrices (NxM), not vectors (Nx1)
            // Check: rank == 2 AND second dimension > 16 (not a column vector)
            bool is2DMatrix = false;
            if (baseType && baseType.getRank() == 2) {
              auto shape = baseType.getShape();
              // Dynamic dimensions are represented as -1 in MLIR
              if (shape.size() == 2 && (shape[1] < 0 || shape[1] > 16)) {
                is2DMatrix = true;
              }
            }

            // Check if FIRST offset uses this loop's IV
            OpFoldResult firstOffset = subviewOffsets[0];
            if (auto firstOffsetVal = firstOffset.dyn_cast<Value>()) {
              if (firstOffsetVal == iv) {
                // Deep prefetch: prefetch 2 iterations ahead for better latency hiding
                // Iteration i+1
                Value nextIV1 = builder.create<AddIOp>(loc, iv, step);
                SmallVector<Value, 2> prefetchIdx1 = {nextIV1, builder.create<arith::ConstantIndexOp>(loc, 0)};
                builder.create<memref::PrefetchOp>(
                    loc, subviewOp.source(), prefetchIdx1,
                    /*isWrite=*/false, /*localityHint=*/localityHint, /*isDataCache=*/true);

                // Iteration i+2
                Value nextIV2 = builder.create<AddIOp>(loc, nextIV1, step);
                SmallVector<Value, 2> prefetchIdx2 = {nextIV2, builder.create<arith::ConstantIndexOp>(loc, 0)};
                builder.create<memref::PrefetchOp>(
                    loc, subviewOp.source(), prefetchIdx2,
                    /*isWrite=*/false, /*localityHint=*/localityHint, /*isDataCache=*/true);
              }
            }

            // Check if SECOND offset uses this loop's IV (inner loop accessing weights)
            if (subviewOffsets.size() >= 2) {
              OpFoldResult secondOffset = subviewOffsets[1];
              if (auto secondOffsetVal = secondOffset.dyn_cast<Value>()) {
                if (secondOffsetVal == iv && !hasNestedLoop) {
                  // Inner loop: prefetch next column tile
                  Value nextIV = builder.create<AddIOp>(loc, iv, step);
                  SmallVector<Value, 2> prefetchIdx = {builder.create<arith::ConstantIndexOp>(loc, 0), nextIV};

                  builder.create<memref::PrefetchOp>(
                      loc, subviewOp.source(), prefetchIdx,
                      /*isWrite=*/false, /*localityHint=*/2, /*isDataCache=*/true);
                }
              }
            }
          }
        }
      }
    });
  }
};

} // namespace

namespace mlir {
namespace simp {

std::unique_ptr<Pass> createInsertPrefetchPass() {
  return std::make_unique<InsertPrefetchPass>();
}

} // namespace simp
} // namespace mlir
