//===- InsertPrefetch.cpp - Insert prefetch operations for tiled loops ----===//
//
// Part of the SimpLang Project
//
// This pass inserts memref.prefetch operations into tiled loops to overlap
// memory access latency with computation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
class InsertPrefetchPass : public PassWrapper<InsertPrefetchPass, OperationPass<func::FuncOp>> {
public:
  StringRef getArgument() const override {
    return "simp-insert-prefetch";
  }

  StringRef getDescription() const override {
    return "Insert prefetch operations into tiled loops";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder builder(func.getContext());

    // Read prefetch distance from annotation attribute (default: 2)
    int64_t prefetchDistance = 2;
    if (auto distAttr = func->getAttrOfType<IntegerAttr>("simp.prefetch_distance")) {
      prefetchDistance = distAttr.getInt();
      llvm::outs() << "[Prefetch] Using annotation distance=" << prefetchDistance
                   << " for " << func.getName() << "\n";
    }

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

      // Find memref.load operations (scalar loads) - only in immediate loop body
      // Skip loads inside nested loops to avoid using values that don't dominate
      SmallVector<memref::LoadOp, 4> loads;
      for (Operation &op : *body) {
        if (auto loadOp = dyn_cast<memref::LoadOp>(&op)) {
          loads.push_back(loadOp);
        }
      }

      for (auto loadOp : loads) {
        auto indices = loadOp.getIndices();
        if (indices.empty()) continue;

        // If first index is the loop IV, prefetch next iteration
        // Also verify all indices dominate the insertion point
        bool allIndicesDominate = true;
        for (auto idx : indices) {
          // Check if idx is defined in this loop body or outside
          if (auto defOp = idx.getDefiningOp()) {
            // If defined inside a nested loop, skip this load
            if (defOp->getParentRegion() != &forOp.getBodyRegion()) {
              // It's fine if defined outside
            }
          } else {
            // Block argument - check if it's from a nested loop
            if (auto blockArg = mlir::dyn_cast<BlockArgument>(idx)) {
              if (blockArg.getOwner() != body) {
                // Not our block - might be inner loop, skip
                allIndicesDominate = false;
                break;
              }
            }
          }
        }

        if (!allIndicesDominate) continue;

        if (indices[0] == iv) {
          SmallVector<Value, 4> prefetchIndices(indices.begin(), indices.end());
          Value nextIV = builder.create<AddIOp>(loc, iv, step);
          prefetchIndices[0] = nextIV;

          builder.create<memref::PrefetchOp>(
              loc, loadOp.getMemRef(), prefetchIndices,
              /*isWrite=*/false, /*localityHint=*/localityHint, /*isDataCache=*/true);
        }
      }

      // Find vector.transfer_read operations (vectorized loads) - only in immediate loop body
      SmallVector<vector::TransferReadOp, 4> vectorReads;
      for (Operation &op : *body) {
        if (auto readOp = dyn_cast<vector::TransferReadOp>(&op)) {
          vectorReads.push_back(readOp);
        }
      }

      for (auto readOp : vectorReads) {
        // Check if any index uses the loop IV - prefetch subview memref
        // For matmul: vector.transfer_read reads from memref.subview results
        // We prefetch the base memref using the outer loop IV
        Value source = readOp.getBase();

        // If source is a subview, get its indices
        if (auto subviewOp = source.getDefiningOp<memref::SubViewOp>()) {
          auto subviewOffsets = subviewOp.getMixedOffsets();
          if (subviewOffsets.size() >= 2) {
            // Get the base memref type to check dimensionality
            auto baseMemref = subviewOp.getSource();
            auto baseType = mlir::dyn_cast<MemRefType>(baseMemref.getType());

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
            if (auto firstOffsetVal = mlir::dyn_cast<Value>(firstOffset)) {
              if (firstOffsetVal == iv) {
                // Deep prefetch: prefetch N iterations ahead based on annotation
                // Generate prefetch for iterations i+1 through i+prefetchDistance
                Value currentIV = iv;
                for (int64_t d = 1; d <= prefetchDistance; ++d) {
                  Value nextIV = builder.create<AddIOp>(loc, currentIV, step);
                  SmallVector<Value, 2> prefetchIdx = {nextIV, builder.create<arith::ConstantIndexOp>(loc, 0)};
                  builder.create<memref::PrefetchOp>(
                      loc, subviewOp.getSource(), prefetchIdx,
                      /*isWrite=*/false, /*localityHint=*/localityHint, /*isDataCache=*/true);
                  currentIV = nextIV;
                }
              }
            }

            // Check if SECOND offset uses this loop's IV (inner loop accessing weights)
            if (subviewOffsets.size() >= 2) {
              OpFoldResult secondOffset = subviewOffsets[1];
              if (auto secondOffsetVal = mlir::dyn_cast<Value>(secondOffset)) {
                if (secondOffsetVal == iv && !hasNestedLoop) {
                  // Inner loop: prefetch next column tile
                  Value nextIV = builder.create<AddIOp>(loc, iv, step);
                  SmallVector<Value, 2> prefetchIdx = {builder.create<arith::ConstantIndexOp>(loc, 0), nextIV};

                  builder.create<memref::PrefetchOp>(
                      loc, subviewOp.getSource(), prefetchIdx,
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
