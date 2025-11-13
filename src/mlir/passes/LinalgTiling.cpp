//===- LinalgTiling.cpp - Configurable Linalg Tiling Pass ----------------===//
//
// Part of the SimpLang Project
//
// This pass provides configurable loop tiling for Linalg operations with
// cache-aware defaults optimized for transformer workloads.
//
// Performance Note: 8×8×8 tiling achieves 45.68 tok/s on Stories110M (optimal)
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {

/// Options for SimpLinalg Tiling Pass
struct SimpLinalgTilingPassOptions {
  int64_t tileSize = 8;            // L1 cache tile size (optimal: 8)
  int64_t outerTileSize = 64;      // L2/L3 cache tile size
  bool hierarchical = false;        // Two-level tiling
  bool parallelLoops = false;       // Generate scf.parallel for OpenMP
};

/// Configurable Linalg tiling pass for SimpLang
///
/// This pass tiles Linalg operations (matmul, conv2d, etc.) to improve
/// cache locality. The default tile size of 8 is optimized for transformer
/// workloads (768×768 matrices) and achieves 45.68 tok/s on Stories110M.
///
class SimpLinalgTilingPass
    : public PassWrapper<SimpLinalgTilingPass, OperationPass<FuncOp>> {
private:
  SimpLinalgTilingPassOptions options;

public:
  SimpLinalgTilingPass() = default;
  SimpLinalgTilingPass(const SimpLinalgTilingPassOptions &opts) : options(opts) {}

  StringRef getArgument() const override {
    return "simp-linalg-tile";
  }

  StringRef getDescription() const override {
    return "Tile Linalg operations for cache locality (default: 8×8×8)";
  }

  void runOnOperation() override {
    FuncOp func = getOperation();
    MLIRContext *context = &getContext();

    // Determine loop type based on parallelization flag
    LinalgTilingLoopType loopType = options.parallelLoops
        ? LinalgTilingLoopType::ParallelLoops
        : LinalgTilingLoopType::Loops;

    if (options.hierarchical) {
      // Two-level hierarchical tiling
      llvm::outs() << "[Hierarchical Tiling] Outer: " << options.outerTileSize
                   << "×" << options.outerTileSize << "×" << options.outerTileSize
                   << ", Inner: " << options.tileSize << "×" << options.tileSize
                   << "×" << options.tileSize << "\n";

      // Level 1: Outer tiling (L2/L3 cache aware)
      llvm::SmallVector<int64_t, 3> outerTileSizes(3, options.outerTileSize);
      PassManager pm1(context);
      pm1.addPass(createLinalgTilingPass(outerTileSizes, loopType));
      pm1.addPass(createCanonicalizerPass());
      if (failed(pm1.run(func))) {
        signalPassFailure();
        return;
      }

      // Level 2: Inner tiling (L1 cache + vectorization)
      llvm::SmallVector<int64_t, 3> innerTileSizes(3, options.tileSize);
      PassManager pm2(context);
      pm2.addPass(createLinalgTilingPass(innerTileSizes, LinalgTilingLoopType::Loops));
      pm2.addPass(createCanonicalizerPass());
      if (failed(pm2.run(func))) {
        signalPassFailure();
        return;
      }

    } else {
      // Single-level tiling
      llvm::SmallVector<int64_t, 3> tileSizes(3, options.tileSize);
      PassManager pm(context);
      pm.addPass(createLinalgTilingPass(tileSizes, loopType));
      pm.addPass(createCanonicalizerPass());
      if (failed(pm.run(func))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

namespace mlir {
namespace simp {

/// Create the Linalg tiling pass with custom options
std::unique_ptr<Pass> createSimpLinalgTilingPass(int64_t tileSize, bool hierarchical, bool parallel) {
  SimpLinalgTilingPassOptions opts;
  opts.tileSize = tileSize;
  opts.hierarchical = hierarchical;
  opts.parallelLoops = parallel;
  return std::make_unique<SimpLinalgTilingPass>(opts);
}

/// Create the Linalg tiling pass with default options (8×8×8, optimal for transformers)
std::unique_ptr<Pass> createSimpLinalgTilingPass() {
  return std::make_unique<SimpLinalgTilingPass>();
}

/// Register the pass for command-line usage
void registerSimpLinalgTilingPass() {
  PassRegistration<SimpLinalgTilingPass>();
}

} // namespace simp
} // namespace mlir
