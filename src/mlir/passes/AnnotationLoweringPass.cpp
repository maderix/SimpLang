//===- AnnotationLoweringPass.cpp - Process optimization annotations ------===//
//
// Part of the SimpLang Project
//
// This pass processes simp.annotated_region_* attributes on functions and
// applies appropriate transformations. Uses simp::AnnotationInfo from
// AnnotationRegistry.h as the single source of truth for annotation data.
//
// Supported annotations:
//   @tile(M, K, N)      - Tile sizes for matrix operations
//   @lower("pattern")   - Lowering pattern (e.g., "vnni.i8_matmul")
//   @align(N)           - Memory alignment
//   @unroll(N)          - Loop unrolling factor
//   @vectorize(width)   - Force SIMD width (128, 256, 512)
//   @prefetch(distance) - Memory prefetch distance
//   @parallel           - Enable outer loop parallelization
//   @fuse               - Enable loop fusion
//
// Composable transforms: Annotations are applied in order of appearance
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/passes/AnnotationRegistry.h"
#include "llvm/Support/Debug.h"
#include <functional>
#include <map>

#define DEBUG_TYPE "annotation-lowering"

using namespace mlir;
using namespace mlir::linalg;
using simp::AnnotationInfo;  // Single source of truth

namespace {

//===----------------------------------------------------------------------===//
// PatternRegistry - Extensible pattern registration system
//===----------------------------------------------------------------------===//

/// Pattern callback type - receives function, annotation info, and rewriter
using PatternCallback = std::function<LogicalResult(
    func::FuncOp, const AnnotationInfo&, IRRewriter&)>;

/// Registry for annotation-driven optimization patterns
class PatternRegistry {
public:
  /// Register a named pattern
  void registerPattern(StringRef name, PatternCallback callback) {
    patterns[name.str()] = std::move(callback);
  }

  /// Check if a pattern exists
  bool hasPattern(StringRef name) const {
    return patterns.find(name.str()) != patterns.end();
  }

  /// Apply a pattern by name
  LogicalResult applyPattern(StringRef name, func::FuncOp func,
                             const AnnotationInfo& info, IRRewriter& rewriter) {
    auto it = patterns.find(name.str());
    if (it == patterns.end()) {
      llvm::errs() << "Warning: Unknown pattern '" << name << "'\n";
      return failure();
    }
    return it->second(func, info, rewriter);
  }

  /// Get all registered pattern names
  std::vector<std::string> getPatternNames() const {
    std::vector<std::string> names;
    for (const auto& p : patterns) {
      names.push_back(p.first);
    }
    return names;
  }

private:
  std::map<std::string, PatternCallback> patterns;
};

//===----------------------------------------------------------------------===//
// VNNI Pattern Implementations
//===----------------------------------------------------------------------===//

/// Apply optimized tiling to matmul operations based on annotation
LogicalResult applyMatmulTilingPattern(func::FuncOp func,
                                       const AnnotationInfo& info,
                                       IRRewriter& rewriter) {
  LLVM_DEBUG(llvm::dbgs() << "Applying matmul tiling to " << func.getName() << "\n");

  // Get tile sizes from annotation (default: 64×64×64)
  SmallVector<int64_t, 3> tileSizes = {64, 64, 64};
  if (info.hasTileSizes() && info.tileSizes.size() >= 3) {
    tileSizes[0] = info.tileSizes[0];
    tileSizes[1] = info.tileSizes[1];
    tileSizes[2] = info.tileSizes[2];
  }

  // Find linalg.matmul operations
  SmallVector<LinalgOp, 4> matmulOps;
  func.walk([&](LinalgOp op) {
    if (isa<linalg::MatmulOp>(op))
      matmulOps.push_back(op);
  });

  if (matmulOps.empty()) {
    return success();  // No matmul to transform
  }

  // Use errs() for thread-safe output (unbuffered) - outs() crashes with multithreading
  llvm::errs() << "[Annotation] Tiling " << matmulOps.size() << " matmul(s) in "
               << func.getName() << " with tile=[" << tileSizes[0] << ","
               << tileSizes[1] << "," << tileSizes[2] << "]\n";

  // Configure and apply tiling
  // NOTE: For matmul, we CANNOT use LinalgTilingLoopType::ParallelLoops because:
  // - Matmul has 3 loops: M (rows), N (cols), K (reduction)
  // - M and N are parallel iterators, K is a reduction iterator
  // - ParallelLoops creates scf.parallel for M,N but scf.for for K inside
  // - SCF-to-OpenMP conversion fails because scf.for is inside omp.loop_nest
  //
  // Instead, we tile with regular loops and mark for parallelization.
  // The outer tile loops (over M and N tiles) can be parallelized later.
  LinalgTilingLoopType loopType = LinalgTilingLoopType::Loops;

  if (info.hasParallel()) {
    llvm::errs() << "[Parallel] Marking " << func.getName()
                 << " for parallelization (outer tile loops)\n";
    // Set attribute for later passes to parallelize outer loops
    func->setAttr("simp.parallelize_outer_tiles", rewriter.getUnitAttr());
  }

  LinalgTilingOptions tilingOptions;
  tilingOptions.setTileSizes(tileSizes);
  tilingOptions.setLoopType(loopType);

  for (LinalgOp op : matmulOps) {
    if (op->getParentOp() == nullptr) continue;

    rewriter.setInsertionPoint(op);
    FailureOr<TiledLinalgOp> tiledOp = tileLinalgOp(rewriter, op, tilingOptions);

    if (succeeded(tiledOp) && tiledOp->op) {
      // Mark as annotation-tiled (prevents double-tiling by default pass)
      tiledOp->op->setAttr("simp.annotation_tiled", rewriter.getUnitAttr());
      tiledOp->op->setAttr("simp.tile_sizes", rewriter.getI64ArrayAttr(tileSizes));
      rewriter.eraseOp(op);

      // If @parallel requested, mark the outer tile loops for late-stage
      // OpenMP conversion. We DON'T generate OpenMP here because:
      // 1. Buffer management passes run after this and insert ops
      // 2. Those ops would end up inside omp.wsloop, violating its constraint
      // 3. omp.wsloop MUST contain exactly one op (omp.loop_nest)
      //
      // Instead, we mark loops with attributes and a late-stage pass
      // (after buffer management) will convert them to OpenMP.
      //
      // For matmul, the iterator types are [parallel, parallel, reduction]:
      //   - Loop 0 (M tile): PARALLEL - can be parallelized
      //   - Loop 1 (N tile): PARALLEL - can be parallelized
      //   - Loop 2 (K tile): REDUCTION - must remain sequential
      if (info.hasParallel() && tiledOp->loops.size() >= 2) {
        auto mLoop = dyn_cast<scf::ForOp>(tiledOp->loops[0]);
        auto nLoop = dyn_cast<scf::ForOp>(tiledOp->loops[1]);

        if (mLoop && nLoop) {
          // Mark both M and N loops for parallelization
          mLoop->setAttr("simp.parallel_loop", rewriter.getUnitAttr());
          mLoop->setAttr("simp.parallel_dim", rewriter.getI64IntegerAttr(0));
          nLoop->setAttr("simp.parallel_loop", rewriter.getUnitAttr());
          nLoop->setAttr("simp.parallel_dim", rewriter.getI64IntegerAttr(1));

          // Mark function for late-stage OpenMP conversion
          func->setAttr("simp.needs_openmp", rewriter.getUnitAttr());

          llvm::errs() << "[Parallel] Marked M and N tile loops for OpenMP (late-stage)\n";
        }
      } else if (info.hasParallel() && tiledOp->loops.size() == 1) {
        auto outerLoop = dyn_cast<scf::ForOp>(tiledOp->loops[0]);
        if (outerLoop) {
          outerLoop->setAttr("simp.parallel_loop", rewriter.getUnitAttr());
          outerLoop->setAttr("simp.parallel_dim", rewriter.getI64IntegerAttr(0));
          func->setAttr("simp.needs_openmp", rewriter.getUnitAttr());
          llvm::errs() << "[Parallel] Marked outer loop for OpenMP (late-stage)\n";
        }
      }
    } else {
      llvm::errs() << "Warning: Failed to tile matmul in " << func.getName() << "\n";
    }
  }

  return success();
}

/// Scalar baseline - no optimization, for benchmarking
LogicalResult applyScalarPattern(func::FuncOp func,
                                 const AnnotationInfo& info,
                                 IRRewriter& rewriter) {
  func->setAttr("simp.scalar_baseline", rewriter.getUnitAttr());
  return success();
}

//===----------------------------------------------------------------------===//
// New Annotation Pattern Implementations
//===----------------------------------------------------------------------===//

/// Apply loop unrolling annotation
/// Marks loops for unrolling with the specified factor
LogicalResult applyUnrollPattern(func::FuncOp func,
                                 const AnnotationInfo& info,
                                 IRRewriter& rewriter) {
  if (!info.hasUnroll()) return success();

  LLVM_DEBUG(llvm::dbgs() << "Applying unroll(" << info.unrollFactor
                          << ") to " << func.getName() << "\n");

  // Set attribute for later LLVM pass to process
  func->setAttr("simp.unroll_factor",
                rewriter.getI64IntegerAttr(info.unrollFactor));

  // Also set on SCF loops within the function
  func.walk([&](scf::ForOp forOp) {
    forOp->setAttr("simp.unroll_factor",
                   rewriter.getI64IntegerAttr(info.unrollFactor));
  });

  return success();
}

/// Apply vectorization annotation
/// Sets preferred SIMD width for vectorization pass
LogicalResult applyVectorizePattern(func::FuncOp func,
                                    const AnnotationInfo& info,
                                    IRRewriter& rewriter) {
  if (!info.hasVectorize()) return success();

  LLVM_DEBUG(llvm::dbgs() << "Applying vectorize(" << info.vectorizeWidth
                          << ") to " << func.getName() << "\n");

  // Validate width (must be 128, 256, or 512)
  int64_t width = info.vectorizeWidth;
  if (width != 128 && width != 256 && width != 512) {
    llvm::errs() << "Warning: Invalid vectorize width " << width
                 << ", must be 128, 256, or 512\n";
    return failure();
  }

  func->setAttr("simp.vectorize_width", rewriter.getI64IntegerAttr(width));
  return success();
}

/// Apply prefetch annotation
/// Sets memory prefetch distance for cache optimization
LogicalResult applyPrefetchPattern(func::FuncOp func,
                                   const AnnotationInfo& info,
                                   IRRewriter& rewriter) {
  if (!info.hasPrefetch()) return success();

  LLVM_DEBUG(llvm::dbgs() << "Applying prefetch(" << info.prefetchDistance
                          << ") to " << func.getName() << "\n");

  func->setAttr("simp.prefetch_distance",
                rewriter.getI64IntegerAttr(info.prefetchDistance));
  return success();
}

/// Apply parallel annotation
/// Marks outer loops for OpenMP-style parallelization
LogicalResult applyParallelPattern(func::FuncOp func,
                                   const AnnotationInfo& info,
                                   IRRewriter& rewriter) {
  if (!info.hasParallel()) return success();

  LLVM_DEBUG(llvm::dbgs() << "Applying @parallel to " << func.getName() << "\n");

  func->setAttr("simp.parallel", rewriter.getUnitAttr());

  // Mark outermost SCF loops for parallelization
  bool firstLoop = true;
  func.walk([&](scf::ForOp forOp) {
    // Only mark the outermost loop(s)
    if (firstLoop) {
      forOp->setAttr("simp.parallel_loop", rewriter.getUnitAttr());
      firstLoop = false;
    }
  });

  return success();
}

/// Apply fuse annotation
/// Enables loop fusion for adjacent operations
LogicalResult applyFusePattern(func::FuncOp func,
                               const AnnotationInfo& info,
                               IRRewriter& rewriter) {
  if (!info.hasFuse()) return success();

  LLVM_DEBUG(llvm::dbgs() << "Applying @fuse to " << func.getName() << "\n");

  func->setAttr("simp.fuse", rewriter.getUnitAttr());
  return success();
}

/// Register all optimization patterns
void registerPatterns(PatternRegistry& registry) {
  // VNNI patterns (x86)
  registry.registerPattern("vnni.i8_matmul", applyMatmulTilingPattern);
  registry.registerPattern("vnni.i8_dot_product", applyMatmulTilingPattern);

  // Generic patterns
  registry.registerPattern("tile", applyMatmulTilingPattern);
  registry.registerPattern("scalar", applyScalarPattern);

  // New annotation patterns
  registry.registerPattern("unroll", applyUnrollPattern);
  registry.registerPattern("vectorize", applyVectorizePattern);
  registry.registerPattern("prefetch", applyPrefetchPattern);
  registry.registerPattern("parallel", applyParallelPattern);
  registry.registerPattern("fuse", applyFusePattern);
}

//===----------------------------------------------------------------------===//
// AnnotationLoweringPass - Main pass implementation
//===----------------------------------------------------------------------===//

class AnnotationLoweringPass
    : public PassWrapper<AnnotationLoweringPass, OperationPass<func::FuncOp>> {
public:
  AnnotationLoweringPass() { registerPatterns(registry); }

  StringRef getArgument() const override { return "simp-annotation-lowering"; }
  StringRef getDescription() const override {
    return "Process annotation attributes and apply optimization patterns";
  }

  // Declare that this pass may create OpenMP dialect operations
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<omp::OpenMPDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    IRRewriter rewriter(&getContext());

    // Find simp.annotated_region_* attributes
    SmallVector<std::pair<StringRef, DictionaryAttr>, 4> regions;
    for (NamedAttribute attr : func->getAttrs()) {
      if (attr.getName().getValue().starts_with("simp.annotated_region_")) {
        if (auto dict = dyn_cast<DictionaryAttr>(attr.getValue()))
          regions.push_back({attr.getName(), dict});
      }
    }

    if (regions.empty())
      return;

    // Process each annotated region
    for (const auto& [attrName, dictAttr] : regions) {
      AnnotationInfo info = extractAnnotationInfo(dictAttr);

      LLVM_DEBUG({
        llvm::dbgs() << "Processing annotations for " << func.getName() << ":\n";
        if (info.hasPattern())
          llvm::dbgs() << "  lower: " << info.lowerPattern << "\n";
        if (info.hasTileSizes())
          llvm::dbgs() << "  tile: [" << info.tileSizes[0] << ","
                       << info.tileSizes[1] << "," << info.tileSizes[2] << "]\n";
        if (info.hasUnroll())
          llvm::dbgs() << "  unroll: " << info.unrollFactor << "\n";
        if (info.hasVectorize())
          llvm::dbgs() << "  vectorize: " << info.vectorizeWidth << "\n";
        if (info.hasPrefetch())
          llvm::dbgs() << "  prefetch: " << info.prefetchDistance << "\n";
        if (info.hasParallel())
          llvm::dbgs() << "  parallel: true\n";
        if (info.hasFuse())
          llvm::dbgs() << "  fuse: true\n";
      });

      // Apply composable transforms in correct order
      applyComposableTransforms(func, info, rewriter);

      // Register for backend passes (VNNI, GPU, etc.)
      if (info.hasAnyOptimization()) {
        simp::AnnotationRegistry::instance().registerFunction(func.getName().str(), info);
      }

      func->removeAttr(attrName);
    }
  }

private:
  PatternRegistry registry;

  /// Extract AnnotationInfo from dictionary attribute
  AnnotationInfo extractAnnotationInfo(DictionaryAttr dict) {
    AnnotationInfo info;

    // Core annotations
    if (auto tiles = dict.getAs<ArrayAttr>("simp.tile_sizes")) {
      for (auto size : tiles) {
        if (auto intAttr = dyn_cast<IntegerAttr>(size))
          info.tileSizes.push_back(intAttr.getInt());
      }
    }

    if (auto align = dict.getAs<IntegerAttr>("simp.alignment"))
      info.alignment = align.getInt();

    if (auto pattern = dict.getAs<StringAttr>("simp.lower_pattern"))
      info.lowerPattern = pattern.getValue().str();

    // New annotations
    if (auto unroll = dict.getAs<IntegerAttr>("simp.unroll_factor"))
      info.unrollFactor = unroll.getInt();

    if (auto vec = dict.getAs<IntegerAttr>("simp.vectorize_width"))
      info.vectorizeWidth = vec.getInt();

    if (auto pf = dict.getAs<IntegerAttr>("simp.prefetch_distance"))
      info.prefetchDistance = pf.getInt();

    if (dict.get("simp.parallel"))
      info.parallel = true;

    if (dict.get("simp.fuse"))
      info.fuse = true;

    // Transform ordering (for composability)
    if (auto order = dict.getAs<ArrayAttr>("simp.transform_order")) {
      for (auto name : order) {
        if (auto strAttr = dyn_cast<StringAttr>(name))
          info.transformOrder.push_back(strAttr.getValue().str());
      }
    }

    return info;
  }

  /// Apply all transforms in composable order
  /// Order: parallel → tile → unroll → vectorize → prefetch → lower
  void applyComposableTransforms(func::FuncOp func, const AnnotationInfo& info,
                                 IRRewriter& rewriter) {
    // If explicit order specified, use it
    if (!info.transformOrder.empty()) {
      for (const auto& transform : info.transformOrder) {
        if (registry.hasPattern(transform)) {
          (void)registry.applyPattern(transform, func, info, rewriter);
        }
      }
      return;
    }

    // Default composable order: parallel → tile → unroll → vectorize → prefetch
    // This order ensures outer transforms don't interfere with inner ones

    // 1. Parallelization (outermost)
    if (info.hasParallel()) {
      (void)registry.applyPattern("parallel", func, info, rewriter);
    }

    // 2. Tiling (after parallel, before unroll)
    if (info.hasPattern() && registry.hasPattern(info.lowerPattern)) {
      (void)registry.applyPattern(info.lowerPattern, func, info, rewriter);
    } else if (info.hasTileSizes()) {
      (void)registry.applyPattern("tile", func, info, rewriter);
    }

    // 3. Unrolling (after tiling)
    if (info.hasUnroll()) {
      (void)registry.applyPattern("unroll", func, info, rewriter);
    }

    // 4. Vectorization hints
    if (info.hasVectorize()) {
      (void)registry.applyPattern("vectorize", func, info, rewriter);
    }

    // 5. Prefetch hints
    if (info.hasPrefetch()) {
      (void)registry.applyPattern("prefetch", func, info, rewriter);
    }

    // 6. Loop fusion (last, operates on transformed loops)
    if (info.hasFuse()) {
      (void)registry.applyPattern("fuse", func, info, rewriter);
    }
  }
};

} // namespace

namespace mlir {
namespace simp {

/// Create the annotation lowering pass
std::unique_ptr<Pass> createAnnotationLoweringPass() {
  return std::make_unique<AnnotationLoweringPass>();
}

/// Register the pass for command-line usage
void registerAnnotationLoweringPass() {
  PassRegistration<AnnotationLoweringPass>();
}

} // namespace simp
} // namespace mlir
