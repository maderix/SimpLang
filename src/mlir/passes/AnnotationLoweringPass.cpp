//===- AnnotationLoweringPass.cpp - Process optimization annotations ------===//
//
// Part of the SimpLang Project
//
// This pass processes simp.annotated_region_* attributes on functions and
// applies appropriate transformations. Uses simp::AnnotationInfo from
// AnnotationRegistry.h as the single source of truth for annotation data.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

  if (matmulOps.empty())
    return success();  // No matmul to transform

  // Configure and apply tiling
  LinalgTilingOptions tilingOptions;
  tilingOptions.setTileSizes(tileSizes);
  tilingOptions.setLoopType(LinalgTilingLoopType::Loops);

  for (LinalgOp op : matmulOps) {
    if (op->getParentOp() == nullptr) continue;

    rewriter.setInsertionPoint(op);
    FailureOr<TiledLinalgOp> tiledOp = tileLinalgOp(rewriter, op, tilingOptions);

    if (succeeded(tiledOp) && tiledOp->op) {
      // Mark as annotation-tiled (prevents double-tiling by default pass)
      tiledOp->op->setAttr("simp.annotation_tiled", rewriter.getUnitAttr());
      tiledOp->op->setAttr("simp.tile_sizes", rewriter.getI64ArrayAttr(tileSizes));
      rewriter.eraseOp(op);
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

/// Register all optimization patterns
void registerPatterns(PatternRegistry& registry) {
  // VNNI patterns (x86)
  registry.registerPattern("vnni.i8_matmul", applyMatmulTilingPattern);
  registry.registerPattern("vnni.i8_dot_product", applyMatmulTilingPattern);

  // Generic patterns
  registry.registerPattern("tile", applyMatmulTilingPattern);
  registry.registerPattern("scalar", applyScalarPattern);
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

      // Apply pattern if specified
      if (info.hasPattern()) {
        if (registry.hasPattern(info.lowerPattern)) {
          if (failed(registry.applyPattern(info.lowerPattern, func, info, rewriter))) {
            llvm::errs() << "Warning: Pattern '" << info.lowerPattern << "' failed\n";
          }
        } else {
          llvm::errs() << "Warning: Unknown pattern '" << info.lowerPattern << "'\n";
        }
      } else if (info.hasTileSizes()) {
        // Default tiling when no pattern specified
        (void)registry.applyPattern("tile", func, info, rewriter);
      }

      // Register for backend passes (VNNI, GPU, etc.)
      if (info.hasPattern() || info.hasTileSizes()) {
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

    return info;
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
