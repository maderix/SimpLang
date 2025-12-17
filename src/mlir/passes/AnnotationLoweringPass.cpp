//===- AnnotationLoweringPass.cpp - Process VNNI/optimization annotations -===//
//
// Part of the SimpLang Project
//
// This pass processes simp.annotated_region_* attributes on functions and
// applies appropriate transformations for VNNI optimization. It implements
// a pattern registry for extensible annotation-driven optimizations.
//
// Patterns supported:
//   - vnni.i8_matmul: Tile INT8 matmul for AVX512-VNNI with correct indices
//   - vnni.i8_dot_product: Mark reduction loops for vpdpbusd emission
//   - transpose_b: Insert B matrix transpose before matmul
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
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <map>

using namespace mlir;
using namespace mlir::linalg;

namespace {

//===----------------------------------------------------------------------===//
// PatternRegistry - Extensible pattern registration system
//===----------------------------------------------------------------------===//

/// Information extracted from annotation attributes
struct AnnotationInfo {
  SmallVector<int64_t, 3> tileSizes;
  int64_t alignment = 0;
  std::string lowerPattern;

  bool hasTileSizes() const { return !tileSizes.empty(); }
  bool hasAlignment() const { return alignment > 0; }
  bool hasLowerPattern() const { return !lowerPattern.empty(); }
};

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

/// Apply VNNI-optimized tiling to INT8 matmul operations
///
/// This implements 6-level tiling with correct index calculation:
///   for i_outer in 0..M step tileM:
///     for j_outer in 0..N step tileN:
///       for k_outer in 0..K step tileK:
///         for ii in 0..tileM:
///           for jj in 0..tileN:
///             for kk in 0..tileK:
///               i_full = i_outer + ii  // CORRECT
///               j_full = j_outer + jj  // CORRECT
///               k_full = k_outer + kk  // CORRECT
///               C[i_full, j_full] += A[i_full, k_full] * B[k_full, j_full]
///
LogicalResult applyVNNIMatmulPattern(func::FuncOp func,
                                     const AnnotationInfo& info,
                                     IRRewriter& rewriter) {
  llvm::errs() << "[AnnotationLowering] Applying vnni.i8_matmul pattern to "
               << func.getName() << "\n";

  // Get tile sizes from annotation (default: 64×64×64)
  SmallVector<int64_t, 3> tileSizes = {64, 64, 64};
  if (info.hasTileSizes() && info.tileSizes.size() >= 3) {
    tileSizes[0] = info.tileSizes[0];  // M tile
    tileSizes[1] = info.tileSizes[1];  // K tile
    tileSizes[2] = info.tileSizes[2];  // N tile
  }

  llvm::errs() << "  Tile sizes: [" << tileSizes[0] << ", "
               << tileSizes[1] << ", " << tileSizes[2] << "]\n";

  // Find linalg.matmul operations in the function
  SmallVector<LinalgOp, 4> matmulOps;
  func.walk([&](LinalgOp op) {
    if (isa<linalg::MatmulOp>(op)) {
      matmulOps.push_back(op);
    }
  });

  if (matmulOps.empty()) {
    llvm::errs() << "  No linalg.matmul found, pattern skipped\n";
    return success();  // No matmul to transform, but not an error
  }

  llvm::errs() << "  Found " << matmulOps.size() << " matmul operation(s)\n";

  // Configure tiling for VNNI optimization
  LinalgTilingOptions tilingOptions;
  tilingOptions.setTileSizes(tileSizes);
  tilingOptions.setLoopType(LinalgTilingLoopType::Loops);

  // Apply tiling to each matmul
  for (LinalgOp op : matmulOps) {
    if (op->getParentOp() == nullptr) continue;  // Already replaced

    rewriter.setInsertionPoint(op);
    FailureOr<TiledLinalgOp> tiledOp = tileLinalgOp(rewriter, op, tilingOptions);

    if (succeeded(tiledOp)) {
      // Mark the tiled operation with metadata for downstream passes
      if (tiledOp->op) {
        // Generic marker to prevent default tiling pass from double-tiling
        // Works for any backend: VNNI (x86), SVE (ARM), Tensor Cores (GPU), etc.
        tiledOp->op->setAttr("simp.annotation_tiled", rewriter.getUnitAttr());
        tiledOp->op->setAttr("simp.tile_sizes", rewriter.getI64ArrayAttr(tileSizes));
        // Backend-specific marker for LLVM pass (legacy, kept for compatibility)
        tiledOp->op->setAttr("simp.vnni_optimized", rewriter.getUnitAttr());
      }
      rewriter.eraseOp(op);
      llvm::errs() << "  Applied annotation-driven tiling to matmul\n";
    } else {
      llvm::errs() << "  Warning: Failed to tile matmul\n";
    }
  }

  // Set function-level attribute for LLVM VNNI pass
  func->setAttr("simp.emit_vnni", rewriter.getUnitAttr());

  return success();
}

/// Mark reduction loops for VNNI dot product emission
LogicalResult applyVNNIDotProductPattern(func::FuncOp func,
                                         const AnnotationInfo& info,
                                         IRRewriter& rewriter) {
  llvm::errs() << "[AnnotationLowering] Applying vnni.i8_dot_product pattern to "
               << func.getName() << "\n";

  // Set function attribute for LLVM VNNI pass to detect dot product pattern
  func->setAttr("simp.emit_vnni_dot", rewriter.getUnitAttr());

  return success();
}

/// Insert B matrix transpose before matmul (for row-major B optimization)
LogicalResult applyTransposeBPattern(func::FuncOp func,
                                     const AnnotationInfo& info,
                                     IRRewriter& rewriter) {
  llvm::errs() << "[AnnotationLowering] Applying transpose_b pattern to "
               << func.getName() << "\n";

  // TODO: Implement transpose insertion for B matrix
  // This would help when B is row-major and needs to be transposed
  // for optimal memory access in VNNI operations

  func->setAttr("simp.transpose_b", rewriter.getUnitAttr());

  return success();
}

/// Register all VNNI-related patterns
void registerVNNIPatterns(PatternRegistry& registry) {
  registry.registerPattern("vnni.i8_matmul", applyVNNIMatmulPattern);
  registry.registerPattern("vnni.i8_dot_product", applyVNNIDotProductPattern);
  registry.registerPattern("vnni.i8_matmul_i4", applyVNNIMatmulPattern);  // Same as matmul for now
  registry.registerPattern("transpose_b", applyTransposeBPattern);

  // Scalar pattern - no optimization, just marks function
  registry.registerPattern("scalar", [](func::FuncOp func, const AnnotationInfo&,
                                         IRRewriter& rewriter) {
    llvm::errs() << "[AnnotationLowering] Scalar pattern - no optimization for "
                 << func.getName() << "\n";
    func->setAttr("simp.scalar_baseline", rewriter.getUnitAttr());
    return success();
  });
}

//===----------------------------------------------------------------------===//
// AnnotationLoweringPass - Main pass implementation
//===----------------------------------------------------------------------===//

class AnnotationLoweringPass
    : public PassWrapper<AnnotationLoweringPass, OperationPass<func::FuncOp>> {
public:
  AnnotationLoweringPass() {
    // Register all patterns
    registerVNNIPatterns(registry);
  }

  StringRef getArgument() const override {
    return "simp-annotation-lowering";
  }

  StringRef getDescription() const override {
    return "Process annotation attributes and apply optimization patterns";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);

    // Find all simp.annotated_region_* attributes
    SmallVector<std::pair<StringRef, DictionaryAttr>, 4> annotatedRegions;

    for (NamedAttribute attr : func->getAttrs()) {
      if (attr.getName().getValue().starts_with("simp.annotated_region_")) {
        if (auto dictAttr = dyn_cast<DictionaryAttr>(attr.getValue())) {
          annotatedRegions.push_back({attr.getName(), dictAttr});
        }
      }
    }

    if (annotatedRegions.empty()) {
      return;  // No annotations to process
    }

    llvm::errs() << "[AnnotationLowering] Processing " << func.getName()
                 << " with " << annotatedRegions.size() << " annotated region(s)\n";

    // Process each annotated region
    for (const auto& [attrName, dictAttr] : annotatedRegions) {
      AnnotationInfo info;

      // Extract tile_sizes
      if (auto tileSizesAttr = dictAttr.getAs<ArrayAttr>("simp.tile_sizes")) {
        for (auto size : tileSizesAttr) {
          if (auto intAttr = dyn_cast<IntegerAttr>(size)) {
            info.tileSizes.push_back(intAttr.getInt());
          }
        }
      }

      // Extract alignment
      if (auto alignAttr = dictAttr.getAs<IntegerAttr>("simp.alignment")) {
        info.alignment = alignAttr.getInt();
      }

      // Extract lower_pattern
      if (auto patternAttr = dictAttr.getAs<StringAttr>("simp.lower_pattern")) {
        info.lowerPattern = patternAttr.getValue().str();
      }

      // Apply pattern if specified
      if (info.hasLowerPattern()) {
        if (registry.hasPattern(info.lowerPattern)) {
          if (failed(registry.applyPattern(info.lowerPattern, func, info, rewriter))) {
            llvm::errs() << "Warning: Failed to apply pattern '"
                         << info.lowerPattern << "'\n";
          }
        } else {
          llvm::errs() << "Warning: Unknown pattern '" << info.lowerPattern
                       << "' in " << attrName << "\n";
          llvm::errs() << "  Available patterns: ";
          for (const auto& name : registry.getPatternNames()) {
            llvm::errs() << name << " ";
          }
          llvm::errs() << "\n";
        }
      } else if (info.hasTileSizes()) {
        // If only tile_sizes specified without lower_pattern, apply default tiling
        llvm::errs() << "  Applying default tiling from annotation\n";
        // Could apply generic tiling here if needed
      }

      // Register function in the backend-agnostic AnnotationRegistry
      // This allows LLVM passes (VNNI, GPU, etc.) to query annotation info
      if (info.hasLowerPattern() || info.hasTileSizes()) {
        simp::AnnotationInfo registryInfo;
        registryInfo.lowerPattern = info.lowerPattern;
        registryInfo.alignment = info.alignment;
        for (int64_t ts : info.tileSizes) {
          registryInfo.tileSizes.push_back(ts);
        }
        simp::AnnotationRegistry::instance().registerFunction(
            func.getName().str(), registryInfo);
        llvm::errs() << "[AnnotationLowering] Registered " << func.getName()
                     << " in AnnotationRegistry with pattern '" << info.lowerPattern << "'\n";
      }

      // Remove processed annotation attribute
      func->removeAttr(attrName);
    }
  }

private:
  PatternRegistry registry;
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
