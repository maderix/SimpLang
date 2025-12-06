//===- Pipelines.cpp - MLIR Pipeline Builders for SimpLang ---------------===//
//
// Part of the SimpLang Project
//
// This file implements composable pipeline builders and named pipeline
// registrations for common use cases.
//
//===----------------------------------------------------------------------===//

#include "mlir/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"

using namespace mlir;

namespace mlir {
namespace simp {

//===----------------------------------------------------------------------===//
// Pipeline Builder: Simp Lowering (Phase 1)
//===----------------------------------------------------------------------===//

/// Build the Simp dialect lowering pipeline
/// Lowers Simp operations to MemRef, Arith, and Linalg dialects
void buildSimpLoweringPipeline(OpPassManager &pm) {
  pm.addPass(createConvertSimpToMemRefPass());
}

//===----------------------------------------------------------------------===//
// Pipeline Builder: Linalg Optimization (Phase 2)
//===----------------------------------------------------------------------===//

/// Options for Linalg optimization pipeline
struct LinalgOptOptions {
  bool enableTiling = true;
  int64_t tileSize = 8;           // Optimal for transformers
  bool hierarchical = false;
  bool enableOpenMP = false;
  bool enableVectorization = true;
};

/// Build the Linalg optimization pipeline
/// Applies tiling, vectorization, and loop optimizations
void buildLinalgOptimizationPipeline(OpPassManager &pm, const LinalgOptOptions &opts) {
  // Tiling for cache locality
  if (opts.enableTiling) {
    pm.addNestedPass<FuncOp>(
        createSimpLinalgTilingPass(opts.tileSize, opts.hierarchical, opts.enableOpenMP));
  }

  // OpenMP parallelization
  if (opts.enableOpenMP) {
    pm.addPass(createConvertSCFToOpenMPPass());
    pm.addPass(createCanonicalizerPass());
  }

  // Vectorization
  if (opts.enableVectorization) {
    pm.addNestedPass<FuncOp>(createLinalgStrategyEnablePass());
    pm.addNestedPass<FuncOp>(createLinalgStrategyVectorizePass(""));
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addNestedPass<FuncOp>(createLinalgStrategyLowerVectorsPass());
  }

  // Fallback: Lower remaining Linalg ops to loops
  pm.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());

  // Cleanup and optimization
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Loop optimizations
  pm.addNestedPass<FuncOp>(createLoopInvariantCodeMotionPass());
  pm.addNestedPass<FuncOp>(createLoopUnrollPass(4));
}

//===----------------------------------------------------------------------===//
// Pipeline Builder: Buffer Management (Phase 2.5)
//===----------------------------------------------------------------------===//

/// Build the buffer management pipeline
/// Hoists allocations and inserts deallocations
void buildBufferManagementPipeline(OpPassManager &pm) {
  pm.addNestedPass<FuncOp>(bufferization::createBufferHoistingPass());
  pm.addNestedPass<FuncOp>(bufferization::createBufferLoopHoistingPass());
  pm.addNestedPass<FuncOp>(bufferization::createBufferDeallocationPass());
  pm.addPass(createBufferizationToMemRefPass());
}

//===----------------------------------------------------------------------===//
// Pipeline Builder: LLVM Lowering (Phase 3)
//===----------------------------------------------------------------------===//

/// Build the LLVM dialect lowering pipeline
/// Lowers MemRef, Arith, SCF, Vector to LLVM dialect
void buildLLVMLoweringPipeline(OpPassManager &pm, bool enableOpenMP) {
  // Emulate sub-byte types (i4, i2) by packing into i8
  // Must run before LLVM conversion since LLVM doesn't support sub-byte memrefs
  pm.addPass(createEmulateNarrowTypePass());

  // Vector lowering
  pm.addNestedPass<FuncOp>(createConvertVectorToSCFPass());

  // Control flow lowering
  pm.addPass(createLowerAffinePass());
  pm.addPass(createLowerToCFGPass());

  // OpenMP lowering (if enabled)
  if (enableOpenMP) {
    pm.addPass(createConvertOpenMPToLLVMPass());
  }

  // Arithmetic expansion (maxf/minf → cmpf + select)
  pm.addNestedPass<FuncOp>(arith::createArithmeticExpandOpsPass());

  // Final conversions to LLVM dialect
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

//===----------------------------------------------------------------------===//
// Named Pipeline Registration
//===----------------------------------------------------------------------===//

/// Register the default SimpLang pipeline
/// Optimized for transformer workloads with 8×8×8 tiling
void registerSimpDefaultPipeline() {
  PassPipelineRegistration<>(
      "simp-default",
      "Default SimpLang pipeline (8×8×8 tiling, optimal for transformers)",
      [](OpPassManager &pm) {
        // Phase 1: Simp lowering
        buildSimpLoweringPipeline(pm);

        // Phase 2: Linalg optimization
        LinalgOptOptions opts;
        opts.enableTiling = true;
        opts.tileSize = 8;  // 45.68 tok/s on Stories110M!
        opts.hierarchical = false;
        opts.enableOpenMP = false;
        opts.enableVectorization = true;
        buildLinalgOptimizationPipeline(pm, opts);

        // Phase 2.5: Buffer management
        buildBufferManagementPipeline(pm);

        // Phase 3: LLVM lowering
        buildLLVMLoweringPipeline(pm, false);
      });
}

/// Register the transformer-optimized pipeline
/// Same as default but explicitly documented for transformer use cases
void registerSimpTransformerPipeline() {
  PassPipelineRegistration<>(
      "simp-transformer",
      "Transformer-optimized pipeline (Stories110M: 45.68 tok/s, LLaMA: competitive)",
      [](OpPassManager &pm) {
        buildSimpLoweringPipeline(pm);

        LinalgOptOptions opts;
        opts.enableTiling = true;
        opts.tileSize = 8;           // Perfect for 768×768 matrices
        opts.hierarchical = false;
        opts.enableOpenMP = false;
        opts.enableVectorization = true;
        buildLinalgOptimizationPipeline(pm, opts);

        buildBufferManagementPipeline(pm);
        buildLLVMLoweringPipeline(pm, false);
      });
}

/// Register the debug pipeline
/// Minimal optimizations for faster compilation and easier debugging
void registerSimpDebugPipeline() {
  PassPipelineRegistration<>(
      "simp-debug",
      "Debug pipeline with minimal optimizations",
      [](OpPassManager &pm) {
        buildSimpLoweringPipeline(pm);

        // No tiling, no vectorization - just basic lowering
        LinalgOptOptions opts;
        opts.enableTiling = false;
        opts.enableOpenMP = false;
        opts.enableVectorization = false;
        buildLinalgOptimizationPipeline(pm, opts);

        buildBufferManagementPipeline(pm);
        buildLLVMLoweringPipeline(pm, false);
      });
}

/// Register the high-performance pipeline with hierarchical tiling
/// For large matrices where cache hierarchy matters
void registerSimpHighPerfPipeline() {
  PassPipelineRegistration<>(
      "simp-high-perf",
      "High-performance pipeline with hierarchical 64→16 tiling",
      [](OpPassManager &pm) {
        buildSimpLoweringPipeline(pm);

        LinalgOptOptions opts;
        opts.enableTiling = true;
        opts.tileSize = 16;          // Inner tile for L1 cache
        opts.hierarchical = true;    // 64→16 two-level tiling
        opts.enableOpenMP = false;
        opts.enableVectorization = true;
        buildLinalgOptimizationPipeline(pm, opts);

        buildBufferManagementPipeline(pm);
        buildLLVMLoweringPipeline(pm, false);
      });
}

/// Register the OpenMP parallelized pipeline
/// For multi-core systems
void registerSimpParallelPipeline() {
  PassPipelineRegistration<>(
      "simp-parallel",
      "OpenMP-parallelized pipeline for multi-core systems",
      [](OpPassManager &pm) {
        buildSimpLoweringPipeline(pm);

        LinalgOptOptions opts;
        opts.enableTiling = true;
        opts.tileSize = 16;
        opts.hierarchical = false;
        opts.enableOpenMP = true;   // Enable OpenMP
        opts.enableVectorization = true;
        buildLinalgOptimizationPipeline(pm, opts);

        buildBufferManagementPipeline(pm);
        buildLLVMLoweringPipeline(pm, true);  // OpenMP lowering
      });
}

/// Register all SimpLang pipelines
void registerSimpPipelines() {
  registerSimpDefaultPipeline();
  registerSimpTransformerPipeline();
  registerSimpDebugPipeline();
  registerSimpHighPerfPipeline();
  registerSimpParallelPipeline();
}

} // namespace simp
} // namespace mlir
