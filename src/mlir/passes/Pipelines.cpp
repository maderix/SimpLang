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
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
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
// Pipeline Builder: Annotation Processing (Phase 1.5)
//===----------------------------------------------------------------------===//

/// Build the annotation processing pipeline
/// Processes simp.annotated_region_* attributes and applies VNNI patterns
void buildAnnotationProcessingPipeline(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createAnnotationLoweringPass());
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
    pm.addNestedPass<func::FuncOp>(
        createSimpLinalgTilingPass(opts.tileSize, opts.hierarchical, opts.enableOpenMP));
  }

  // OpenMP parallelization
  if (opts.enableOpenMP) {
    pm.addPass(createConvertSCFToOpenMPPass());
    pm.addPass(createCanonicalizerPass());
  }

  // Vectorization - Strategy passes were removed in LLVM 15+
  // The tiling pass now handles vectorization preparation, and
  // vector lowering happens automatically during the conversion pipeline
  if (opts.enableVectorization) {
    // Apply vector lowering transformations
    pm.addNestedPass<func::FuncOp>(vector::createLowerVectorMultiReductionPass());
    pm.addNestedPass<func::FuncOp>(vector::createLowerVectorMaskPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }

  // Fallback: Lower remaining Linalg ops to loops
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());

  // Cleanup and optimization
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Loop optimizations
  pm.addPass(createLoopInvariantCodeMotionPass());
  pm.addNestedPass<func::FuncOp>(affine::createLoopUnrollPass());
}

//===----------------------------------------------------------------------===//
// Pipeline Builder: Buffer Management (Phase 2.5)
//===----------------------------------------------------------------------===//

/// Build the buffer management pipeline
/// Hoists allocations and inserts deallocations
void buildBufferManagementPipeline(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(bufferization::createBufferHoistingPass());
  pm.addNestedPass<func::FuncOp>(bufferization::createBufferLoopHoistingPass());
  // Use ownership-based deallocation (replaces old buffer deallocation pass)
  pm.addPass(bufferization::createOwnershipBasedBufferDeallocationPass());
  pm.addPass(bufferization::createLowerDeallocationsPass());
  pm.addPass(createConvertBufferizationToMemRefPass());
}

//===----------------------------------------------------------------------===//
// Pipeline Builder: LLVM Lowering (Phase 3)
//===----------------------------------------------------------------------===//

/// Build the LLVM dialect lowering pipeline
/// Lowers MemRef, Arith, SCF, Vector to LLVM dialect
void buildLLVMLoweringPipeline(OpPassManager &pm, bool enableOpenMP) {
  // Vector lowering
  pm.addNestedPass<func::FuncOp>(createConvertVectorToSCFPass());

  // Control flow lowering
  pm.addPass(createLowerAffinePass());
  pm.addPass(createSCFToControlFlowPass());

  // OpenMP lowering (if enabled)
  if (enableOpenMP) {
    pm.addPass(createConvertOpenMPToLLVMPass());
  }

  // Arithmetic expansion (maxf/minf → cmpf + select)
  pm.addNestedPass<func::FuncOp>(arith::createArithExpandOpsPass());

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

        // Phase 1.5: Annotation processing (VNNI patterns)
        buildAnnotationProcessingPipeline(pm);

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
