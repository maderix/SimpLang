//===- mlir_pipeline.cpp - MLIR Compilation Pipeline Implementation ------===//
//
// Part of the SimpLang Project
//
// This file implements the MLIRCompilationPipeline class for progressive
// lowering of MLIR modules to LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/mlir_pipeline.hpp"
#include "mlir/Passes.h"
#include "mlir/simp_dialect.hpp"
#include "mlir/cache_info.hpp"

// MLIR Pass Infrastructure
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

// MLIR Dialects
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"  // For LinalgTilingLoopType
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"  // OpenMP dialect
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// MLIR Conversion Passes
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"  // OpenMP to LLVM conversion
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"  // SCF to OpenMP conversion
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/Passes.h"   // For createConvertFuncToLLVMPass, createConvertArithToLLVMPass, etc.
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/DialectConversion.h"

// MLIR to LLVM IR Translation
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

// MLIR Verification
#include "mlir/IR/Verifier.h"

// LLVM
#include "llvm/IR/Module.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::simp;

//===----------------------------------------------------------------------===//
// Constructor
//===----------------------------------------------------------------------===//

MLIRCompilationPipeline::MLIRCompilationPipeline(mlir::ModuleOp module)
    : module(module) {
  if (!module) {
    llvm::errs() << "Error: MLIRCompilationPipeline initialized with null module\n";
  }
}

//===----------------------------------------------------------------------===//
// Pipeline Execution
//===----------------------------------------------------------------------===//

bool MLIRCompilationPipeline::runPasses() {
  if (!module) {
    llvm::errs() << "Error: Cannot run passes on null module\n";
    return false;
  }

  // Disable multi-threading if IR dumping is enabled (required by MLIR)
  if (dumpIntermediateIR) {
    module.getContext()->disableMultithreading();
  }

  // Phase 1: Lower Simp dialect to MemRef + Arith + Linalg
  {
    mlir::PassManager pm(module.getContext());
    if (dumpIntermediateIR) pm.enableIRPrinting();
    buildPhase1_SimpLowering(pm);

    if (failed(pm.run(module))) {
      llvm::errs() << "Error: Phase 1 (Simp lowering) failed\n";
      module.dump();
      return false;
    }

    if (dumpIntermediateIR && !outputPath.empty()) {
      std::string dumpPath = outputPath + "_phase1_simp_lowering.mlir";
      std::error_code EC;
      llvm::raw_fd_ostream out(dumpPath, EC);
      if (!EC) {
        module.print(out);
        llvm::outs() << "Phase 1 dump: " << dumpPath << "\n";
      }
    }
  }

  // Phase 2: Linalg optimization (tiling, vectorization, loop lowering)
  // This phase should create scf.for loops and insert memref.prefetch
  {
    mlir::PassManager pm(module.getContext());
    if (dumpIntermediateIR) pm.enableIRPrinting();
    buildPhase2_LinalgOptimization(pm);

    if (failed(pm.run(module))) {
      llvm::errs() << "Error: Phase 2 (Linalg optimization) failed\n";
      module.dump();
      return false;
    }

    if (dumpIntermediateIR && !outputPath.empty()) {
      std::string dumpPath = outputPath + "_phase2_with_prefetch.mlir";
      std::error_code EC;
      llvm::raw_fd_ostream out(dumpPath, EC);
      if (!EC) {
        module.print(out);
        llvm::outs() << "Phase 2 dump (should have scf.for + memref.prefetch): " << dumpPath << "\n";
      }
    }
  }

  // Phase 2.5: Buffer management (hoisting, deallocation)
  {
    mlir::PassManager pm(module.getContext());
    if (dumpIntermediateIR) pm.enableIRPrinting();
    buildPhase2_5_BufferManagement(pm);

    if (failed(pm.run(module))) {
      llvm::errs() << "Error: Phase 2.5 (Buffer management) failed\n";
      module.dump();
      return false;
    }

    if (dumpIntermediateIR && !outputPath.empty()) {
      std::string dumpPath = outputPath + "_phase2_5_buffer_mgmt.mlir";
      std::error_code EC;
      llvm::raw_fd_ostream out(dumpPath, EC);
      if (!EC) {
        module.print(out);
        llvm::outs() << "Phase 2.5 dump: " << dumpPath << "\n";
      }
    }
  }

  // Phase 3: Lower to LLVM dialect (vector, control flow, arithmetic)
  {
    mlir::PassManager pm(module.getContext());
    if (dumpIntermediateIR) pm.enableIRPrinting();
    buildPhase3_LLVMDialectLowering(pm);

    if (failed(pm.run(module))) {
      llvm::errs() << "Error: Phase 3 (LLVM dialect lowering) failed\n";
      module.dump();
      return false;
    }

    if (dumpIntermediateIR && !outputPath.empty()) {
      std::string dumpPath = outputPath + "_phase3_llvm_dialect.mlir";
      std::error_code EC;
      llvm::raw_fd_ostream out(dumpPath, EC);
      if (!EC) {
        module.print(out);
        llvm::outs() << "Phase 3 dump (LLVM dialect): " << dumpPath << "\n";
      }
    }
  }

  // Apply vector lowering patterns and LLVM dialect conversion
  // This must be done AFTER the pass pipeline since it requires direct module access
  if (!applyLLVMDialectConversion()) {
    llvm::errs() << "Error: LLVM dialect conversion failed\n";
    return false;
  }

  // Try to reconcile unrealized casts after pattern conversion
  // With OpenMP, some casts inside omp regions will remain (expected behavior)
  {
    mlir::PassManager reconcilePM(module.getContext());
    reconcilePM.addPass(mlir::createReconcileUnrealizedCastsPass());
    // Ignore failures - with OpenMP some casts can't be reconciled
    (void)reconcilePM.run(module);
  }

  // Verify the final module
  // NOTE: When OpenMP is enabled, unrealized_conversion_cast ops are expected
  // (see mlir/test/Conversion/OpenMPToLLVM/convert-to-llvmir.mlir)
  if (!enableOpenMP && failed(mlir::verify(module))) {
    llvm::errs() << "Error: Module verification failed after lowering\n";
    module.dump();
    return false;
  }

  return true;
}

std::unique_ptr<llvm::Module> MLIRCompilationPipeline::translateToLLVMIR(
    llvm::LLVMContext& llvmContext) {
  if (!module) {
    llvm::errs() << "Error: Cannot translate null module\n";
    return nullptr;
  }

  // LLVM 21: Register dialect translations for LLVM IR generation
  mlir::DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  if (enableOpenMP) {
    mlir::registerOpenMPDialectTranslation(registry);
  }
  module.getContext()->appendDialectRegistry(registry);

  // Translate MLIR LLVM dialect to LLVM IR
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Error: Failed to translate MLIR to LLVM IR\n";
    return nullptr;
  }

  // Add debug information if enabled
  if (enableDebugInfo) {
    addDebugInfoToModule(llvmModule.get());
  }

  return llvmModule;
}

//===----------------------------------------------------------------------===//
// Phase 1: Simp → MemRef + Arith + Linalg
//===----------------------------------------------------------------------===//

void MLIRCompilationPipeline::buildPhase1_SimpLowering(mlir::OpPassManager& pm) {
  // Add our custom Simp → MemRef + Arith + Linalg lowering pass
  pm.addPass(mlir::simp::createConvertSimpToMemRefPass());
}

//===----------------------------------------------------------------------===//
// Phase 2: Linalg Vectorization and Loop Lowering
//===----------------------------------------------------------------------===//

void MLIRCompilationPipeline::buildPhase2_LinalgOptimization(mlir::OpPassManager& pm) {

  // OPTIONAL: Tile matmul BEFORE vectorization (enabled by default)
  // Vectorizing large matrices (768x768) directly causes 1.6GB memory usage and hangs
  // Tiling first breaks the matrix into small chunks that vectorize efficiently
  if (enableTiling) {
    // Choose loop type: scf.parallel for OpenMP, scf.for for sequential
    linalg::LinalgTilingLoopType loopType = enableOpenMP
        ? linalg::LinalgTilingLoopType::ParallelLoops  // scf.parallel → OpenMP
        : linalg::LinalgTilingLoopType::Loops;         // scf.for (sequential)

    if (enableOpenMP) {
      llvm::outs() << "[OpenMP] Generating parallel loops for multi-threading\n";
    }

    // SMART OPENMP: Force hierarchical tiling when OpenMP is enabled
    // This avoids creating parallel regions for every tiny tile (massive overhead)
    // Instead: parallelize coarse outer tiles, keep inner tiles sequential
    bool useHierarchicalTiling = enableHierarchicalTiling || enableOpenMP;

    if (useHierarchicalTiling) {
      // HIERARCHICAL TILING: Two-level cache-aware tiling
      // Query actual CPU cache sizes
      CacheInfo cache = CacheInfo::query();

      // TWO-LEVEL TILING to reduce code explosion:
      // - Outer tile: L2/L3 cache (16-32)
      // - Inner tile: L1 cache + vectorization (4-8)
      //
      // This avoids the exponential code explosion of 3-level tiling
      // while still providing cache benefits.

      int element_size = 4;  // f32
      int l1_tile = 8, l2_tile = 32, l3_tile = 128;
      cache.computeMatmulTileSizes(element_size, l1_tile, l2_tile, l3_tile);

      // CRITICAL: Ensure inner_tile_count ≥ 4 to amortize loop overhead
      // 64→16 gives 4×4×4=64 inner iterations (perfect balance)
      // 24→8 gives 3×3×3=27 inner iterations (too much overhead!)
      int outer_tile, inner_tile;

      if (enableOpenMP) {
        // OpenMP AUTO-TUNING: Compute tile size based on target work per thread
        // Goal: Each parallel tile should have enough work to amortize ~50μs thread overhead
        //
        // Heuristic: Target ~2000 FLOPS minimum per thread (50μs @ 40 GFLOPS = 2M FLOPS)
        // For matmul: FLOPS ≈ 2 * tile³, so tile ≥ ∛(1M) ≈ 100
        //
        // Conservative: Use L2/L3 cache size as upper bound (avoid thrashing)
        int min_tile_for_overhead = 128;  // ~4M FLOPS per tile
        int max_tile_for_cache = std::min((int)l3_tile, 256);  // Cache-friendly upper bound

        outer_tile = std::min(min_tile_for_overhead, max_tile_for_cache);
        inner_tile = 8;  // Optimal for vectorization + L1 cache

        llvm::outs() << "[OpenMP Auto-tuned] Outer (parallel): " << outer_tile
                     << "×" << outer_tile << ", Inner (sequential): " << inner_tile << "×" << inner_tile
                     << " [min_work=" << min_tile_for_overhead << ", max_cache=" << max_tile_for_cache << "]\n";

        llvm::outs() << "\n[OpenMP WARNING] Multi-threading may hurt performance for memory-bound workloads:\n"
                     << "  - Cache coherence overhead can cause 10-25× more cache misses\n"
                     << "  - Beneficial for: batch processing, large matrices (>2048), CPU-bound tasks\n"
                     << "  - May degrade: single-sequence inference, memory-bound matmuls\n"
                     << "  - Measured: 2-3× slower for Stories110M single-sequence (cache miss rate: 2% → 25%)\n"
                     << "  Use OMP_NUM_THREADS to control thread count. Benchmark your specific workload.\n\n";
      } else {
        // Regular hierarchical: balanced for cache hierarchy
        // Use 32×32 outer tiles for 768×768 matrices (fits better in L2)
        outer_tile = 32;   // L2 cache blocking (4KB per tile)
        inner_tile = 16;   // L1 cache + vectorization
        llvm::outs() << "[Hierarchical Tiling] Outer: " << outer_tile
                     << "×" << outer_tile << "×" << outer_tile
                     << ", Inner: " << inner_tile << "×" << inner_tile << "×" << inner_tile << "\n";
      }

      // Level 1: Outer tiling (L2/L3 cache aware)
      // In LLVM 21+, use custom SimpLinalgTilingPass
      pm.addNestedPass<mlir::func::FuncOp>(
          mlir::simp::createSimpLinalgTilingPass(outer_tile, true, enableOpenMP));
      if (!enableDebugInfo) pm.addPass(mlir::createCanonicalizerPass());

      // Level 2: Inner tiling is handled by hierarchical flag in custom pass
      if (!enableDebugInfo) pm.addPass(mlir::createCanonicalizerPass());
    } else {
      // SINGLE-LEVEL TILING: Use configured tile size
      // For matmul(MxK, KxN, MxN), we tile all 3 dimensions
      pm.addNestedPass<mlir::func::FuncOp>(
          mlir::simp::createSimpLinalgTilingPass(tileSize, false, enableOpenMP));

      // Canonicalize after tiling (skip in debug mode to preserve operations)
      if (!enableDebugInfo) pm.addPass(mlir::createCanonicalizerPass());
    }
  }

  // PARALLELIZATION: Convert scf.parallel to OpenMP if enabled
  if (enableOpenMP) {
    llvm::outs() << "[OpenMP] Converting scf.parallel to omp.parallel\n";
    pm.addPass(mlir::createConvertSCFToOpenMPPass());
    if (!enableDebugInfo) pm.addPass(mlir::createCanonicalizerPass());
  }

  // VECTORIZATION: In LLVM 21+, strategy passes were removed
  // Use vector lowering passes instead for progressive vector lowering

  // Vector multi-reduction and mask lowering (LLVM 21+ API)
  pm.addNestedPass<mlir::func::FuncOp>(mlir::vector::createLowerVectorMultiReductionPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::vector::createLowerVectorMaskPass());

  // Canonicalize after vectorization (skip in debug mode)
  if (!enableDebugInfo) {
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
  }

  // Lower remaining Linalg ops (non-vectorized) to SCF loops
  // Only operations that couldn't be vectorized will go through this
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());

  // Apply general canonicalization and CSE (skip in debug mode)
  if (!enableDebugInfo) {
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
  }

  // INSERT PREFETCH: Add prefetch operations to hide memory latency
  // NOTE: Prefetch is disabled when using LLVM vectorization path because
  // InsertPrefetchPass can generate invalid IR with certain loop structures.
  if (enablePrefetch && !enableDebugInfo && !skipMLIRVectorization) {
    llvm::outs() << "[Prefetch] Inserting prefetch operations into loops\n";
    pm.addNestedPass<mlir::func::FuncOp>(mlir::simp::createInsertPrefetchPass());
    pm.addPass(mlir::createCanonicalizerPass());
  }

  // OPTIMIZATION: Skip optimizations in debug mode
  if (!enableDebugInfo) {
    // Apply loop invariant code motion to hoist invariant operations
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createLoopInvariantCodeMotionPass());

    // Apply loop unrolling for better ILP
    pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createLoopUnrollPass(/*unrollFactor=*/4));
  }
}

//===----------------------------------------------------------------------===//
// Phase 2.5: Insert Buffer Deallocations
//===----------------------------------------------------------------------===//

void MLIRCompilationPipeline::buildPhase2_5_BufferManagement(mlir::OpPassManager& pm) {
  // First, hoist buffer allocations out of loops to avoid repeated allocation
  pm.addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createBufferHoistingPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createBufferLoopHoistingPass());

  // LLVM 21+: Use ownership-based buffer deallocation (replaces createBufferDeallocationPass)
  pm.addPass(mlir::bufferization::createOwnershipBasedBufferDeallocationPass());
  pm.addPass(mlir::bufferization::createLowerDeallocationsPass());

  // Convert bufferization ops (like bufferization.clone) to memref ops
  pm.addPass(mlir::createConvertBufferizationToMemRefPass());
}

//===----------------------------------------------------------------------===//
// Phase 3: MemRef + Arith + SCF + Standard → LLVM Dialect
//===----------------------------------------------------------------------===//

void MLIRCompilationPipeline::buildPhase3_LLVMDialectLowering(mlir::OpPassManager& pm) {
  // Follow the Toy Chapter 6 pattern for lowering to LLVM dialect
  // This uses pattern-based conversion with full dialect conversion

  // 1. First, lower Vector transfer ops to simpler vector operations
  // Lower vector.transfer_read/write to simpler vector operations
  // This converts high-level vector ops to operations LLVM can handle
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertVectorToSCFPass());

  // 1.5. CRITICAL: Expand strided metadata (subview, reinterpret_cast, etc.)
  // This MUST run BEFORE LowerAffinePass because it creates affine.apply ops
  // SubViewOp must be expanded before LLVM conversion
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());

  // 2. Lower Affine and SCF to Standard dialect control flow
  // Lower Affine ops (affine.min, affine.apply, etc.) to Standard
  pm.addPass(mlir::createLowerAffinePass());

  // Always lower SCF to CF, even when OpenMP is enabled
  // The SCF-to-OpenMP pass only converts scf.parallel, leaving other SCF ops
  // (scf.while, scf.for, scf.if) that need to be converted to CF
  pm.addPass(mlir::createSCFToControlFlowPass());

  // CRITICAL: Run createConvertOpenMPToLLVMPass() AFTER createLowerToCFGPass()!
  // This fixes index/i64 type mismatches in branches inside OpenMP regions
  if (enableOpenMP) {
    llvm::outs() << "[OpenMP] Converting OpenMP index types to i64\n";
    pm.addPass(mlir::createConvertOpenMPToLLVMPass());
  }

  // 2.5. Expand Arithmetic ops (maxf, minf) to operations that can be lowered to LLVM
  //      MaxFOp/MinFOp don't have direct LLVM lowering - they must be expanded to CmpF + Select
  pm.addNestedPass<mlir::func::FuncOp>(mlir::arith::createArithExpandOpsPass());

  // 3. Lower MemRef ops to LLVM (including subview, cast, etc.)
  // This must happen before the pattern-based final conversion
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());

  // 4. Lower Arith ops to LLVM
  pm.addPass(mlir::createArithToLLVMConversionPass());

  // 5. Lower ControlFlow ops to LLVM
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());

  // 6. Lower Func ops to LLVM
  pm.addPass(mlir::createConvertFuncToLLVMPass());

  // 6.5. Lower Index ops to LLVM (handles index type conversions)
  pm.addPass(mlir::createConvertIndexToLLVMPass());

  // 7. Add canonicalization and CSE passes for cleanup (skip in debug mode)
  if (!enableDebugInfo) {
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
  }

  // 8. Final reconciliation of unrealized casts
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}

//===----------------------------------------------------------------------===//
// Helper: Apply Vector Lowering Patterns and LLVM Dialect Conversion
//===----------------------------------------------------------------------===//

bool MLIRCompilationPipeline::applyLLVMDialectConversion() {
  // This method applies pattern-based conversions that require direct module access
  // It must be called AFTER the main pass pipeline has run

  // Skip vector optimization patterns in debug mode to preserve variable mappings
  if (!enableDebugInfo) {
    // Step 1: Lower vector operations to simpler forms BEFORE LLVM conversion
    // LLVM 21+: Many vector lowering patterns have been reorganized
    {
      mlir::RewritePatternSet patterns(module.getContext());

      // Use available canonicalization patterns
      mlir::vector::populateVectorToVectorCanonicalizationPatterns(patterns);
      // Vector transfer ops - use full/partial patterns with options
      mlir::vector::VectorTransformsOptions vectorOptions;
      mlir::vector::populateVectorTransferFullPartialPatterns(patterns, vectorOptions);

      if (failed(mlir::applyPatternsGreedily(module, std::move(patterns)))) {
        llvm::errs() << "Warning: Vector lowering patterns failed\n";
      }
    }

    // Progressive lowering of vector operations
    // Run iteratively to handle nested operations
    for (int iteration = 0; iteration < 5; ++iteration) {
      mlir::RewritePatternSet patterns(module.getContext());
      mlir::vector::VectorTransformsOptions vectorOptions;
      mlir::vector::populateVectorToVectorCanonicalizationPatterns(patterns);
      mlir::vector::populateVectorTransferFullPartialPatterns(patterns, vectorOptions);

      if (failed(mlir::applyPatternsGreedily(module, std::move(patterns)))) {
        llvm::errs() << "Warning: Vector lowering iteration " << iteration << " had issues\n";
      }
    }
  } // end if (!enableDebugInfo)

  // Step 2: Define the conversion target - only LLVM dialect operations are legal
  // DON'T run createConvertVectorToLLVMPass as a separate pass
  // This creates unrealized_conversion_cast that applyFullConversion marks as illegal
  // Instead, we add vector conversion patterns to the main pattern set below

  // Define the conversion target - only LLVM dialect operations are legal
  mlir::LLVMConversionTarget target(*module.getContext());
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalOp<mlir::UnrealizedConversionCastOp>();  // Allow temporary casts

  // OpenMP operations are NOT converted to LLVM dialect - they are translated
  // directly to LLVM IR during the MLIR-to-LLVMIR translation phase
  if (enableOpenMP) {
    target.addLegalDialect<mlir::omp::OpenMPDialect>();
  }

  // Type converter for lowering types to LLVM types
  mlir::LLVMTypeConverter typeConverter(module.getContext());

  // Populate conversion patterns from built-in passes
  mlir::RewritePatternSet patterns(module.getContext());

  // Add patterns for Arith → LLVM
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);

  // Add patterns for Math → LLVM
  mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);

  // Add patterns for MemRef → LLVM
  mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);

  // NOTE: OpenMP operations remain as omp.* and are translated to LLVM IR
  // during the mlir-translate phase, NOT during LLVM dialect lowering

  // Add patterns for Vector → LLVM (CRITICAL for vectorization)
  // Match LLVM's ConvertVectorToLLVMPass
  mlir::vector::VectorTransformsOptions vectorOpts;
  mlir::vector::populateVectorMaskMaterializationPatterns(patterns, /*indexOptimizations=*/false);
  mlir::vector::populateVectorTransferFullPartialPatterns(patterns, vectorOpts);
  mlir::populateVectorToLLVMMatrixConversionPatterns(typeConverter, patterns);
  mlir::populateVectorToLLVMConversionPatterns(typeConverter, patterns);

  // Add patterns for Func → LLVM (converts func.func, func.call, func.return)
  mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);

  // Add patterns for ControlFlow → LLVM (converts cf.br, cf.cond_br)
  mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

  // Add index to LLVM conversion patterns
  mlir::index::populateIndexToLLVMConversionPatterns(typeConverter, patterns);

  // Use partial conversion (allows unrealized casts temporarily)
  if (failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
    llvm::errs() << "Error: Failed to convert to LLVM dialect\n";
    module.dump();
    return false;
  }

  // Run reconcile pass to clean up unrealized casts after pattern conversion
  {
    mlir::PassManager reconcilePM(module.getContext());
    reconcilePM.addPass(mlir::createReconcileUnrealizedCastsPass());
    if (failed(reconcilePM.run(module))) {
      // Non-fatal - some casts may remain which is expected for certain patterns
      llvm::errs() << "Warning: Some unrealized casts could not be reconciled\n";
    }
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Debug Info Generation
//===----------------------------------------------------------------------===//

void MLIRCompilationPipeline::addDebugInfoToModule(llvm::Module* llvmModule) {
  if (!llvmModule || sourceFileName.empty()) {
    return;
  }

  // Check if the module already has debug info from MLIR translation
  // MLIR's translation creates debug info from FileLineColLoc locations
  bool hasExistingDebugInfo = false;
  for (llvm::Function& func : *llvmModule) {
    if (func.getSubprogram()) {
      hasExistingDebugInfo = true;
      break;
    }
  }

  // Helper lambda to ensure module flags are set
  auto ensureModuleFlags = [&llvmModule]() {
    bool hasDwarfVersion = false;
    bool hasDebugInfoVersion = false;

    if (auto* flags = llvmModule->getModuleFlagsMetadata()) {
      for (const auto& flag : flags->operands()) {
        if (auto* md = dyn_cast<llvm::MDNode>(flag)) {
          if (md->getNumOperands() >= 2) {
            if (auto* str = dyn_cast<llvm::MDString>(md->getOperand(1))) {
              if (str->getString() == "Dwarf Version") hasDwarfVersion = true;
              if (str->getString() == "Debug Info Version") hasDebugInfoVersion = true;
            }
          }
        }
      }
    }

    if (!hasDwarfVersion) {
      llvmModule->addModuleFlag(llvm::Module::Warning, "Dwarf Version", 4);
    }
    if (!hasDebugInfoVersion) {
      llvmModule->addModuleFlag(llvm::Module::Warning, "Debug Info Version",
                                llvm::DEBUG_METADATA_VERSION);
    }
  };

  // If MLIR already generated debug info from our source locations,
  // enhance it with argument/variable debug info
  if (hasExistingDebugInfo) {
    // The MLIR translation already added DICompileUnit and DISubprograms.
    // We need to get the existing compile unit to add variable debug info.
    llvm::DICompileUnit* existingCU = nullptr;
    if (llvm::NamedMDNode* cuMD = llvmModule->getNamedMetadata("llvm.dbg.cu")) {
      if (cuMD->getNumOperands() > 0) {
        existingCU = llvm::dyn_cast<llvm::DICompileUnit>(cuMD->getOperand(0));
      }
    }

    if (!existingCU) {
      // No compile unit found, just ensure flags are set
      ensureModuleFlags();
      return;
    }

    // Create DIBuilder with the existing compile unit
    // Use AllowUnresolved=false since CU is already fully resolved
    llvm::DIBuilder dbuilder(*llvmModule, /*AllowUnresolved=*/false, existingCU);

    // Add debug info for function arguments in each function
    for (llvm::Function& func : *llvmModule) {
      if (func.isDeclaration()) continue;

      llvm::DISubprogram* sp = func.getSubprogram();
      if (!sp) continue;

      // Get the file from the subprogram
      llvm::DIFile* file = sp->getFile();
      unsigned line = sp->getLine();
      if (!file) continue;

      // Add debug info for function arguments
      unsigned argNo = 0;
      for (llvm::Argument& arg : func.args()) {
        argNo++;

        // Get or create type for the argument
        llvm::DIType* argDIType = nullptr;
        llvm::Type* argType = arg.getType();

        if (argType->isFloatTy()) {
          argDIType = dbuilder.createBasicType("float", 32, llvm::dwarf::DW_ATE_float);
        } else if (argType->isDoubleTy()) {
          argDIType = dbuilder.createBasicType("double", 64, llvm::dwarf::DW_ATE_float);
        } else if (argType->isIntegerTy(32)) {
          argDIType = dbuilder.createBasicType("int", 32, llvm::dwarf::DW_ATE_signed);
        } else if (argType->isIntegerTy(64)) {
          argDIType = dbuilder.createBasicType("long", 64, llvm::dwarf::DW_ATE_signed);
        } else if (argType->isPointerTy()) {
          llvm::DIType* elementType = dbuilder.createBasicType("float", 32, llvm::dwarf::DW_ATE_float);
          argDIType = dbuilder.createPointerType(elementType, 64);
        } else {
          argDIType = dbuilder.createBasicType("unknown", 64, llvm::dwarf::DW_ATE_unsigned);
        }

        // Create argument name
        std::string argName = "arg" + std::to_string(argNo - 1);
        if (arg.hasName()) {
          argName = arg.getName().str();
        }

        // Create DILocalVariable for the argument
        llvm::DILocalVariable* argVar = dbuilder.createParameterVariable(
            sp, argName, argNo, file, line, argDIType, true);

        // Insert dbg.value intrinsic at function entry
        if (!func.empty()) {
          llvm::BasicBlock& entryBB = func.getEntryBlock();
          llvm::Instruction* firstInst = &*entryBB.getFirstInsertionPt();
          llvm::DILocation* debugLoc = llvm::DILocation::get(
              llvmModule->getContext(), line, 0, sp);
          dbuilder.insertDbgValueIntrinsic(
              &arg, argVar, dbuilder.createExpression(), debugLoc, firstInst);
        }
      }

      // Add debug info for local variables
      // Look up variable info for this function
      std::string funcName = func.getName().str();
      auto varIt = functionVariables.find(funcName);
      if (varIt != functionVariables.end()) {
        // Collect non-argument variables in declaration order
        std::vector<VarDebugInfo> localVars;
        for (const auto& varInfo : varIt->second) {
          if (!varInfo.isArg) {
            localVars.push_back(varInfo);
          }
        }

        // Collect value-producing instructions (excluding debug intrinsics and MLIR artifacts)
        std::vector<std::pair<llvm::Instruction*, llvm::Instruction*>> valueInsts;
        for (llvm::BasicBlock& bb : func) {
          for (llvm::Instruction& inst : bb) {
            // Skip debug intrinsics and void-returning instructions
            if (llvm::isa<llvm::DbgInfoIntrinsic>(&inst)) continue;
            if (inst.getType()->isVoidTy()) continue;
            // Skip branch/phi as they're control flow, not variable defs
            if (llvm::isa<llvm::BranchInst>(&inst)) continue;
            if (llvm::isa<llvm::PHINode>(&inst)) continue;
            // Skip insertvalue/extractvalue - these are MLIR memref artifacts
            if (llvm::isa<llvm::InsertValueInst>(&inst)) continue;
            if (llvm::isa<llvm::ExtractValueInst>(&inst)) continue;
            // Skip struct types - these are memref descriptors, not source vars
            if (inst.getType()->isStructTy()) continue;

            llvm::Instruction* insertPoint = inst.getNextNode();
            valueInsts.push_back({&inst, insertPoint});
          }
        }

        // Match local variables to instructions by order
        // This assumes variables are declared in the same order as their values appear
        size_t numToMatch = std::min(localVars.size(), valueInsts.size());
        for (size_t i = 0; i < numToMatch; ++i) {
          const auto& varInfo = localVars[i];
          llvm::Instruction* inst = valueInsts[i].first;
          llvm::Instruction* insertPoint = valueInsts[i].second;

          // Create DIType for the variable
          llvm::DIType* varDIType = nullptr;
          llvm::Type* valType = inst->getType();
          if (valType->isFloatTy()) {
            varDIType = dbuilder.createBasicType("float", 32, llvm::dwarf::DW_ATE_float);
          } else if (valType->isDoubleTy()) {
            varDIType = dbuilder.createBasicType("double", 64, llvm::dwarf::DW_ATE_float);
          } else if (valType->isIntegerTy(32)) {
            varDIType = dbuilder.createBasicType("int", 32, llvm::dwarf::DW_ATE_signed);
          } else if (valType->isIntegerTy(64)) {
            varDIType = dbuilder.createBasicType("long", 64, llvm::dwarf::DW_ATE_signed);
          } else if (valType->isPointerTy()) {
            llvm::DIType* elemType = dbuilder.createBasicType("float", 32, llvm::dwarf::DW_ATE_float);
            varDIType = dbuilder.createPointerType(elemType, 64);
          } else {
            varDIType = dbuilder.createBasicType("unknown", 64, llvm::dwarf::DW_ATE_unsigned);
          }

          // Create DILocalVariable
          llvm::DILocalVariable* localVar = dbuilder.createAutoVariable(
              sp, varInfo.name, file, varInfo.line, varDIType, true);

          // Insert dbg.value intrinsic after the instruction
          llvm::DILocation* varDebugLoc = llvm::DILocation::get(
              llvmModule->getContext(), varInfo.line, varInfo.col, sp);
          if (insertPoint) {
            dbuilder.insertDbgValueIntrinsic(
                inst, localVar, dbuilder.createExpression(), varDebugLoc, insertPoint);
          }
        }
      }
    }

    dbuilder.finalize();
    ensureModuleFlags();
    return;
  }

  // If no existing debug info, create it from scratch
  llvm::DIBuilder dbuilder(*llvmModule);

  // Extract directory and filename
  std::string directory = ".";
  std::string filename = sourceFileName;
  size_t lastSlash = sourceFileName.find_last_of("/\\");
  if (lastSlash != std::string::npos) {
    directory = sourceFileName.substr(0, lastSlash);
    filename = sourceFileName.substr(lastSlash + 1);
  }

  // Create file descriptor
  llvm::DIFile* file = dbuilder.createFile(filename, directory);

  // Create compile unit
  dbuilder.createCompileUnit(
      llvm::dwarf::DW_LANG_C,           // Language (closest to SimpleLang)
      file,                              // File
      "SimpleLang MLIR Compiler",        // Producer
      false,                             // isOptimized
      "",                                // Flags
      0,                                 // RuntimeVersion
      "",                                // SplitName
      llvm::DICompileUnit::FullDebug,    // DebugEmissionKind
      0                                  // DWOId
  );

  // Create a basic subroutine type (void function type for simplicity)
  llvm::SmallVector<llvm::Metadata*, 1> types;
  types.push_back(nullptr); // Return type (void)
  llvm::DISubroutineType* funcType = dbuilder.createSubroutineType(
      dbuilder.getOrCreateTypeArray(types));

  // For each function in the module, create a DISubprogram
  for (llvm::Function& func : *llvmModule) {
    if (func.isDeclaration()) {
      continue; // Skip external declarations
    }

    // Skip if function already has debug info
    if (func.getSubprogram()) {
      continue;
    }

    // Try to get line number from first instruction's debug location
    unsigned line = 1; // Default to line 1 if no debug info available
    for (llvm::BasicBlock& bb : func) {
      for (llvm::Instruction& inst : bb) {
        if (const llvm::DebugLoc& dl = inst.getDebugLoc()) {
          line = dl.getLine();
          if (line > 0) break;
        }
      }
      if (line > 1) break;
    }

    // Create DISubprogram for this function
    llvm::DISubprogram* sp = dbuilder.createFunction(
        file,                              // Scope (file)
        func.getName(),                    // Name
        func.getName(),                    // Linkage name
        file,                              // File
        line,                              // Line number
        funcType,                          // Type
        line,                              // ScopeLine
        llvm::DINode::FlagPrototyped,      // Flags
        llvm::DISubprogram::SPFlagDefinition // SPFlags
    );

    // Attach the subprogram to the function
    func.setSubprogram(sp);

    // Add debug info for function arguments
    unsigned argNo = 0;
    for (llvm::Argument& arg : func.args()) {
      argNo++;

      // Get or create type for the argument
      llvm::DIType* argDIType = nullptr;
      llvm::Type* argType = arg.getType();

      if (argType->isFloatTy()) {
        argDIType = dbuilder.createBasicType("float", 32, llvm::dwarf::DW_ATE_float);
      } else if (argType->isDoubleTy()) {
        argDIType = dbuilder.createBasicType("double", 64, llvm::dwarf::DW_ATE_float);
      } else if (argType->isIntegerTy(32)) {
        argDIType = dbuilder.createBasicType("int", 32, llvm::dwarf::DW_ATE_signed);
      } else if (argType->isIntegerTy(64)) {
        argDIType = dbuilder.createBasicType("long", 64, llvm::dwarf::DW_ATE_signed);
      } else if (argType->isPointerTy()) {
        // For pointers (arrays, tensors), create a pointer type
        llvm::DIType* elementType = dbuilder.createBasicType("float", 32, llvm::dwarf::DW_ATE_float);
        argDIType = dbuilder.createPointerType(elementType, 64);
      } else {
        // Default to a generic type
        argDIType = dbuilder.createBasicType("unknown", 64, llvm::dwarf::DW_ATE_unsigned);
      }

      // Create argument name (use "arg0", "arg1", etc. since original names are lost)
      std::string argName = "arg" + std::to_string(argNo - 1);
      if (arg.hasName()) {
        argName = arg.getName().str();
      }

      // Create DILocalVariable for the argument
      llvm::DILocalVariable* argVar = dbuilder.createParameterVariable(
          sp,                    // Scope
          argName,               // Name
          argNo,                 // ArgNo (1-indexed)
          file,                  // File
          line,                  // Line
          argDIType,             // Type
          true                   // AlwaysPreserve
      );

      // Insert dbg.value intrinsic at the function entry
      if (!func.empty()) {
        llvm::BasicBlock& entryBB = func.getEntryBlock();
        llvm::Instruction* firstInst = &*entryBB.getFirstInsertionPt();

        // Create a debug location for the intrinsic
        llvm::DILocation* debugLoc = llvm::DILocation::get(
            llvmModule->getContext(), line, 0, sp);

        // Insert dbg.value intrinsic
        dbuilder.insertDbgValueIntrinsic(
            &arg,                                // Value
            argVar,                              // Variable
            dbuilder.createExpression(),         // Expression
            debugLoc,                            // DebugLoc
            firstInst                            // InsertBefore
        );
      }
    }
  }

  // Add DWARF version and debug info version module flags
  llvmModule->addModuleFlag(llvm::Module::Warning, "Dwarf Version", 4);
  llvmModule->addModuleFlag(llvm::Module::Warning, "Debug Info Version",
                            llvm::DEBUG_METADATA_VERSION);

  // Finalize the debug info
  dbuilder.finalize();
}
