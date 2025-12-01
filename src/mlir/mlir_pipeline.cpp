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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"  // For LinalgTilingLoopType
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"  // OpenMP dialect
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// MLIR Conversion Passes
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"  // OpenMP to LLVM conversion
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"  // SCF to OpenMP conversion
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/DialectConversion.h"

// MLIR to LLVM IR Translation
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"  // OpenMP IR translation
#include "mlir/Target/LLVMIR/Export.h"

// MLIR Verification
#include "mlir/IR/Verifier.h"

// LLVM
#include "llvm/IR/Module.h"
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

  // Register LLVM dialect translation
  mlir::registerLLVMDialectTranslation(*module.getContext());

  // Register OpenMP dialect translation (if OpenMP is enabled)
  if (enableOpenMP) {
    mlir::registerOpenMPDialectTranslation(*module.getContext());
  }

  // Translate MLIR LLVM dialect to LLVM IR
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Error: Failed to translate MLIR to LLVM IR\n";
    return nullptr;
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
      llvm::SmallVector<int64_t, 3> outerTileSizes = {outer_tile, outer_tile, outer_tile};
      pm.addNestedPass<mlir::FuncOp>(
          mlir::createLinalgTilingPass(outerTileSizes, loopType));
      pm.addPass(mlir::createCanonicalizerPass());

      // Level 2: Inner tiling (L1 cache + vectorization)
      // Inner loops should be sequential for vectorization
      llvm::SmallVector<int64_t, 3> innerTileSizes = {inner_tile, inner_tile, inner_tile};
      pm.addNestedPass<mlir::FuncOp>(
          mlir::createLinalgTilingPass(innerTileSizes, linalg::LinalgTilingLoopType::Loops));
      pm.addPass(mlir::createCanonicalizerPass());
    } else {
      // SINGLE-LEVEL TILING: Use configured tile size
      // For matmul(MxK, KxN, MxN), we tile all 3 dimensions
      llvm::SmallVector<int64_t, 3> tileSizes = {tileSize, tileSize, tileSize};
      pm.addNestedPass<mlir::FuncOp>(
          mlir::createLinalgTilingPass(tileSizes, loopType));

      // Canonicalize after tiling
      pm.addPass(mlir::createCanonicalizerPass());
    }
  }

  // PARALLELIZATION: Convert scf.parallel to OpenMP if enabled
  if (enableOpenMP) {
    llvm::outs() << "[OpenMP] Converting scf.parallel to omp.parallel\n";
    pm.addPass(mlir::createConvertSCFToOpenMPPass());
    pm.addPass(mlir::createCanonicalizerPass());
  }

  // VECTORIZATION: Convert linalg operations to vector dialect
  // The strategy passes work together: Enable → Vectorize → LowerVectors

  // Step 1: Enable vectorization strategy (marks ops for vectorization)
  pm.addNestedPass<mlir::FuncOp>(
      mlir::createLinalgStrategyEnablePass());

  // Step 2: Vectorize marked linalg operations
  pm.addNestedPass<mlir::FuncOp>(
      mlir::createLinalgStrategyVectorizePass(""));

  // Canonicalize after vectorization to clean up
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // Step 3: Lower vector dialect operations
  // These passes convert high-level vector ops (like vector.contract) to LLVM-compatible ops
  pm.addNestedPass<mlir::FuncOp>(
      mlir::createLinalgStrategyLowerVectorsPass());

  // Apply canonicalization after vector lowering
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // Lower remaining Linalg ops (non-vectorized) to SCF loops
  // Only operations that couldn't be vectorized will go through this
  pm.addNestedPass<mlir::FuncOp>(mlir::createConvertLinalgToLoopsPass());

  // Apply general canonicalization and CSE to clean up generated code
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // INSERT PREFETCH: Add prefetch operations to hide memory latency
  // Run after linalg-to-loops so we can find memref.load in all loops
  if (enablePrefetch) {
    llvm::outs() << "[Prefetch] Inserting prefetch operations into loops\n";
    pm.addNestedPass<mlir::FuncOp>(mlir::simp::createInsertPrefetchPass());
    pm.addPass(mlir::createCanonicalizerPass());
  }

  // OPTIMIZATION: Optimize loops for better performance
  // Apply loop invariant code motion to hoist invariant operations
  pm.addNestedPass<mlir::FuncOp>(mlir::createLoopInvariantCodeMotionPass());

  // Apply loop unrolling for better ILP
  pm.addNestedPass<mlir::FuncOp>(mlir::createLoopUnrollPass(4));
}

//===----------------------------------------------------------------------===//
// Phase 2.5: Insert Buffer Deallocations
//===----------------------------------------------------------------------===//

void MLIRCompilationPipeline::buildPhase2_5_BufferManagement(mlir::OpPassManager& pm) {
  // First, hoist buffer allocations out of loops to avoid repeated allocation
  pm.addNestedPass<mlir::FuncOp>(mlir::bufferization::createBufferHoistingPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::bufferization::createBufferLoopHoistingPass());

  // Then add buffer deallocation pass
  // This inserts memref.dealloc operations after the last use of allocated buffers
  pm.addNestedPass<mlir::FuncOp>(mlir::bufferization::createBufferDeallocationPass());

  // Convert bufferization ops (like bufferization.clone) to memref ops
  pm.addPass(mlir::createBufferizationToMemRefPass());
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
  pm.addNestedPass<mlir::FuncOp>(mlir::createConvertVectorToSCFPass());

  // 2. Lower Affine and SCF to Standard dialect control flow
  // Lower Affine ops (affine.min, etc.) to Standard
  pm.addPass(mlir::createLowerAffinePass());

  // Always lower SCF to CF, even when OpenMP is enabled
  // The SCF-to-OpenMP pass only converts scf.parallel, leaving other SCF ops
  // (scf.while, scf.for, scf.if) that need to be converted to CF
  pm.addPass(mlir::createLowerToCFGPass());

  // CRITICAL: Run createConvertOpenMPToLLVMPass() AFTER createLowerToCFGPass()!
  // This fixes index/i64 type mismatches in branches inside OpenMP regions
  if (enableOpenMP) {
    llvm::outs() << "[OpenMP] Converting OpenMP index types to i64\n";
    pm.addPass(mlir::createConvertOpenMPToLLVMPass());
  }

  // 2.5. Expand Arithmetic ops (maxf, minf) to operations that can be lowered to LLVM
  //      MaxFOp/MinFOp don't have direct LLVM lowering - they must be expanded to CmpF + Select
  pm.addNestedPass<mlir::FuncOp>(mlir::arith::createArithmeticExpandOpsPass());

  // 3. Add canonicalization and CSE passes for cleanup
  // Pattern-based conversions will be applied after running the pipeline
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // 4. Final reconciliation of unrealized casts
  // Skip when OpenMP is enabled - casts inside omp regions can't be reconciled until LLVM IR translation
  if (!enableOpenMP) {
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  }
}

//===----------------------------------------------------------------------===//
// Helper: Apply Vector Lowering Patterns and LLVM Dialect Conversion
//===----------------------------------------------------------------------===//

bool MLIRCompilationPipeline::applyLLVMDialectConversion() {
  // This method applies pattern-based conversions that require direct module access
  // It must be called AFTER the main pass pipeline has run

  // Step 1: Lower vector operations to simpler forms BEFORE LLVM conversion
  {
    mlir::RewritePatternSet patterns(module.getContext());

    // Lower vector.contract to outer products and vector.transpose to shuffles
    mlir::vector::VectorTransformsOptions vectorOptions;
    vectorOptions.setVectorTransformsOptions(mlir::vector::VectorContractLowering::OuterProduct);
    vectorOptions.setVectorTransposeLowering(mlir::vector::VectorTransposeLowering::Shuffle);
    mlir::vector::populateVectorContractLoweringPatterns(patterns, vectorOptions);
    mlir::vector::populateVectorTransposeLoweringPatterns(patterns, vectorOptions);

    // Lower vector.transpose, vector.broadcast, vector.shape_cast, etc.
    mlir::vector::populateVectorToVectorCanonicalizationPatterns(patterns);
    mlir::vector::populateVectorBroadcastLoweringPatterns(patterns);
    mlir::vector::populateVectorMaskOpLoweringPatterns(patterns);
    mlir::vector::populateVectorShapeCastLoweringPatterns(patterns);

    if (failed(mlir::applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      llvm::errs() << "Warning: Vector lowering patterns failed\n";
    }
  }

  // Progressive lowering of vector operations (matches LLVM's ConvertVectorToLLVMPass line 64-75)
  // Run iteratively to handle nested broadcasts (e.g., 1D->2D broadcasts in 1x1 matmul)
  for (int iteration = 0; iteration < 5; ++iteration) {
    mlir::RewritePatternSet patterns(module.getContext());
    mlir::vector::populateVectorToVectorCanonicalizationPatterns(patterns);
    mlir::vector::populateVectorBroadcastLoweringPatterns(patterns);
    mlir::vector::populateVectorContractLoweringPatterns(patterns);
    mlir::vector::populateVectorMaskOpLoweringPatterns(patterns);
    mlir::vector::populateVectorShapeCastLoweringPatterns(patterns);
    mlir::vector::populateVectorTransposeLoweringPatterns(patterns);
    // Vector transfer ops with rank > 1 should be lowered with VectorToSCF
    mlir::vector::populateVectorTransferLoweringPatterns(patterns, /*maxTransferRank=*/1);

    if (failed(mlir::applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      llvm::errs() << "Warning: Vector lowering iteration " << iteration << " had issues\n";
    }
  }

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
  mlir::arith::populateArithmeticToLLVMConversionPatterns(typeConverter, patterns);

  // Add patterns for Math → LLVM
  mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);

  // Add patterns for MemRef → LLVM
  mlir::populateMemRefToLLVMConversionPatterns(typeConverter, patterns);

  // NOTE: OpenMP operations remain as omp.* and are translated to LLVM IR
  // during the mlir-translate phase, NOT during LLVM dialect lowering

  // Add patterns for Vector → LLVM (CRITICAL for vectorization)
  // Match LLVM's ConvertVectorToLLVMPass lines 80-85
  mlir::vector::populateVectorMaskMaterializationPatterns(patterns, /*indexOptimizations=*/false);
  mlir::vector::populateVectorTransferLoweringPatterns(patterns);
  mlir::populateVectorToLLVMMatrixConversionPatterns(typeConverter, patterns);
  mlir::populateVectorToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateVectorToLLVMMatrixConversionPatterns(typeConverter, patterns);

  // Add patterns for Standard → LLVM
  mlir::populateStdToLLVMConversionPatterns(typeConverter, patterns);

  // Use partial conversion (allows unrealized casts temporarily)
  if (failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
    llvm::errs() << "Error: Failed to convert to LLVM dialect\n";
    module.dump();
    return false;
  }

  // Note: createReconcileUnrealizedCastsPass() is already added in buildPhase3_LLVMDialectLowering
  // and will run after the main pipeline completes

  return true;
}
