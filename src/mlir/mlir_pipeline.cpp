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

// MLIR Pass Infrastructure
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

// MLIR Dialects
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

// MLIR Conversion Passes
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

// MLIR to LLVM IR Translation
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
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

  // Phase 1: Lower Simp dialect to MemRef + Arith + Linalg
  // This converts all Simp operations, including matmul → linalg.matmul
  if (!runSimpToMemRefLowering()) {
    llvm::errs() << "Error: Failed to lower Simp to MemRef + Linalg\n";
    return false;
  }
  if (dumpIntermediateIR) {
    llvm::outs() << "\n=== After Phase 1: Simp → MemRef + Linalg ===\n";
    module.print(llvm::outs());
    llvm::outs() << "\n";
  }

  // Phase 2: Lower Linalg ops to SCF loops
  if (!runLinalgToLoopsLowering()) {
    llvm::errs() << "Error: Failed to lower Linalg to loops\n";
    return false;
  }
  if (dumpIntermediateIR) {
    llvm::outs() << "\n=== After Phase 2: Linalg → SCF Loops ===\n";
    module.print(llvm::outs());
    llvm::outs() << "\n";
  }

  // Phase 2.5: Insert buffer deallocations
  // This adds memref.dealloc operations after the last use of allocated buffers
  if (!runBufferDeallocationPass()) {
    llvm::errs() << "Error: Failed to insert buffer deallocations\n";
    return false;
  }
  if (dumpIntermediateIR) {
    llvm::outs() << "\n=== After Phase 2.5: Buffer Deallocation ===\n";
    module.print(llvm::outs());
    llvm::outs() << "\n";
  }

  // Phase 3: Lower MemRef + Arith + SCF + Standard to LLVM dialect
  if (!runToLLVMDialectLowering()) {
    llvm::errs() << "Error: Failed to lower to LLVM dialect\n";
    return false;
  }
  if (dumpIntermediateIR) {
    llvm::outs() << "\n=== After Phase 3: → LLVM Dialect ===\n";
    module.print(llvm::outs());
    llvm::outs() << "\n";
  }

  // Verify the final module
  if (failed(mlir::verify(module))) {
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

bool MLIRCompilationPipeline::runSimpToMemRefLowering() {
  mlir::PassManager pm(module.getContext());

  // Add our custom Simp → MemRef + Arith + Linalg lowering pass
  pm.addPass(mlir::simp::createConvertSimpToMemRefPass());

  // Run the pass
  if (failed(pm.run(module))) {
    llvm::errs() << "Error: ConvertSimpToMemRef pass failed\n";
    module.dump();
    return false;
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Phase 2: Linalg → SCF Loops (with optimizations)
//===----------------------------------------------------------------------===//

bool MLIRCompilationPipeline::runLinalgToLoopsLowering() {
  mlir::PassManager pm(module.getContext());

  // Enable IR printing for debugging (comment out for production)
  // pm.enableIRPrinting();

  // OPTIONAL: Tile matmul for better cache locality (enabled via --enable-tiling flag)
  if (enableTiling) {
    // Tile sizes: 32x32x32 is good for L1 cache (32*32*4 bytes = 4KB per tile)
    // For matmul(MxK, KxN, MxN), we tile all 3 dimensions
    llvm::SmallVector<int64_t, 3> tileSizes = {32, 32, 32};
    pm.addNestedPass<mlir::FuncOp>(
        mlir::createLinalgTilingPass(tileSizes)); // Tile to 32x32x32 blocks
  }

  // Lower Linalg ops (matmul, fill) to SCF loops
  // NOTE: Affine lowering doesn't work even with static shapes for linalg.matmul
  // TODO: Custom matmul lowering with iter_args for proper accumulator handling
  pm.addNestedPass<mlir::FuncOp>(mlir::createConvertLinalgToLoopsPass());

  // Apply general canonicalization and CSE to clean up generated loops
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // Run the pass
  if (failed(pm.run(module))) {
    llvm::errs() << "Error: Linalg to Loops pass failed\n";
    module.dump();
    return false;
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Phase 2.5: Insert Buffer Deallocations
//===----------------------------------------------------------------------===//

bool MLIRCompilationPipeline::runBufferDeallocationPass() {
  mlir::PassManager pm(module.getContext());

  // First, hoist buffer allocations out of loops to avoid repeated allocation
  pm.addNestedPass<mlir::FuncOp>(mlir::bufferization::createBufferHoistingPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::bufferization::createBufferLoopHoistingPass());

  // Then add buffer deallocation pass
  // This inserts memref.dealloc operations after the last use of allocated buffers
  pm.addNestedPass<mlir::FuncOp>(mlir::bufferization::createBufferDeallocationPass());

  // Convert bufferization ops (like bufferization.clone) to memref ops
  pm.addPass(mlir::createBufferizationToMemRefPass());

  // Run the passes
  if (failed(pm.run(module))) {
    llvm::errs() << "Error: Buffer deallocation/conversion pass failed\n";
    module.dump();
    return false;
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Phase 3: MemRef + Arith + SCF + Standard → LLVM Dialect
//===----------------------------------------------------------------------===//

bool MLIRCompilationPipeline::runToLLVMDialectLowering() {
  // Follow the Toy Chapter 6 pattern for lowering to LLVM dialect
  // This uses pattern-based conversion with full dialect conversion

  // 1. First, lower Affine and SCF to Standard dialect control flow
  {
    mlir::PassManager pm(module.getContext());
    // Lower Affine ops (affine.min, etc.) to Standard
    pm.addPass(mlir::createLowerAffinePass());
    // Lower SCF to Standard
    pm.addPass(mlir::createLowerToCFGPass());
    if (failed(pm.run(module))) {
      llvm::errs() << "Error: Affine/SCF to Standard lowering failed\n";
      module.dump();
      return false;
    }
  }

  // 2. Lower MemRef, Arith, and Standard to LLVM dialect
  //    Using pattern-based conversion like Toy Ch6

  // Define the conversion target - only LLVM dialect operations are legal
  mlir::LLVMConversionTarget target(*module.getContext());
  target.addLegalOp<mlir::ModuleOp>();

  // Type converter for lowering types to LLVM types
  mlir::LLVMTypeConverter typeConverter(module.getContext());

  // Populate conversion patterns from built-in passes
  mlir::RewritePatternSet patterns(module.getContext());

  // Add patterns for Arith → LLVM
  mlir::arith::populateArithmeticToLLVMConversionPatterns(typeConverter, patterns);

  // Add patterns for MemRef → LLVM
  mlir::populateMemRefToLLVMConversionPatterns(typeConverter, patterns);

  // Add patterns for Standard → LLVM
  mlir::populateStdToLLVMConversionPatterns(typeConverter, patterns);

  // Apply full conversion (all operations must be legal after this)
  if (failed(mlir::applyFullConversion(module, target, std::move(patterns)))) {
    llvm::errs() << "Error: Failed to convert to LLVM dialect\n";
    module.dump();
    return false;
  }

  return true;
}
