//===- mlir_pipeline.hpp - MLIR Compilation Pipeline ------------*- C++ -*-===//
//
// Part of the SimpLang Project
//
// This file defines the MLIRCompilationPipeline class for managing the
// progressive lowering of MLIR modules to LLVM IR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PIPELINE_HPP
#define MLIR_PIPELINE_HPP

#include "mlir/IR/BuiltinOps.h"
#include <memory>

// Forward declarations
namespace llvm {
class Module;
class LLVMContext;
}

namespace mlir {
class MLIRContext;
namespace simp {

//===----------------------------------------------------------------------===//
// MLIRCompilationPipeline
//===----------------------------------------------------------------------===//

/// Manages the progressive lowering pipeline for MLIR modules.
///
/// Pipeline stages:
/// 1. Simp dialect → MemRef + Arith + Linalg (ConvertSimpToMemRef)
/// 2. Linalg ops → SCF loops (ConvertLinalgToLoops)
/// 3. MemRef + Arith + SCF + Standard → LLVM dialect
/// 4. LLVM dialect → LLVM IR module
///
/// Usage:
///   MLIRCompilationPipeline pipeline(mlirModule);
///   if (pipeline.runPasses()) {
///     auto llvmModule = pipeline.translateToLLVMIR(llvmContext);
///   }
class MLIRCompilationPipeline {
public:
  explicit MLIRCompilationPipeline(mlir::ModuleOp module);
  ~MLIRCompilationPipeline() = default;

  // Disable copy/move
  MLIRCompilationPipeline(const MLIRCompilationPipeline&) = delete;
  MLIRCompilationPipeline& operator=(const MLIRCompilationPipeline&) = delete;

  //===--------------------------------------------------------------------===//
  // Pipeline Execution
  //===--------------------------------------------------------------------===//

  /// Run the complete lowering pipeline (Simp → LLVM dialect)
  /// Returns true on success, false on failure
  bool runPasses();

  /// Translate the MLIR LLVM dialect module to LLVM IR
  /// Must be called after runPasses() succeeds
  /// Returns nullptr on failure
  std::unique_ptr<llvm::Module> translateToLLVMIR(llvm::LLVMContext& llvmContext);

  //===--------------------------------------------------------------------===//
  // Accessors
  //===--------------------------------------------------------------------===//

  /// Get the MLIR module (after passes have been applied)
  mlir::ModuleOp getModule() const { return module; }

  /// Get the MLIR context
  mlir::MLIRContext* getMLIRContext() { return module.getContext(); }

  //===--------------------------------------------------------------------===//
  // Configuration
  //===--------------------------------------------------------------------===//

  /// Enable/disable loop tiling optimization for matmul
  void setEnableTiling(bool enable) { enableTiling = enable; }

  /// Enable/disable IR dumping at each pipeline stage
  void setDumpIntermediateIR(bool enable) { dumpIntermediateIR = enable; }

private:
  //===--------------------------------------------------------------------===//
  // Private Members
  //===--------------------------------------------------------------------===//

  /// The MLIR module being compiled
  mlir::ModuleOp module;

  /// Enable loop tiling optimization (default: false)
  bool enableTiling = false;

  /// Enable IR dumping at each pipeline stage (default: false)
  bool dumpIntermediateIR = false;

  //===--------------------------------------------------------------------===//
  // Private Helpers
  //===--------------------------------------------------------------------===//

  /// Run Phase 1: Simp → MemRef + Arith + Linalg
  bool runSimpToMemRefLowering();

  /// Run Phase 2: Linalg ops → SCF loops
  bool runLinalgToLoopsLowering();

  /// Run Phase 2.5: Insert buffer deallocations
  bool runBufferDeallocationPass();

  /// Run Phase 3: MemRef + Arith + SCF + Standard → LLVM dialect
  bool runToLLVMDialectLowering();
};

} // namespace simp
} // namespace mlir

#endif // MLIR_PIPELINE_HPP
