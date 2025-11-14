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
class OpPassManager;

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

  /// Set tile size for matmul optimization (default: 8)
  void setTileSize(int size) { tileSize = size; }

  /// Enable hierarchical (multi-level) tiling for cache optimization
  void setEnableHierarchicalTiling(bool enable) { enableHierarchicalTiling = enable; }

  /// Enable/disable OpenMP parallelization (multi-threading)
  void setEnableOpenMP(bool enable) { enableOpenMP = enable; }

  /// Enable/disable IR dumping at each pipeline stage
  void setDumpIntermediateIR(bool enable) { dumpIntermediateIR = enable; }

  /// Set output path for intermediate IR dumps
  void setOutputPath(const std::string& path) { outputPath = path; }

private:
  //===--------------------------------------------------------------------===//
  // Private Members
  //===--------------------------------------------------------------------===//

  /// The MLIR module being compiled
  mlir::ModuleOp module;

  /// Enable loop tiling optimization (default: true)
  bool enableTiling = true;

  /// Tile size for matmul optimization (default: 8)
  int tileSize = 8;

  /// Enable hierarchical tiling (default: false for compatibility)
  bool enableHierarchicalTiling = false;

  /// Enable OpenMP parallelization (default: false)
  bool enableOpenMP = false;

  /// Enable IR dumping at each pipeline stage (default: false)
  bool dumpIntermediateIR = false;

  /// Output path for intermediate IR dumps
  std::string outputPath;

  //===--------------------------------------------------------------------===//
  // Private Helpers: Pipeline Builders
  //===--------------------------------------------------------------------===//

  /// Build Phase 1 passes: Simp → MemRef + Arith + Linalg
  void buildPhase1_SimpLowering(mlir::OpPassManager& pm);

  /// Build Phase 2 passes: Linalg ops → SCF loops (with tiling, vectorization)
  void buildPhase2_LinalgOptimization(mlir::OpPassManager& pm);

  /// Build Phase 2.5 passes: Buffer management (hoisting, deallocation)
  void buildPhase2_5_BufferManagement(mlir::OpPassManager& pm);

  /// Build Phase 3 passes: MemRef + Arith + SCF + Standard → LLVM dialect
  void buildPhase3_LLVMDialectLowering(mlir::OpPassManager& pm);

  /// Apply vector lowering patterns and LLVM dialect conversion
  /// Must be called after the pass pipeline runs (requires direct module access)
  bool applyLLVMDialectConversion();
};

} // namespace simp
} // namespace mlir

#endif // MLIR_PIPELINE_HPP
