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

  /// Enable/disable prefetch insertion for memory latency hiding
  void setEnablePrefetch(bool enable) { enablePrefetch = enable; }

  /// Skip MLIR vectorization and let LLVM handle it (better for INT8/INT4)
  void setSkipMLIRVectorization(bool skip) { skipMLIRVectorization = skip; }

  /// Set output path for intermediate IR dumps
  void setOutputPath(const std::string& path) { outputPath = path; }

  //===--------------------------------------------------------------------===//
  // GPU Backend Configuration (requires USE_CUDA)
  //===--------------------------------------------------------------------===//

  /// Enable/disable GPU code generation (requires CUDA)
  void setEnableGPU(bool enable) { enableGPU = enable; }

  /// Set target CUDA architecture (e.g., "sm_80" for A100, "sm_90" for H100)
  void setCudaArch(const std::string& arch) { cudaArch = arch; }

  /// Set GPU matmul strategy: "cublas" (default), "mlir", or "auto"
  /// - cublas: Use cuBLAS for large matrices (M,N,K >= 128)
  /// - mlir: Use MLIR-generated GPU kernels
  /// - auto: Automatically choose based on matrix size
  void setGPUMatMulStrategy(const std::string& strategy) { gpuMatMulStrategy = strategy; }

  /// Check if GPU code generation is enabled
  bool isGPUEnabled() const { return enableGPU; }

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

  /// Enable prefetch insertion (default: true)
  bool enablePrefetch = true;

  /// Output path for intermediate IR dumps
  std::string outputPath;

  /// Skip MLIR vectorization passes and let LLVM handle vectorization
  /// Better for INT8/INT4 (avoids shuffle overhead from MLIR vector dialect)
  bool skipMLIRVectorization = false;

  //===--------------------------------------------------------------------===//
  // GPU Configuration Members (requires USE_CUDA)
  //===--------------------------------------------------------------------===//

  /// Enable GPU code generation (default: false)
  bool enableGPU = false;

  /// Target CUDA architecture (default: sm_80 for A100)
  std::string cudaArch = "sm_80";

  /// GPU matmul strategy: "cublas", "mlir", or "auto" (default: cublas)
  std::string gpuMatMulStrategy = "cublas";

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

  //===--------------------------------------------------------------------===//
  // GPU Pipeline Builders (requires USE_CUDA)
  //===--------------------------------------------------------------------===//

#ifdef USE_CUDA
  /// Build Phase 2.7 passes: GPU lowering (parallel loops → GPU kernels)
  /// Maps parallel loops to GPU thread hierarchy and outlines GPU kernels
  void buildPhase2_7_GPULowering(mlir::OpPassManager& pm);

  /// Build Phase 4 passes: NVVM lowering (GPU → NVVM → PTX/CUBIN)
  /// Converts GPU dialect to NVVM dialect and serializes to PTX/CUBIN
  void buildPhase4_NVVMLowering(mlir::OpPassManager& pm);
#endif
};

} // namespace simp
} // namespace mlir

#endif // MLIR_PIPELINE_HPP
