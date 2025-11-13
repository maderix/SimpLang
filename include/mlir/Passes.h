//===- Passes.h - Simp dialect passes --------------------------*- C++ -*-===//
//
// Part of the SimpLang Project
//
// This file declares the passes for the Simp dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SIMP_PASSES_H
#define MLIR_DIALECT_SIMP_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace simp {

/// Create a pass that converts Simp dialect operations to MemRef and Arith
/// dialect operations. This is the first lowering step in the compilation
/// pipeline.
std::unique_ptr<Pass> createConvertSimpToMemRefPass();

/// Register the ConvertSimpToMemRef pass for command-line usage
void registerConvertSimpToMemRefPass();

/// Create a pass that converts Simp MatMul operations to Linalg matmul.
/// This enables high-level linear algebra optimizations before lowering to loops.
std::unique_ptr<Pass> createConvertSimpToLinalgPass();

/// Register the ConvertSimpToLinalg pass for command-line usage
void registerConvertSimpToLinalgPass();

/// Create a pass that specializes dynamic memref shapes to static shapes
/// when dimensions are known at compile time. This enables affine optimizations.
std::unique_ptr<Pass> createSpecializeShapesPass();

/// Register the SpecializeShapes pass for command-line usage
void registerSpecializeShapesPass();

/// Create a configurable Linalg tiling pass optimized for cache locality.
/// Default 8×8×8 tiling achieves 45.68 tok/s on Stories110M (optimal for transformers).
std::unique_ptr<Pass> createSimpLinalgTilingPass();

/// Create a configurable Linalg tiling pass with custom parameters.
std::unique_ptr<Pass> createSimpLinalgTilingPass(int64_t tileSize, bool hierarchical, bool parallel);

/// Register the SimpLinalgTiling pass for command-line usage
void registerSimpLinalgTilingPass();

/// Create a pass that inserts prefetch operations into tiled loops
std::unique_ptr<Pass> createInsertPrefetchPass();

/// Register all Simp dialect passes for command-line usage
void registerSimpPasses();

/// Register all Simp dialect pipelines for command-line usage
/// Pipelines: simp-default, simp-transformer, simp-debug, simp-high-perf, simp-parallel
void registerSimpPipelines();

} // namespace simp
} // namespace mlir

#endif // MLIR_DIALECT_SIMP_PASSES_H
