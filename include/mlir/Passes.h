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

/// Create a pass that converts Simp MatMul operations to Linalg matmul.
/// This enables high-level linear algebra optimizations before lowering to loops.
std::unique_ptr<Pass> createConvertSimpToLinalgPass();

/// Create a pass that specializes dynamic memref shapes to static shapes
/// when dimensions are known at compile time. This enables affine optimizations.
std::unique_ptr<Pass> createSpecializeShapesPass();

} // namespace simp
} // namespace mlir

#endif // MLIR_DIALECT_SIMP_PASSES_H
