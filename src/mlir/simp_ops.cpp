//===- simp_ops.cpp - Simp dialect operations implementation --------------===//
//
// Part of the SimpLang Project
//
// This file implements methods for Simp dialect operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/simp_ops.hpp"
#include "mlir/simp_dialect.hpp"
#include "mlir/simp_types.hpp"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::simp;

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

mlir::OpFoldResult ConstantOp::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  // Constants fold to themselves
  return value();
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

void AddOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                         mlir::MLIRContext *context) {
  // No canonicalization patterns for now
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

void SubOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                         mlir::MLIRContext *context) {
  // No canonicalization patterns for now
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

void MulOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                         mlir::MLIRContext *context) {
  // No canonicalization patterns for now
}

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

void DivOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                         mlir::MLIRContext *context) {
  // No canonicalization patterns for now
}
