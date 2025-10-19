//===- simp_types.cpp - Simp dialect types implementation -----------------===//
//
// Part of the SimpLang Project
//
// This file implements the types for the Simp dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/simp_types.hpp"
#include "mlir/simp_dialect.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::simp;

//===----------------------------------------------------------------------===//
// ArrayType Implementation
//===----------------------------------------------------------------------===//

ArrayType ArrayType::get(mlir::MLIRContext *context, mlir::Type elementType) {
  return Base::get(context, elementType);
}

mlir::Type ArrayType::getElementType() const {
  return getImpl()->elementType;
}

//===----------------------------------------------------------------------===//
// SimpTensorType Implementation
//===----------------------------------------------------------------------===//

SimpTensorType SimpTensorType::get(mlir::MLIRContext *context,
                                    llvm::ArrayRef<int64_t> shape,
                                    mlir::Type elementType) {
  return Base::get(context, shape, elementType);
}

llvm::ArrayRef<int64_t> SimpTensorType::getShape() const {
  return getImpl()->shape;
}

mlir::Type SimpTensorType::getElementType() const {
  return getImpl()->elementType;
}

int64_t SimpTensorType::getRank() const {
  return getShape().size();
}

bool SimpTensorType::isDynamicDim(unsigned idx) const {
  assert(idx < getRank() && "Index out of range");
  return getShape()[idx] == -1;
}

bool SimpTensorType::hasStaticShape() const {
  for (int64_t dim : getShape()) {
    if (dim == -1)
      return false;
  }
  return true;
}
