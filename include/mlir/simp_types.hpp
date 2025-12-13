//===- simp_types.hpp - Simp dialect types ---------------------*- C++ -*-===//
//
// Part of the SimpLang Project
//
// This file declares the types for the Simp dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SIMP_TYPES_HPP
#define MLIR_SIMP_TYPES_HPP

#include "mlir/IR/Types.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include <vector>

namespace mlir {
namespace simp {

//===----------------------------------------------------------------------===//
// ArrayType Storage
//===----------------------------------------------------------------------===//

namespace detail {

struct ArrayTypeStorage : public mlir::TypeStorage {
  using KeyTy = mlir::Type; // Element type is the key

  ArrayTypeStorage(mlir::Type elementType) : elementType(elementType) {}

  /// Equality comparison
  bool operator==(const KeyTy &key) const {
    return key == elementType;
  }

  /// Hash key for type uniquing
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key.getAsOpaquePointer());
  }

  /// Construct storage instance
  static ArrayTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    return new (allocator.allocate<ArrayTypeStorage>()) ArrayTypeStorage(key);
  }

  mlir::Type elementType;
};

} // namespace detail

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

/// Represents a 1D dynamically-sized array: !simp.array<element-type>
class ArrayType : public mlir::Type::TypeBase<ArrayType, mlir::Type,
                                               detail::ArrayTypeStorage> {
public:
  using Base::Base;

  /// Type name for MLIR type registration (required in LLVM 15+)
  static constexpr llvm::StringLiteral name = "simp.array";

  /// Get or create an instance of ArrayType.
  static ArrayType get(mlir::MLIRContext *context, mlir::Type elementType);

  /// Get the element type of this array.
  mlir::Type getElementType() const;
};

//===----------------------------------------------------------------------===//
// SimpTensorType Storage
//===----------------------------------------------------------------------===//

namespace detail {

struct SimpTensorTypeStorage : public mlir::TypeStorage {
  using KeyTy = std::pair<llvm::ArrayRef<int64_t>, mlir::Type>; // (shape, element type)

  SimpTensorTypeStorage(llvm::ArrayRef<int64_t> shape, mlir::Type elementType)
      : shape(shape.begin(), shape.end()), elementType(elementType) {}

  /// Equality comparison
  bool operator==(const KeyTy &key) const {
    return key.first == llvm::ArrayRef<int64_t>(shape) &&
           key.second == elementType;
  }

  /// Hash key for type uniquing
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(llvm::hash_combine_range(key.first.begin(),
                                                        key.first.end()),
                              key.second);
  }

  /// Construct storage instance
  static SimpTensorTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                           const KeyTy &key) {
    // Allocate space for shape array
    llvm::ArrayRef<int64_t> shape = allocator.copyInto(key.first);
    return new (allocator.allocate<SimpTensorTypeStorage>())
        SimpTensorTypeStorage(shape, key.second);
  }

  llvm::SmallVector<int64_t, 4> shape;
  mlir::Type elementType;
};

} // namespace detail

//===----------------------------------------------------------------------===//
// SimpTensorType
//===----------------------------------------------------------------------===//

/// Represents an N-dimensional tensor: !simp.tensor<shape, element-type>
/// Shape can be static (e.g., 10x20) or dynamic (e.g., ?x?)
/// NOTE: Named SimpTensorType to avoid conflict with MLIR's built-in TensorType
class SimpTensorType : public mlir::Type::TypeBase<SimpTensorType, mlir::Type,
                                                    detail::SimpTensorTypeStorage> {
public:
  using Base::Base;

  /// Type name for MLIR type registration (required in LLVM 15+)
  static constexpr llvm::StringLiteral name = "simp.tensor";

  /// Get or create an instance of SimpTensorType.
  /// @param context MLIR context
  /// @param shape Shape of the tensor (use -1 for dynamic dimensions)
  /// @param elementType Element type of the tensor
  static SimpTensorType get(mlir::MLIRContext *context,
                            llvm::ArrayRef<int64_t> shape,
                            mlir::Type elementType);

  /// Get the shape of this tensor.
  llvm::ArrayRef<int64_t> getShape() const;

  /// Get the element type of this tensor.
  mlir::Type getElementType() const;

  /// Get the rank (number of dimensions) of this tensor.
  int64_t getRank() const;

  /// Check if a dimension is dynamic (runtime-determined).
  bool isDynamicDim(unsigned idx) const;

  /// Check if all dimensions are static.
  bool hasStaticShape() const;
};

} // namespace simp
} // namespace mlir

#endif // MLIR_SIMP_TYPES_HPP
