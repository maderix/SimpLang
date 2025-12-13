//===- SimpTypes.td - Simp dialect types -------------------*- tablegen -*-===//
//
// Part of the SimpLang Project
//
// This file defines the types for the Simp dialect.
//
//===----------------------------------------------------------------------===//

#ifndef SIMP_TYPES
#define SIMP_TYPES

include "SimpBase.td"

//===----------------------------------------------------------------------===//
// Simp Type Definitions
//===----------------------------------------------------------------------===//

// Simp Array Type: !simp.array<element-type>
//
// Represents a 1D dynamically-sized array of elements. This is the primary
// data structure for Phase 1 implementation.
//
// Example:
//   !simp.array<f64>  - Array of 64-bit floats
//   !simp.array<i32>  - Array of 32-bit integers
//   !simp.array<f32>  - Array of 32-bit floats
//
// Design notes:
// - Dynamic size (runtime-determined length)
// - Element type is statically known
// - SSA-pure: operations return new arrays
// - Lowers to memref.alloc in lowering pass
//
def Simp_ArrayType : DialectType<Simp_Dialect,
    CPred<"::mlir::isa<::mlir::simp::ArrayType>($_self)">,
    "Simp array type",
    "::mlir::Type"> {

  let description = [{
    Array type representing a 1D dynamically-sized array of elements.

    Syntax: `!simp.array<element-type>`

    This type represents arrays in the Simp dialect before lowering to
    memref. It preserves high-level array semantics for optimization.

    During lowering (Phase 1):
      !simp.array<f64> → memref<?xf64>
      !simp.array<i32> → memref<?xi32>
  }];
}

// Type constraint for any Simp array (used in operation definitions)
def Simp_AnyArray : Type<Simp_ArrayType.predicate, "any simp array">;

//===----------------------------------------------------------------------===//
// Simp Tensor Type
//===----------------------------------------------------------------------===//

// Simp Tensor Type: !simp.tensor<shape, element-type>
//
// Represents an N-dimensional tensor with statically or dynamically known shape.
// This type is used for Phase 2+ to support multi-dimensional array operations
// and higher-level tensor operations.
//
// NOTE: Named SimpTensorType (not TensorType) to avoid conflict with MLIR's
//       built-in mlir::TensorType class.
//
// Example:
//   !simp.tensor<10x20xf64>     - 2D tensor, static shape 10×20, f64 elements
//   !simp.tensor<?x?xf32>       - 2D tensor, dynamic shape, f32 elements
//   !simp.tensor<3x224x224xf32> - 3D tensor for CNN input (MobileNet)
//
// Design notes:
// - Shape can be static (compile-time known) or dynamic (runtime-determined)
// - Element type is statically known
// - SSA-pure: operations return new tensors
// - Lowers to memref for memory representation
// - Phase 1: Multi-dim arrays flatten to 1D arrays
// - Phase 2+: Full tensor support with shape inference
//
def Simp_SimpTensorType : DialectType<Simp_Dialect,
    CPred<"::mlir::isa<::mlir::simp::SimpTensorType>($_self)">,
    "Simp tensor type",
    "::mlir::Type"> {

  let description = [{
    Tensor type representing an N-dimensional tensor with static or dynamic shape.

    Syntax: `!simp.tensor<shape-spec, element-type>`

    Examples:
      !simp.tensor<10x20xf64>     - 2D static tensor
      !simp.tensor<?x?xf32>       - 2D dynamic tensor
      !simp.tensor<3x224x224xf32> - 3D static tensor

    This type represents tensors in the Simp dialect before lowering to
    memref. It preserves high-level tensor semantics for optimization.

    During lowering (Phase 2+):
      !simp.tensor<10x20xf64> → memref<10x20xf64>
      !simp.tensor<?x?xf32>   → memref<?x?xf32>
  }];
}

// Type constraint for any Simp tensor (used in operation definitions)
def Simp_AnySimpTensor : Type<Simp_SimpTensorType.predicate, "any simp tensor">;

#endif // SIMP_TYPES
