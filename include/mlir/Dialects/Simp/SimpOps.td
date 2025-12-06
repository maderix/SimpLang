//===- SimpOps.td - Simp dialect operations ----------------*- tablegen -*-===//
//
// Part of the SimpLang Project
//
// This file defines the operations for the Simp dialect.
//
//===----------------------------------------------------------------------===//

#ifndef SIMP_OPS
#define SIMP_OPS

include "SimpBase.td"
include "SimpTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Simp Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Simp_ConstantOp
//===----------------------------------------------------------------------===//

def Simp_ConstantOp : Simp_Op<"constant", [NoSideEffect, ConstantLike]> {
  let summary = "Constant value";

  let description = [{
    The `simp.constant` operation produces a constant SSA value from a literal
    attribute. This is used for all compile-time constants in SimpLang.

    Examples:
    ```mlir
    %0 = simp.constant 42 : i64
    %1 = simp.constant 3.14 : f64
    %2 = simp.constant 100.0 : f32
    ```

    Lowering:
    - Phase 1: Lower to arith.constant
    - Canonicalization: Fold constant operations
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$result);

  // Custom assembly format: simp.constant(<value>) : <type>
  let assemblyFormat = "`(` $value `)` attr-dict `:` type($result)";

  // Enable constant folding
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Binary Arithmetic Operations
//===----------------------------------------------------------------------===//

def Simp_AddOp : Simp_Op<"add", [NoSideEffect, Commutative]> {
  let summary = "Addition operation";

  let description = [{
    The `simp.add` operation performs element-wise addition.

    Examples:
    ```mlir
    %0 = simp.add %a, %b : (f64, f64) -> f64
    %1 = simp.add %x, %y : (i32, i32) -> i32
    ```

    Properties:
    - Commutative (a + b = b + a)
    - No side effects
    - Type of result matches operand types

    Lowering:
    - Phase 1: Lower to arith.addf (floats) or arith.addi (integers)
    - Canonicalization: Fold constant additions, identity elimination
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
  let hasCanonicalizer = 1;
}

def Simp_SubOp : Simp_Op<"sub", [NoSideEffect]> {
  let summary = "Subtraction operation";

  let description = [{
    The `simp.sub` operation performs element-wise subtraction.

    Examples:
    ```mlir
    %0 = simp.sub %a, %b : (f64, f64) -> f64
    %1 = simp.sub %x, %y : (i32, i32) -> i32
    ```

    Lowering:
    - Phase 1: Lower to arith.subf (floats) or arith.subi (integers)
    - Canonicalization: Fold constant subtractions, identity elimination
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
  let hasCanonicalizer = 1;
}

def Simp_MulOp : Simp_Op<"mul", [NoSideEffect, Commutative]> {
  let summary = "Multiplication operation";

  let description = [{
    The `simp.mul` operation performs element-wise multiplication.

    Examples:
    ```mlir
    %0 = simp.mul %a, %b : (f64, f64) -> f64
    %1 = simp.mul %x, %y : (i32, i32) -> i32
    ```

    Properties:
    - Commutative (a * b = b * a)
    - No side effects

    Lowering:
    - Phase 1: Lower to arith.mulf (floats) or arith.muli (integers)
    - Canonicalization: Fold constant multiplications, identity elimination
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
  let hasCanonicalizer = 1;
}

def Simp_DivOp : Simp_Op<"div", [NoSideEffect]> {
  let summary = "Division operation";

  let description = [{
    The `simp.div` operation performs element-wise division.

    Examples:
    ```mlir
    %0 = simp.div %a, %b : (f64, f64) -> f64
    %1 = simp.div %x, %y : (i32, i32) -> i32
    ```

    Lowering:
    - Phase 1: Lower to arith.divf (floats) or arith.divsi (signed integers)
    - Canonicalization: Fold constant divisions, identity elimination
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
  let hasCanonicalizer = 1;
}

def Simp_ModOp : Simp_Op<"mod", [NoSideEffect]> {
  let summary = "Modulo operation";

  let description = [{
    The `simp.mod` operation performs the modulo (remainder) operation.

    Examples:
    ```mlir
    %0 = simp.mod %a, %b : (f64, f64) -> f64
    %1 = simp.mod %x, %y : (i32, i32) -> i32
    ```

    Lowering:
    - Phase 1: Lower to arith.remf (floats) or arith.remsi (signed integers)
    - Canonicalization: Fold constant modulos
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Simp_NegOp
//===----------------------------------------------------------------------===//

def Simp_NegOp : Simp_Op<"neg", [NoSideEffect]> {
  let summary = "Unary negation operation";
  let description = [{
    The `simp.neg` operation performs unary negation on a value.

    Example:
    ```mlir
    %result = simp.neg %value : f64 -> f64
    ```
  }];

  let arguments = (ins AnyType:$operand);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Simp_ArrayCreateOp
//===----------------------------------------------------------------------===//

def Simp_ArrayCreateOp : Simp_Op<"array_create", [NoSideEffect]> {
  let summary = "Create array with given size";

  let description = [{
    The `simp.array_create` operation allocates a new array with the
    specified size. The array is uninitialized.

    Examples:
    ```mlir
    %size = simp.constant 100 : i64
    %arr = simp.array_create %size : !simp.array<f64>
    ```

    SSA-Pure Semantics:
    - Returns a new SSA value (the array)
    - No global state mutation
    - Size is a runtime value (dynamic allocation)

    Lowering:
    - Phase 1: Lower to memref.alloc
      simp.array_create %N : !simp.array<f64>
      → memref.alloc(%N) : memref<?xf64>
  }];

  let arguments = (ins I64:$size);
  let results = (outs Simp_ArrayType:$result);

  // Custom assembly format: simp.array_create <size> : <array-type>
  let assemblyFormat = "$size attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Simp_ArrayGetOp
//===----------------------------------------------------------------------===//

def Simp_ArrayGetOp : Simp_Op<"array_get", [NoSideEffect]> {
  let summary = "Get element from array (load)";

  let description = [{
    The `simp.array_get` operation loads an element from an array at the
    specified index. This is a pure operation (no side effects).

    Examples:
    ```mlir
    %elem = simp.array_get %arr[%idx] : !simp.array<f64> -> f64
    ```

    SSA-Pure Semantics:
    - Reads value without modifying the array
    - Returns element value as SSA result
    - Array indexing is 0-based

    Lowering:
    - Phase 1: Lower to memref.load
      simp.array_get %arr[%idx] : !simp.array<f64>
      → memref.load %memref[%idx] : memref<?xf64>

    Future Optimizations (Phase 2):
    - Affine analysis can detect access patterns
    - Vectorization can group adjacent loads
  }];

  let arguments = (ins Simp_ArrayType:$array, I64:$index);
  let results = (outs AnyType:$result);

  // Custom assembly format: simp.array_get <array>[<index>] : <array-type> -> <elem-type>
  let assemblyFormat = "$array `[` $index `]` attr-dict `:` type($array) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Simp_ArraySetOp
//===----------------------------------------------------------------------===//

def Simp_ArraySetOp : Simp_Op<"array_set", [NoSideEffect]> {
  let summary = "Set element in array (functional)";

  let description = [{
    The `simp.array_set` operation returns a new array with the specified
    element updated. This is a functional (SSA-pure) operation that does
    not mutate the input array.

    Examples:
    ```mlir
    %arr2 = simp.array_set %arr1[%idx], %val : !simp.array<f64>
    ```

    SSA-Pure Semantics:
    - Returns a NEW array with updated element
    - Does not modify the input array
    - Enables MLIR optimization passes to reason about data flow

    Lowering:
    - Phase 1: Lower to memref.store on the memref representation
      Note: After lowering to memref, the SSA-pure semantics are relaxed
      since memref operations can have side effects.

    Future Optimizations (Phase 2):
    - Copy-on-write optimizations
    - Escape analysis to eliminate unnecessary copies
  }];

  let arguments = (ins Simp_ArrayType:$array, I64:$index, AnyType:$value);
  let results = (outs Simp_ArrayType:$result);

  // Custom assembly format: simp.array_set <array>[<index>], <value> : <array-type>
  let assemblyFormat = "$array `[` $index `]` `,` $value attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Tensor Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Simp_TensorCreateOp
//===----------------------------------------------------------------------===//

def Simp_TensorCreateOp : Simp_Op<"tensor_create", [NoSideEffect]> {
  let summary = "Create tensor with given shape";

  let description = [{
    The `simp.tensor_create` operation allocates a new tensor with the
    specified shape. The tensor is uninitialized.

    Examples:
    ```mlir
    %tensor = simp.tensor_create : !simp.tensor<2x3x4xf32>
    ```

    SSA-Pure Semantics:
    - Returns a new SSA value (the tensor)
    - No global state mutation
    - Shape is compile-time constant

    Lowering:
    - Phase 1: Lower to memref.alloc
      simp.tensor_create : !simp.tensor<2x3x4xf32>
      → memref.alloc() : memref<2x3x4xf32>
  }];

  let results = (outs Simp_SimpTensorType:$result);

  // Custom assembly format: simp.tensor_create : <tensor-type>
  let assemblyFormat = "attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Simp_TensorGetOp
//===----------------------------------------------------------------------===//

def Simp_TensorGetOp : Simp_Op<"tensor_get", [NoSideEffect]> {
  let summary = "Get element from tensor (load)";

  let description = [{
    The `simp.tensor_get` operation loads an element from a tensor at the
    specified multi-dimensional index.

    Examples:
    ```mlir
    %elem = simp.tensor_get %tensor[%i, %j, %k] : !simp.tensor<2x3x4xf32> -> f32
    ```

    SSA-Pure Semantics:
    - Reads value without modifying the tensor
    - Returns element value as SSA result
    - Tensor indexing is 0-based
    - Multi-dimensional indexing supported

    Lowering:
    - Phase 1: Lower to memref.load
      simp.tensor_get %tensor[%i, %j, %k] : !simp.tensor<2x3x4xf32>
      → memref.load %memref[%i, %j, %k] : memref<2x3x4xf32>
  }];

  let arguments = (ins Simp_SimpTensorType:$tensor, Variadic<I64>:$indices);
  let results = (outs AnyType:$result);

  // Custom assembly format: simp.tensor_get <tensor>[<indices>] : <types>
  let assemblyFormat = "$tensor `[` $indices `]` attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Simp_TensorSetOp
//===----------------------------------------------------------------------===//

def Simp_TensorSetOp : Simp_Op<"tensor_set", [NoSideEffect]> {
  let summary = "Set element in tensor (functional)";

  let description = [{
    The `simp.tensor_set` operation returns a new tensor with the specified
    element updated. This is a functional (SSA-pure) operation.

    Examples:
    ```mlir
    %tensor2 = simp.tensor_set %tensor1[%i, %j, %k], %val : !simp.tensor<2x3x4xf32>
    ```

    SSA-Pure Semantics:
    - Returns a NEW tensor with updated element
    - Does not modify the input tensor
    - Multi-dimensional indexing supported

    Lowering:
    - Phase 1: Lower to memref.store on the memref representation
      Note: After lowering to memref, the SSA-pure semantics are relaxed.
  }];

  let arguments = (ins Simp_SimpTensorType:$tensor, Variadic<I64>:$indices, AnyType:$value);
  let results = (outs Simp_SimpTensorType:$result);

  // Custom assembly format: simp.tensor_set <tensor>[<indices>], <value> : <types>
  let assemblyFormat = "$tensor `[` $indices `]` `,` $value attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Simp_TensorFromArrayOp
//===----------------------------------------------------------------------===//

def Simp_TensorFromArrayOp : Simp_Op<"tensor_from_array", [NoSideEffect]> {
  let summary = "Create tensor from array (bulk conversion with optional offset)";

  let description = [{
    The `simp.tensor_from_array` operation creates a tensor from an existing
    array by reinterpreting its memory layout. This is a zero-copy operation
    that wraps the array data in a tensor view.

    Examples:
    ```mlir
    // Without offset (start from beginning)
    %tensor = simp.tensor_from_array %array, %c0 : !simp.array<f32>, i64 -> !simp.tensor<768x1024xf32>

    // With offset (for multi-layer weights)
    %offset = simp.constant(589824 : i64) : i64  // layer 1 offset
    %tensor = simp.tensor_from_array %array, %offset : !simp.array<f32>, i64 -> !simp.tensor<768x768xf32>
    ```

    Use Cases:
    - Loading large weight matrices (e.g., 32000×768 embeddings)
    - Slicing multi-layer weights (e.g., wq[layer, :, :] from flat array)
    - Zero-copy reshaping for ML operations

    SSA-Pure Semantics:
    - Returns a tensor view of the array data starting at offset
    - No data copying (reinterpret cast with offset)
    - offset + tensor_size must not exceed array size

    Lowering:
    - Phase 1: Lower to memref.reinterpret_cast with offset
      simp.tensor_from_array %array, %offset : !simp.array<f32>, i64 -> !simp.tensor<MxNxf32>
      → memref.reinterpret_cast %memref : memref<?xf32> to memref<MxNxf32> (offset = %offset)
  }];

  let arguments = (ins Simp_ArrayType:$array, I64:$offset);
  let results = (outs Simp_SimpTensorType:$result);

  // Custom assembly format: simp.tensor_from_array <array>, <offset> : <types> -> <tensor-type>
  let assemblyFormat = "$array `,` $offset attr-dict `:` type($array) `,` type($offset) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Tensor Element-wise Operations
//===----------------------------------------------------------------------===//

def Simp_TensorAddOp : Simp_Op<"tensor_add", [NoSideEffect, Commutative]> {
  let summary = "Element-wise tensor addition";
  let description = [{
    Performs element-wise addition of two tensors with the same shape.
    Returns a new tensor with the result.
  }];
  let arguments = (ins Simp_SimpTensorType:$lhs, Simp_SimpTensorType:$rhs);
  let results = (outs Simp_SimpTensorType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)";
}

def Simp_TensorMulOp : Simp_Op<"tensor_mul", [NoSideEffect, Commutative]> {
  let summary = "Element-wise tensor multiplication";
  let description = [{
    Performs element-wise multiplication of two tensors with the same shape.
    Returns a new tensor with the result.
  }];
  let arguments = (ins Simp_SimpTensorType:$lhs, Simp_SimpTensorType:$rhs);
  let results = (outs Simp_SimpTensorType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)";
}

def Simp_TensorSubOp : Simp_Op<"tensor_sub", [NoSideEffect]> {
  let summary = "Element-wise tensor subtraction";
  let description = [{
    Performs element-wise subtraction of two tensors with the same shape.
    Returns a new tensor with the result.
  }];
  let arguments = (ins Simp_SimpTensorType:$lhs, Simp_SimpTensorType:$rhs);
  let results = (outs Simp_SimpTensorType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)";
}

def Simp_TensorDivOp : Simp_Op<"tensor_div", [NoSideEffect]> {
  let summary = "Element-wise tensor division";
  let description = [{
    Performs element-wise division of two tensors with the same shape.
    Returns a new tensor with the result.
  }];
  let arguments = (ins Simp_SimpTensorType:$lhs, Simp_SimpTensorType:$rhs);
  let results = (outs Simp_SimpTensorType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)";
}

def Simp_TensorReluOp : Simp_Op<"tensor_relu", [NoSideEffect]> {
  let summary = "Element-wise ReLU activation";
  let description = [{
    Performs element-wise ReLU (Rectified Linear Unit) activation.
    ReLU(x) = max(0, x). Returns a new tensor with the result.
  }];
  let arguments = (ins Simp_SimpTensorType:$input);
  let results = (outs Simp_SimpTensorType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def Simp_TensorSigmoidOp : Simp_Op<"tensor_sigmoid", [NoSideEffect]> {
  let summary = "Element-wise sigmoid activation";
  let description = [{
    Performs element-wise sigmoid activation.
    sigmoid(x) = 1 / (1 + e^(-x)). Returns a new tensor with the result.
  }];
  let arguments = (ins Simp_SimpTensorType:$input);
  let results = (outs Simp_SimpTensorType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def Simp_TensorTanhOp : Simp_Op<"tensor_tanh", [NoSideEffect]> {
  let summary = "Element-wise tanh activation";
  let description = [{
    Performs element-wise hyperbolic tangent activation.
    tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x)). Returns a new tensor with the result.
  }];
  let arguments = (ins Simp_SimpTensorType:$input);
  let results = (outs Simp_SimpTensorType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Tensor Reduction Operations
//===----------------------------------------------------------------------===//

def Simp_TensorSumOp : Simp_Op<"tensor_sum", [NoSideEffect]> {
  let summary = "Tensor sum reduction";
  let description = [{
    Reduces a tensor by summing all elements (full reduction) or along a specific axis.

    Full reduction (returns scalar):
      %sum = simp.tensor_sum %tensor : !simp.tensor<2x3xf32> -> f32

    Axis reduction (returns tensor with reduced dimension):
      %sum = simp.tensor_sum %tensor, %axis : (!simp.tensor<2x3xf32>, i64) -> !simp.tensor<2xf32>
  }];
  let arguments = (ins Simp_SimpTensorType:$input, Optional<I64>:$axis);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input (`,` $axis^)? attr-dict `:` type($input) `->` type($result)";
}

def Simp_TensorMeanOp : Simp_Op<"tensor_mean", [NoSideEffect]> {
  let summary = "Tensor mean reduction";
  let description = [{
    Reduces a tensor by computing the mean of all elements (full reduction) or along a specific axis.

    Full reduction (returns scalar):
      %mean = simp.tensor_mean %tensor : !simp.tensor<2x3xf32> -> f32
  }];
  let arguments = (ins Simp_SimpTensorType:$input, Optional<I64>:$axis);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input (`,` $axis^)? attr-dict `:` type($input) `->` type($result)";
}

def Simp_TensorMaxOp : Simp_Op<"tensor_max", [NoSideEffect]> {
  let summary = "Tensor max reduction";
  let description = [{
    Reduces a tensor by finding the maximum element (full reduction) or along a specific axis.

    Full reduction (returns scalar):
      %max = simp.tensor_max %tensor : !simp.tensor<2x3xf32> -> f32
  }];
  let arguments = (ins Simp_SimpTensorType:$input, Optional<I64>:$axis);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input (`,` $axis^)? attr-dict `:` type($input) `->` type($result)";
}

def Simp_TensorMinOp : Simp_Op<"tensor_min", [NoSideEffect]> {
  let summary = "Tensor min reduction";
  let description = [{
    Reduces a tensor by finding the minimum element (full reduction) or along a specific axis.

    Full reduction (returns scalar):
      %min = simp.tensor_min %tensor : !simp.tensor<2x3xf32> -> f32
  }];
  let arguments = (ins Simp_SimpTensorType:$input, Optional<I64>:$axis);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input (`,` $axis^)? attr-dict `:` type($input) `->` type($result)";
}

def Simp_TensorArgmaxOp : Simp_Op<"tensor_argmax", [NoSideEffect]> {
  let summary = "Tensor argmax reduction";
  let description = [{
    Reduces a tensor by finding the index of the maximum element (full reduction) or along a specific axis.

    Full reduction (returns scalar index):
      %argmax = simp.tensor_argmax %tensor : !simp.tensor<2x3xf32> -> i64
  }];
  let arguments = (ins Simp_SimpTensorType:$input, Optional<I64>:$axis);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input (`,` $axis^)? attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Tensor Memory Operations
//===----------------------------------------------------------------------===//

def Simp_TensorReshapeOp : Simp_Op<"tensor_reshape", [NoSideEffect]> {
  let summary = "Reshape tensor without copying data";
  let description = [{
    Changes the shape of a tensor without reordering elements. Total number of elements must remain the same.

    Example:
      %reshaped = simp.tensor_reshape %input, %dim0, %dim1 : (!simp.tensor<6xf32>, i64, i64) -> !simp.tensor<2x3xf32>

    The data layout (row-major) is preserved, only the interpretation changes.
  }];
  let arguments = (ins Simp_SimpTensorType:$input, Variadic<I64>:$new_shape);
  let results = (outs Simp_SimpTensorType:$result);
  let assemblyFormat = "$input `,` $new_shape attr-dict `:` `(` type($input) `,` type($new_shape) `)` `->` type($result)";
}

def Simp_TensorTransposeOp : Simp_Op<"tensor_transpose", [NoSideEffect]> {
  let summary = "Transpose tensor dimensions";
  let description = [{
    Swaps tensor dimensions according to a permutation.

    For 2D tensors (no permutation specified):
      %transposed = simp.tensor_transpose %input : !simp.tensor<2x3xf32> -> !simp.tensor<3x2xf32>

    For N-D tensors with explicit permutation:
      %transposed = simp.tensor_transpose %input, %perm0, %perm1, %perm2 : (!simp.tensor<2x3x4xf32>, i64, i64, i64) -> !simp.tensor<4x2x3xf32>

    Requires data reordering (creates a copy).
  }];
  let arguments = (ins Simp_SimpTensorType:$input, Variadic<I64>:$permutation);
  let results = (outs Simp_SimpTensorType:$result);
  let assemblyFormat = "$input (`,` $permutation^)? attr-dict `:` `(` type($input) (`,` type($permutation)^)? `)` `->` type($result)";
}

def Simp_TensorSliceOp : Simp_Op<"tensor_slice", [NoSideEffect]> {
  let summary = "Extract sub-tensor slice";
  let description = [{
    Extracts a contiguous sub-tensor using start and end indices for each dimension.

    Example:
      %slice = simp.tensor_slice %input, %start0, %end0, %start1, %end1 : (!simp.tensor<4x5xf32>, i64, i64, i64, i64) -> !simp.tensor<2x2xf32>

    For each dimension i, extracts elements [start_i, end_i) (end is exclusive).
    Creates a copy of the extracted region.
  }];
  let arguments = (ins Simp_SimpTensorType:$input, Variadic<I64>:$indices);
  let results = (outs Simp_SimpTensorType:$result);
  let assemblyFormat = "$input `,` $indices attr-dict `:` `(` type($input) `,` type($indices) `)` `->` type($result)";
}

def Simp_TensorGatherOp : Simp_Op<"tensor_gather", [NoSideEffect]> {
  let summary = "Gather slices along specified axis (N-D)";
  let description = [{
    Gathers slices from source tensor along the specified axis using indices.
    Supports arbitrary N-dimensional tensors with optimized lowering.

    The indices tensor must be 1D (rank-1) containing integer indices.
    The result shape is the source shape with the gather axis dimension
    replaced by the length of the indices tensor.

    Examples:
      // 1D gather (embedding lookup)
      %result = simp.tensor_gather %embeddings, %tokens, %axis
        : (!simp.tensor<10000xf32>, !simp.tensor<512xi64>, i64)
        -> !simp.tensor<512xf32>

      // 2D gather along axis 0 (select rows)
      %result = simp.tensor_gather %matrix, %row_indices, %axis
        : (!simp.tensor<1000x512xf32>, !simp.tensor<32xi64>, i64)
        -> !simp.tensor<32x512xf32>

      // 3D gather along axis 1
      %result = simp.tensor_gather %tensor3d, %indices, %axis
        : (!simp.tensor<8x100x64xf32>, !simp.tensor<16xi64>, i64)
        -> !simp.tensor<8x16x64xf32>

      // 5D gather along axis 3
      %result = simp.tensor_gather %tensor5d, %indices, %axis
        : (!simp.tensor<2x4x8x16x32xf32>, !simp.tensor<10xi64>, i64)
        -> !simp.tensor<2x4x8x10x32xf32>

    Optimization Notes:
    - Lowering uses nested loops with innermost dimension vectorizable
    - Memory access pattern optimized for cache locality when axis >= rank-2
    - Supports all numeric types (f32, f64, i32, i64, etc.)
    - No bounds checking for maximum performance (undefined if out-of-bounds)

    Lowering Path:
    ```
    simp.tensor_gather
      → nested scf.for loops with memref.load
      → [vectorization + loop tiling optimizations]
      → LLVM IR with SIMD instructions
    ```
  }];

  let arguments = (ins
    Simp_SimpTensorType:$source,
    Simp_SimpTensorType:$indices,  // Must be 1D tensor of i64
    Optional<I64>:$axis             // Default: 0
  );
  let results = (outs Simp_SimpTensorType:$result);

  let assemblyFormat = "$source `,` $indices (`,` $axis^)? attr-dict `:` `(` type($source) `,` type($indices) (`,` type($axis)^)? `)` `->` type($result)";
}

def Simp_TensorScatterOp : Simp_Op<"tensor_scatter", [NoSideEffect]> {
  let summary = "Scatter values along specified axis (N-D)";
  let description = [{
    Scatters values into destination tensor along the specified axis using indices.
    Supports arbitrary N-dimensional tensors with optimized lowering.

    The indices tensor must be 1D (rank-1) containing integer indices.
    The values tensor shape must match the destination shape with the scatter
    axis dimension equal to the length of the indices tensor.

    Semantics: Last write wins if indices contain duplicates.

    Examples:
      // 1D scatter (sparse update)
      %result = simp.tensor_scatter %dst, %indices, %values, %axis
        : (!simp.tensor<10000xf32>, !simp.tensor<16xi64>, !simp.tensor<16xf32>, i64)
        -> !simp.tensor<10000xf32>

      // 2D scatter along axis 0 (update rows)
      %result = simp.tensor_scatter %matrix, %row_indices, %new_rows, %axis
        : (!simp.tensor<1000x512xf32>, !simp.tensor<8xi64>, !simp.tensor<8x512xf32>, i64)
        -> !simp.tensor<1000x512xf32>

      // 3D scatter along axis 2
      %result = simp.tensor_scatter %tensor3d, %indices, %updates, %axis
        : (!simp.tensor<4x8x100xf32>, !simp.tensor<12xi64>, !simp.tensor<4x8x12xf32>, i64)
        -> !simp.tensor<4x8x100xf32>

      // 4D scatter along axis 1
      %result = simp.tensor_scatter %tensor4d, %indices, %updates, %axis
        : (!simp.tensor<2x50x16x32xf32>, !simp.tensor<5xi64>, !simp.tensor<2x5x16x32xf32>, i64)
        -> !simp.tensor<2x50x16x32xf32>

    Optimization Notes:
    - Lowering uses nested loops with innermost dimension vectorizable
    - Minimizes memory copies (single allocation + scatter writes)
    - Memory access pattern optimized for cache locality
    - Supports all numeric types (f32, f64, i32, i64, etc.)
    - No bounds checking for maximum performance (undefined if out-of-bounds)

    Lowering Path:
    ```
    simp.tensor_scatter
      → memref.copy (dst to result) + nested scf.for loops with memref.store
      → [vectorization + loop tiling optimizations]
      → LLVM IR with SIMD instructions
    ```
  }];

  let arguments = (ins
    Simp_SimpTensorType:$dst,
    Simp_SimpTensorType:$indices,  // Must be 1D tensor of i64
    Simp_SimpTensorType:$values,
    Optional<I64>:$axis             // Default: 0
  );
  let results = (outs Simp_SimpTensorType:$result);

  let assemblyFormat = "$dst `,` $indices `,` $values (`,` $axis^)? attr-dict `:` `(` type($dst) `,` type($indices) `,` type($values) (`,` type($axis)^)? `)` `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Simp_MatMulOp
//===----------------------------------------------------------------------===//

def Simp_MatMulOp : Simp_Op<"matmul", [NoSideEffect]> {
  let summary = "Matrix multiplication (GEMM)";

  let description = [{
    The `simp.matmul` operation performs matrix multiplication (GEMM).

    Examples:
    ```mlir
    %C = simp.matmul %A, %B, %m, %k, %n : (!simp.array<f32>, !simp.array<f32>, i64, i64, i64) -> !simp.array<f32>
    ```

    Mathematical Operation:
      C = A × B
      Where A is MxK (stored as 1D array of M*K elements)
            B is KxN (stored as 1D array of K*N elements)
            C is MxN (stored as 1D array of M*N elements)

    Arrays are interpreted as row-major matrices.

    Lowering Path:
    ```
    simp.matmul
      → linalg.matmul (via ConvertSimpToLinalg)
      → [fusion + tiling optimizations]
      → loops + arith ops
    ```

    Session 9 Implementation:
    - Lower to linalg.matmul for optimization
    - Use memref.reinterpret_cast for 2D view creation
  }];

  let arguments = (ins
    Simp_ArrayType:$lhs,    // A: MxK matrix as 1D array
    Simp_ArrayType:$rhs,    // B: KxN matrix as 1D array
    Simp_ArrayType:$output, // C: MxN matrix as 1D array (pre-allocated output buffer)
    I64:$m,                 // Rows of A
    I64:$k,                 // Cols of A / Rows of B
    I64:$n,                 // Cols of B
    I64:$lhs_offset,        // Offset into lhs array
    I64:$rhs_offset,        // Offset into rhs array
    I64:$output_offset      // Offset into output array
  );
  let results = (outs Simp_ArrayType:$result);  // Returns the same output buffer (for chaining)

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $output `,` $m `,` $k `,` $n `,` $lhs_offset `,` $rhs_offset `,` $output_offset attr-dict `:`
    `(` type($lhs) `,` type($rhs) `,` type($output) `,` type($m) `,` type($k) `,` type($n) `,` type($lhs_offset) `,` type($rhs_offset) `,` type($output_offset) `)` `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Simp_Conv2DOp
//===----------------------------------------------------------------------===//

def Simp_Conv2DOp : Simp_Op<"conv2d", [NoSideEffect]> {
  let summary = "2D Convolution operation";

  let description = [{
    The `simp.conv2d` operation performs 2D convolution with support for
    strides, padding, and multiple data types.

    Examples:
    ```mlir
    %output = simp.conv2d %input, %weights, %bias, %out_buf,
                          %batch, %in_h, %in_w, %in_c,
                          %out_c, %k_h, %k_w,
                          %stride_h, %stride_w,
                          %pad_h, %pad_w
      : (!simp.array<f32>, !simp.array<f32>, !simp.array<f32>, !simp.array<f32>,
         i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)
      -> !simp.array<f32>
    ```

    Mathematical Operation:
      output[b, oh, ow, oc] = bias[oc] +
        Σ(kh, kw, ic) input[b, oh*stride_h + kh - pad_h, ow*stride_w + kw - pad_w, ic]
                      * weights[oc, kh, kw, ic]

    Layout:
    - Input: NHWC [batch, height, width, in_channels]
    - Weights: [out_channels, kernel_h, kernel_w, in_channels]
    - Bias: [out_channels]
    - Output: NHWC [batch, out_height, out_width, out_channels]

    Where:
      out_height = (in_height + 2*pad_h - kernel_h) / stride_h + 1
      out_width = (in_width + 2*pad_w - kernel_w) / stride_w + 1

    Arrays are stored as flattened 1D arrays with row-major layout.

    Lowering Path:
    ```
    simp.conv2d
      → nested affine loops with memref operations
      → [tiling + vectorization optimizations]
      → LLVM IR
    ```

    Dtype Support:
    Supports all numeric types: f64, f32, f16, i64, i32, i16, i8
  }];

  let arguments = (ins
    Simp_ArrayType:$input,     // Input: NHWC as 1D array
    Simp_ArrayType:$weights,   // Weights: [OC, KH, KW, IC] as 1D array
    Simp_ArrayType:$bias,      // Bias: [OC] as 1D array
    Simp_ArrayType:$output,    // Output: NHWC as 1D array (pre-allocated)
    I64:$batch,                // Batch size (N)
    I64:$in_h,                 // Input height
    I64:$in_w,                 // Input width
    I64:$in_c,                 // Input channels
    I64:$out_c,                // Output channels
    I64:$kernel_h,             // Kernel height
    I64:$kernel_w,             // Kernel width
    I64:$stride_h,             // Vertical stride
    I64:$stride_w,             // Horizontal stride
    I64:$pad_h,                // Vertical padding
    I64:$pad_w                 // Horizontal padding
  );
  let results = (outs Simp_ArrayType:$result);  // Returns the output buffer

  let assemblyFormat = [{
    $input `,` $weights `,` $bias `,` $output `,`
    $batch `,` $in_h `,` $in_w `,` $in_c `,`
    $out_c `,` $kernel_h `,` $kernel_w `,`
    $stride_h `,` $stride_w `,` $pad_h `,` $pad_w
    attr-dict `:`
    `(` type($input) `,` type($weights) `,` type($bias) `,` type($output) `,`
    type($batch) `,` type($in_h) `,` type($in_w) `,` type($in_c) `,`
    type($out_c) `,` type($kernel_h) `,` type($kernel_w) `,`
    type($stride_h) `,` type($stride_w) `,` type($pad_h) `,` type($pad_w) `)`
    `->` type($result)
  }];
}

// ============================================================================
// LLM/Transformer Operations
// ============================================================================

def Simp_RMSNormOp : Simp_Op<"rmsnorm", [NoSideEffect]> {
  let summary = "Root Mean Square Layer Normalization";
  let description = [{
    Computes RMSNorm: output = (input / rms) * weight
    where rms = sqrt(mean(input^2) + epsilon)

    Used in LLaMA and other modern transformers instead of LayerNorm.
  }];

  let arguments = (ins
    Simp_ArrayType:$input,    // Input array [size]
    Simp_ArrayType:$weight,   // Learned weights [size]
    Simp_ArrayType:$output,   // Output array [size] (pre-allocated)
    I64:$size,                // Vector size
    AnyFloat:$epsilon,        // Numerical stability (typically 1e-5)
    I64:$weight_offset        // Offset into weight array for layer-specific weights
  );

  let results = (outs Simp_ArrayType:$result);

  let assemblyFormat = [{
    `(` $input `,` $weight `,` $output `,` $size `,` $epsilon `,` $weight_offset `)`
    attr-dict `:`
    `(` type($input) `,` type($weight) `,` type($output) `,` type($size) `,` type($epsilon) `,` type($weight_offset) `)`
    `->` type($result)
  }];
}

def Simp_SoftmaxOp : Simp_Op<"softmax", [NoSideEffect]> {
  let summary = "Numerically stable softmax";
  let description = [{
    Computes softmax with max subtraction for numerical stability:
    1. max_val = max(input[input_offset:input_offset+size])
    2. exp_vals = exp(input - max_val)
    3. sum_exp = sum(exp_vals)
    4. output[output_offset:output_offset+size] = exp_vals / sum_exp

    Used in attention mechanisms.
  }];

  let arguments = (ins
    Simp_ArrayType:$input,    // Input logits [size]
    Simp_ArrayType:$output,   // Output probabilities [size] (pre-allocated)
    I64:$size,                // Vector size
    I64:$input_offset,        // Offset into input array
    I64:$output_offset        // Offset into output array
  );

  let results = (outs Simp_ArrayType:$result);

  let assemblyFormat = [{
    `(` $input `,` $output `,` $size `,` $input_offset `,` $output_offset `)`
    attr-dict `:`
    `(` type($input) `,` type($output) `,` type($size) `,` type($input_offset) `,` type($output_offset) `)`
    `->` type($result)
  }];
}

def Simp_SiLUOp : Simp_Op<"silu", [NoSideEffect]> {
  let summary = "SiLU (Swish) activation function";
  let description = [{
    Computes SiLU activation: output = x / (1 + exp(-x))
    Also known as Swish activation.

    Used in SwiGLU feedforward networks.
  }];

  let arguments = (ins
    Simp_ArrayType:$input,    // Input array [size]
    Simp_ArrayType:$output,   // Output array [size] (pre-allocated)
    I64:$size                 // Vector size
  );

  let results = (outs Simp_ArrayType:$result);

  let assemblyFormat = [{
    `(` $input `,` $output `,` $size `)`
    attr-dict `:`
    `(` type($input) `,` type($output) `,` type($size) `)`
    `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Quantization Operations
//===----------------------------------------------------------------------===//

def Simp_DequantW4Op : Simp_Op<"dequant_w4", [NoSideEffect]> {
  let summary = "Dequantize W4 (4-bit) weight value";
  let description = [{
    Dequantizes a single 4-bit weight value from packed int8 array using per-group scales and zeros.

    Formula:
      qval = extract_4bit(qweights[idx/2], idx%2)
      result = (qval - zeros[idx/group_size]) * scales[idx/group_size]

    Used in quantized inference to convert 4-bit weights to FP32 on-the-fly.
  }];

  let arguments = (ins
    Simp_ArrayType:$qweights,    // Packed int8 array (2 weights per byte)
    Simp_ArrayType:$scales,      // FP32 scales per group
    Simp_ArrayType:$zeros,       // FP32 zero points per group
    I64:$idx,                    // Weight index to dequantize
    I64:$group_size              // Quantization group size (e.g., 128)
  );

  let results = (outs F32:$result);  // Dequantized FP32 value

  let assemblyFormat = [{
    `(` $qweights `,` $scales `,` $zeros `,` $idx `,` $group_size `)`
    attr-dict `:`
    `(` type($qweights) `,` type($scales) `,` type($zeros) `,` type($idx) `,` type($group_size) `)`
    `->` type($result)
  }];
}

def Simp_MatMulQuantOp : Simp_Op<"matmul_quant", [NoSideEffect]> {
  let summary = "Quantized matrix multiplication with on-the-fly dequantization";
  let description = [{
    Performs matrix multiplication with 4-bit quantized weights:
      output = qweights @ input

    Where:
      - qweights: MxK quantized matrix (W4 format, packed in int8 array)
      - input: Kx1 FP32 vector
      - output: Mx1 FP32 vector

    The operation dequantizes weights on-the-fly using per-group scales and zeros:
      for i in range(M):
        sum = 0
        for j in range(K):
          w_idx = offset + i*K + j
          w_val = dequant_w4(qweights, scales, zeros, w_idx, group_size)
          sum += w_val * input[j]
        output[i] = sum

    Lowering Path:
    ```
    simp.matmul_quant
      → tiled loops with dequantization
      → [tiling + vectorization optimizations]
      → loops + arith ops
    ```
  }];

  let arguments = (ins
    Simp_ArrayType:$qweights,    // Packed W4 weights as int8 array
    Simp_ArrayType:$scales,      // FP32 scales per group
    Simp_ArrayType:$zeros,       // FP32 zero points per group
    Simp_ArrayType:$input,       // FP32 input vector/matrix
    Simp_ArrayType:$output,      // FP32 output buffer (pre-allocated)
    I64:$rows,                   // M: rows of weight matrix
    I64:$cols,                   // K: cols of weight matrix / length of input
    I64:$group_size,             // Quantization group size
    I64:$offset                  // Offset into qweights for layer-specific weights
  );

  let results = (outs Simp_ArrayType:$result);  // Returns output buffer

  let assemblyFormat = [{
    `(` $qweights `,` $scales `,` $zeros `,` $input `,` $output `,`
        $rows `,` $cols `,` $group_size `,` $offset `)`
    attr-dict `:`
    `(` type($qweights) `,` type($scales) `,` type($zeros) `,` type($input) `,` type($output) `,`
        type($rows) `,` type($cols) `,` type($group_size) `,` type($offset) `)`
    `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// TensorMatMulOp - Matrix multiplication for tensor types (2D/3D/4D)
//===----------------------------------------------------------------------===//

def Simp_TensorMatMulOp : Simp_Op<"tensor_matmul", [NoSideEffect]> {
  let summary = "Matrix multiplication for tensors (GEMM, batched GEMM)";

  let description = [{
    Matrix multiplication for SimpTensorType with support for:
    - 2D matmul: Standard GEMM
    - 3D matmul: Batched GEMM (broadcast semantics)
    - 4D matmul: NHWC-aware batched operations for CNNs

    Supported cases:

    1. Standard 2D GEMM:
       A: tensor<M, K>, B: tensor<K, N> → C: tensor<M, N>
       ```simplang
       f32<128, 256> A;
       f32<256, 512> B;
       f32<128, 512> C = tensor_matmul(A, B);
       ```

    2. Batched 3D GEMM:
       A: tensor<B, M, K>, B: tensor<B, K, N> → C: tensor<B, M, N>
       ```simplang
       f32<32, 128, 256> A;  // 32 batches
       f32<32, 256, 512> B;
       f32<32, 128, 512> C = tensor_matmul(A, B);
       ```

    3. 4D NHWC-aware operations (for CNNs):
       Input: tensor<N, H, W, C_in>
       Weights: tensor<C_out, C_in>  (fully connected layer)
       Output: tensor<N, H, W, C_out>

       The operation reshapes to (N*H*W, C_in) × (C_in, C_out) → (N*H*W, C_out)
       and reshapes back, optimizing for NHWC layout (channels innermost).

    Layout attribute:
    - "NCHW": Channels first (PyTorch style)
    - "NHWC": Channels last (TensorFlow style, better cache/vectorization)

    Lowering strategy:
    - 2D: Use linalg.matmul with cache-aware tiling
    - 3D: Use linalg.batch_matmul with loop fusion
    - 4D NHWC: Reshape + matmul + reshape with layout-aware optimization
    - Enable vectorization on innermost (contiguous) dimension
    - Competitive with Intel MKL for small-medium sizes
  }];

  let arguments = (ins
    Simp_SimpTensorType:$lhs,           // A: input tensor (2D/3D/4D)
    Simp_SimpTensorType:$rhs,           // B: weight tensor (2D/3D/4D)
    OptionalAttr<StrAttr>:$layout   // "NCHW" or "NHWC" (default: infer from shapes)
  );

  let results = (outs Simp_SimpTensorType:$result);  // C: output tensor

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// TensorMatMulNTOp - Matrix multiplication with pre-transposed B (No Transpose)
//===----------------------------------------------------------------------===//

def Simp_TensorMatMulNTOp : Simp_Op<"tensor_matmul_nt", [NoSideEffect]> {
  let summary = "Matrix multiplication with pre-transposed B matrix";

  let description = [{
    Matrix multiplication where B is already transposed: A[M,K] × B_T[N,K] → C[M,N]

    This avoids runtime transpose overhead. The host should pre-transpose weights
    once at model load time.

    Usage:
    ```simplang
    // B_T is pre-transposed: shape [N, K] instead of [K, N]
    i8<1, 2048> x = tensor_from_array(x_arr, 0i);
    i8<2048, 2048> Wq_T = tensor_from_array(wq_transposed, 0i);  // [N, K]
    var Q = tensor_matmul_nt(x, Wq_T);  // No transpose needed!
    ```

    For VNNI optimization, K-innermost loop is used directly on B_T.
  }];

  let arguments = (ins
    Simp_SimpTensorType:$lhs,    // A: [M, K]
    Simp_SimpTensorType:$rhs     // B_T: [N, K] (pre-transposed)
  );

  let results = (outs Simp_SimpTensorType:$result);  // C: [M, N]

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// TensorMatMulIntoOp - In-place matrix multiplication (no allocation)
//===----------------------------------------------------------------------===//

def Simp_TensorMatMulIntoOp : Simp_Op<"tensor_matmul_into", []> {
  let summary = "In-place matrix multiplication for pre-allocated output";

  let description = [{
    Matrix multiplication that writes to a pre-allocated output tensor.
    This avoids internal allocation overhead, critical for performance.

    Usage:
    ```simplang
    i32<M, N> C = tensor_from_array(c_arr, 0i);  // Pre-allocated from host
    tensor_matmul_into(A, B, C);  // Write directly to C
    ```

    Same semantics as tensor_matmul but output must be pre-allocated.
  }];

  let arguments = (ins
    Simp_SimpTensorType:$lhs,    // A: input tensor
    Simp_SimpTensorType:$rhs,    // B: weight tensor
    Simp_SimpTensorType:$output  // C: pre-allocated output tensor
  );

  let results = (outs);  // No result - writes in-place

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $output attr-dict `:` `(` type($lhs) `,` type($rhs) `,` type($output) `)`
  }];
}

//===----------------------------------------------------------------------===//
// TensorMatVecMulOp - Matrix × skinny matrix with i-k-j loop order
//===----------------------------------------------------------------------===//

def Simp_TensorMatVecMulOp : Simp_Op<"tensor_matvecmul", [NoSideEffect]> {
  let summary = "Matrix × skinny matrix multiplication (optimized for N << K)";

  let description = [{
    Matrix multiplication optimized for when the second matrix has few columns (N << K).
    Uses i-k-j loop order instead of i-j-k, which is more efficient for:
    - Attention × V in transformers: [seq, seq] × [seq, head_dim]
    - Matrix × vector operations
    - Any A[M,K] × B[K,N] where N is small (e.g., N <= 128)

    Why i-k-j is better for small N:
    - i-j-k requires horizontal reduction after each K-loop (expensive)
    - i-k-j accumulates into C[i,:] across K iterations (no reduction per element)
    - J-innermost loop vectorizes across N columns

    Usage:
    ```simplang
    i8<1024, 1024> Attn;  // Attention weights
    i8<1024, 64> V;       // Value matrix (head_dim=64)
    i32<1024, 64> out = tensor_matvecmul(Attn, V);  // Uses i-k-j order
    ```

    Mathematical Operation:
      C[i,j] = sum_k(A[i,k] * B[k,j])

    Loop order: for i, for k, for j (J innermost)
  }];

  let arguments = (ins
    Simp_SimpTensorType:$lhs,           // A: [M, K] input tensor
    Simp_SimpTensorType:$rhs            // B: [K, N] weight tensor (N should be small)
  );

  let results = (outs Simp_SimpTensorType:$result);  // C: [M, N] output tensor

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// TensorDotOp - Dot product for 1D tensors
//===----------------------------------------------------------------------===//

def Simp_TensorDotOp : Simp_Op<"tensor_dot", [NoSideEffect]> {
  let summary = "Dot product for 1D tensors";

  let description = [{
    Computes dot product of two 1D tensors.

    Computes: result = sum(a[i] * b[i])

    Where:
      a: tensor<N>  (1D tensor, N elements)
      b: tensor<N>  (1D tensor, N elements)
      result: scalar

    Example:
    ```simplang
    f32<1024> a;
    f32<1024> b;
    f32 result = tensor_dot(a, b);
    ```

    Lowering strategy:
    - Tight vectorized loop with reduction
    - Auto-vectorization friendly
    - Minimal overhead vs hand-written SIMD
  }];

  let arguments = (ins
    Simp_SimpTensorType:$lhs,  // Vector a
    Simp_SimpTensorType:$rhs   // Vector b
  );

  let results = (outs AnyType:$result);  // Scalar result (type matches tensor element type)

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
}

#endif // SIMP_OPS
