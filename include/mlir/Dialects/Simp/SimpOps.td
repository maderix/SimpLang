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

#endif // SIMP_OPS
