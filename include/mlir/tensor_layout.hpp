//===- tensor_layout.hpp - Tensor Layout Definitions ------------*- C++ -*-===//
//
// Part of the SimpLang Project
//
// This file defines tensor layout types for multi-dimensional array operations,
// particularly for deep learning operations like convolution.
//
//===----------------------------------------------------------------------===//

#ifndef AST_TYPE_TENSOR_LAYOUT_HPP
#define AST_TYPE_TENSOR_LAYOUT_HPP

#include <string>

namespace simp {

/// Tensor layout specification for multi-dimensional arrays
/// Defines how logical indices map to physical memory layout
enum class TensorLayout {
    /// Row-major layout (default C-style)
    /// 2D: [i, j] → i*N + j
    /// Used for general matrix operations
    RowMajor,

    /// NHWC layout (Channels Last)
    /// Logical: [N, C, H, W] → Physical: [N, H, W, C]
    /// Memory stride: [H*W*C, W*C, C, 1]
    /// Optimal for:
    ///   - GPU Tensor Cores (NVIDIA)
    ///   - Coalesced memory access
    ///   - TensorFlow default layout
    NHWC,

    /// NCHW layout (Channels First)
    /// Logical: [N, C, H, W] → Physical: [N, C, H, W] (identity)
    /// Memory stride: [C*H*W, H*W, W, 1]
    /// Optimal for:
    ///   - Some CPU operations
    ///   - PyTorch default layout
    NCHW,

    /// Custom layout defined by user-provided affine map
    /// Allows arbitrary permutations and transformations
    Custom
};

/// Convert tensor layout enum to string representation
inline std::string tensorLayoutToString(TensorLayout layout) {
    switch (layout) {
        case TensorLayout::RowMajor:  return "row_major";
        case TensorLayout::NHWC:      return "nhwc";
        case TensorLayout::NCHW:      return "nchw";
        case TensorLayout::Custom:    return "custom";
        default:                      return "unknown";
    }
}

/// Parse string to tensor layout enum
inline TensorLayout tensorLayoutFromString(const std::string& str) {
    if (str == "row_major") return TensorLayout::RowMajor;
    if (str == "nhwc")      return TensorLayout::NHWC;
    if (str == "nchw")      return TensorLayout::NCHW;
    if (str == "custom")    return TensorLayout::Custom;
    return TensorLayout::RowMajor; // Default
}

/// Check if layout is valid for given rank
inline bool isValidLayoutForRank(TensorLayout layout, size_t rank) {
    switch (layout) {
        case TensorLayout::RowMajor:
            return true; // Works for any rank
        case TensorLayout::NHWC:
        case TensorLayout::NCHW:
            return rank == 4; // Requires 4D tensors
        case TensorLayout::Custom:
            return true; // Custom can handle anything
        default:
            return false;
    }
}

} // namespace simp

#endif // AST_TYPE_TENSOR_LAYOUT_HPP
