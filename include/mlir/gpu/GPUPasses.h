#pragma once

#ifdef USE_CUDA

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace simp {

/// Creates a pass that converts Simp tensor operations to GPU dialect operations.
/// This includes:
/// - tensor_create -> gpu.alloc
/// - tensor_matmul -> gpu.launch with matmul kernel (or cuBLAS call)
/// - Element-wise ops -> gpu.launch with element-wise kernels
std::unique_ptr<Pass> createConvertSimpToGPUPass();

/// Creates a pass that outlines GPU kernels from gpu.launch regions
std::unique_ptr<Pass> createGPUKernelOutliningPass();

/// Creates a pass that lowers GPU dialect to NVVM dialect
std::unique_ptr<Pass> createConvertGPUToNVVMPass();

/// Creates a pass that serializes GPU modules to CUBIN
std::unique_ptr<Pass> createGPUToCubinPass(const std::string& arch = "sm_80");

/// Register all GPU passes
void registerGPUPasses();

} // namespace simp
} // namespace mlir

#endif // USE_CUDA
