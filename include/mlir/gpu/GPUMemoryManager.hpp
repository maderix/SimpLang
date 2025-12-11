#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>

namespace simp {
namespace gpu {

/// GPU memory allocation tracking
struct GPUAllocation {
    void* devicePtr;
    size_t size;
    std::string tag;  // For debugging
    bool isPooled;
};

/// KV Cache configuration for transformer models
struct KVCacheConfig {
    int maxSeqLen;
    int nLayers;
    int nHeads;
    int headDim;
    int batchSize;

    size_t totalSize() const {
        // 2 for K and V, float32
        return 2 * maxSeqLen * nLayers * nHeads * headDim * batchSize * sizeof(float);
    }
};

/// GPU Memory Manager for SimpLang GPU backend
/// Handles device memory allocation, transfers, and memory pooling
class GPUMemoryManager {
public:
    GPUMemoryManager();
    ~GPUMemoryManager();

    // Disable copy
    GPUMemoryManager(const GPUMemoryManager&) = delete;
    GPUMemoryManager& operator=(const GPUMemoryManager&) = delete;

    /// Initialize CUDA context and cuBLAS handle
    bool initialize(int deviceId = 0);

    /// Shutdown and free all resources
    void shutdown();

    /// Check if initialized
    bool isInitialized() const { return initialized_; }

    // === Memory Allocation ===

    /// Allocate device memory
    void* allocateDevice(size_t size, const std::string& tag = "");

    /// Allocate pinned host memory (for faster H2D/D2H transfers)
    void* allocatePinned(size_t size, const std::string& tag = "");

    /// Free device memory
    void freeDevice(void* ptr);

    /// Free pinned host memory
    void freePinned(void* ptr);

    /// Free all allocations
    void freeAll();

    // === Memory Transfers ===

    /// Copy from host to device
    void hostToDevice(void* dst, const void* src, size_t size);

    /// Copy from device to host
    void deviceToHost(void* dst, const void* src, size_t size);

    /// Copy device to device
    void deviceToDevice(void* dst, const void* src, size_t size);

    /// Async copy from host to device
    void hostToDeviceAsync(void* dst, const void* src, size_t size, cudaStream_t stream = 0);

    /// Async copy from device to host
    void deviceToHostAsync(void* dst, const void* src, size_t size, cudaStream_t stream = 0);

    // === Memory Pooling for LLaMA ===

    /// Pre-allocate KV cache pool for transformer inference
    void* createKVCachePool(const KVCacheConfig& config);

    /// Get KV cache slice for a specific layer
    void* getKVCacheSlice(void* pool, int layer, bool isKey, int seqPos = 0);

    /// Pre-allocate activation buffers (reused across layers)
    void* createActivationPool(size_t maxActivationSize);

    // === cuBLAS Integration ===

    /// Get cuBLAS handle for matrix operations
    cublasHandle_t getCublasHandle() const { return cublasHandle_; }

    /// SGEMM wrapper: C = alpha * A * B + beta * C
    void sgemm(bool transA, bool transB,
               int M, int N, int K,
               float alpha,
               const float* A, int lda,
               const float* B, int ldb,
               float beta,
               float* C, int ldc);

    /// Batched SGEMM for multi-head attention
    void sgemmBatched(bool transA, bool transB,
                      int M, int N, int K,
                      float alpha,
                      const float** A, int lda,
                      const float** B, int ldb,
                      float beta,
                      float** C, int ldc,
                      int batchCount);

    // === Utility ===

    /// Synchronize device
    void synchronize();

    /// Get device properties
    cudaDeviceProp getDeviceProperties() const { return deviceProps_; }

    /// Get total allocated memory
    size_t getTotalAllocated() const { return totalAllocated_; }

    /// Get device memory info
    void getMemoryInfo(size_t* free, size_t* total) const;

    /// Print allocation stats
    void printStats() const;

private:
    bool initialized_ = false;
    int deviceId_ = 0;
    cudaDeviceProp deviceProps_;
    cublasHandle_t cublasHandle_ = nullptr;

    // Allocation tracking
    std::unordered_map<void*, GPUAllocation> deviceAllocations_;
    std::unordered_map<void*, GPUAllocation> pinnedAllocations_;
    size_t totalAllocated_ = 0;

    // KV Cache pool info
    KVCacheConfig kvCacheConfig_;
    void* kvCachePool_ = nullptr;

    // Error checking helper
    void checkCudaError(cudaError_t err, const char* msg);
    void checkCublasError(cublasStatus_t status, const char* msg);
};

/// Global GPU memory manager instance
GPUMemoryManager& getGPUMemoryManager();

} // namespace gpu
} // namespace simp

#endif // USE_CUDA
