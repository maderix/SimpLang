#ifdef USE_CUDA

#include "mlir/gpu/GPUMemoryManager.hpp"
#include <iostream>
#include <stdexcept>
#include <cstring>

namespace simp {
namespace gpu {

// Global instance
static std::unique_ptr<GPUMemoryManager> globalGPUMemoryManager;

GPUMemoryManager& getGPUMemoryManager() {
    if (!globalGPUMemoryManager) {
        globalGPUMemoryManager = std::make_unique<GPUMemoryManager>();
    }
    return *globalGPUMemoryManager;
}

GPUMemoryManager::GPUMemoryManager() = default;

GPUMemoryManager::~GPUMemoryManager() {
    if (initialized_) {
        shutdown();
    }
}

void GPUMemoryManager::checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "[GPU Error] " << msg << ": " << cudaGetErrorString(err) << std::endl;
        std::abort();  // No exceptions in MLIR code
    }
}

void GPUMemoryManager::checkCublasError(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[cuBLAS Error] " << msg << ": cuBLAS error " << status << std::endl;
        std::abort();  // No exceptions in MLIR code
    }
}

bool GPUMemoryManager::initialize(int deviceId) {
    if (initialized_) {
        return true;
    }

    deviceId_ = deviceId;

    // Set device
    cudaError_t err = cudaSetDevice(deviceId_);
    if (err != cudaSuccess) {
        std::cerr << "[GPU] Failed to set device " << deviceId_ << ": "
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // Get device properties
    err = cudaGetDeviceProperties(&deviceProps_, deviceId_);
    if (err != cudaSuccess) {
        std::cerr << "[GPU] Failed to get device properties: "
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }

    std::cout << "[GPU] Initialized device " << deviceId_ << ": "
              << deviceProps_.name << std::endl;
    std::cout << "[GPU] Compute capability: " << deviceProps_.major << "."
              << deviceProps_.minor << std::endl;
    std::cout << "[GPU] Total memory: " << (deviceProps_.totalGlobalMem / (1024*1024))
              << " MB" << std::endl;
    std::cout << "[GPU] SM count: " << deviceProps_.multiProcessorCount << std::endl;

    // Create cuBLAS handle
    cublasStatus_t cublasStatus = cublasCreate(&cublasHandle_);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[GPU] Failed to create cuBLAS handle" << std::endl;
        return false;
    }

    // Set cuBLAS to use Tensor Cores when available
    cublasSetMathMode(cublasHandle_, CUBLAS_DEFAULT_MATH);

    initialized_ = true;
    return true;
}

void GPUMemoryManager::shutdown() {
    if (!initialized_) return;

    // Free all allocations
    freeAll();

    // Destroy cuBLAS handle
    if (cublasHandle_) {
        cublasDestroy(cublasHandle_);
        cublasHandle_ = nullptr;
    }

    cudaDeviceReset();
    initialized_ = false;

    std::cout << "[GPU] Shutdown complete" << std::endl;
}

void* GPUMemoryManager::allocateDevice(size_t size, const std::string& tag) {
    if (!initialized_) {
        std::cerr << "[GPU Error] GPU not initialized" << std::endl;
        return nullptr;
    }

    void* ptr = nullptr;
    checkCudaError(cudaMalloc(&ptr, size), "cudaMalloc failed");

    GPUAllocation alloc{ptr, size, tag, false};
    deviceAllocations_[ptr] = alloc;
    totalAllocated_ += size;

    return ptr;
}

void* GPUMemoryManager::allocatePinned(size_t size, const std::string& tag) {
    if (!initialized_) {
        std::cerr << "[GPU Error] GPU not initialized" << std::endl;
        return nullptr;
    }

    void* ptr = nullptr;
    checkCudaError(cudaMallocHost(&ptr, size), "cudaMallocHost failed");

    GPUAllocation alloc{ptr, size, tag, false};
    pinnedAllocations_[ptr] = alloc;

    return ptr;
}

void GPUMemoryManager::freeDevice(void* ptr) {
    if (!ptr) return;

    auto it = deviceAllocations_.find(ptr);
    if (it != deviceAllocations_.end()) {
        totalAllocated_ -= it->second.size;
        cudaFree(ptr);
        deviceAllocations_.erase(it);
    }
}

void GPUMemoryManager::freePinned(void* ptr) {
    if (!ptr) return;

    auto it = pinnedAllocations_.find(ptr);
    if (it != pinnedAllocations_.end()) {
        cudaFreeHost(ptr);
        pinnedAllocations_.erase(it);
    }
}

void GPUMemoryManager::freeAll() {
    // Free device allocations
    for (auto& [ptr, alloc] : deviceAllocations_) {
        cudaFree(ptr);
    }
    deviceAllocations_.clear();

    // Free pinned allocations
    for (auto& [ptr, alloc] : pinnedAllocations_) {
        cudaFreeHost(ptr);
    }
    pinnedAllocations_.clear();

    // Reset KV cache pool
    kvCachePool_ = nullptr;
    totalAllocated_ = 0;
}

void GPUMemoryManager::hostToDevice(void* dst, const void* src, size_t size) {
    checkCudaError(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D failed");
}

void GPUMemoryManager::deviceToHost(void* dst, const void* src, size_t size) {
    checkCudaError(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H failed");
}

void GPUMemoryManager::deviceToDevice(void* dst, const void* src, size_t size) {
    checkCudaError(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice),
                   "cudaMemcpy D2D failed");
}

void GPUMemoryManager::hostToDeviceAsync(void* dst, const void* src, size_t size, cudaStream_t stream) {
    checkCudaError(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync H2D failed");
}

void GPUMemoryManager::deviceToHostAsync(void* dst, const void* src, size_t size, cudaStream_t stream) {
    checkCudaError(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync D2H failed");
}

void* GPUMemoryManager::createKVCachePool(const KVCacheConfig& config) {
    kvCacheConfig_ = config;
    size_t totalSize = config.totalSize();

    std::cout << "[GPU] Creating KV cache pool:" << std::endl;
    std::cout << "  Max seq len: " << config.maxSeqLen << std::endl;
    std::cout << "  Layers: " << config.nLayers << std::endl;
    std::cout << "  Heads: " << config.nHeads << std::endl;
    std::cout << "  Head dim: " << config.headDim << std::endl;
    std::cout << "  Total size: " << (totalSize / (1024*1024)) << " MB" << std::endl;

    kvCachePool_ = allocateDevice(totalSize, "kv_cache_pool");

    // Zero-initialize
    checkCudaError(cudaMemset(kvCachePool_, 0, totalSize), "cudaMemset KV cache failed");

    return kvCachePool_;
}

void* GPUMemoryManager::getKVCacheSlice(void* pool, int layer, bool isKey, int seqPos) {
    if (!pool || pool != kvCachePool_) {
        std::cerr << "[GPU Error] Invalid KV cache pool" << std::endl;
        return nullptr;
    }

    // Layout: [2, nLayers, maxSeqLen, nHeads, headDim]
    size_t keyOffset = isKey ? 0 : 1;
    size_t layerSize = kvCacheConfig_.maxSeqLen * kvCacheConfig_.nHeads *
                       kvCacheConfig_.headDim * sizeof(float);
    size_t offset = (keyOffset * kvCacheConfig_.nLayers + layer) * layerSize +
                    seqPos * kvCacheConfig_.nHeads * kvCacheConfig_.headDim * sizeof(float);

    return static_cast<char*>(pool) + offset;
}

void* GPUMemoryManager::createActivationPool(size_t maxActivationSize) {
    return allocateDevice(maxActivationSize, "activation_pool");
}

void GPUMemoryManager::sgemm(bool transA, bool transB,
                              int M, int N, int K,
                              float alpha,
                              const float* A, int lda,
                              const float* B, int ldb,
                              float beta,
                              float* C, int ldc) {
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

    // Note: cuBLAS uses column-major, so we swap A and B
    checkCublasError(
        cublasSgemm(cublasHandle_, opB, opA, N, M, K,
                    &alpha, B, ldb, A, lda, &beta, C, ldc),
        "cublasSgemm failed"
    );
}

void GPUMemoryManager::sgemmBatched(bool transA, bool transB,
                                     int M, int N, int K,
                                     float alpha,
                                     const float** A, int lda,
                                     const float** B, int ldb,
                                     float beta,
                                     float** C, int ldc,
                                     int batchCount) {
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

    checkCublasError(
        cublasSgemmBatched(cublasHandle_, opB, opA, N, M, K,
                           &alpha, B, ldb, A, lda, &beta, C, ldc, batchCount),
        "cublasSgemmBatched failed"
    );
}

void GPUMemoryManager::synchronize() {
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
}

void GPUMemoryManager::getMemoryInfo(size_t* free, size_t* total) const {
    cudaError_t err = cudaMemGetInfo(free, total);
    if (err != cudaSuccess) {
        std::cerr << "[GPU Error] cudaMemGetInfo failed: " << cudaGetErrorString(err) << std::endl;
    }
}

void GPUMemoryManager::printStats() const {
    std::cout << "\n=== GPU Memory Stats ===" << std::endl;
    std::cout << "Device allocations: " << deviceAllocations_.size() << std::endl;
    std::cout << "Pinned allocations: " << pinnedAllocations_.size() << std::endl;
    std::cout << "Total allocated: " << (totalAllocated_ / (1024*1024)) << " MB" << std::endl;

    size_t free, total;
    cudaMemGetInfo(&free, &total);
    std::cout << "Device memory: " << (free / (1024*1024)) << " / "
              << (total / (1024*1024)) << " MB free" << std::endl;

    if (!deviceAllocations_.empty()) {
        std::cout << "\nAllocations:" << std::endl;
        for (const auto& [ptr, alloc] : deviceAllocations_) {
            std::cout << "  " << ptr << ": " << (alloc.size / 1024) << " KB";
            if (!alloc.tag.empty()) {
                std::cout << " (" << alloc.tag << ")";
            }
            std::cout << std::endl;
        }
    }
    std::cout << "========================\n" << std::endl;
}

} // namespace gpu
} // namespace simp

#endif // USE_CUDA
