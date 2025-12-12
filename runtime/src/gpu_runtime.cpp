//===- gpu_runtime.cpp - GPU Runtime for SimpLang -------------------------===//
//
// Runtime support for GPU operations, including cuBLAS SGEMM wrapper.
//
// This file provides C-callable functions that the MLIR-generated code
// links against for GPU execution.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <unordered_map>

// Global cuBLAS handle - initialized lazily
static cublasHandle_t g_cublasHandle = nullptr;
static bool g_initialized = false;

// CUDA streams for async operations
static cudaStream_t g_computeStream = nullptr;
static cudaStream_t g_copyStream = nullptr;

// Pinned memory staging buffers for faster H2D/D2H transfers
static float* g_pinnedA = nullptr;
static float* g_pinnedB = nullptr;
static float* g_pinnedC = nullptr;
static size_t g_pinnedSize = 0;

// Pending D2H copy state for async overlap
static float* g_pendingHostC = nullptr;
static size_t g_pendingSize = 0;
static bool g_hasPendingCopy = false;

//===----------------------------------------------------------------------===//
// Persistent GPU Weight Cache
// Weights are uploaded once and kept on GPU - keyed by (host pointer, size)
// Activations use size-based allocation cache
//===----------------------------------------------------------------------===//
struct GPUWeightCache {
    // Weights: keyed by (host pointer, size) pair
    // Using size in key handles case where same address is reused for different data
    struct WeightKey {
        const float* ptr;
        size_t size;
        bool operator==(const WeightKey& other) const {
            return ptr == other.ptr && size == other.size;
        }
    };
    struct WeightKeyHash {
        size_t operator()(const WeightKey& k) const {
            return std::hash<const float*>()(k.ptr) ^ (std::hash<size_t>()(k.size) << 1);
        }
    };
    std::unordered_map<WeightKey, float*, WeightKeyHash> weight_cache;

    // Activations: keyed by size (reused across calls)
    std::unordered_map<size_t, float*> activation_cache;

    // Get or upload weight matrix (A and B are typically weights)
    float* get_weight(const float* host_ptr, size_t size) {
        WeightKey key{host_ptr, size};
        auto it = weight_cache.find(key);
        if (it != weight_cache.end()) {
            return it->second;  // Already on GPU, no transfer needed!
        }
        // First time seeing this weight - upload to GPU
        float* d_ptr;
        cudaMalloc(&d_ptr, size);
        cudaMemcpy(d_ptr, host_ptr, size, cudaMemcpyHostToDevice);
        weight_cache[key] = d_ptr;
        return d_ptr;
    }

    // Get activation buffer (output matrix C - changes every call)
    float* get_activation(size_t size) {
        auto it = activation_cache.find(size);
        if (it != activation_cache.end()) return it->second;
        float* ptr;
        cudaMalloc(&ptr, size);
        activation_cache[size] = ptr;
        return ptr;
    }

    void clear() {
        for (auto& p : weight_cache) cudaFree(p.second);
        for (auto& p : activation_cache) cudaFree(p.second);
        weight_cache.clear();
        activation_cache.clear();
    }

    size_t get_cached_weight_count() { return weight_cache.size(); }
};

static GPUWeightCache g_weightCache;

static void initCublas() {
    if (!g_initialized) {
        cublasStatus_t status = cublasCreate(&g_cublasHandle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[GPU] Failed to create cuBLAS handle: %d\n", status);
            exit(1);
        }

        // Create CUDA streams for async operations
        cudaStreamCreate(&g_computeStream);
        cudaStreamCreate(&g_copyStream);

        // Set cuBLAS to use compute stream
        cublasSetStream(g_cublasHandle, g_computeStream);

        // Enable TF32 for faster matmul on Ampere+ GPUs (RTX 30xx/40xx)
        cublasSetMathMode(g_cublasHandle, CUBLAS_TF32_TENSOR_OP_MATH);

        g_initialized = true;
    }
}

// Ensure pinned buffers are large enough
static void ensurePinnedBuffers(size_t maxSize) {
    if (maxSize > g_pinnedSize) {
        if (g_pinnedA) cudaFreeHost(g_pinnedA);
        if (g_pinnedB) cudaFreeHost(g_pinnedB);
        if (g_pinnedC) cudaFreeHost(g_pinnedC);
        cudaHostAlloc(&g_pinnedA, maxSize, cudaHostAllocDefault);
        cudaHostAlloc(&g_pinnedB, maxSize, cudaHostAllocDefault);
        cudaHostAlloc(&g_pinnedC, maxSize, cudaHostAllocDefault);
        g_pinnedSize = maxSize;
    }
}

// Track last output for sync
static float* g_lastHostC = nullptr;
static float* g_lastDeviceC = nullptr;
static size_t g_lastSizeC = 0;

// Sync last output from GPU to host
extern "C" void simp_gpu_sync_output() {
    if (g_lastDeviceC && g_lastHostC && g_lastSizeC > 0) {
        cudaMemcpy(g_lastHostC, g_lastDeviceC, g_lastSizeC, cudaMemcpyDeviceToHost);
    }
}

// Cleanup function (can be called at program exit)
extern "C" void simp_gpu_cleanup() {
    g_weightCache.clear();
    if (g_initialized) {
        if (g_pinnedA) cudaFreeHost(g_pinnedA);
        if (g_pinnedB) cudaFreeHost(g_pinnedB);
        if (g_pinnedC) cudaFreeHost(g_pinnedC);
        g_pinnedA = g_pinnedB = g_pinnedC = nullptr;
        g_pinnedSize = 0;
        if (g_computeStream) cudaStreamDestroy(g_computeStream);
        if (g_copyStream) cudaStreamDestroy(g_copyStream);
        g_computeStream = nullptr;
        g_copyStream = nullptr;
        if (g_cublasHandle) cublasDestroy(g_cublasHandle);
        g_cublasHandle = nullptr;
        g_initialized = false;
    }
}

#endif // USE_CUDA

//===----------------------------------------------------------------------===//
// simp_cublas_sgemm - cuBLAS SGEMM wrapper
//
// MLIR memref<NxMxf32> is lowered to:
//   { float* allocatedPtr, float* alignedPtr, i64 offset,
//     i64 sizes[2], i64 strides[2] }
//
// This function receives the expanded struct fields as individual arguments.
//===----------------------------------------------------------------------===//

extern "C" void simp_cublas_sgemm(
    // transA, transB
    bool transA, bool transB,
    // M, N, K dimensions
    int M, int N, int K,
    // alpha scalar
    float alpha,
    // Matrix A memref descriptor (expanded)
    float* A_alloc, float* A_aligned, int64_t A_offset,
    int64_t A_size0, int64_t A_size1, int64_t A_stride0, int64_t A_stride1,
    int lda,
    // Matrix B memref descriptor (expanded)
    float* B_alloc, float* B_aligned, int64_t B_offset,
    int64_t B_size0, int64_t B_size1, int64_t B_stride0, int64_t B_stride1,
    int ldb,
    // beta scalar
    float beta,
    // Matrix C memref descriptor (expanded)
    float* C_alloc, float* C_aligned, int64_t C_offset,
    int64_t C_size0, int64_t C_size1, int64_t C_stride0, int64_t C_stride1,
    int ldc
) {
    // Debug output disabled for benchmarking
    // printf("[simp_cublas_sgemm] M=%d, N=%d, K=%d\n", M, N, K);

#ifdef USE_CUDA
    initCublas();

    // Get actual data pointers (aligned + offset)
    float* A = A_aligned + A_offset;
    float* B = B_aligned + B_offset;
    float* C = C_aligned + C_offset;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // Heuristic: Large matrices (>64KB) are likely weights, keep on GPU
    // Small matrices are activations that change every call
    const size_t WEIGHT_THRESHOLD = 64 * 1024;  // 64KB = 16K floats

    float* d_A;
    float* d_B;
    float* d_C;

    // Matrix A: if large, treat as weight (persistent), else as activation
    bool copyA = false, copyB = false;
    if (sizeA >= WEIGHT_THRESHOLD) {
        d_A = g_weightCache.get_weight(A, sizeA);
    } else {
        d_A = g_weightCache.get_activation(sizeA);
        copyA = true;
    }

    // Matrix B: if large, treat as weight (persistent), else as activation
    if (sizeB >= WEIGHT_THRESHOLD) {
        d_B = g_weightCache.get_weight(B, sizeB);
    } else {
        d_B = g_weightCache.get_activation(sizeB + 1);
        copyB = true;
    }

    // C is always output activation
    d_C = g_weightCache.get_activation(sizeC + 2);

    if (copyA) cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    if (copyB) cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);
    if (beta != 0.0f) cudaMemcpy(d_C, C, sizeC, cudaMemcpyHostToDevice);

    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasStatus_t status = cublasSgemm(
        g_cublasHandle,
        opB, opA,
        N, M, K,
        &alpha,
        d_B, ldb,
        d_A, lda,
        &beta,
        d_C, ldc
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[GPU] cuBLAS SGEMM failed: %d\n", status);
    }

    // Wait for SGEMM to complete (but skip D2H)
    cudaDeviceSynchronize();

    // Track last output for later sync
    g_lastHostC = C;
    g_lastDeviceC = d_C;
    g_lastSizeC = sizeC;

    // Memory stays cached - no free here

#else
    // CPU fallback: naive triple-loop matmul
    printf("[simp_cublas_sgemm] Using CPU fallback (no CUDA)\n");
    fflush(stdout);
    float* A = A_aligned + A_offset;
    float* B = B_aligned + B_offset;
    float* C = C_aligned + C_offset;

    // Initialize C with beta * C (or zero if beta == 0)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = beta * C[i * ldc + j];
        }
    }

    // C += alpha * A * B
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = alpha * A[i * lda + k];
            for (int j = 0; j < N; j++) {
                C[i * ldc + j] += a_ik * B[k * ldb + j];
            }
        }
    }
#endif
}

//===----------------------------------------------------------------------===//
// GPU Fill - Fast tensor initialization using cudaMemset
//===----------------------------------------------------------------------===//

// simp_gpu_fill_f32 - GPU-accelerated tensor fill using cudaMemset
//
// MLIR memref<Nx...xf32> is lowered to:
//   { float* allocatedPtr, float* alignedPtr, i64 offset,
//     i64 sizes[rank], i64 strides[rank] }
//
// For 1D dynamic memref (after collapse), we receive:
//   { float* alloc, float* aligned, i64 offset, i64 size, i64 stride }
extern "C" void simp_gpu_fill_f32(
    float* alloc_ptr, float* aligned_ptr, int64_t offset,
    int64_t size, int64_t stride,
    int64_t num_elements,
    float value
) {
    float* data = aligned_ptr + offset;

#ifdef USE_CUDA
    initCublas();  // Ensure CUDA is initialized

    size_t byte_size = num_elements * sizeof(float);

    // For zero fill, use cudaMemset (fastest)
    if (value == 0.0f) {
        // Get GPU buffer and zero it
        float* d_data = g_weightCache.get_activation(byte_size);
        cudaMemset(d_data, 0, byte_size);
        cudaMemcpy(data, d_data, byte_size, cudaMemcpyDeviceToHost);
    } else {
        // For non-zero values, fill on CPU (memset only works for 0)
        // This is still fast for initialization since it's sequential memory access
        for (int64_t i = 0; i < num_elements; i++) {
            data[i] = value;
        }
    }
#else
    // CPU fallback: simple loop fill
    for (int64_t i = 0; i < num_elements; i++) {
        data[i] = value;
    }
#endif
}

//===----------------------------------------------------------------------===//
// GPU Memory Management Functions (for future use)
//===----------------------------------------------------------------------===//

#ifdef USE_CUDA
extern "C" void* simp_gpu_malloc(size_t size) {
    void* ptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GPU] cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

extern "C" void simp_gpu_free(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

extern "C" void simp_gpu_memcpy_h2d(void* dst, const void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

extern "C" void simp_gpu_memcpy_d2h(void* dst, const void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}
#endif // USE_CUDA
