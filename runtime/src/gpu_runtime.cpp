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

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <unordered_map>

// Global cuBLAS handle - initialized lazily
static cublasHandle_t g_cublasHandle = nullptr;
static bool g_initialized = false;

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
        g_initialized = true;
    }
}

// Cleanup function (can be called at program exit)
extern "C" void simp_gpu_cleanup() {
    g_weightCache.clear();
    if (g_initialized && g_cublasHandle) {
        cublasDestroy(g_cublasHandle);
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
    if (sizeA >= WEIGHT_THRESHOLD) {
        d_A = g_weightCache.get_weight(A, sizeA);
    } else {
        d_A = g_weightCache.get_activation(sizeA);
        cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    }

    // Matrix B: if large, treat as weight (persistent), else as activation
    if (sizeB >= WEIGHT_THRESHOLD) {
        d_B = g_weightCache.get_weight(B, sizeB);
    } else {
        d_B = g_weightCache.get_activation(sizeB + 1);  // +1 to differentiate from A
        cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);
    }

    // C is always output activation
    d_C = g_weightCache.get_activation(sizeC + 2);  // +2 to differentiate

    // Transfer C if beta != 0 (accumulating into existing values)
    if (beta != 0.0f) {
        cudaMemcpy(d_C, C, sizeC, cudaMemcpyHostToDevice);
    }

    // cuBLAS uses column-major, so we compute C^T = B^T * A^T
    // This gives us C in row-major (which is what we want)
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

    // SGEMM: C = alpha * A * B + beta * C
    // In column-major terms: C^T = alpha * B^T * A^T + beta * C^T
    // So we swap A and B, and swap M/N
    cublasStatus_t status = cublasSgemm(
        g_cublasHandle,
        opB, opA,           // Swap operations for row-major
        N, M, K,            // Swap M and N for row-major
        &alpha,
        d_B, ldb,           // B first (column-major trick)
        d_A, lda,           // A second
        &beta,
        d_C, ldc
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[GPU] cuBLAS SGEMM failed: %d\n", status);
    }

    // Copy result back to host
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);

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
