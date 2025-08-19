/**
 * @file kernels_f32.c
 * @brief AVX2 optimized implementations for simpblas kernels
 */

#include <immintrin.h>
#include <string.h>

/**
 * @brief AVX2 element-wise addition: C = A + B
 * Processes 8 floats at a time using AVX2
 */
void sb_ew_add_f32_avx2(const float* A, const float* B, float* C, size_t elems) {
    size_t i = 0;
    
    // Process 8 elements at a time with AVX2
    for (; i + 8 <= elems; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(&A[i]);
        __m256 b_vec = _mm256_loadu_ps(&B[i]);
        __m256 c_vec = _mm256_add_ps(a_vec, b_vec);
        _mm256_storeu_ps(&C[i], c_vec);
    }
    
    // Handle remaining elements with scalar code
    for (; i < elems; i++) {
        C[i] = A[i] + B[i];
    }
}

/**
 * @brief AVX2 element-wise multiplication: C = A * B
 * Processes 8 floats at a time using AVX2
 */
void sb_ew_mul_f32_avx2(const float* A, const float* B, float* C, size_t elems) {
    size_t i = 0;
    
    // Process 8 elements at a time with AVX2
    for (; i + 8 <= elems; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(&A[i]);
        __m256 b_vec = _mm256_loadu_ps(&B[i]);
        __m256 c_vec = _mm256_mul_ps(a_vec, b_vec);
        _mm256_storeu_ps(&C[i], c_vec);
    }
    
    // Handle remaining elements with scalar code
    for (; i < elems; i++) {
        C[i] = A[i] * B[i];
    }
}

/**
 * @brief AVX2 element-wise ReLU: C = max(0, A)
 * Processes 8 floats at a time using AVX2
 */
void sb_ew_relu_f32_avx2(const float* A, float* C, size_t elems) {
    size_t i = 0;
    __m256 zero = _mm256_setzero_ps();
    
    // Process 8 elements at a time with AVX2
    for (; i + 8 <= elems; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(&A[i]);
        __m256 c_vec = _mm256_max_ps(a_vec, zero);
        _mm256_storeu_ps(&C[i], c_vec);
    }
    
    // Handle remaining elements with scalar code
    for (; i < elems; i++) {
        C[i] = A[i] > 0.0f ? A[i] : 0.0f;
    }
}

/**
 * @brief AVX2 GEMM: C = A × B
 * Simple blocked implementation with AVX2 vectorization
 */
void sb_gemm_f32_avx2(
    int M, int N, int K,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc) {
    
    // Initialize C to zero
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = 0.0f;
        }
    }
    
    // Simple blocking for better cache utilization
    const int BLOCK_SIZE = 64;
    
    for (int ii = 0; ii < M; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < K; kk += BLOCK_SIZE) {
                
                int i_max = (ii + BLOCK_SIZE < M) ? ii + BLOCK_SIZE : M;
                int j_max = (jj + BLOCK_SIZE < N) ? jj + BLOCK_SIZE : N;
                int k_max = (kk + BLOCK_SIZE < K) ? kk + BLOCK_SIZE : K;
                
                // Inner kernel with AVX2 vectorization
                for (int i = ii; i < i_max; i++) {
                    for (int k = kk; k < k_max; k++) {
                        __m256 a_broadcast = _mm256_broadcast_ss(&A[i * lda + k]);
                        int j = jj;
                        
                        // Process 8 columns at a time
                        for (; j + 8 <= j_max; j += 8) {
                            __m256 b_vec = _mm256_loadu_ps(&B[k * ldb + j]);
                            __m256 c_vec = _mm256_loadu_ps(&C[i * ldc + j]);
                            c_vec = _mm256_fmadd_ps(a_broadcast, b_vec, c_vec);
                            _mm256_storeu_ps(&C[i * ldc + j], c_vec);
                        }
                        
                        // Handle remaining columns
                        for (; j < j_max; j++) {
                            C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
                        }
                    }
                }
            }
        }
    }
}