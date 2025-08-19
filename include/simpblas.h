/**
 * @file simpblas.h
 * @brief simpblas - Low-level, architecture-aware kernel library
 * 
 * simpblas is the kernel layer that powers both SimpLang programs and the
 * upcoming simpnn model compiler. It delivers hand-tuned micro-kernels for
 * GEMM, convolution, and element-wise ops, exposed via a stable C ABI.
 */

#ifndef SIMPBLAS_H
#define SIMPBLAS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Datatype enum aligns with SimpLang TensorDType
 */
typedef enum {
    SB_F32 = 0,  /**< 32-bit floating point */
    SB_F16 = 1,  /**< 16-bit floating point */
    SB_I8  = 2,  /**< 8-bit integer */
} sb_dtype_t;

/**
 * @brief Initialize simpblas runtime and dispatch tables
 * Must be called once before using any simpblas functions
 * @return 0 on success, non-zero on error
 */
int sb_init(void);

/**
 * @brief General Matrix Multiply: C = A × B
 * All matrices are row-major layout
 * 
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C  
 * @param K Number of columns in A and rows in B
 * @param A Input matrix A [M x K]
 * @param lda Leading dimension of A (>= K)
 * @param B Input matrix B [K x N]
 * @param ldb Leading dimension of B (>= N)
 * @param C Output matrix C [M x N]
 * @param ldc Leading dimension of C (>= N)
 */
void sb_gemm_f32(
    int M, int N, int K,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc);

/**
 * @brief 3x3 Convolution with stride 1, padding 1
 * Input/output in NCHW packed format
 * 
 * @param N Batch size
 * @param C Input channels
 * @param H Input height
 * @param W Input width
 * @param input Input tensor [N, C, H, W]
 * @param weights Weight tensor [out_c, C, 3, 3]
 * @param bias Bias tensor [out_c] (may be NULL)
 * @param out_c Output channels
 * @param output Output tensor [N, out_c, H, W]
 */
void sb_conv3x3_s1_p1_f32(
    int N, int C, int H, int W,
    const float* input,
    const float* weights,
    const float* bias,
    int out_c,
    float* output);

/**
 * @brief Element-wise addition: C = A + B
 * 
 * @param A Input array A
 * @param B Input array B
 * @param C Output array C
 * @param elems Number of elements to process
 */
void sb_ew_add_f32(const float* A, const float* B, float* C, size_t elems);

/**
 * @brief Element-wise multiplication: C = A * B
 * 
 * @param A Input array A
 * @param B Input array B
 * @param C Output array C
 * @param elems Number of elements to process
 */
void sb_ew_mul_f32(const float* A, const float* B, float* C, size_t elems);

/**
 * @brief Element-wise ReLU: C = max(0, A)
 * 
 * @param A Input array A
 * @param C Output array C
 * @param elems Number of elements to process
 */
void sb_ew_relu_f32(const float* A, float* C, size_t elems);

/**
 * @brief Get version string
 * @return Version string (e.g., "0.1.0-alpha")
 */
const char* sb_get_version(void);

/**
 * @brief Get information about available kernels
 * @return String describing available ISA variants
 */
const char* sb_get_kernel_info(void);

#ifdef __cplusplus
}
#endif

#endif /* SIMPBLAS_H */