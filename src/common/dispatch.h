/**
 * @file dispatch.h
 * @brief Runtime dispatch table for simpblas kernels
 */

#ifndef SIMPBLAS_DISPATCH_H
#define SIMPBLAS_DISPATCH_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Function pointer types for all kernel variants
 */
typedef void (*sb_gemm_f32_func_t)(
    int M, int N, int K,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc);

typedef void (*sb_conv3x3_s1_p1_f32_func_t)(
    int N, int C, int H, int W,
    const float* input,
    const float* weights,
    const float* bias,
    int out_c,
    float* output);

typedef void (*sb_ew_add_f32_func_t)(const float* A, const float* B, float* C, size_t elems);
typedef void (*sb_ew_mul_f32_func_t)(const float* A, const float* B, float* C, size_t elems);
typedef void (*sb_ew_relu_f32_func_t)(const float* A, float* C, size_t elems);

/**
 * @brief Global dispatch table - resolved at sb_init()
 */
struct sb_dispatch {
    sb_gemm_f32_func_t gemm_f32;
    sb_conv3x3_s1_p1_f32_func_t conv3x3_s1_p1_f32;
    sb_ew_add_f32_func_t ew_add_f32;
    sb_ew_mul_f32_func_t ew_mul_f32;
    sb_ew_relu_f32_func_t ew_relu_f32;
    
    const char* variant_name;  // e.g., "avx2", "neon", "scalar"
};

extern struct sb_dispatch sb;

/**
 * @brief Initialize dispatch table based on CPU features
 * @return 0 on success, -1 on error
 */
int sb_init_dispatch(void);

#ifdef __cplusplus
}
#endif

#endif /* SIMPBLAS_DISPATCH_H */