/**
 * @file init.c
 * @brief simpblas initialization and dispatch table setup
 */

#include "../include/simpblas.h"
#include "common/cpuid.h"
#include "common/dispatch.h"
#include <stdio.h>
#include <string.h>

// Declare scalar kernel functions
extern void sb_gemm_f32_scalar(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc);
extern void sb_conv3x3_s1_p1_f32_scalar(int N, int C, int H, int W, const float* input, const float* weights, const float* bias, int out_c, float* output);
extern void sb_ew_add_f32_scalar(const float* A, const float* B, float* C, size_t elems);
extern void sb_ew_mul_f32_scalar(const float* A, const float* B, float* C, size_t elems);
extern void sb_ew_relu_f32_scalar(const float* A, float* C, size_t elems);

// Declare AVX2 kernel functions (to be implemented)
#ifdef __AVX2__
extern void sb_gemm_f32_avx2(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc);
extern void sb_ew_add_f32_avx2(const float* A, const float* B, float* C, size_t elems);
extern void sb_ew_mul_f32_avx2(const float* A, const float* B, float* C, size_t elems);
extern void sb_ew_relu_f32_avx2(const float* A, float* C, size_t elems);
#endif

// Declare NEON kernel functions (to be implemented)
#ifdef __ARM_NEON
extern void sb_gemm_f32_neon(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc);
extern void sb_ew_add_f32_neon(const float* A, const float* B, float* C, size_t elems);
extern void sb_ew_mul_f32_neon(const float* A, const float* B, float* C, size_t elems);
extern void sb_ew_relu_f32_neon(const float* A, float* C, size_t elems);
#endif

// Global dispatch table
struct sb_dispatch sb = {0};

// Static flag to track initialization
static int sb_initialized = 0;

int sb_init(void) {
    if (sb_initialized) {
        return 0;  // Already initialized
    }
    
    // Detect CPU features
    sb_cpu_features_t features;
    if (sb_detect_cpu_features(&features) != 0) {
        fprintf(stderr, "simpblas: Failed to detect CPU features\n");
        return -1;
    }
    
    // Print detected features for debugging
    char* feature_str = sb_cpu_features_to_string(&features);
    if (feature_str) {
        printf("simpblas: %s\n", feature_str);
        free(feature_str);
    }
    
    // Select best available kernel variant
#ifdef __AVX2__
    if (features.avx2 && features.fma) {
        sb.gemm_f32 = sb_gemm_f32_avx2;
        sb.ew_add_f32 = sb_ew_add_f32_avx2;
        sb.ew_mul_f32 = sb_ew_mul_f32_avx2;
        sb.ew_relu_f32 = sb_ew_relu_f32_avx2;
        sb.conv3x3_s1_p1_f32 = sb_conv3x3_s1_p1_f32_scalar;  // Fallback to scalar for now
        sb.variant_name = "avx2";
        printf("simpblas: Using AVX2 kernels\n");
    } else
#endif
#ifdef __ARM_NEON
    if (features.neon) {
        sb.gemm_f32 = sb_gemm_f32_neon;
        sb.ew_add_f32 = sb_ew_add_f32_neon;
        sb.ew_mul_f32 = sb_ew_mul_f32_neon;
        sb.ew_relu_f32 = sb_ew_relu_f32_neon;
        sb.conv3x3_s1_p1_f32 = sb_conv3x3_s1_p1_f32_scalar;  // Fallback to scalar for now
        sb.variant_name = "neon";
        printf("simpblas: Using NEON kernels\n");
    } else
#endif
    {
        // Fallback to scalar kernels
        sb.gemm_f32 = sb_gemm_f32_scalar;
        sb.conv3x3_s1_p1_f32 = sb_conv3x3_s1_p1_f32_scalar;
        sb.ew_add_f32 = sb_ew_add_f32_scalar;
        sb.ew_mul_f32 = sb_ew_mul_f32_scalar;
        sb.ew_relu_f32 = sb_ew_relu_f32_scalar;
        sb.variant_name = "scalar";
        printf("simpblas: Using scalar kernels\n");
    }
    
    sb_initialized = 1;
    return 0;
}

// Public API implementations - these just call the dispatch table

void sb_gemm_f32(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    if (!sb_initialized) {
        sb_init();
    }
    sb.gemm_f32(M, N, K, A, lda, B, ldb, C, ldc);
}

void sb_conv3x3_s1_p1_f32(int N, int C, int H, int W, const float* input, const float* weights, const float* bias, int out_c, float* output) {
    if (!sb_initialized) {
        sb_init();
    }
    sb.conv3x3_s1_p1_f32(N, C, H, W, input, weights, bias, out_c, output);
}

void sb_ew_add_f32(const float* A, const float* B, float* C, size_t elems) {
    if (!sb_initialized) {
        sb_init();
    }
    sb.ew_add_f32(A, B, C, elems);
}

void sb_ew_mul_f32(const float* A, const float* B, float* C, size_t elems) {
    if (!sb_initialized) {
        sb_init();
    }
    sb.ew_mul_f32(A, B, C, elems);
}

void sb_ew_relu_f32(const float* A, float* C, size_t elems) {
    if (!sb_initialized) {
        sb_init();
    }
    sb.ew_relu_f32(A, C, elems);
}

const char* sb_get_version(void) {
    return "0.1.0-alpha";
}

const char* sb_get_kernel_info(void) {
    if (!sb_initialized) {
        sb_init();
    }
    return sb.variant_name;
}