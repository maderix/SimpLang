/**
 * @file kernels_f32.c
 * @brief Scalar reference implementations for all simpblas kernels
 * 
 * These are the fallback implementations that work on any platform.
 * They serve as reference for correctness testing and baseline performance.
 */

#include <string.h>
#include <math.h>

/**
 * @brief Scalar GEMM: C = A × B
 * Basic three-loop implementation
 */
void sb_gemm_f32_scalar(
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
    
    // C[i,j] += A[i,k] * B[k,j]
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}

/**
 * @brief Scalar 3x3 convolution via im2col + GEMM
 * This is a simplified version - production would use optimized im2col
 */
void sb_conv3x3_s1_p1_f32_scalar(
    int N, int C, int H, int W,
    const float* input,
    const float* weights,
    const float* bias,
    int out_c,
    float* output) {
    
    // For each sample in batch
    for (int n = 0; n < N; n++) {
        // For each output channel
        for (int oc = 0; oc < out_c; oc++) {
            // For each output position
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    float sum = 0.0f;
                    
                    // Convolve 3x3 kernel
                    for (int ic = 0; ic < C; ic++) {
                        for (int kh = 0; kh < 3; kh++) {
                            for (int kw = 0; kw < 3; kw++) {
                                int ih = h + kh - 1;  // -1 for padding
                                int iw = w + kw - 1;
                                
                                float input_val = 0.0f;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    // NCHW layout: input[n][ic][ih][iw]
                                    int input_idx = n * (C * H * W) + ic * (H * W) + ih * W + iw;
                                    input_val = input[input_idx];
                                }
                                
                                // weights[oc][ic][kh][kw]
                                int weight_idx = oc * (C * 3 * 3) + ic * (3 * 3) + kh * 3 + kw;
                                sum += input_val * weights[weight_idx];
                            }
                        }
                    }
                    
                    // Add bias if provided
                    if (bias) {
                        sum += bias[oc];
                    }
                    
                    // output[n][oc][h][w]
                    int output_idx = n * (out_c * H * W) + oc * (H * W) + h * W + w;
                    output[output_idx] = sum;
                }
            }
        }
    }
}

/**
 * @brief Scalar element-wise addition
 */
void sb_ew_add_f32_scalar(const float* A, const float* B, float* C, size_t elems) {
    for (size_t i = 0; i < elems; i++) {
        C[i] = A[i] + B[i];
    }
}

/**
 * @brief Scalar element-wise multiplication
 */
void sb_ew_mul_f32_scalar(const float* A, const float* B, float* C, size_t elems) {
    for (size_t i = 0; i < elems; i++) {
        C[i] = A[i] * B[i];
    }
}

/**
 * @brief Scalar element-wise ReLU
 */
void sb_ew_relu_f32_scalar(const float* A, float* C, size_t elems) {
    for (size_t i = 0; i < elems; i++) {
        C[i] = fmaxf(0.0f, A[i]);
    }
}