/**
 * @file simpblas_test.cpp
 * @brief Unit tests for simpblas kernels
 */

#include <gtest/gtest.h>
#include "simpblas.h"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

class SimpBlasTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize simpblas
        ASSERT_EQ(sb_init(), 0) << "Failed to initialize simpblas";
        
        // Set up random number generator
        rng.seed(42);
    }
    
    std::mt19937 rng;
    
    // Helper function to compare floating point arrays with tolerance
    void expectArrayNear(const float* a, const float* b, size_t size, float tolerance = 1e-5f) {
        for (size_t i = 0; i < size; i++) {
            EXPECT_NEAR(a[i], b[i], tolerance) 
                << "Arrays differ at index " << i 
                << ": " << a[i] << " vs " << b[i];
        }
    }
    
    // Generate random float array
    void fillRandom(float* data, size_t size, float min = -1.0f, float max = 1.0f) {
        std::uniform_real_distribution<float> dist(min, max);
        for (size_t i = 0; i < size; i++) {
            data[i] = dist(rng);
        }
    }
};

TEST_F(SimpBlasTest, InitializationTest) {
    // Test version info
    const char* version = sb_get_version();
    EXPECT_STREQ(version, "0.1.0-alpha");
    
    // Test kernel info
    const char* kernel_info = sb_get_kernel_info();
    EXPECT_TRUE(kernel_info != nullptr);
    EXPECT_TRUE(strlen(kernel_info) > 0);
    
    std::cout << "simpblas version: " << version << std::endl;
    std::cout << "Available kernels: " << kernel_info << std::endl;
}

TEST_F(SimpBlasTest, ElementWiseAdd) {
    const size_t size = 1000;
    std::vector<float> A(size), B(size), C(size), expected(size);
    
    // Fill with random data
    fillRandom(A.data(), size);
    fillRandom(B.data(), size);
    
    // Compute expected result
    for (size_t i = 0; i < size; i++) {
        expected[i] = A[i] + B[i];
    }
    
    // Test simpblas implementation
    sb_ew_add_f32(A.data(), B.data(), C.data(), size);
    
    // Compare results
    expectArrayNear(C.data(), expected.data(), size);
}

TEST_F(SimpBlasTest, ElementWiseMultiply) {
    const size_t size = 1000;
    std::vector<float> A(size), B(size), C(size), expected(size);
    
    fillRandom(A.data(), size);
    fillRandom(B.data(), size);
    
    // Compute expected result
    for (size_t i = 0; i < size; i++) {
        expected[i] = A[i] * B[i];
    }
    
    // Test simpblas implementation
    sb_ew_mul_f32(A.data(), B.data(), C.data(), size);
    
    expectArrayNear(C.data(), expected.data(), size);
}

TEST_F(SimpBlasTest, ElementWiseReLU) {
    const size_t size = 1000;
    std::vector<float> A(size), C(size), expected(size);
    
    fillRandom(A.data(), size, -2.0f, 2.0f);
    
    // Compute expected result
    for (size_t i = 0; i < size; i++) {
        expected[i] = std::max(0.0f, A[i]);
    }
    
    // Test simpblas implementation
    sb_ew_relu_f32(A.data(), C.data(), size);
    
    expectArrayNear(C.data(), expected.data(), size);
}

TEST_F(SimpBlasTest, SmallGEMM) {
    // Test small matrix multiplication
    const int M = 4, N = 4, K = 4;
    std::vector<float> A(M * K), B(K * N), C(M * N), expected(M * N);
    
    fillRandom(A.data(), M * K);
    fillRandom(B.data(), K * N);
    
    // Compute expected result using naive implementation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            expected[i * N + j] = sum;
        }
    }
    
    // Test simpblas implementation
    sb_gemm_f32(M, N, K, A.data(), K, B.data(), N, C.data(), N);
    
    expectArrayNear(C.data(), expected.data(), M * N);
}

TEST_F(SimpBlasTest, LargerGEMM) {
    // Test larger matrix multiplication
    const int M = 64, N = 64, K = 64;
    std::vector<float> A(M * K), B(K * N), C(M * N), expected(M * N);
    
    fillRandom(A.data(), M * K);
    fillRandom(B.data(), K * N);
    
    // Initialize C and expected to zero
    std::fill(C.begin(), C.end(), 0.0f);
    std::fill(expected.begin(), expected.end(), 0.0f);
    
    // Compute expected result
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                expected[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
    
    // Test simpblas implementation
    sb_gemm_f32(M, N, K, A.data(), K, B.data(), N, C.data(), N);
    
    expectArrayNear(C.data(), expected.data(), M * N, 1e-4f);  // Slightly relaxed tolerance
}

TEST_F(SimpBlasTest, SimpleConv3x3) {
    // Test simple 3x3 convolution
    const int N = 1, C = 2, H = 4, W = 4, out_c = 1;
    
    std::vector<float> input(N * C * H * W);
    std::vector<float> weights(out_c * C * 3 * 3);
    std::vector<float> bias(out_c);
    std::vector<float> output(N * out_c * H * W);
    
    // Simple test data
    fillRandom(input.data(), N * C * H * W, 0.0f, 1.0f);
    fillRandom(weights.data(), out_c * C * 3 * 3, -0.1f, 0.1f);
    fillRandom(bias.data(), out_c, -0.1f, 0.1f);
    
    // Test convolution (we don't compute expected here as it's complex)
    // This test mainly checks that the function doesn't crash
    sb_conv3x3_s1_p1_f32(N, C, H, W, input.data(), weights.data(), bias.data(), out_c, output.data());
    
    // Basic sanity check - output should not be all zeros or contain NaN/inf
    bool hasNonZero = false;
    for (float val : output) {
        EXPECT_TRUE(std::isfinite(val)) << "Output contains non-finite value: " << val;
        if (val != 0.0f) hasNonZero = true;
    }
    EXPECT_TRUE(hasNonZero) << "Convolution output is all zeros (likely incorrect)";
}

// Performance benchmark tests
TEST_F(SimpBlasTest, DISABLED_BenchmarkElementWise) {
    const size_t size = 1000000;  // 1M elements
    std::vector<float> A(size), B(size), C(size);
    
    fillRandom(A.data(), size);
    fillRandom(B.data(), size);
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        sb_ew_add_f32(A.data(), B.data(), C.data(), size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Element-wise add benchmark: " 
              << duration.count() / 100.0 << " μs per operation" << std::endl;
}

TEST_F(SimpBlasTest, DISABLED_BenchmarkGEMM) {
    const int M = 256, N = 256, K = 256;
    std::vector<float> A(M * K), B(K * N), C(M * N);
    
    fillRandom(A.data(), M * K);
    fillRandom(B.data(), K * N);
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        sb_gemm_f32(M, N, K, A.data(), K, B.data(), N, C.data(), N);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double gflops = (2.0 * M * N * K * 10) / (duration.count() / 1000.0) / 1e9;
    
    std::cout << "GEMM benchmark (" << M << "x" << N << "x" << K << "): " 
              << gflops << " GFLOPS" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}