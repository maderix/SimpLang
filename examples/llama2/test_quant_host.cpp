#include <iostream>
#include <dlfcn.h>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <iomanip>

#define MEMREF_PARAMS float*, float*, int64_t, int64_t, int64_t
#define MEMREF_PARAMS_I8 int8_t*, int8_t*, int64_t, int64_t, int64_t
#define PASS_MEMREF(vec) vec.data(), vec.data(), 0, (int64_t)vec.size(), 1
#define PASS_MEMREF_I8(vec) vec.data(), vec.data(), 0, (int64_t)vec.size(), 1

typedef float (*MatmulW4Func)(
    MEMREF_PARAMS,      // act
    MEMREF_PARAMS_I8,   // qweights
    MEMREF_PARAMS,      // scales
    MEMREF_PARAMS,      // zeros
    MEMREF_PARAMS,      // output
    int64_t, int64_t, int64_t, int64_t  // M, K, N, group_size
);

// Quantize weights to W4
void quantize_w4(const std::vector<float>& weights, std::vector<int8_t>& qweights,
                 std::vector<float>& scales, std::vector<float>& zeros,
                 int group_size) {
    int total = weights.size();
    int num_groups = (total + group_size - 1) / group_size;

    scales.resize(num_groups);
    zeros.resize(num_groups);
    qweights.resize((total + 1) / 2);  // 2 values per byte

    for (int g = 0; g < num_groups; g++) {
        int start = g * group_size;
        int end = std::min(start + group_size, total);

        // Find min/max in group
        float min_val = weights[start];
        float max_val = weights[start];
        for (int i = start; i < end; i++) {
            min_val = std::min(min_val, weights[i]);
            max_val = std::max(max_val, weights[i]);
        }

        float scale = (max_val - min_val) / 15.0f;
        if (scale < 1e-8f) scale = 1e-8f;

        scales[g] = scale;
        zeros[g] = min_val;

        // Quantize values in group
        for (int i = start; i < end; i++) {
            int qval = std::round((weights[i] - min_val) / scale);
            qval = std::max(0, std::min(15, qval));

            int byte_idx = i / 2;
            if (i % 2 == 0) {
                qweights[byte_idx] = qval;
            } else {
                qweights[byte_idx] |= (qval << 4);
            }
        }
    }
}

// FP32 reference matmul
void matmul_fp32(const std::vector<float>& A, const std::vector<float>& B,
                 std::vector<float>& C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

float compute_error(const std::vector<float>& a, const std::vector<float>& b) {
    float mse = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = a[i] - b[i];
        mse += diff * diff;
    }
    return std::sqrt(mse / a.size());
}

int main() {
    std::cout << "=== W4 Quantized Matmul Test ===" << std::endl;

    // Test dimensions
    int M = 128, K = 256, N = 128;
    int group_size = 128;

    std::cout << "Matrix sizes: A[" << M << "," << K << "] x B[" << K << "," << N << "]" << std::endl;
    std::cout << "Quantization: W4, group_size=" << group_size << std::endl;

    // Initialize matrices
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);

    std::mt19937 gen(42);
    std::normal_distribution<float> dis(0.0f, 0.02f);

    for (auto& v : A) v = dis(gen);
    for (auto& v : B) v = dis(gen);

    // Quantize B
    std::vector<int8_t> B_quant;
    std::vector<float> B_scales;
    std::vector<float> B_zeros;

    std::cout << "Quantizing weights..." << std::flush;
    quantize_w4(B, B_quant, B_scales, B_zeros, group_size);
    std::cout << " Done" << std::endl;

    size_t fp32_bytes = B.size() * sizeof(float);
    size_t w4_bytes = B_quant.size() + B_scales.size() * sizeof(float) + B_zeros.size() * sizeof(float);
    std::cout << "Memory: FP32=" << fp32_bytes/1024 << " KB, W4=" << w4_bytes/1024
              << " KB (" << (float)fp32_bytes/w4_bytes << "x compression)" << std::endl;

    // FP32 baseline
    std::vector<float> C_fp32(M * N);
    auto start = std::chrono::high_resolution_clock::now();
    matmul_fp32(A, B, C_fp32, M, K, N);
    auto end = std::chrono::high_resolution_clock::now();
    auto fp32_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "\nFP32 baseline: " << fp32_time / 1000.0 << " ms" << std::endl;

    // Load quantized kernel
    void* handle = dlopen("/tmp/test_quant.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Error loading library: " << dlerror() << std::endl;
        return 1;
    }

    MatmulW4Func matmul_w4 = (MatmulW4Func)dlsym(handle, "matmul_w4");
    if (!matmul_w4) {
        std::cerr << "Error loading symbol: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }

    // Run quantized matmul
    std::vector<float> C_quant(M * N);

    start = std::chrono::high_resolution_clock::now();
    matmul_w4(
        PASS_MEMREF(A),
        PASS_MEMREF_I8(B_quant),
        PASS_MEMREF(B_scales),
        PASS_MEMREF(B_zeros),
        PASS_MEMREF(C_quant),
        M, K, N, group_size
    );
    end = std::chrono::high_resolution_clock::now();
    auto quant_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "W4 quantized: " << quant_time / 1000.0 << " ms" << std::endl;
    std::cout << "Slowdown: " << (float)quant_time / fp32_time << "x" << std::endl;

    // Measure error
    float rmse = compute_error(C_fp32, C_quant);
    std::cout << "\nQuantization RMSE: " << std::scientific << rmse << std::endl;

    // Check a few values
    std::cout << "\nSample outputs:" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < 3; i++) {
        std::cout << "  C[" << i << "]: FP32=" << C_fp32[i]
                  << ", W4=" << C_quant[i]
                  << ", diff=" << std::abs(C_fp32[i] - C_quant[i]) << std::endl;
    }

    dlclose(handle);

    std::cout << "\n✓ Test completed successfully!" << std::endl;
    return 0;
}
