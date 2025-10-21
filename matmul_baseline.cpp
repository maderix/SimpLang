#include <iostream>
#include <chrono>
#include <vector>
#include <cstring>

// Naive matrix multiplication (for baseline comparison)
void matmul_naive(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] += sum;  // Accumulate (to match MLIR behavior)
        }
    }
}

int main() {
    const int M = 256, K = 256, N = 256;
    const int size_A = M * K;
    const int size_B = K * N;
    const int size_C = M * N;
    const int iterations = 100;

    // Allocate matrices
    std::vector<float> A(size_A, 1.0f);
    std::vector<float> B(size_B, 2.0f);
    std::vector<float> C(size_C, 0.0f);

    // Warm-up
    matmul_naive(A.data(), B.data(), C.data(), M, K, N);
    memset(C.data(), 0, size_C * sizeof(float));

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; iter++) {
        matmul_naive(A.data(), B.data(), C.data(), M, K, N);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "C++ Naive Baseline (ijkorder):\n";
    std::cout << "  Matrix size: " << M << "x" << K << " * " << K << "x" << N << "\n";
    std::cout << "  Iterations: " << iterations << "\n";
    std::cout << "  Total time: " << duration.count() << " ms\n";
    std::cout << "  Time per iteration: " << (duration.count() / (double)iterations) << " ms\n";
    std::cout << "  Result C[0]: " << C[0] << " (expected: 51200.0)\n";
    std::cout << "  GFLOPS: " << (2.0 * M * K * N * iterations) / (duration.count() / 1000.0) / 1e9 << "\n";

    return 0;
}
