#include <iostream>
#include <chrono>
#include <vector>

// Naive matrix multiplication (for baseline comparison)
void matmul_naive(const float* A, const float* B, float* C, int M, int K, int N) {
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

int main() {
    const int N = 64;
    const int size = N * N;
    const int iterations = 10;

    // Allocate matrices
    std::vector<float> A(size, 1.0f);
    std::vector<float> B(size, 2.0f);
    std::vector<float> C(size, 0.0f);

    // Warm-up
    matmul_naive(A.data(), B.data(), C.data(), N, N, N);

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; iter++) {
        matmul_naive(A.data(), B.data(), C.data(), N, N, N);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "C++ Naive Baseline:\n";
    std::cout << "  Matrix size: " << N << "x" << N << "\n";
    std::cout << "  Iterations: " << iterations << "\n";
    std::cout << "  Total time: " << duration.count() << " ms\n";
    std::cout << "  Time per iteration: " << (duration.count() / (double)iterations) << " ms\n";
    std::cout << "  Result C[0]: " << C[0] << " (expected: 128.0)\n";
    std::cout << "  GFLOPS: " << (2.0 * N * N * N * iterations) / (duration.count() / 1000.0) / 1e9 << "\n";

    return 0;
}
