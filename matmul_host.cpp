#include <iostream>
#include <chrono>
#include <dlfcn.h>
#include <cstring>

// MLIR memref<?xf32> expands to: (allocated_ptr, aligned_ptr, offset, size, stride)
// So matmul_benchmark(memref A, memref B, memref C, i64 m, i64 k, i64 n, i64 iters)
// becomes: (A_alloc, A_align, A_off, A_sz, A_stride, B_alloc, B_align, B_off, B_sz, B_stride,
//           C_alloc, C_align, C_off, C_sz, C_stride, m, k, n, iters) -> float
typedef float (*MatmulKernelFunc)(
    float*, float*, int64_t, int64_t, int64_t,  // A: alloc, align, offset, size, stride
    float*, float*, int64_t, int64_t, int64_t,  // B: alloc, align, offset, size, stride
    float*, float*, int64_t, int64_t, int64_t,  // C: alloc, align, offset, size, stride
    int64_t, int64_t, int64_t, int64_t);        // m, k, n, iters

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <kernel.so>\n";
        return 1;
    }

    // Matrix dimensions
    const int64_t M = 256, K = 256, N = 256;
    const int64_t size_A = M * K;  // 65536
    const int64_t size_B = K * N;  // 65536
    const int64_t size_C = M * N;  // 65536
    const int64_t iterations = 100;

    // HOST ALLOCATES ALL BUFFERS
    float* A = new float[size_A];
    float* B = new float[size_B];
    float* C = new float[size_C];

    // Initialize A with 1.0
    for (int64_t i = 0; i < size_A; i++) {
        A[i] = 1.0f;
    }

    // Initialize B with 2.0
    for (int64_t i = 0; i < size_B; i++) {
        B[i] = 2.0f;
    }

    // Initialize C with 0.0 (important since linalg.matmul accumulates)
    memset(C, 0, size_C * sizeof(float));

    // Load kernel
    void* handle = dlopen(argv[1], RTLD_LAZY);
    if (!handle) {
        std::cerr << "Error loading library: " << dlerror() << "\n";
        delete[] A; delete[] B; delete[] C;
        return 1;
    }

    MatmulKernelFunc kernel = (MatmulKernelFunc)dlsym(handle, "matmul_benchmark");
    if (!kernel) {
        std::cerr << "Error finding matmul_benchmark: " << dlerror() << "\n";
        dlclose(handle);
        delete[] A; delete[] B; delete[] C;
        return 1;
    }

    // Warm-up - pass memref descriptors (alloc_ptr, align_ptr, offset, size, stride)
    kernel(
        A, A, 0, size_A, 1,  // memref A
        B, B, 0, size_B, 1,  // memref B
        C, C, 0, size_C, 1,  // memref C
        M, K, N, 1);         // m, k, n, iterations
    memset(C, 0, size_C * sizeof(float));  // Reset C

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    float result = kernel(
        A, A, 0, size_A, 1,  // memref A
        B, B, 0, size_B, 1,  // memref B
        C, C, 0, size_C, 1,  // memref C
        M, K, N, iterations); // m, k, n, iterations
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "SimpLang MLIR Matmul (Host-Kernel Model):\n";
    std::cout << "  Matrix size: " << M << "x" << K << " * " << K << "x" << N << "\n";
    std::cout << "  Iterations: " << iterations << "\n";
    std::cout << "  Total time: " << duration.count() << " ms\n";
    std::cout << "  Time per iteration: " << (duration.count() / (double)iterations) << " ms\n";
    std::cout << "  Result C[0]: " << result << " (expected: 51200.0 for 100 iterations)\n";
    std::cout << "  GFLOPS: " << (2.0 * M * K * N * iterations) / (duration.count() / 1000.0) / 1e9 << "\n";

    // Expected: 256*2.0*100 = 51200.0 (since C accumulates over 100 iterations)
    float expected = 256.0f * 2.0f * iterations;
    std::cout << "  " << (std::abs(result - expected) < 1.0f ? "✓" : "✗") << " Correctness\n";

    // HOST FREES ALL BUFFERS
    delete[] A;
    delete[] B;
    delete[] C;
    dlclose(handle);

    return 0;
}
