// W8A16 MatMul Benchmark Runner
// Tests Hi-Lo split correctness and performance
//
// Compile:
//   g++ -O3 -march=native -o /tmp/bench_w8a16 bench_matmul_w8a16_runner.cpp -ldl
//
// Run:
//   /tmp/bench_w8a16 /tmp/bench_w8a16.so

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>
#include <dlfcn.h>
#include <cmath>
#include <iomanip>

// MLIR memref ABI
#define MEMREF_I8  int8_t*, int8_t*, int64_t, int64_t, int64_t
#define MEMREF_I16 int16_t*, int16_t*, int64_t, int64_t, int64_t
#define MEMREF_I32 int32_t*, int32_t*, int64_t, int64_t, int64_t
#define PASS_I8(p, s)  p, p, 0LL, (int64_t)(s), 1LL
#define PASS_I16(p, s) p, p, 0LL, (int64_t)(s), 1LL
#define PASS_I32(p, s) p, p, 0LL, (int64_t)(s), 1LL

// Function types
using w8a16_fn = int32_t(*)(MEMREF_I16, MEMREF_I8, MEMREF_I32, MEMREF_I8, MEMREF_I8);
using ref_fn = int32_t(*)(MEMREF_I16, MEMREF_I8, MEMREF_I32);

template<typename T>
T* alloc_aligned(size_t count) {
    return (T*)aligned_alloc(64, count * sizeof(T));
}

// Reference INT16 x INT8 matmul (C++ for correctness check)
void matmul_ref_cpp(const int16_t* A, const int8_t* W_T, int32_t* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                acc += (int32_t)A[i * K + k] * (int32_t)W_T[j * K + k];
            }
            C[i * N + j] = acc;
        }
    }
}

// Test correctness of Hi-Lo split
bool test_correctness(w8a16_fn hilo_fn, int M, int N, int K) {
    auto A = alloc_aligned<int16_t>(M * K);
    auto W_T = alloc_aligned<int8_t>(N * K);
    auto C_hilo = alloc_aligned<int32_t>(M * N);
    auto C_ref = alloc_aligned<int32_t>(M * N);
    auto A_hi = alloc_aligned<int8_t>(M * K);
    auto A_lo = alloc_aligned<int8_t>(M * K);

    // Initialize with test values
    srand(42);
    for (int i = 0; i < M * K; i++) {
        // INT16 values in reasonable range for Q8 (-128 to 127 scaled)
        A[i] = (int16_t)((rand() % 512) - 256);
    }
    for (int i = 0; i < N * K; i++) {
        W_T[i] = (int8_t)((rand() % 256) - 128);
    }

    // Compute reference
    matmul_ref_cpp(A, W_T, C_ref, M, N, K);

    // Compute Hi-Lo
    hilo_fn(PASS_I16(A, M * K), PASS_I8(W_T, N * K), PASS_I32(C_hilo, M * N),
            PASS_I8(A_hi, M * K), PASS_I8(A_lo, M * K));

    // Compare
    int errors = 0;
    int32_t max_diff = 0;
    for (int i = 0; i < M * N; i++) {
        int32_t diff = abs(C_hilo[i] - C_ref[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 0) errors++;
    }

    bool passed = (errors == 0);
    std::cout << "  " << M << "x" << N << "x" << K << ": ";
    if (passed) {
        std::cout << "PASS (exact match)" << std::endl;
    } else {
        std::cout << "FAIL (" << errors << " errors, max_diff=" << max_diff << ")" << std::endl;
    }

    free(A); free(W_T); free(C_hilo); free(C_ref); free(A_hi); free(A_lo);
    return passed;
}

// Benchmark function
void benchmark(const char* name, w8a16_fn fn, int M, int N, int K, int iters) {
    auto A = alloc_aligned<int16_t>(M * K);
    auto W_T = alloc_aligned<int8_t>(N * K);
    auto C = alloc_aligned<int32_t>(M * N);
    auto A_hi = alloc_aligned<int8_t>(M * K);
    auto A_lo = alloc_aligned<int8_t>(M * K);

    // Initialize
    for (int i = 0; i < M * K; i++) A[i] = (int16_t)(i % 256 - 128);
    for (int i = 0; i < N * K; i++) W_T[i] = (int8_t)(i % 256 - 128);

    // Warmup
    for (int i = 0; i < 3; i++) {
        fn(PASS_I16(A, M * K), PASS_I8(W_T, N * K), PASS_I32(C, M * N),
           PASS_I8(A_hi, M * K), PASS_I8(A_lo, M * K));
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        fn(PASS_I16(A, M * K), PASS_I8(W_T, N * K), PASS_I32(C, M * N),
           PASS_I8(A_hi, M * K), PASS_I8(A_lo, M * K));
    }
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    double gops = 2.0 * M * N * K / 1e9;
    double gops_s = gops / (ms / 1000.0);

    std::cout << std::setw(20) << name << " | "
              << std::setw(5) << M << "x" << std::setw(5) << N << "x" << std::setw(5) << K << " | "
              << std::setw(10) << std::fixed << std::setprecision(3) << ms << " ms | "
              << std::setw(8) << std::fixed << std::setprecision(2) << gops_s << " GOP/s"
              << std::endl;

    free(A); free(W_T); free(C); free(A_hi); free(A_lo);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.so>\n";
        return 1;
    }

    void* h = dlopen(argv[1], RTLD_NOW);
    if (!h) {
        std::cerr << "Error: " << dlerror() << "\n";
        return 1;
    }

    std::cout << "============================================================" << std::endl;
    std::cout << "   W8A16 Hi-Lo Split MatMul Benchmark" << std::endl;
    std::cout << "============================================================" << std::endl;

    // Test correctness
    std::cout << "\n--- Correctness Tests ---" << std::endl;

    auto fn_64 = (w8a16_fn)dlsym(h, "bench_w8a16_64x64");
    auto fn_128 = (w8a16_fn)dlsym(h, "bench_w8a16_128x128");
    auto fn_256 = (w8a16_fn)dlsym(h, "bench_w8a16_256x256");
    auto fn_512 = (w8a16_fn)dlsym(h, "bench_w8a16_512x512");
    auto fn_1024 = (w8a16_fn)dlsym(h, "bench_w8a16_1024x1024");
    auto fn_2048 = (w8a16_fn)dlsym(h, "bench_w8a16_2048x2048");

    if (fn_64) test_correctness(fn_64, 64, 64, 64);
    if (fn_128) test_correctness(fn_128, 128, 128, 128);
    if (fn_256) test_correctness(fn_256, 256, 256, 256);

    // Performance benchmarks
    std::cout << "\n--- Performance Benchmarks ---" << std::endl;
    std::cout << std::setw(20) << "Name" << " | "
              << std::setw(19) << "Size" << " | "
              << std::setw(13) << "Time" << " | "
              << std::setw(10) << "Throughput" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    if (fn_64) benchmark("W8A16 64x64", fn_64, 64, 64, 64, 1000);
    if (fn_128) benchmark("W8A16 128x128", fn_128, 128, 128, 128, 500);
    if (fn_256) benchmark("W8A16 256x256", fn_256, 256, 256, 256, 100);
    if (fn_512) benchmark("W8A16 512x512", fn_512, 512, 512, 512, 20);
    if (fn_1024) benchmark("W8A16 1024x1024", fn_1024, 1024, 1024, 1024, 5);
    if (fn_2048) benchmark("W8A16 2048x2048", fn_2048, 2048, 2048, 2048, 2);

    // LLaMA-like sizes
    std::cout << "\n--- LLaMA-like Sizes (prefill) ---" << std::endl;
    auto fn_32x2048 = (w8a16_fn)dlsym(h, "bench_w8a16_32x2048x2048");
    auto fn_64x2048 = (w8a16_fn)dlsym(h, "bench_w8a16_64x2048x2048");
    auto fn_128x2048 = (w8a16_fn)dlsym(h, "bench_w8a16_128x2048x2048");

    if (fn_32x2048) benchmark("W8A16 32x2048", fn_32x2048, 32, 2048, 2048, 5);
    if (fn_64x2048) benchmark("W8A16 64x2048", fn_64x2048, 64, 2048, 2048, 3);
    if (fn_128x2048) benchmark("W8A16 128x2048", fn_128x2048, 128, 2048, 2048, 2);

    std::cout << "\n============================================================" << std::endl;

    dlclose(h);
    return 0;
}
