/**
 * INT8 MatMul Parallel Runner - All Sizes
 * Tests SimpLang parallel matmul at multiple sizes with OpenMP
 */

#include <iostream>
#include <chrono>
#include <dlfcn.h>
#include <iomanip>
#include <vector>
#include <omp.h>

typedef int32_t (*ChunkFunc)(int8_t*, int8_t*, int64_t, int64_t, int64_t,
                             int8_t*, int8_t*, int64_t, int64_t, int64_t,
                             int32_t*, int32_t*, int64_t, int64_t, int64_t,
                             int64_t);

template<typename T>
inline void DoNotOptimize(T const& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

struct Result {
    int N;
    double single_ms, par_ms;
    double single_giops, par_giops;
    double speedup;
};

Result benchmark_size(int N, int chunk_size, ChunkFunc chunk_fn, int iterations) {
    Result r = {N, 0, 0, 0, 0, 0};
    int num_chunks = N / chunk_size;

    std::vector<int8_t> A(N * N), B(N * N);
    std::vector<int32_t> C(N * N, 0);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int val = ((i * N + j) % 127) - 64;
            A[i * N + j] = (int8_t)val;
            B[j * N + i] = (int8_t)val;
        }
    }

    // Single-threaded (sequential chunks)
    for (int w = 0; w < 2; w++) {
        std::fill(C.begin(), C.end(), 0);
        for (int c = 0; c < num_chunks; c++) {
            chunk_fn(A.data(), A.data(), 0, N*N, 1,
                     B.data(), B.data(), 0, N*N, 1,
                     C.data(), C.data(), 0, N*N, 1,
                     c * chunk_size);
        }
    }
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; iter++) {
        std::fill(C.begin(), C.end(), 0);
        for (int c = 0; c < num_chunks; c++) {
            chunk_fn(A.data(), A.data(), 0, N*N, 1,
                     B.data(), B.data(), 0, N*N, 1,
                     C.data(), C.data(), 0, N*N, 1,
                     c * chunk_size);
        }
        DoNotOptimize(C[0]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    r.single_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

    // Parallel
    for (int w = 0; w < 2; w++) {
        std::fill(C.begin(), C.end(), 0);
        #pragma omp parallel for
        for (int c = 0; c < num_chunks; c++) {
            chunk_fn(A.data(), A.data(), 0, N*N, 1,
                     B.data(), B.data(), 0, N*N, 1,
                     C.data(), C.data(), 0, N*N, 1,
                     c * chunk_size);
        }
    }
    start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; iter++) {
        std::fill(C.begin(), C.end(), 0);
        #pragma omp parallel for
        for (int c = 0; c < num_chunks; c++) {
            chunk_fn(A.data(), A.data(), 0, N*N, 1,
                     B.data(), B.data(), 0, N*N, 1,
                     C.data(), C.data(), 0, N*N, 1,
                     c * chunk_size);
        }
        DoNotOptimize(C[0]);
    }
    end = std::chrono::high_resolution_clock::now();
    r.par_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

    double ops = 2.0 * N * N * N;
    r.single_giops = (ops / 1e9) / (r.single_ms / 1000.0);
    r.par_giops = (ops / 1e9) / (r.par_ms / 1000.0);
    r.speedup = r.single_ms / r.par_ms;
    return r;
}

int main(int argc, char* argv[]) {
    void* handle = dlopen("/tmp/bench_int8_parallel.so", RTLD_LAZY);
    if (!handle) { std::cerr << "Load failed\n"; return 1; }

    int threads = omp_get_max_threads();
    std::cout << "==========================================================================\n";
    std::cout << "   SimpLang INT8 VNNI Parallel Benchmark (" << threads << " threads)\n";
    std::cout << "==========================================================================\n\n";
    std::cout << "   Size   | Single GIOP/s | Parallel GIOP/s | Speedup\n";
    std::cout << "----------|---------------|-----------------|--------\n";

    struct { int N; int chunk; const char* fn; int iters; } sizes[] = {
        {256, 32, "int8_chunk_256", 20},
        {512, 64, "int8_chunk_512", 10},
        {768, 96, "int8_chunk_768", 5},
        {1024, 128, "int8_chunk_1024", 3},
        {2048, 256, "int8_chunk_2048", 1},
    };

    for (auto& s : sizes) {
        ChunkFunc fn = (ChunkFunc)dlsym(handle, s.fn);
        if (!fn) { std::cout << s.N << " - missing\n"; continue; }
        Result r = benchmark_size(s.N, s.chunk, fn, s.iters);
        printf(" %4dx%4d |    %8.2f   |     %8.2f    |  %.2fx\n",
               r.N, r.N, r.single_giops, r.par_giops, r.speedup);
    }

    std::cout << "==========================================================================\n";
    dlclose(handle);
    return 0;
}
