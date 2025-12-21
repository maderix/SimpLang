/**
 * Matmul Tile Sweep Benchmark Runner
 * 
 * Generates kernels for each tile configuration, compiles them,
 * and runs benchmarks with proper warmup.
 * 
 * Build:
 *   g++ -O3 -fopenmp -o bench_matmul_sweep tests/bench_matmul_sweep_runner.cpp -ldl -std=c++17
 * 
 * Run:
 *   ./bench_matmul_sweep
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <dlfcn.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

// MLIR memref descriptor
#define MEMREF_F32_PARAMS float*, float*, int64_t, int64_t, int64_t
#define MEMREF_F32(ptr, size) (ptr), (ptr), 0, (int64_t)(size), 1

typedef float (*KernelFunc)(MEMREF_F32_PARAMS, MEMREF_F32_PARAMS, MEMREF_F32_PARAMS);

struct TileConfig {
    int m, n, k;
    bool parallel;
    const char* name;
};

// Configurations from the benchmark document
std::vector<TileConfig> configs = {
    // Sequential configs
    {8, 16, 4, false, "seq"},
    {8, 32, 4, false, "seq"},
    // Parallel configs (best performers)
    {64, 256, 4, true, "par"},
    {32, 128, 4, true, "par"},
    {64, 128, 4, true, "par"},
    {128, 256, 8, true, "par"},
};

std::string generateKernel(const TileConfig& cfg, int M, int N, int K) {
    std::string anno = cfg.parallel
        ? "@parallel @tile(" + std::to_string(cfg.m) + ", " + std::to_string(cfg.n) + ", " + std::to_string(cfg.k) + ")"
        : "@tile(" + std::to_string(cfg.m) + ", " + std::to_string(cfg.n) + ", " + std::to_string(cfg.k) + ")";

    std::string ms = std::to_string(M);
    std::string ns = std::to_string(N);
    std::string ks = std::to_string(K);

    return "fn bench(f32[] A_arr, f32[] B_arr, f32[] C_arr) -> f32 {\n"
           "    f32<" + ms + ", " + ks + "> A = tensor_from_array(A_arr, 0i);\n"
           "    f32<" + ks + ", " + ns + "> B = tensor_from_array(B_arr, 0i);\n"
           "    " + anno + "\n"
           "    var C = tensor_matmul(A, B);\n"
           "    var i = 0i;\n"
           "    while (i < " + ms + "i) {\n"
           "        var j = 0i;\n"
           "        while (j < " + ns + "i) {\n"
           "            C_arr[i * " + ns + "i + j] = C[i, j];\n"
           "            j = j + 1i;\n"
           "        }\n"
           "        i = i + 1i;\n"
           "    }\n"
           "    return C[0i, 0i];\n"
           "}\n"
           "fn kernel_main(f32[] A, f32[] B, f32[] C) -> f32 { return bench(A, B, C); }\n";
}

bool compileKernel(const TileConfig& cfg, const std::string& simplang_path, int M, int N, int K) {
    // Write kernel
    std::ofstream sl("/tmp/sweep_kernel.sl");
    sl << generateKernel(cfg, M, N, K);
    sl.close();
    
    // Compile
    std::string cmd = simplang_path + " /tmp/sweep_kernel.sl --emit-mlir -o /tmp/sweep_kernel.o 2>/dev/null";
    if (system(cmd.c_str()) != 0) return false;
    
    // Link
    std::string link = cfg.parallel 
        ? "gcc -shared -fopenmp -o /tmp/sweep_kernel.so /tmp/sweep_kernel.o -lm 2>/dev/null"
        : "gcc -shared -o /tmp/sweep_kernel.so /tmp/sweep_kernel.o -lm 2>/dev/null";
    return system(link.c_str()) == 0;
}

double runBenchmark(KernelFunc kernel, float* A, float* B, float* C, size_t size, int threads) {
    // Set thread count
    if (threads > 0) {
        std::string env = "OMP_NUM_THREADS=" + std::to_string(threads);
        putenv(const_cast<char*>(env.c_str()));
    }
    
    // Proper warmup: 5 iterations
    for (int i = 0; i < 5; i++) {
        kernel(MEMREF_F32(A, size), MEMREF_F32(B, size), MEMREF_F32(C, size));
    }
    
    // Benchmark: 10 iterations, take median
    std::vector<double> times;
    for (int i = 0; i < 10; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        kernel(MEMREF_F32(A, size), MEMREF_F32(B, size), MEMREF_F32(C, size));
        auto end = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    
    // Return median
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

struct MatrixShape {
    int M, N, K;
    const char* name;
};

void runBenchmarkForShape(const MatrixShape& shape, const std::string& simplang) {
    const size_t sizeA = shape.M * shape.K;
    const size_t sizeB = shape.K * shape.N;
    const size_t sizeC = shape.M * shape.N;
    const double flops = 2.0 * shape.M * shape.N * shape.K;

    // Allocate matrices
    float* A = (float*)aligned_alloc(64, sizeA * sizeof(float));
    float* B = (float*)aligned_alloc(64, sizeB * sizeof(float));
    float* C = (float*)aligned_alloc(64, sizeC * sizeof(float));

    for (size_t i = 0; i < sizeA; i++) A[i] = 1.0f;
    for (size_t i = 0; i < sizeB; i++) B[i] = 1.0f;

    std::cout << "\n### " << shape.name << " (" << shape.M << "x" << shape.K << " @ " << shape.K << "x" << shape.N << ")\n";
    std::cout << "FLOP: " << std::fixed << std::setprecision(2) << flops/1e9 << " GFLOP\n\n";

    std::cout << std::setw(6) << "Type"
              << std::setw(6) << "TileM" << std::setw(6) << "TileN" << std::setw(6) << "TileK"
              << std::setw(8) << "Threads"
              << std::setw(10) << "ms"
              << std::setw(12) << "GFLOP/s" << "\n";
    std::cout << std::string(54, '-') << "\n";

    for (const auto& cfg : configs) {
        // Skip configs with tiles larger than matrix dims
        if (cfg.m > shape.M || cfg.n > shape.N || cfg.k > shape.K) continue;

        if (!compileKernel(cfg, simplang, shape.M, shape.N, shape.K)) {
            std::cerr << "Failed to compile: " << cfg.m << "," << cfg.n << "," << cfg.k << "\n";
            continue;
        }

        void* handle = dlopen("/tmp/sweep_kernel.so", RTLD_NOW);
        if (!handle) {
            std::cerr << "dlopen failed: " << dlerror() << "\n";
            continue;
        }

        auto kernel = (KernelFunc)dlsym(handle, "kernel_main");
        if (!kernel) {
            std::cerr << "dlsym failed\n";
            dlclose(handle);
            continue;
        }

        std::vector<int> thread_counts = cfg.parallel ? std::vector<int>{4, 8} : std::vector<int>{1};

        for (int threads : thread_counts) {
            memset(C, 0, sizeC * sizeof(float));
            double ms = runBenchmark(kernel, A, B, C, sizeC, threads);
            double gflops = (flops / 1e9) / (ms / 1000.0);

            std::cout << std::setw(6) << cfg.name
                      << std::setw(6) << cfg.m << std::setw(6) << cfg.n << std::setw(6) << cfg.k
                      << std::setw(8) << threads
                      << std::setw(10) << std::fixed << std::setprecision(2) << ms
                      << std::setw(12) << std::setprecision(1) << gflops << "\n";
        }

        dlclose(handle);
    }

    // Verify
    float expected = (float)shape.K;
    std::cout << "Verify: C[0,0]=" << C[0] << " (expected: " << expected << ")\n";

    free(A); free(B); free(C);
}

int main(int argc, char** argv) {
    std::string simplang = "./build_mlir/src/simplang";
    if (argc > 1) simplang = argv[1];

    std::cout << "============================================================\n";
    std::cout << "  SimpLang Matmul Multi-Shape Benchmark\n";
    std::cout << "============================================================\n";

    // Different matrix shapes to test
    std::vector<MatrixShape> shapes = {
        {512, 512, 512, "Small Square"},
        {1024, 1024, 1024, "Medium Square"},
        {2048, 2048, 2048, "Large Square"},
        {4096, 4096, 4096, "XL Square"},
        // Non-square shapes (common in transformers)
        {1024, 4096, 1024, "Wide (1024x1024 @ 1024x4096)"},
        {4096, 1024, 1024, "Tall (4096x1024 @ 1024x1024)"},
        // Batch-like: simulate B batches of MxK @ KxN
        {768, 768, 768, "Transformer dim=768"},
        {2048, 8192, 2048, "LLaMA FFN (2048x2048 @ 2048x8192)"},
    };

    for (const auto& shape : shapes) {
        runBenchmarkForShape(shape, simplang);
    }

    return 0;
}
