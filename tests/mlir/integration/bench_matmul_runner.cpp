#include <iostream>
#include <chrono>
#include <dlfcn.h>

typedef float (*KernelMainFunc)();

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <shared_object.so>\n";
        return 1;
    }

    void* handle = dlopen(argv[1], RTLD_LAZY);
    if (!handle) {
        std::cerr << "Error loading library: " << dlerror() << "\n";
        return 1;
    }

    KernelMainFunc kernel_main = (KernelMainFunc)dlsym(handle, "main");
    if (!kernel_main) {
        std::cerr << "Error finding main function: " << dlerror() << "\n";
        dlclose(handle);
        return 1;
    }

    // Warm-up
    kernel_main();

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    float result = kernel_main();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double duration_ms = duration_us.count() / 1000.0;

    std::cout << "SimpLang MLIR+Linalg:\n";
    std::cout << "  Matrix size: 64x64\n";
    std::cout << "  Iterations: 10\n";
    std::cout << "  Total time: " << duration_ms << " ms\n";
    std::cout << "  Time per iteration: " << (duration_ms / 10.0) << " ms\n";
    std::cout << "  Result C[0]: " << result << " (expected: 128.0)\n";
    std::cout << "  GFLOPS: " << (2.0 * 64 * 64 * 64 * 10) / (duration_ms / 1000.0) / 1e9 << "\n";

    if (std::abs(result - 128.0f) < 0.01f) {
        std::cout << "  ✓ Correctness: PASS\n";
    } else {
        std::cout << "  ✗ Correctness: FAIL\n";
    }

    dlclose(handle);
    return 0;
}
