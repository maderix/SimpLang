#include <iostream>
#include <dlfcn.h>
#include <chrono>
#include <vector>
#include <cmath>

typedef double (*kernel_func)();

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <kernel.so>" << std::endl;
        return 1;
    }

    // Load the shared library
    void* handle = dlopen(argv[1], RTLD_LAZY);
    if (!handle) {
        std::cerr << "Cannot open library: " << dlerror() << std::endl;
        return 1;
    }

    // Get the kernel function
    kernel_func kernel = (kernel_func)dlsym(handle, "kernel_main");
    if (!kernel) {
        std::cerr << "Cannot load symbol 'kernel_main': " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }

    // Warm up
    for (int i = 0; i < 10; i++) {
        kernel();
    }

    // Benchmark
    const int iterations = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    
    double result = 0;
    for (int i = 0; i < iterations; i++) {
        result = kernel();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Result: " << result << std::endl;
    std::cout << "Time for " << iterations << " iterations: " << duration.count() << " μs" << std::endl;
    std::cout << "Average time per iteration: " << duration.count() / (double)iterations << " μs" << std::endl;

    dlclose(handle);
    return 0;
}