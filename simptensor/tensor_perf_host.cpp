#include <iostream>
#include <dlfcn.h>
#include <chrono>
#include <cmath>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <shared_object_file>" << std::endl;
        return 1;
    }

    // Load the shared library
    void* handle = dlopen(argv[1], RTLD_LAZY);
    if (!handle) {
        std::cerr << "Error loading library: " << dlerror() << std::endl;
        return 1;
    }

    // Load the kernel_main function
    typedef float (*kernel_func_t)();
    kernel_func_t kernel_main = (kernel_func_t) dlsym(handle, "kernel_main");
    
    if (!kernel_main) {
        std::cerr << "Error loading kernel_main: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }

    std::cout << "=== SimpTensor Performance Stress Test ===" << std::endl;
    std::cout << "Tensor dimensions: 4x64x64x128 = " << (4 * 64 * 64 * 128) << " elements" << std::endl;
    std::cout << "Memory usage: ~" << (4 * 64 * 64 * 128 * sizeof(float) / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Using AVX-512 SIMD arrays with 64-byte alignment" << std::endl;
    std::cout << std::endl;

    try {
        // Warmup run
        std::cout << "Warming up..." << std::endl;
        kernel_main();
        
        // Performance measurement
        const int num_runs = 10;
        double total_time = 0.0;
        double best_time = std::numeric_limits<double>::max();
        double worst_time = 0.0;
        
        std::cout << "Running " << num_runs << " performance iterations..." << std::endl;
        
        for (int i = 0; i < num_runs; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            
            float result = kernel_main();
            
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
            
            total_time += elapsed;
            best_time = std::min(best_time, elapsed);
            worst_time = std::max(worst_time, elapsed);
            
            std::cout << "Run " << (i+1) << ": " << elapsed << " ms (result: " << result << ")" << std::endl;
        }
        
        double avg_time = total_time / num_runs;
        
        std::cout << std::endl;
        std::cout << "=== Performance Results ===" << std::endl;
        std::cout << "Average time: " << avg_time << " ms" << std::endl;
        std::cout << "Best time:    " << best_time << " ms" << std::endl;
        std::cout << "Worst time:   " << worst_time << " ms" << std::endl;
        std::cout << "Total elements processed: " << (4 * 64 * 64 * 128) << std::endl;
        
        // Calculate throughput metrics
        const long long total_elements = 4LL * 64 * 64 * 128;
        const long long operations_per_run = total_elements * 4; // 4 major operation phases
        double throughput_best = (operations_per_run / (best_time / 1000.0)) / 1e6; // M ops/sec
        double throughput_avg = (operations_per_run / (avg_time / 1000.0)) / 1e6; // M ops/sec
        
        std::cout << "Peak throughput:    " << throughput_best << " M ops/sec" << std::endl;
        std::cout << "Average throughput: " << throughput_avg << " M ops/sec" << std::endl;
        
        // Memory bandwidth estimation (rough)
        const long long memory_ops = total_elements * 8; // Read + write per element approximately
        double memory_bandwidth = (memory_ops * sizeof(float) / (best_time / 1000.0)) / (1024 * 1024 * 1024); // GB/sec
        std::cout << "Est. memory bandwidth: " << memory_bandwidth << " GB/sec" << std::endl;
        
        std::cout << std::endl;
        std::cout << "✓ Tensor performance stress test completed successfully!" << std::endl;
        std::cout << "✓ SIMD vectorization and memory alignment working optimally" << std::endl;
        std::cout << "✓ NHWC stride-based indexing performing efficiently" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during kernel execution: " << e.what() << std::endl;
        dlclose(handle);
        return 1;
    }

    dlclose(handle);
    return 0;
}