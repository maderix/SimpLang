#include "kernel_runner.hpp"
#include "simd_types.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <immintrin.h>
#include <chrono>

// Helper function to print timing results with proper scaling
void printTimings(const std::string& operation, double time_ms, size_t iterations) {
    double time_per_iter = time_ms / iterations;
    double ops_per_sec = iterations / (time_ms / 1000.0);

    std::cout << std::fixed 
              << operation << ":\n"
              << "  Total time:     " << std::setprecision(3) << time_ms << " ms\n"
              << "  Time per iter:  " << std::setprecision(6) << time_per_iter << " ms\n"
              << "  Operations/sec: " << std::setprecision(0) << ops_per_sec << "\n";
}

void print_vector(const char* label, double* data, size_t elements) {
    std::cout << label << ": [";
    for (size_t i = 0; i < elements; i++) {
        std::cout << std::fixed << std::setprecision(2) << data[i];
        if (i < elements - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <kernel.so>" << std::endl;
        return 1;
    }

    KernelRunner runner;
    
    try {
        // Allocate aligned memory for SIMD operations
        const size_t VECTOR_SIZE = 1024;
        const size_t sse_elements = VECTOR_SIZE * SSESlice::VECTOR_SIZE;
        const size_t avx_elements = VECTOR_SIZE * AVXSlice::VECTOR_SIZE;
        
        double* sse_data = (double*)aligned_alloc(16, sse_elements * sizeof(double));
        double* avx_data = (double*)aligned_alloc(32, avx_elements * sizeof(double));

        if (!sse_data || !avx_data) {
            throw std::runtime_error("Failed to allocate aligned memory");
        }

        // Cast to SIMD vector types
        sse_vector_t* sse_vec_data = reinterpret_cast<sse_vector_t*>(sse_data);
        avx_vector_t* avx_vec_data = reinterpret_cast<avx_vector_t*>(avx_data);

        // Initialize slices
        SSESlice sse_slice = {sse_vec_data, VECTOR_SIZE, VECTOR_SIZE};
        AVXSlice avx_slice = {avx_vec_data, VECTOR_SIZE, VECTOR_SIZE};

        // Load and run kernel
        runner.loadLibrary(argv[1]);
        
        const size_t WARMUP_ITERATIONS = 100;
        const size_t ITERATIONS = 10000;

        // Warmup runs
        for (size_t i = 0; i < WARMUP_ITERATIONS; i++) {
            runner.runKernel(&sse_slice, &avx_slice);
        }
        
        // Timed runs
        auto start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < ITERATIONS; i++) {
            runner.runKernel(&sse_slice, &avx_slice);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        double time_ms = duration.count() / 1e6; // Convert ns to ms
        
        // Print results
        std::cout << "\nPerformance Results:\n"
                  << "==================\n";
        printTimings("Kernel execution", time_ms, ITERATIONS);
        
        // Print sample results
        print_vector("SSE Results", sse_data, 2);
        print_vector("AVX Results", avx_data, 8);

        // Cleanup
        free(sse_data);
        free(avx_data);
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 