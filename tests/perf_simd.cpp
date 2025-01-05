#include "kernel_runner.hpp"
#include "simd_types.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <immintrin.h>
#include <chrono>

// Helper function to print timing results
void printTimings(const std::string& operation, double time_ms, size_t iterations) {
    std::cout << std::fixed << std::setprecision(3)
              << operation << ": "
              << time_ms << " ms total, "
              << (time_ms / iterations) << " ms/iter, "
              << (iterations / (time_ms / 1000.0)) << " ops/sec"
              << std::endl;
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

        // Before slice initialization
        // Cast the raw double pointers to the appropriate SIMD vector types
        sse_vector_t* sse_vec_data = reinterpret_cast<sse_vector_t*>(sse_data);
        avx_vector_t* avx_vec_data = reinterpret_cast<avx_vector_t*>(avx_data);

        // Initialize slices with properly typed pointers
        SSESlice sse_slice = {sse_vec_data, VECTOR_SIZE, VECTOR_SIZE};
        AVXSlice avx_slice = {avx_vec_data, VECTOR_SIZE, VECTOR_SIZE};

        // Load kernel
        runner.loadLibrary(argv[1]);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Run kernel multiple times
        const size_t ITERATIONS = 10000;
        for (size_t i = 0; i < ITERATIONS; i++) {
            runner.runKernel(&sse_slice, &avx_slice);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double time_ms = duration.count() / 1000.0;
        
        // Print timing results
        std::cout << "\nPerformance Results:\n"
                  << "==================\n";
        printTimings("Total kernel execution", time_ms, ITERATIONS);
        
        // Verify a few results
        std::cout << "\nVerifying results (first few elements):\n";
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