#include "kernel_runner.hpp"
#include "simd_types.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <immintrin.h>

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
        const size_t num_vectors = 4;  // Space for all operations (ADD, SUB, MUL, DIV)
        const size_t sse_elements = num_vectors * SSESlice::VECTOR_SIZE;
        const size_t avx_elements = num_vectors * AVXSlice::VECTOR_SIZE;
        
        double* sse_data = (double*)aligned_alloc(16, sse_elements * sizeof(double));
        double* avx_data = (double*)aligned_alloc(32, avx_elements * sizeof(double));

        if (!sse_data || !avx_data) {
            throw std::runtime_error("Failed to allocate aligned memory");
        }

        // Initialize slices
        SSESlice sse_slice = {(float*)sse_data, num_vectors};
        AVXSlice avx_slice = {(float*)avx_data, num_vectors};

        // Run kernel
        runner.loadLibrary(argv[1]);
        double result = runner.runKernel(&sse_slice, &avx_slice);
        
        // Print results
        std::cout << "\nSSE Results (128-bit, 2 doubles):\n";
        print_vector("ADD", &sse_data[0], 2);
        print_vector("SUB", &sse_data[2], 2);
        print_vector("MUL", &sse_data[4], 2);
        print_vector("DIV", &sse_data[6], 2);

        std::cout << "\nAVX Results (512-bit, 8 doubles):\n";
        print_vector("ADD", &avx_data[0], 8);
        print_vector("SUB", &avx_data[8], 8);
        print_vector("MUL", &avx_data[16], 8);
        print_vector("DIV", &avx_data[24], 8);

        // Cleanup
        free(sse_data);
        free(avx_data);
        
        return result == 1.0 ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 