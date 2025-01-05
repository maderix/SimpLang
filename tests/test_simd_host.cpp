#include "kernel_runner.hpp"
#include "simd_types.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <immintrin.h>

// Global variables
sse_vector_t* sse_data = nullptr;
avx_vector_t* avx_data = nullptr;

// Number of vectors in each slice
#define NUM_SSE_VECTORS 4  // Number of SSE vectors (each vector is 2 doubles)
#define NUM_AVX_VECTORS 4  // Number of AVX vectors (each vector is 8 doubles)

// Vector sizes (for reference)
#define SSE_VECTOR_SIZE 2  // Each SSE vector holds 2 doubles
#define AVX_VECTOR_SIZE 8  // Each AVX vector holds 8 doubles

void print_vector(const char* label, double* data, size_t elements) {
    std::cout << label << ": [";
    for (size_t i = 0; i < elements; i++) {
        std::cout << std::fixed << std::setprecision(2) << data[i];
        if (i < elements - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

// Allocate memory for test data
void allocateTestData() {
    // Allocate SSE vectors
    size_t sse_size = NUM_SSE_VECTORS * sizeof(sse_vector_t);
    sse_data = (sse_vector_t*)aligned_alloc(16, sse_size);
    
    // Allocate AVX vectors
    size_t avx_size = NUM_AVX_VECTORS * sizeof(avx_vector_t);
    avx_data = (avx_vector_t*)aligned_alloc(64, avx_size);

    // Initialize test data (using vector types)
    for (size_t i = 0; i < NUM_SSE_VECTORS; i++) {
        sse_data[i] = (sse_vector_t){static_cast<double>(i*2 + 1),
                                    static_cast<double>(i*2 + 2)};
    }
    
    for (size_t i = 0; i < NUM_AVX_VECTORS; i++) {
        avx_data[i] = (avx_vector_t){
            static_cast<double>(i*8 + 1),
            static_cast<double>(i*8 + 2),
            static_cast<double>(i*8 + 3),
            static_cast<double>(i*8 + 4),
            static_cast<double>(i*8 + 5),
            static_cast<double>(i*8 + 6),
            static_cast<double>(i*8 + 7),
            static_cast<double>(i*8 + 8)
        };
    }
}

// Clean up function
void cleanupTestData() {
    if (sse_data) {
        free(sse_data);
        sse_data = nullptr;
    }
    if (avx_data) {
        free(avx_data);
        avx_data = nullptr;
    }
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

        // Cast the raw double pointers to the appropriate SIMD vector types
        sse_vector_t* sse_vec_data = reinterpret_cast<sse_vector_t*>(sse_data);
        avx_vector_t* avx_vec_data = reinterpret_cast<avx_vector_t*>(avx_data);

        // Initialize slices
        SSESlice sse_slice = {sse_vec_data, num_vectors, num_vectors};
        AVXSlice avx_slice = {avx_vec_data, num_vectors, num_vectors};

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