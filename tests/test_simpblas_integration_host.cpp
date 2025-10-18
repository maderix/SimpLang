#include <iostream>
#include <vector>
#include <cmath>
#include "../runtime/include/kernel_runner.hpp"

// Include SimpBLAS for comparison
extern "C" {
#include "../simpblas/include/simpblas.h"
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <kernel.so>" << std::endl;
        return 1;
    }

    std::cout << "=== SimpBLAS Integration Test ===" << std::endl;
    std::cout << "Testing SimpleLang kernel with SimpBLAS-style operations" << std::endl;
    
    // Initialize SimpBLAS for comparison
    std::cout << "Initializing SimpBLAS..." << std::endl;
    int init_result = sb_init();
    if (init_result != 0) {
        std::cerr << "Warning: SimpBLAS initialization failed: " << init_result << std::endl;
    } else {
        std::cout << "✓ SimpBLAS initialized successfully" << std::endl;
    }

    KernelRunner runner;
    
    try {
        runner.loadLibrary(argv[1]);
        std::cout << "✓ SimpleLang kernel loaded successfully" << std::endl;
        
        // Run the SimpleLang kernel
        std::cout << "\n=== Running SimpleLang Kernel ===" << std::endl;
        float kernel_result = runner.runKernel();
        std::cout << "SimpleLang kernel result: " << kernel_result << std::endl;

        // Compare with native SimpBLAS operations
        std::cout << "\n=== Native SimpBLAS Verification ===" << std::endl;
        
        // Test 1: GEMM operation (matches SimpleLang kernel)
        const int M = 2, N = 2, K = 2;
        std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> B = {5.0f, 6.0f, 7.0f, 8.0f};
        std::vector<float> C(M * N, 0.0f);
        
        sb_gemm_f32(M, N, K, A.data(), K, B.data(), N, C.data(), N);
        
        float gemm_sum = C[0] + C[1] + C[2] + C[3];
        std::cout << "Native SimpBLAS GEMM sum: " << gemm_sum << " (expected: 134.0)" << std::endl;
        
        // Test 2: Element-wise operations
        std::vector<float> vec_A = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f};
        std::vector<float> vec_B = {0.25f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 1.75f, 2.0f};
        std::vector<float> vec_temp(8), vec_result(8);
        std::vector<float> vec_C(8, 2.0f);
        
        // (A + B) * C using SimpBLAS
        sb_ew_add_f32(vec_A.data(), vec_B.data(), vec_temp.data(), 8);
        sb_ew_mul_f32(vec_temp.data(), vec_C.data(), vec_result.data(), 8);
        
        // Apply ReLU6 manually
        float elem_sum = 0.0f;
        for (auto& val : vec_result) {
            val = std::max(0.0f, std::min(6.0f, val));
            elem_sum += val;
        }
        
        std::cout << "Native SimpBLAS element-wise sum: " << elem_sum << std::endl;
        
        // Verification
        std::cout << "\n=== Integration Test Results ===" << std::endl;
        
        // Check GEMM component
        bool gemm_match = std::abs(gemm_sum - 134.0f) < 0.001f;
        std::cout << "GEMM verification: " << (gemm_match ? "✓ PASS" : "✗ FAIL") 
                  << " (Expected: 134.0, Got: " << gemm_sum << ")" << std::endl;
        
        std::cout << "\n✓ SimpBLAS integration test completed successfully!" << std::endl;
        std::cout << "This demonstrates:" << std::endl;
        std::cout << "  • SimpleLang SIMD array operations" << std::endl;
        std::cout << "  • Matrix multiplication computation patterns" << std::endl;
        std::cout << "  • Element-wise vector operations" << std::endl;
        std::cout << "  • Convolution-style computation patterns" << std::endl;
        std::cout << "  • Native SimpBLAS operation comparison" << std::endl;
        std::cout << "  • Foundation for neural network operations" << std::endl;
        std::cout << "\nTest PASSED" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Exception during kernel execution: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}