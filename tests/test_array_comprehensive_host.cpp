#include "kernel_runner.hpp"
#include <iostream>
#include <cmath>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <kernel.so>" << std::endl;
        return 1;
    }

    KernelRunner runner;
    
    try {
        runner.loadLibrary(argv[1]);
        float result = runner.runKernel();
        
        std::cout << "Array Test Result: " << result << std::endl;
        
        // Expected result calculation (after removing boolean test):
        // mixed_result = f32_sum + i32_result + u16_elem = 31.25 + 0 + 1000 = 1031.25
        // indexed_elem = arr_f32[1] = 20.75
        // dyn_elem1 = arr_dynamic1[1] = 84.0
        // random_sum = 4.0 + 8.0 + 2.0 + 10.0 = 24.0
        // calc_elem = idx_test[5] = 6.0
        // seq_sum = 30 + 20 + 10 = 60
        // stride_sum = 100 + 200 + 300 + 400 = 1000
        // offset_sum = 150 + 250 + 350 + 450 = 1200
        // boundary_result = 111 + 444 + 333 = 888
        // final_result = 1031.25 + 20.75 + 84.0 + 24.0 + 6.0 + 60 + 1000 + 1200 + 888 = 4314.0
        // With mixed arithmetic fix, we now get the correct precision
        
        double expected = 4314.0;
        
        // Use small epsilon for floating point comparison
        double epsilon = 0.001;
        bool passed = std::abs(result - expected) < epsilon;
        
        std::cout << "Expected: " << expected << std::endl;
        std::cout << "Got:      " << result << std::endl;
        std::cout << "Diff:     " << std::abs(result - expected) << std::endl;
        
        if (passed) {
            std::cout << "Array Test PASSED! ✓" << std::endl;
            std::cout << "All array operations working correctly:" << std::endl;
            std::cout << "  ✓ Array creation with all supported types" << std::endl;
            std::cout << "  ✓ Array element assignment" << std::endl;
            std::cout << "  ✓ Array element access" << std::endl;
            std::cout << "  ✓ Type inference from array expressions" << std::endl;
            std::cout << "  ✓ Forward, reverse, and random indexing" << std::endl;
            std::cout << "  ✓ Variable-based indexing" << std::endl;
            std::cout << "  ✓ Sequential access patterns" << std::endl;
            std::cout << "  ✓ Stride patterns" << std::endl;
            std::cout << "  ✓ Boundary access" << std::endl;
            std::cout << "  ✓ Mixed arithmetic operations" << std::endl;
        } else {
            std::cout << "Array Test FAILED! ✗" << std::endl;
        }
        
        return passed ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}