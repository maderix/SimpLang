#include "kernel_runner.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <kernel.so>" << std::endl;
        return 1;
    }

    KernelRunner runner;
    
    try {
        runner.loadLibrary(argv[1]);
        double result = runner.runKernel();
        
        std::cout << "Result: " << result << std::endl;
        
        // Verify the result is 72.0
        bool passed = (result == 72.0);
        std::cout << "Test " << (passed ? "PASSED! ✓" : "FAILED! ✗") << std::endl;
        
        return passed ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 