#include "kernel_runner.hpp"
#include <iostream>

double host_main() {
    try {
        KernelRunner runner;
        runner.loadLibrary("./test_fibonacci.so");
        return runner.runKernel();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1.0;
    }
}

int main() {
    double result = host_main();
    
    std::cout << "Result: " << result << std::endl;
    
    // Verify the result is 110.0 (fibonacci(10))
    bool passed = (result == 110.0);
    std::cout << "Test " << (passed ? "PASSED! ✓" : "FAILED! ✗") << std::endl;
    
    return passed ? 0 : 1;
} 