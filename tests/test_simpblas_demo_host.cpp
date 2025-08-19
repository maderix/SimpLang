/**
 * @file test_simpblas_demo_host.cpp
 * @brief Host program that initializes simpblas and runs SimpLang kernel
 */

#include <iostream>
#include <dlfcn.h>
#include <chrono>
#include <iomanip>
#include "kernel_runner.hpp"
#include "simpblas.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <kernel.so>" << std::endl;
        return 1;
    }

    std::cout << "=== SimpLang + SimpBLAS Integration Test ===" << std::endl;
    
    // Initialize simpblas first
    std::cout << "Initializing simpblas..." << std::endl;
    if (sb_init() != 0) {
        std::cerr << "Failed to initialize simpblas!" << std::endl;
        return 1;
    }
    
    std::cout << "SimpBLAS initialized successfully" << std::endl;
    std::cout << "Version: " << sb_get_version() << std::endl;
    std::cout << "Kernels: " << sb_get_kernel_info() << std::endl;
    
    // Load and run the SimpLang kernel
    std::string kernelPath = argv[1];
    std::cout << "\nLoading SimpLang kernel: " << kernelPath << std::endl;
    
    try {
        KernelRunner runner;
        runner.loadLibrary(kernelPath);
        
        auto start = std::chrono::high_resolution_clock::now();
        double result = runner.runKernel();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Kernel executed successfully" << std::endl;
        std::cout << "Result: " << std::fixed << std::setprecision(1) << result << std::endl;
        std::cout << "Execution time: " << duration.count() << " μs" << std::endl;
        
        // Verify expected result
        if (result == 15.0) {
            std::cout << "✓ Test PASSED! SimpLang + SimpBLAS integration working correctly." << std::endl;
            return 0;
        } else {
            std::cout << "✗ Test FAILED! Expected 15.0, got " << result << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error running kernel: " << e.what() << std::endl;
        return 1;
    }
}