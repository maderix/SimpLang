#include "kernel_runner.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

// C++ implementation matching SimpleLang exactly
double bounded_sum(double n) {
    double sum = 0.0;
    double i = 1.0;
    
    while (i <= n) {
        sum = fmod(sum + i, 10000.0);
        i = i + 1.0;
    }
    return sum;
}

double bounded_multiply(double n) {
    double result = 1.0;
    double i = 1.0;
    
    while (i <= n) {
        result = fmod(result * i, 10000.0);
        i = i + 1.0;
    }
    return result;
}

double cpp_kernel() {
    double n = 100000.0;  // Increased to 100k iterations
    double sum_result = bounded_sum(n);
    double mul_result = bounded_multiply(n);
    return sum_result + mul_result;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <kernel.so>" << std::endl;
        return 1;
    }

    const int WARMUP_ITERATIONS = 100;
    const int NUM_ITERATIONS = 1000;
    std::vector<double> cpp_times;
    std::vector<double> sl_times;
    
    try {
        // Verify results match
        KernelRunner runner;
        runner.loadLibrary(argv[1]);
        double sl_result = runner.runKernel();
        double cpp_result = cpp_kernel();
        
        std::cout << "=== Result Verification ===" << std::endl;
        std::cout << "C++ Result:       " << cpp_result << std::endl;
        std::cout << "SimpleLang Result: " << sl_result << std::endl;
        if (std::abs(cpp_result - sl_result) < 1e-10) {
            std::cout << "Results match! ✓\n" << std::endl;
        } else {
            std::cerr << "Results don't match! ✗\n" << std::endl;
            return 1;
        }

        std::cout << "=== Performance Test ===" << std::endl;
        std::cout << "Running " << NUM_ITERATIONS << " iterations with N=100000" << std::endl;

        // Warmup for C++
        std::cout << "\nWarming up C++..." << std::endl;
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            cpp_kernel();
        }

        // Test C++ implementation
        std::cout << "Running C++ Implementation:" << std::endl;
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            cpp_kernel();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            cpp_times.push_back(duration.count());
        }

        // Warmup for SimpleLang
        std::cout << "\nWarming up SimpleLang..." << std::endl;
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            runner.runKernel();
        }
        
        // Test SimpleLang implementation
        std::cout << "Running SimpleLang Implementation:" << std::endl;
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            runner.runKernel();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            sl_times.push_back(duration.count());
        }

        // Calculate statistics
        auto cpp_avg = std::accumulate(cpp_times.begin(), cpp_times.end(), 0.0) / NUM_ITERATIONS;
        auto cpp_min = *std::min_element(cpp_times.begin(), cpp_times.end());
        auto cpp_max = *std::max_element(cpp_times.begin(), cpp_times.end());

        auto sl_avg = std::accumulate(sl_times.begin(), sl_times.end(), 0.0) / NUM_ITERATIONS;
        auto sl_min = *std::min_element(sl_times.begin(), sl_times.end());
        auto sl_max = *std::max_element(sl_times.begin(), sl_times.end());

        // Print results
        std::cout << "\nResults:" << std::endl;
        std::cout << "C++ Implementation:" << std::endl;
        std::cout << "  Average: " << cpp_avg << " microseconds" << std::endl;
        std::cout << "  Min: " << cpp_min << " microseconds" << std::endl;
        std::cout << "  Max: " << cpp_max << " microseconds" << std::endl;
        std::cout << "  Total: " << (cpp_avg * NUM_ITERATIONS / 1000.0) << " milliseconds" << std::endl;

        std::cout << "\nSimpleLang Implementation:" << std::endl;
        std::cout << "  Average: " << sl_avg << " microseconds" << std::endl;
        std::cout << "  Min: " << sl_min << " microseconds" << std::endl;
        std::cout << "  Max: " << sl_max << " microseconds" << std::endl;
        std::cout << "  Total: " << (sl_avg * NUM_ITERATIONS / 1000.0) << " milliseconds" << std::endl;

        std::cout << "\nPerformance Ratio (SimpleLang/C++): " << (sl_avg / cpp_avg) << "x" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}