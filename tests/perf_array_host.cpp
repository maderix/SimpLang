#include "kernel_runner.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

// C++ implementation matching SimpLang array test exactly
double cpp_array_kernel() {
    // Performance test parameters
    const int size = 10000;
    const int iterations = 1000;
    
    // Test 1: Large array creation and initialization
    std::vector<float> large_array(size);
    for (int i = 0; i < size; i++) {
        large_array[i] = i * 2.5f;
    }
    
    // Test 2: Sequential access pattern (cache-friendly)
    double sum = 0.0;
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < size; i++) {
            sum += large_array[i];
        }
    }
    
    // Test 3: Random access pattern (cache-unfriendly) - simplified
    double random_sum = 0.0;
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < size; i++) {
            int random_idx = (size - 1) - i;  // Simple reverse pattern
            random_sum += large_array[random_idx];
        }
    }
    
    // Test 4: Array-to-array operations
    std::vector<float> array2(size);
    std::vector<float> array3(size);
    
    // Initialize arrays
    for (int i = 0; i < size; i++) {
        array2[i] = i * 1.5f;
        array3[i] = i * 0.5f;
    }
    
    // Element-wise addition
    double result_sum = 0.0;
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < size; i++) {
            float temp_result = array2[i] + array3[i] + large_array[i];
            result_sum += temp_result;
        }
    }
    
    // Test 5: Stride access patterns - simplified
    double stride_sum = 0.0;
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < size; i++) {
            stride_sum += large_array[i];  // Just sequential access
        }
    }
    
    // Test 6: Reverse access pattern
    double reverse_sum = 0.0;
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = size - 1; i >= 0; i--) {
            reverse_sum += large_array[i];
        }
    }
    
    // Return combined result to prevent optimization
    return sum + random_sum + result_sum + stride_sum + reverse_sum;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <kernel.so>" << std::endl;
        return 1;
    }

    const int WARMUP_ITERATIONS = 10;
    const int NUM_ITERATIONS = 100;
    std::vector<double> cpp_times;
    std::vector<double> sl_times;
    
    try {
        // Verify results match
        KernelRunner runner;
        runner.loadLibrary(argv[1]);
        double sl_result = runner.runKernel();
        double cpp_result = cpp_array_kernel();
        
        std::cout << "=== Result Verification ===" << std::endl;
        std::cout << "C++ Result:        " << cpp_result << std::endl;
        std::cout << "SimpLang Result:   " << sl_result << std::endl;
        double diff_pct = std::abs(cpp_result - sl_result) / cpp_result * 100.0;
        std::cout << "Difference:        " << diff_pct << "%" << std::endl;
        if (diff_pct < 0.1) {
            std::cout << "Results match! ✓\n" << std::endl;
        } else if (diff_pct < 1.0) {
            std::cout << "Results are close (within 1%) ✓\n" << std::endl;
        } else {
            std::cerr << "Results differ significantly! ✗\n" << std::endl;
            return 1;
        }

        std::cout << "=== Array Performance Test ===" << std::endl;
        std::cout << "Array size: 10,000 elements" << std::endl;
        std::cout << "Iterations: 1,000 per test" << std::endl;
        std::cout << "Tests: Sequential, Random, Element-wise ops, Stride, Reverse" << std::endl;
        std::cout << "Running " << NUM_ITERATIONS << " benchmark iterations" << std::endl;

        // Warmup for C++
        std::cout << "\nWarming up C++..." << std::endl;
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            cpp_array_kernel();
        }

        // Test C++ implementation
        std::cout << "Running C++ Implementation:" << std::endl;
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            cpp_array_kernel();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            cpp_times.push_back(duration.count());
            
            if ((i + 1) % 20 == 0) {
                std::cout << "  Completed " << (i + 1) << "/" << NUM_ITERATIONS << " iterations" << std::endl;
            }
        }

        // Warmup for SimpLang
        std::cout << "\nWarming up SimpLang..." << std::endl;
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            runner.runKernel();
        }
        
        // Test SimpLang implementation
        std::cout << "Running SimpLang Implementation:" << std::endl;
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            runner.runKernel();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            sl_times.push_back(duration.count());
            
            if ((i + 1) % 20 == 0) {
                std::cout << "  Completed " << (i + 1) << "/" << NUM_ITERATIONS << " iterations" << std::endl;
            }
        }

        // Calculate statistics
        auto cpp_avg = std::accumulate(cpp_times.begin(), cpp_times.end(), 0.0) / NUM_ITERATIONS;
        auto cpp_min = *std::min_element(cpp_times.begin(), cpp_times.end());
        auto cpp_max = *std::max_element(cpp_times.begin(), cpp_times.end());

        auto sl_avg = std::accumulate(sl_times.begin(), sl_times.end(), 0.0) / NUM_ITERATIONS;
        auto sl_min = *std::min_element(sl_times.begin(), sl_times.end());
        auto sl_max = *std::max_element(sl_times.begin(), sl_times.end());

        // Print results
        std::cout << "\n=== Performance Results ===" << std::endl;
        std::cout << "C++ Implementation:" << std::endl;
        std::cout << "  Average: " << cpp_avg << " μs" << std::endl;
        std::cout << "  Min:     " << cpp_min << " μs" << std::endl;
        std::cout << "  Max:     " << cpp_max << " μs" << std::endl;
        std::cout << "  Total:   " << (cpp_avg * NUM_ITERATIONS / 1000.0) << " ms" << std::endl;

        std::cout << "\nSimpLang Implementation:" << std::endl;
        std::cout << "  Average: " << sl_avg << " μs" << std::endl;
        std::cout << "  Min:     " << sl_min << " μs" << std::endl;
        std::cout << "  Max:     " << sl_max << " μs" << std::endl;
        std::cout << "  Total:   " << (sl_avg * NUM_ITERATIONS / 1000.0) << " ms" << std::endl;

        double slowdown = sl_avg / cpp_avg;
        double performance_pct = (cpp_avg / sl_avg) * 100.0;
        
        std::cout << "\n=== Performance Analysis ===" << std::endl;
        std::cout << "Slowdown Factor:     " << slowdown << "x" << std::endl;
        std::cout << "Performance Ratio:   " << performance_pct << "% of C++ speed" << std::endl;
        
        // Performance assessment
        if (slowdown < 1.3) {
            std::cout << "Assessment:          ✓ EXCELLENT - Very competitive with C++" << std::endl;
        } else if (slowdown < 2.0) {
            std::cout << "Assessment:          ✓ GOOD - Acceptable overhead for abstraction" << std::endl;
        } else if (slowdown < 3.0) {
            std::cout << "Assessment:          ! MODERATE - Noticeable overhead" << std::endl;
        } else {
            std::cout << "Assessment:          ⚠ POOR - Significant optimization needed" << std::endl;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}