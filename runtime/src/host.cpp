#include "kernel.h"
#include "kernel_debugger/debugger.hpp"
#include <iostream>
#include <stdexcept>
#include <memory>
#include <dlfcn.h>  // For dynamic linking debug info

class SliceRAII {
    void* slice;
public:
    explicit SliceRAII(void* s) : slice(s) {}
    ~SliceRAII() { if (slice) free_slice(slice); }
    
    void* get() const { return slice; }
    void* release() { void* tmp = slice; slice = nullptr; return tmp; }
    
    SliceRAII(const SliceRAII&) = delete;
    SliceRAII& operator=(const SliceRAII&) = delete;
};

class KernelRunner {
    static constexpr size_t SLICE_SIZE = 4;  // Increased to store reference results
    
    SliceRAII sse_slice;
    SliceRAII avx_slice;
    KernelDebugger* debugger;
    bool is_simd_test;
    bool debug_mode;
    
public:
    KernelRunner() 
        : sse_slice(make_sse_slice(SLICE_SIZE))
        , avx_slice(make_avx_slice(SLICE_SIZE))
        , debugger(KernelDebugger::getInstance())
        , is_simd_test(false)
        , debug_mode(true)  // Set to true for SIMD debugging
    {
        if (!sse_slice.get() || !avx_slice.get()) {
            throw std::runtime_error("Failed to allocate slices");
        }

        debugger->start();
        debugger->setMode(KernelDebugger::Mode::STEP);
    }

    ~KernelRunner() {
        debugger->stop();
    }
    
    double run() {
        is_simd_test = true;
        double result = 0.0;
        
        try {
            std::cout << "Starting kernel execution..." << std::endl;

            // Try to locate kernel_main symbol
            #if !defined(TEST_SIMD)
                using KernelMainFunc = double(*)();
                KernelMainFunc kernel_main_ptr = reinterpret_cast<KernelMainFunc>(dlsym(RTLD_DEFAULT, "kernel_main"));
                if (!kernel_main_ptr) {
                    std::cerr << "Failed to find kernel_main: " << dlerror() << std::endl;
                    throw std::runtime_error("kernel_main not found");
                }
                std::cout << "Found kernel_main at " << (void*)kernel_main_ptr << std::endl;
            #endif

            // Convert void* to proper slice types
            auto sse = static_cast<sse_slice_t*>(sse_slice.get());
            auto avx = static_cast<avx_slice_t*>(avx_slice.get());

            // Call the appropriate kernel function based on test type
            #ifdef TEST_SIMD
                std::cout << "Calling SIMD kernel_main..." << std::endl;
                kernel_main(sse, avx);  // Special SIMD test
            #else
                std::cout << "Calling arithmetic kernel_main..." << std::endl;
                result = kernel_main();  // Regular arithmetic test
            #endif
            
            if (debug_mode) {
                std::cout << "Kernel execution completed successfully\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "Kernel execution failed: " << e.what() << "\n";
            throw;
        }

        return result;
    }
    
    void print_results() const {
        if (!is_simd_test) return;

        std::cout << "=== Test Results ===\n\n";
        
        // Print SSE Tests (4-wide vectors)
        std::cout << "SSE Tests (4-wide vectors):\n";
        std::cout << "----------------------------\n";
        auto sse = static_cast<sse_slice_t*>(sse_slice.get());
        
        std::cout << "1. Addition Test:\n";
        std::cout << "   Input1: [1.0, 2.0, 3.0, 4.0]\n";
        std::cout << "   Input2: [5.0, 6.0, 7.0, 8.0]\n";
        std::cout << "   SIMD Result:     ";
        print_sse_vector(sse->data[0]);
        std::cout << "   Expected Result: ";
        print_sse_vector(sse->data[2]);
        std::cout << "\n";
        
        std::cout << "2. Multiplication Test:\n";
        std::cout << "   Input1: [2.0, 3.0, 4.0, 5.0]\n";
        std::cout << "   Input2: [3.0, 4.0, 5.0, 6.0]\n";
        std::cout << "   SIMD Result:     ";
        print_sse_vector(sse->data[1]);
        std::cout << "   Expected Result: ";
        print_sse_vector(sse->data[3]);
        std::cout << "\n";
        
        // Print AVX Tests (8-wide vectors)
        std::cout << "\nAVX Tests (8-wide vectors):\n";
        std::cout << "----------------------------\n";
        auto avx = static_cast<avx_slice_t*>(avx_slice.get());
        
        std::cout << "1. Addition Test:\n";
        std::cout << "   Input1: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]\n";
        std::cout << "   Input2: [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]\n";
        std::cout << "   SIMD Result:     ";
        print_avx_vector(avx->data[0]);
        std::cout << "   Expected Result: ";
        print_avx_vector(avx->data[2]);
        std::cout << "\n";
        
        std::cout << "2. Multiplication Test:\n";
        std::cout << "   Input1: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]\n";
        std::cout << "   Input2: [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]\n";
        std::cout << "   SIMD Result:     ";
        print_avx_vector(avx->data[1]);
        std::cout << "   Expected Result: ";
        print_avx_vector(avx->data[3]);
        std::cout << "\n";
    }
};

// Host entry point called by SimpleLang main
double host_main() {
    try {
        KernelRunner runner;
        double result = runner.run();
        runner.print_results();
        return result;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1.0;
    }
}

// Program entry point
int main() {
    double result = host_main();

    #if defined(EXPECTED_RESULT)
        // Get expected result from CMake definition
        const double expected = EXPECTED_RESULT;
        const double tolerance = 0.01;  // 1% tolerance for floating point comparison
        
        // Compare with tolerance
        if (std::abs(result - expected) <= tolerance * std::abs(expected)) {
            std::cout << "Test passed with result: " << result << std::endl;
            return 0;
        } else {
            std::cerr << "Test failed: Expected " << expected 
                      << " but got " << result << std::endl;
            return 1;
        }
    #else
        // For SIMD tests or when no expected result is defined
        return static_cast<int>(result);
    #endif
}