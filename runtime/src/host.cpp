#include "kernel.h"
#include "kernel_debugger/debugger.hpp"
#include <iostream>
#include <stdexcept>
#include <memory>
#include <dlfcn.h>
#include <iomanip>
#include <cmath>

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
    static constexpr size_t SLICE_SIZE = 2;  // Changed to match test size
    
    SliceRAII sse_slice;
    SliceRAII avx_slice;
    KernelDebugger& debugger;
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

        debugger.start();
        debugger.setMode(KernelDebugger::Mode::STEP);
    }

    ~KernelRunner() = default;
    
    double run() {
        #if defined(TEST_SIMD)
            is_simd_test = true;
        #else
            is_simd_test = false;
        #endif
        double result = 0.0;
        
        try {
            std::cout << "Starting kernel execution..." << std::endl;

            #if defined(TEST_SIMD)
                std::cout << "Calling SIMD kernel_main..." << std::endl;
                auto* sse = static_cast<sse_slice_t*>(sse_slice.get());
                auto* avx = static_cast<avx_slice_t*>(avx_slice.get());
                kernel_main(sse, avx);  // Special SIMD test
                result = 1.0;  // Return 1.0 for successful SIMD test
            #else
                using KernelMainFunc = double(*)();
                KernelMainFunc kernel_main_ptr = reinterpret_cast<KernelMainFunc>(dlsym(RTLD_DEFAULT, "kernel_main"));
                if (!kernel_main_ptr) {
                    std::cerr << "Failed to find kernel_main: " << dlerror() << std::endl;
                    throw std::runtime_error("kernel_main not found");
                }
                std::cout << "Found kernel_main at " << (void*)kernel_main_ptr << std::endl;
                result = kernel_main_ptr();
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

        std::cout << "\n=== Detailed Test Results ===\n\n";
        
        /* Comment out SSE tests
        // Print SSE Tests
        std::cout << "SSE Tests:\n";
        std::cout << "----------------------------\n";
        auto* sse = static_cast<sse_slice_t*>(sse_slice.get());
        std::cout << "SSE Slice Info:\n";
        std::cout << "  Length: " << sse->len << "\n";
        std::cout << "  Capacity: " << sse->cap << "\n";
        std::cout << "  Data pointer: " << sse->data << "\n\n";
        
        std::cout << "SSE Vector Contents:\n";
        for (size_t i = 0; i < SLICE_SIZE; i++) {
            std::cout << "  Vector " << i << " (at " << &sse->data[i] << "): ";
            print_sse_vector(sse->data[i]);
        }
        std::cout << "\n";
        */
        
        // Print AVX Tests
        std::cout << "\nAVX Tests:\n";
        std::cout << "----------------------------\n";
        auto* avx = static_cast<avx_slice_t*>(avx_slice.get());
        std::cout << "AVX Slice Info:\n";
        std::cout << "  Length: " << avx->len << "\n";
        std::cout << "  Capacity: " << avx->cap << "\n";
        std::cout << "  Data pointer: " << avx->data << "\n\n";
        
        std::cout << "AVX Vector Contents:\n";
        for (size_t i = 0; i < SLICE_SIZE; i++) {
            std::cout << "  Vector " << i << " (at " << &avx->data[i] << "): ";
            print_avx_vector(avx->data[i]);
            
            // Add raw memory dump
            alignas(64) double values[8];
            _mm512_store_pd(values, avx->data[i]);
            std::cout << "    Raw values: ";
            for (int j = 0; j < 8; j++) {
                std::cout << values[j] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
};

double host_main() {
    try {
        KernelRunner runner;
        double result = runner.run();
        
        // Print test results
        std::cout << "\nKernel returned: " << std::fixed << std::setprecision(1) << result << std::endl;
        
        // Expected results:
        // sum_to_n(5.0) = 15.0 (1+2+3+4+5)
        // factorial(5.0) = 120.0 (5*4*3*2*1)
        // Total should be 135.0
        const double expected = 135.0;
        const double epsilon = 0.0001;
        
        if (std::abs(result - expected) < epsilon) {
            std::cout << "Test PASSED! ✓" << std::endl;
        } else {
            std::cout << "Test FAILED! ✗" << std::endl;
            std::cout << "Expected: " << expected << std::endl;
            std::cout << "Got: " << result << std::endl;
        }
        
        runner.print_results();
        return result;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1.0;
    }
}

int main() {
    double result = host_main();
    return static_cast<int>(result);
}