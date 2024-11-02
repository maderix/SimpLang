#include "kernel.h"
#include <iostream>
#include <stdexcept>
#include <memory>

class SliceRAII {
    void* slice;
public:
    explicit SliceRAII(void* s) : slice(s) {}
    ~SliceRAII() { if (slice) free_slice(slice); }
    
    void* get() const { return slice; }
    
    SliceRAII(const SliceRAII&) = delete;
    SliceRAII& operator=(const SliceRAII&) = delete;
};

class KernelRunner {
    static constexpr size_t SLICE_SIZE = 4;  // Increased to store reference results
    
    SliceRAII sse_slice;
    SliceRAII avx_slice;
    
public:
    KernelRunner() 
        : sse_slice(make_sse_slice(SLICE_SIZE))
        , avx_slice(make_avx_slice(SLICE_SIZE))
    {
        if (!sse_slice.get() || !avx_slice.get()) {
            throw std::runtime_error("Failed to allocate slices");
        }
    }
    
    void run() {
        kernel_main(
            static_cast<sse_slice_t*>(sse_slice.get()),
            static_cast<avx_slice_t*>(avx_slice.get())
        );
    }
    
    void print_results() const {
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

int main() {
    try {
        KernelRunner runner;
        runner.run();
        runner.print_results();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}