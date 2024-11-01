#include "kernel.h"
#include <iostream>
#include <stdexcept>
#include <memory>

class SliceRAII {
    void* slice;
public:
    explicit SliceRAII(void* s) : slice(s) {}
    ~SliceRAII() { if (slice) free_slice(slice); }
    void* get() { return slice; }
    
    SliceRAII(const SliceRAII&) = delete;
    SliceRAII& operator=(const SliceRAII&) = delete;
};

class KernelRunner {
    static constexpr size_t SLICE_SIZE = 2;  // Space for results
    
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
        std::cout << "SSE Results:\n";
        print_sse_slice(static_cast<sse_slice_t*>(sse_slice.get()));
        
        std::cout << "\nAVX Results:\n";
        print_avx_slice(static_cast<avx_slice_t*>(avx_slice.get()));
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