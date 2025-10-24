// Host runner for dtype tests
#include <iostream>
#include <dlfcn.h>
#include <cmath>

typedef double (*TestFunc)();

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <kernel.so>" << std::endl;
        return 1;
    }

    // Load compiled kernel
    void* handle = dlopen(argv[1], RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load kernel: " << dlerror() << std::endl;
        return 1;
    }

    auto kernel = (TestFunc)dlsym(handle, "kernel_main");
    if (!kernel) {
        std::cerr << "Failed to find function: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }

    // Call kernel
    double result = kernel();

    // Verify result (approximately)
    double expected = 10823.858;  // 3.14 + 2.718 + 42 + 9999 + 777
    bool passed = std::abs(result - expected) < 1.0;  // Allow small tolerance

    std::cout << "Dtype Support Test" << std::endl;
    std::cout << "  Expected: ~" << expected << std::endl;
    std::cout << "  Got:      " << result << std::endl;
    std::cout << "  Status:   " << (passed ? "PASSED ✓" : "FAILED ✗") << std::endl;

    dlclose(handle);
    return passed ? 0 : 1;
}
