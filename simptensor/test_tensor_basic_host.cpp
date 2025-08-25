#include <iostream>
#include <dlfcn.h>
#include <cmath>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <shared_object_file>" << std::endl;
        return 1;
    }

    // Load the shared library
    void* handle = dlopen(argv[1], RTLD_LAZY);
    if (!handle) {
        std::cerr << "Error loading library: " << dlerror() << std::endl;
        return 1;
    }

    // Load the kernel_main function
    typedef float (*kernel_func_t)();
    kernel_func_t kernel_main = (kernel_func_t) dlsym(handle, "kernel_main");
    
    if (!kernel_main) {
        std::cerr << "Error loading kernel_main: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }

    try {
        // Run the kernel
        float result = kernel_main();
        std::cout << "Tensor test result: " << result << std::endl;
        
        // Expected result should be 28.0 (0 + 1 + 3 + 24)
        const float expected = 28.0f;
        if (std::abs(result - expected) < 0.001f) {
            std::cout << "✓ NHWC stride calculations correct!" << std::endl;
            std::cout << "  Tensor creation and basic operations working" << std::endl;
            std::cout << "  SIMD array backend integration successful" << std::endl;
        } else {
            std::cout << "✗ Unexpected result (expected " << expected << ", got " << result << ")" << std::endl;
            dlclose(handle);
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception during kernel execution: " << e.what() << std::endl;
        dlclose(handle);
        return 1;
    }

    dlclose(handle);
    return 0;
}