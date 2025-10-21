#include <iostream>
#include <dlfcn.h>

// Function pointer type for kernel_main
typedef float (*KernelMainFunc)();

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <shared_object.so>\n";
        return 1;
    }

    // Load the shared object
    void* handle = dlopen(argv[1], RTLD_LAZY);
    if (!handle) {
        std::cerr << "Error loading library: " << dlerror() << "\n";
        return 1;
    }

    // Get the main function
    KernelMainFunc kernel_main = (KernelMainFunc)dlsym(handle, "main");
    if (!kernel_main) {
        std::cerr << "Error finding main function: " << dlerror() << "\n";
        dlclose(handle);
        return 1;
    }

    // Execute and get result
    float result = kernel_main();

    std::cout << "Result: " << result << "\n";

    // Expected: 19.0 (first element of result matrix)
    // C = A × B where A = [[1,2],[3,4]] and B = [[5,6],[7,8]]
    // C[0,0] = 1*5 + 2*7 = 5 + 14 = 19
    float expected = 19.0f;

    if (std::abs(result - expected) < 0.001f) {
        std::cout << "✓ PASS: Result matches expected value " << expected << "\n";
        dlclose(handle);
        return 0;
    } else {
        std::cout << "✗ FAIL: Expected " << expected << " but got " << result << "\n";
        dlclose(handle);
        return 1;
    }
}
