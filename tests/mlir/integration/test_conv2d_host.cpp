#include <iostream>
#include <dlfcn.h>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_conv2d.so>" << std::endl;
        return 1;
    }

    // Load the shared library
    void* handle = dlopen(argv[1], RTLD_LAZY);
    if (!handle) {
        std::cerr << "Error loading library: " << dlerror() << std::endl;
        return 1;
    }

    // Load the test_conv2d_simple function
    using TestFunc = float(*)();
    auto test_conv2d = (TestFunc)dlsym(handle, "test_conv2d_simple");
    if (!test_conv2d) {
        std::cerr << "Error finding symbol: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }

    // Run the test
    std::cout << "Running Conv2D test..." << std::endl;
    float result = test_conv2d();
    std::cout << "Result (first output element): " << result << std::endl;

    // Expected: With input 0-15, weights all 1.0, bias 0.0,
    // First 3x3 window (top-left of 4x4 input) sums:
    // 0 + 1 + 2 + 4 + 5 + 6 + 8 + 9 + 10 = 45.0
    std::cout << "Expected: 45.0" << std::endl;

    if (result == 45.0f) {
        std::cout << "✓ Conv2D test PASSED!" << std::endl;
    } else {
        std::cout << "✗ Conv2D test FAILED!" << std::endl;
        std::cout << "Difference: " << (result - 45.0f) << std::endl;
    }

    dlclose(handle);
    return (result == 45.0f) ? 0 : 1;
}
