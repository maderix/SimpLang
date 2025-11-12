#include <iostream>
#include <dlfcn.h>
#include <cstdlib>

// Function signature for test_tensor_basic
typedef float (*TestFunc)();

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path-to-shared-lib>" << std::endl;
        return 1;
    }

    // Load the shared library
    void* handle = dlopen(argv[1], RTLD_LAZY);
    if (!handle) {
        std::cerr << "Error: Cannot load library: " << dlerror() << std::endl;
        return 1;
    }

    // Get the function
    TestFunc test_tensor_basic = (TestFunc)dlsym(handle, "test_tensor_basic");
    if (!test_tensor_basic) {
        std::cerr << "Error: Cannot find function test_tensor_basic: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }

    // Call the function
    float result = test_tensor_basic();

    std::cout << "Result: " << result << std::endl;

    // Verify result
    if (result == 5.0f) {
        std::cout << "✓ Test PASSED: tensor[0,1] correctly returns 5.0" << std::endl;
    } else {
        std::cerr << "✗ Test FAILED: expected 5.0, got " << result << std::endl;
        dlclose(handle);
        return 1;
    }

    dlclose(handle);
    return 0;
}
