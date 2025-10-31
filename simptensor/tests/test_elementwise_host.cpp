#include <iostream>
#include <dlfcn.h>
#include <cmath>
#include <cstdlib>

typedef float (*TestFunction)();

bool runTest(void* handle, const char* funcName, float expected) {
    TestFunction test = (TestFunction)dlsym(handle, funcName);
    if (!test) {
        std::cerr << "Failed to load function: " << funcName << std::endl;
        std::cerr << "Error: " << dlerror() << std::endl;
        return false;
    }

    float result = test();
    bool passed = std::abs(result - expected) < 0.001f;

    std::cout << funcName << ": ";
    if (passed) {
        std::cout << "PASS (result=" << result << ", expected=" << expected << ")" << std::endl;
    } else {
        std::cout << "FAIL (result=" << result << ", expected=" << expected << ")" << std::endl;
    }

    return passed;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <test_elementwise.so>" << std::endl;
        return 1;
    }

    void* handle = dlopen(argv[1], RTLD_NOW);
    if (!handle) {
        std::cerr << "Failed to load shared library: " << argv[1] << std::endl;
        std::cerr << "Error: " << dlerror() << std::endl;
        return 1;
    }

    std::cout << "=== Tensor Element-wise Operations Test ===" << std::endl;
    std::cout << std::endl;

    int passed = 0;
    int total = 0;

    // Test tensor addition: result[0,0] should be 11.0
    total++;
    if (runTest(handle, "test_tensor_add", 11.0f)) passed++;

    // Test tensor multiplication: result[0,1] should be 30.0
    total++;
    if (runTest(handle, "test_tensor_mul", 30.0f)) passed++;

    // Test tensor subtraction: result[1,0] should be 30.0
    total++;
    if (runTest(handle, "test_tensor_sub", 30.0f)) passed++;

    // Test tensor division: result[1,1] should be 10.0
    total++;
    if (runTest(handle, "test_tensor_div", 10.0f)) passed++;

    // Test combined operations: result[1,0] should be 20.0
    total++;
    if (runTest(handle, "test_tensor_combined", 20.0f)) passed++;

    // Test chained operations: result[1] should be 5.0
    total++;
    if (runTest(handle, "test_tensor_chained", 5.0f)) passed++;

    // Test main runner: should return sum = 106.0
    total++;
    if (runTest(handle, "test_elementwise_main", 106.0f)) passed++;

    std::cout << std::endl;
    std::cout << "=== Results ===" << std::endl;
    std::cout << "Passed: " << passed << "/" << total << std::endl;

    dlclose(handle);

    return (passed == total) ? 0 : 1;
}
