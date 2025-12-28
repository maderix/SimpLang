// Generic test runner for simple tests that return a double value
// Usage: ./generic_runner <kernel.so> <function_name> <expected_value> [tolerance]
#include <iostream>
#include <dlfcn.h>
#include <cmath>
#include <cstdlib>
#include <cstring>

typedef double (*TestFuncDouble)();
typedef float (*TestFuncFloat)();

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <kernel.so> <function_name> <expected_value> [tolerance]" << std::endl;
        return 1;
    }

    const char* so_path = argv[1];
    const char* func_name = argv[2];
    double expected = std::atof(argv[3]);
    double tolerance = (argc > 4) ? std::atof(argv[4]) : 0.001;

    // Load compiled kernel
    void* handle = dlopen(so_path, RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load kernel: " << dlerror() << std::endl;
        return 1;
    }

    // Try to find function
    void* sym = dlsym(handle, func_name);
    if (!sym) {
        std::cerr << "Failed to find function '" << func_name << "': " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }

    // Call function (assume double return type)
    auto func = (TestFuncDouble)sym;
    double result = func();

    // Verify result
    bool passed = std::abs(result - expected) < tolerance;

    std::cout << "Test: " << func_name << std::endl;
    std::cout << "  Expected: " << expected << std::endl;
    std::cout << "  Got:      " << result << std::endl;
    std::cout << "  Tolerance: " << tolerance << std::endl;
    std::cout << "  Status:   " << (passed ? "PASSED" : "FAILED") << std::endl;

    dlclose(handle);
    return passed ? 0 : 1;
}
