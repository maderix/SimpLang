//===- test_scatter_gather_host.cpp - Scatter/Gather Test Runner ---------===//
//
// Test runner for N-D scatter/gather operations
// Loads compiled SimpLang kernels and verifies correctness
//
//===----------------------------------------------------------------------===//

#include <dlfcn.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>

typedef float (*TestFunc)();

struct TestCase {
    const char* name;
    const char* func_name;
    float expected;
};

bool runTest(void* handle, const TestCase& test) {
    TestFunc func = (TestFunc)dlsym(handle, test.func_name);
    if (!func) {
        std::cerr << "  ✗ " << test.name << ": Function not found - " << dlerror() << "\n";
        return false;
    }

    float result = func();
    float diff = std::abs(result - test.expected);
    bool passed = diff < 1e-4f;

    if (passed) {
        std::cout << "  ✓ " << test.name << ": " << result << " (expected " << test.expected << ")\n";
    } else {
        std::cerr << "  ✗ " << test.name << ": " << result << " (expected " << test.expected << ", diff=" << diff << ")\n";
    }

    return passed;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <shared_library.so>\n";
        return 1;
    }

    // Load the compiled kernel
    void* handle = dlopen(argv[1], RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load library: " << dlerror() << "\n";
        return 1;
    }

    std::cout << "========================================\n";
    std::cout << "Scatter/Gather Correctness Tests\n";
    std::cout << "========================================\n\n";

    TestCase tests[] = {
        // Gather tests
        {"1D gather (embedding lookup)", "test_gather_1d", 12.0f},
        {"1D gather (reversed indices)", "test_gather_1d_reversed", 120.0f},
        {"2D gather along axis 0 (select rows)", "test_gather_2d_axis0", 21.0f},
        {"2D gather along axis 1 (select columns)", "test_gather_2d_axis1", 27.0f},
        {"3D gather along axis 0", "test_gather_3d_axis0", 25.0f},
        {"3D gather along axis 1", "test_gather_3d_axis1", 25.0f},
        {"3D gather along axis 2", "test_gather_3d_axis2", 32.0f},
        {"4D gather along axis 2", "test_gather_4d_axis2", 49.0f},

        // Scatter tests
        {"1D scatter (sparse update)", "test_scatter_1d", 100.0f},
        {"1D scatter with overwrite", "test_scatter_1d_overwrite", 603.0f},
        {"2D scatter along axis 0 (update rows)", "test_scatter_2d_axis0", 151.0f},
        {"2D scatter along axis 1 (update columns)", "test_scatter_2d_axis1", 133.0f},
        {"3D scatter along axis 0", "test_scatter_3d_axis0", 167.0f},
        {"3D scatter along axis 2", "test_scatter_3d_axis2", 148.0f},
        {"4D scatter along axis 1", "test_scatter_4d_axis1", 168.0f},

        // Combined operations
        {"Gather then scatter (combined)", "test_gather_scatter_combined", 39.0f}
    };

    int passed = 0;
    int total = sizeof(tests) / sizeof(TestCase);

    std::cout << "GATHER TESTS\n";
    std::cout << "------------\n";
    for (int i = 0; i < 8; i++) {
        if (runTest(handle, tests[i])) passed++;
    }

    std::cout << "\nSCATTER TESTS\n";
    std::cout << "-------------\n";
    for (int i = 8; i < 15; i++) {
        if (runTest(handle, tests[i])) passed++;
    }

    std::cout << "\nCOMBINED TESTS\n";
    std::cout << "--------------\n";
    for (int i = 15; i < total; i++) {
        if (runTest(handle, tests[i])) passed++;
    }

    std::cout << "\n========================================\n";
    std::cout << "Results: " << passed << "/" << total << " tests passed\n";
    std::cout << "========================================\n";

    dlclose(handle);
    return (passed == total) ? 0 : 1;
}
