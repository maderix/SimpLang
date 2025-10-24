#include <iostream>
#include <dlfcn.h>
#include <cmath>

typedef float (*TestFunc)();

int main() {
    // Load the compiled kernel
    void* handle = dlopen("/tmp/test_rmsnorm.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load kernel: " << dlerror() << std::endl;
        return 1;
    }

    // Get the function pointer
    TestFunc test_rmsnorm = (TestFunc)dlsym(handle, "test_rmsnorm");
    if (!test_rmsnorm) {
        std::cerr << "Failed to find function: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }

    // Execute the function
    float result = test_rmsnorm();

    std::cout << "RMSNorm result: " << result << std::endl;

    // Expected calculation:
    // Input: [1.0, 2.0, 3.0, 4.0]
    // Weight: [1.0, 1.0, 1.0, 1.0]
    // Sum of squares: 1 + 4 + 9 + 16 = 30
    // Mean: 30/4 = 7.5
    // RMS: sqrt(7.5 + 1e-5) ≈ 2.7386
    // Normalized first element: 1.0 / 2.7386 ≈ 0.3651
    // Scaled by weight (1.0): 0.3651

    float expected = 1.0f / sqrtf(7.5f + 1e-5f);
    std::cout << "Expected result: " << expected << std::endl;

    // Check if result is close to expected
    float diff = fabsf(result - expected);
    if (diff < 1e-4) {
        std::cout << "✓ RMSNorm test PASSED!" << std::endl;
    } else {
        std::cout << "✗ RMSNorm test FAILED! Difference: " << diff << std::endl;
    }

    dlclose(handle);
    return (diff < 1e-4) ? 0 : 1;
}
