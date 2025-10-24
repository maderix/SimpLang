#include <iostream>
#include <dlfcn.h>
#include <cmath>

typedef float (*TestFunc)();

int main() {
    // Convert object to shared library first
    system("gcc -shared -o /tmp/test_silu.so /tmp/test_silu -lm");

    void* handle = dlopen("/tmp/test_silu.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load kernel: " << dlerror() << std::endl;
        return 1;
    }

    TestFunc test_silu = (TestFunc)dlsym(handle, "test_silu");
    if (!test_silu) {
        std::cerr << "Failed to find function: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }

    float result = test_silu();

    std::cout << "SiLU result[0]: " << result << std::endl;

    // Expected calculation for input[0] = 1.0:
    // SiLU(x) = x / (1 + exp(-x))
    // SiLU(1.0) = 1.0 / (1 + exp(-1.0)) = 1.0 / (1 + 0.3679) ≈ 0.7311

    float x = 1.0f;
    float expected = x / (1.0f + expf(-x));
    std::cout << "Expected result[0]: " << expected << std::endl;

    float diff = fabsf(result - expected);
    if (diff < 1e-4) {
        std::cout << "✓ SiLU test PASSED!" << std::endl;
    } else {
        std::cout << "✗ SiLU test FAILED! Difference: " << diff << std::endl;
    }

    dlclose(handle);
    return (diff < 1e-4) ? 0 : 1;
}
