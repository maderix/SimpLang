#include <iostream>
#include <dlfcn.h>
#include <cmath>

typedef float (*TestFunc)();

int main(int argc, char** argv) {
    const char* so_path = (argc > 1) ? argv[1] : "/tmp/test_silu.so";

    void* handle = dlopen(so_path, RTLD_LAZY);
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
