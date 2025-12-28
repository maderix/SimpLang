#include <iostream>
#include <dlfcn.h>
#include <cmath>

typedef float (*TestFunc)();

int main(int argc, char** argv) {
    const char* so_path = (argc > 1) ? argv[1] : "/tmp/test_softmax.so";

    void* handle = dlopen(so_path, RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load kernel: " << dlerror() << std::endl;
        return 1;
    }

    TestFunc test_softmax = (TestFunc)dlsym(handle, "test_softmax");
    if (!test_softmax) {
        std::cerr << "Failed to find function: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }

    float result = test_softmax();

    std::cout << "Softmax result[0]: " << result << std::endl;

    // Expected calculation for input [1.0, 2.0, 3.0, 4.0]:
    // max = 4.0
    // shifted = [-3.0, -2.0, -1.0, 0.0]
    // exp = [0.0498, 0.1353, 0.3679, 1.0]
    // sum = 1.5530
    // softmax[0] = 0.0498 / 1.5530 ≈ 0.0321

    float expected = expf(-3.0f) / (expf(-3.0f) + expf(-2.0f) + expf(-1.0f) + expf(0.0f));
    std::cout << "Expected result[0]: " << expected << std::endl;

    float diff = fabsf(result - expected);
    if (diff < 1e-4) {
        std::cout << "✓ Softmax test PASSED!" << std::endl;
    } else {
        std::cout << "✗ Softmax test FAILED! Difference: " << diff << std::endl;
    }

    dlclose(handle);
    return (diff < 1e-4) ? 0 : 1;
}
