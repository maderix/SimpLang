#include <iostream>
#include <dlfcn.h>
#include <cmath>

typedef float (*TestFunc)();

int main() {
    system("gcc -shared -o /tmp/test_attention.so /tmp/test_attention -lm");

    void* handle = dlopen("/tmp/test_attention.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load kernel: " << dlerror() << std::endl;
        return 1;
    }

    TestFunc test_attention = (TestFunc)dlsym(handle, "test_attention_simple");
    if (!test_attention) {
        std::cerr << "Failed to find function: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }

    float result = test_attention();
    std::cout << "Attention result: " << result << std::endl;

    // Expected calculation:
    // Q = [[1, 0], [0, 1]]
    // K = [[1, 0], [0, 1]]
    // V = [[2, 0], [0, 3]]
    // QK^T = [[1, 0], [0, 1]]
    // softmax([1, 0]) = [0.7311, 0.2689]
    // output[0] = 0.7311 * 2 + 0.2689 * 0 = 1.4621

    float exp1 = expf(1.0f);
    float exp0 = expf(0.0f);
    float softmax0 = exp1 / (exp1 + exp0);
    float expected = softmax0 * 2.0f;

    std::cout << "Expected result: " << expected << std::endl;

    float diff = fabsf(result - expected);
    if (diff < 1e-3) {
        std::cout << "✓ Attention test PASSED!" << std::endl;
    } else {
        std::cout << "✗ Attention test FAILED! Difference: " << diff << std::endl;
    }

    dlclose(handle);
    return (diff < 1e-3) ? 0 : 1;
}
