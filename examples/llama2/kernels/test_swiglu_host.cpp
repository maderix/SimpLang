#include <iostream>
#include <dlfcn.h>
#include <cmath>

typedef float (*TestFunc)();

int main() {
    system("gcc -shared -o /tmp/test_swiglu_ffn.so /tmp/test_swiglu_ffn -lm");

    void* handle = dlopen("/tmp/test_swiglu_ffn.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load kernel: " << dlerror() << std::endl;
        return 1;
    }

    TestFunc test_swiglu = (TestFunc)dlsym(handle, "test_swiglu");
    if (!test_swiglu) {
        std::cerr << "Failed to find function: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }

    float result = test_swiglu();
    std::cout << "SwiGLU result: " << result << std::endl;

    // Expected calculation:
    // x[0] = 1.0
    // gate[0] = 1.0 * 0.5 = 0.5
    // up[0] = 1.0 * 2.0 = 2.0
    // gate_silu[0] = SiLU(0.5) = 0.5 / (1 + exp(-0.5))
    // result[0] = gate_silu[0] * up[0]

    float gate = 0.5f;
    float gate_silu = gate / (1.0f + expf(-gate));
    float up = 2.0f;
    float expected = gate_silu * up;

    std::cout << "Expected result: " << expected << std::endl;

    float diff = fabsf(result - expected);
    if (diff < 1e-4) {
        std::cout << "✓ SwiGLU test PASSED!" << std::endl;
    } else {
        std::cout << "✗ SwiGLU test FAILED! Difference: " << diff << std::endl;
    }

    dlclose(handle);
    return (diff < 1e-4) ? 0 : 1;
}
