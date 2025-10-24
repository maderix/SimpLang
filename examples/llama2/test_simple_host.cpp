#include <iostream>
#include <dlfcn.h>
#include <vector>

typedef float (*TestFunc)(
    float*, float*, int64_t, int64_t, int64_t,  // a: alloc, align, off, size, stride
    float*, float*, int64_t, int64_t, int64_t,  // b
    float*, float*, int64_t, int64_t, int64_t,  // c
    int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t
);

int main() {
    std::vector<float> a = {1.0, 2.0};
    std::vector<float> b = {3.0, 4.0};
    std::vector<float> c = {5.0, 6.0};

    void* handle = dlopen("/tmp/test_simple.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Error: " << dlerror() << std::endl;
        return 1;
    }

    TestFunc func = (TestFunc)dlsym(handle, "test_params");
    if (!func) {
        std::cerr << "Error: " << dlerror() << std::endl;
        return 1;
    }

    float result = func(
        a.data(), a.data(), 0, 2, 1,  // a: alloc, align, offset, size, stride
        b.data(), b.data(), 0, 2, 1,  // b
        c.data(), c.data(), 0, 2, 1,  // c
        64, 4, 128, 8, 1,
        4, 4, 512, 64, 100
    );

    std::cout << "Result: " << result << std::endl;
    std::cout << "Expected: " << (a[0] + b[0] + c[0]) << std::endl;

    dlclose(handle);
    return 0;
}
