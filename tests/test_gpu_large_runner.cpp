// Test runner for GPU large matmul test (256x256)
// Expected result: C[0,0] = 256 * 1.0 * 2.0 = 512.0

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <dlfcn.h>

typedef float (*kernel_main_fn)();

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <kernel.so>\n", argv[0]);
        return 1;
    }

    // Load the compiled kernel
    void* handle = dlopen(argv[1], RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "Error loading kernel: %s\n", dlerror());
        return 1;
    }

    // Find kernel_main
    kernel_main_fn kernel_main = (kernel_main_fn)dlsym(handle, "kernel_main");
    if (!kernel_main) {
        fprintf(stderr, "Error finding kernel_main: %s\n", dlerror());
        dlclose(handle);
        return 1;
    }

    printf("Running 256x256 matmul test...\n");

    // Run the kernel
    float result = kernel_main();

    // Expected: C[0,0] = sum(A[0,k] * B[k,0]) for k=0..255
    // A[i,j] = 1.0, B[i,j] = 2.0
    // So C[0,0] = 256 * 1.0 * 2.0 = 512.0
    float expected = 512.0f;

    printf("Result: %f\n", result);
    printf("Expected: %f\n", expected);

    if (fabs(result - expected) < 0.001f) {
        printf("TEST PASSED!\n");
        dlclose(handle);
        return 0;
    } else {
        printf("TEST FAILED! Difference: %f\n", fabs(result - expected));
        dlclose(handle);
        return 1;
    }
}
