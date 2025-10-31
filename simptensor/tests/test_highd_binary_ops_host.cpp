// Host program to test high-dimensional binary operations
#include <dlfcn.h>
#include <stdio.h>
#include <cmath>

typedef float (*TestFuncF32)();

struct TestCase {
    const char* name;
    const char* funcName;
    double expected;
};

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <shared_library.so>\n", argv[0]);
        return 1;
    }

    void* handle = dlopen(argv[1], RTLD_LAZY);
    if (!handle) {
        printf("Error: Cannot open library: %s\n", dlerror());
        return 1;
    }

    TestCase tests[] = {
        {"test_4d_add", "test_4d_add", 37.0},
        {"test_4d_sub", "test_4d_sub", 117.5},
        {"test_4d_mul", "test_4d_mul", 25.0},
        {"test_4d_div", "test_4d_div", 13.0},
        {"test_4d_complex", "test_4d_complex", 9.0},
        {"test_5d_add", "test_5d_add", 233.0},
        {"test_5d_mul", "test_5d_mul", 30.0},
        {"test_5d_div", "test_5d_div", 21.0},
        {"test_highd_binary_main", "test_highd_binary_main", 485.5},
    };

    printf("=== High-Dimensional Binary Operations Test ===\n\n");

    int passed = 0;
    int failed = 0;

    for (const auto& test : tests) {
        void* func_ptr = dlsym(handle, test.funcName);
        if (!func_ptr) {
            printf("%s: SKIP (function not found)\n", test.name);
            failed++;
            continue;
        }

        TestFuncF32 func = (TestFuncF32)func_ptr;
        double result = (double)func();

        bool success = fabs(result - test.expected) < 0.01;
        if (success) {
            printf("%s: PASS (result=%.2f, expected=%.2f)\n", test.name, result, test.expected);
            passed++;
        } else {
            printf("%s: FAIL (result=%.2f, expected=%.2f)\n", test.name, result, test.expected);
            failed++;
        }
    }

    printf("\n=== Results ===\n");
    printf("Passed: %d/%d\n", passed, passed + failed);
    printf("Failed: %d/%d\n", failed, passed + failed);

    dlclose(handle);
    return (failed == 0) ? 0 : 1;
}
