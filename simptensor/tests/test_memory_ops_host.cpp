// Host program to test memory operations (reshape, transpose, slice)
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
        {"test_reshape_1d_to_2d", "test_reshape_1d_to_2d", 7.0},
        {"test_reshape_2d_to_1d", "test_reshape_2d_to_1d", 70.0},
        {"test_reshape_2d_to_2d", "test_reshape_2d_to_2d", 7.0},
        {"test_transpose_2d", "test_transpose_2d", 7.0},
        {"test_transpose_3d", "test_transpose_3d", 25.0},
        {"test_slice_2d", "test_slice_2d", 22.0},
        {"test_slice_3d", "test_slice_3d", 19.0},
        {"test_combined_ops", "test_combined_ops", 8.0},
        {"test_reshape_4d", "test_reshape_4d", 17.0},
        {"test_transpose_4d", "test_transpose_4d", 17.0},
        {"test_slice_4d", "test_slice_4d", 41.0},
        {"test_reshape_5d", "test_reshape_5d", 33.0},
        {"test_transpose_5d", "test_transpose_5d", 33.0},
        {"test_slice_5d", "test_slice_5d", 1.0},
        {"test_memory_ops_main", "test_memory_ops_main", 307.0},
    };

    printf("=== Tensor Memory Operations Test ===\n\n");

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
