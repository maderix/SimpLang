#include <dlfcn.h>
#include <cstdio>
#include <cmath>

typedef float (*TestFuncF32)();
typedef int (*TestFuncI32)();
typedef long long (*TestFuncI64)();

struct TestCase {
    const char* name;
    const char* funcName;
    enum { F32, I32, I64 } returnType;
    double expected;
};

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <shared_library.so>\n", argv[0]);
        return 1;
    }

    void* handle = dlopen(argv[1], RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "Failed to load library: %s\n", dlerror());
        return 1;
    }

    TestCase tests[] = {
        {"test_sum_2d_axis0_f32", "test_sum_2d_axis0_f32", TestCase::F32, 21.0},
        {"test_sum_2d_axis1_f32", "test_sum_2d_axis1_f32", TestCase::F32, 21.0},
        {"test_mean_2d_axis0_f32", "test_mean_2d_axis0_f32", TestCase::F32, 21.0},
        {"test_mean_2d_axis1_f32", "test_mean_2d_axis1_f32", TestCase::F32, 9.0},
        {"test_max_2d_axis0_f32", "test_max_2d_axis0_f32", TestCase::F32, 21.0},
        {"test_max_2d_axis1_f32", "test_max_2d_axis1_f32", TestCase::F32, 14.0},
        {"test_min_2d_axis0_f32", "test_min_2d_axis0_f32", TestCase::F32, 6.0},
        {"test_min_2d_axis1_f32", "test_min_2d_axis1_f32", TestCase::F32, 5.0},
        {"test_argmax_2d_axis0_f32", "test_argmax_2d_axis0_f32", TestCase::I64, 2.0},
        {"test_argmax_2d_axis1_f32", "test_argmax_2d_axis1_f32", TestCase::I64, 3.0},
        {"test_sum_3d_axis0_f32", "test_sum_3d_axis0_f32", TestCase::F32, 78.0},
        {"test_sum_3d_axis1_f32", "test_sum_3d_axis1_f32", TestCase::F32, 78.0},
        {"test_sum_3d_axis2_f32", "test_sum_3d_axis2_f32", TestCase::F32, 78.0},
        {"test_sum_2d_axis0_i32", "test_sum_2d_axis0_i32", TestCase::I32, 210.0},
        {"test_max_2d_axis0_i32", "test_max_2d_axis0_i32", TestCase::I32, 210.0},
        {"test_axis_reductions_main", "test_axis_reductions_main", TestCase::F32, 352.0},
    };

    printf("=== Tensor Axis Reduction Operations Test ===\n\n");

    int passed = 0;
    int failed = 0;

    for (const auto& test : tests) {
        void* func_ptr = dlsym(handle, test.funcName);
        if (!func_ptr) {
            printf("%s: SKIP (function not found)\n", test.name);
            failed++;
            continue;
        }

        double result;
        if (test.returnType == TestCase::F32) {
            TestFuncF32 func = (TestFuncF32)func_ptr;
            result = (double)func();
        } else if (test.returnType == TestCase::I32) {
            TestFuncI32 func = (TestFuncI32)func_ptr;
            result = (double)func();
        } else { // I64
            TestFuncI64 func = (TestFuncI64)func_ptr;
            result = (double)func();
        }

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
