#include <iostream>
#include <dlfcn.h>
#include <cmath>
#include <iomanip>

// Test function signatures
typedef float (*TestFuncF32)();
typedef int32_t (*TestFuncI32)();
typedef int64_t (*TestFuncI64)();

struct TestCase {
    const char* name;
    const char* funcName;
    enum { F32, I32, I64 } returnType;
    double expected;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <library.so>\n";
        return 1;
    }

    std::cout << "=== Tensor Reduction Operations Test ===\n\n";

    // Load the shared library
    void* handle = dlopen(argv[1], RTLD_LAZY);
    if (!handle) {
        std::cerr << "Error loading library: " << dlerror() << "\n";
        return 1;
    }

    // Define test cases
    TestCase tests[] = {
        {"test_sum_f32", "test_sum_f32", TestCase::F32, 21.0},
        {"test_mean_f32", "test_mean_f32", TestCase::F32, 3.5},
        {"test_max_f32", "test_max_f32", TestCase::F32, 9.2},
        {"test_min_f32", "test_min_f32", TestCase::F32, 1.2},
        {"test_argmax_f32", "test_argmax_f32", TestCase::I64, 2.0},
        {"test_sum_i32", "test_sum_i32", TestCase::I32, 210.0},
        {"test_max_i32", "test_max_i32", TestCase::I32, 42.0},
        {"test_min_i32", "test_min_i32", TestCase::I32, 8.0},
        {"test_argmax_i32", "test_argmax_i32", TestCase::I64, 4.0},
        {"test_sum_i64", "test_sum_i64", TestCase::I64, 1000.0},
        {"test_max_i64", "test_max_i64", TestCase::I64, 200.0},
        {"test_min_i64", "test_min_i64", TestCase::I64, 50.0},
        {"test_sum_3d", "test_sum_3d", TestCase::F32, 36.0},
        {"test_single_element", "test_single_element", TestCase::F32, 42.5},
    };

    int passed = 0;
    int failed = 0;

    for (const auto& test : tests) {
        double result = 0.0;

        if (test.returnType == TestCase::F32) {
            auto func = (TestFuncF32)dlsym(handle, test.funcName);
            if (!func) {
                std::cerr << "Error loading function " << test.funcName << ": " << dlerror() << "\n";
                failed++;
                continue;
            }
            result = func();
        } else if (test.returnType == TestCase::I32) {
            auto func = (TestFuncI32)dlsym(handle, test.funcName);
            if (!func) {
                std::cerr << "Error loading function " << test.funcName << ": " << dlerror() << "\n";
                failed++;
                continue;
            }
            result = func();
        } else { // I64
            auto func = (TestFuncI64)dlsym(handle, test.funcName);
            if (!func) {
                std::cerr << "Error loading function " << test.funcName << ": " << dlerror() << "\n";
                failed++;
                continue;
            }
            result = func();
        }

        // Check result with tolerance for floating point
        double tolerance = (test.returnType == TestCase::F32) ? 1e-5 : 0.0;
        bool pass = std::abs(result - test.expected) <= tolerance;

        std::cout << test.name << ": ";
        if (pass) {
            std::cout << "PASS";
            passed++;
        } else {
            std::cout << "FAIL";
            failed++;
        }
        std::cout << " (result=" << std::fixed << std::setprecision(2) << result
                  << ", expected=" << test.expected << ")\n";
    }

    // Test main function
    std::cout << "\ntest_reductions_main: ";
    auto mainFunc = (TestFuncF32)dlsym(handle, "test_reductions_main");
    if (!mainFunc) {
        std::cerr << "FAIL (Error loading function: " << dlerror() << ")\n";
        failed++;
    } else {
        float mainResult = mainFunc();
        double expected = 1629.4;
        bool pass = std::abs(mainResult - expected) < 0.1;

        if (pass) {
            std::cout << "PASS";
            passed++;
        } else {
            std::cout << "FAIL";
            failed++;
        }
        std::cout << " (result=" << std::fixed << std::setprecision(2) << mainResult
                  << ", expected=" << expected << ")\n";
    }

    std::cout << "\n=== Results ===\n";
    std::cout << "Passed: " << passed << "/" << (passed + failed) << "\n";
    std::cout << "Failed: " << failed << "/" << (passed + failed) << "\n";

    dlclose(handle);
    return failed > 0 ? 1 : 0;
}
