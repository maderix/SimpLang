// Gating test runner for annotation framework
// Validates accuracy (checksums) and tracks performance baselines
// Build: g++ -O3 -march=native -o /tmp/gate_runner tests/gate_annotations_runner.cpp -ldl -std=c++17
// Run: /tmp/gate_runner /tmp/gate.so

#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>

// Performance baseline thresholds (GIOP/s)
// These are minimum acceptable values - actual should be higher
constexpr double PERF_THRESHOLD_256 = 50.0;   // Minimum GIOP/s for 256x256
constexpr double PERF_THRESHOLD_512 = 100.0;  // Minimum GIOP/s for 512x512
constexpr double PERF_THRESHOLD_1024 = 150.0; // Minimum GIOP/s for 1024x1024

// Accuracy tests with expected results
struct AccuracyTest {
    const char* name;
    const char* symbol;
    int expected;
};

// Performance tests with size info
struct PerfTest {
    const char* name;
    const char* symbol;
    int M, K, N;
    double min_giops;
};

AccuracyTest accuracy_tests[] = {
    {"64x64 tile=8x8x8", "gate_accuracy_tile8", 64},
    {"64x64 tile=32x32x32", "gate_accuracy_tile32", 64},
    {"256x256 tile=16x16x16", "gate_accuracy_256", 256},
    {"512x512 tile=64x64x64", "gate_accuracy_512", 512},
};

PerfTest perf_tests[] = {
    {"256x256 tile=16", "gate_perf_256_t16", 256, 256, 256, PERF_THRESHOLD_256},
    {"512x512 tile=64", "gate_perf_512_t64", 512, 512, 512, PERF_THRESHOLD_512},
    {"1024x1024 tile=128", "gate_perf_1024_t128", 1024, 1024, 1024, PERF_THRESHOLD_1024},
};

using KernelFn = int (*)();

double measure_giops(KernelFn fn, int M, int K, int N, int warmup = 2, int iters = 5) {
    // Warmup
    for (int i = 0; i < warmup; i++) {
        fn();
    }

    // Measure
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        fn();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double ops_per_iter = 2.0 * M * K * N;  // 2 ops per element (multiply + add)
    double total_ops = ops_per_iter * iters;
    double giops = (total_ops / 1e9) / (elapsed_ms / 1000.0);

    return giops;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <kernel.so>\n", argv[0]);
        return 1;
    }

    void* handle = dlopen(argv[1], RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "Failed to load %s: %s\n", argv[1], dlerror());
        return 1;
    }

    printf("=================================================================\n");
    printf("ANNOTATION FRAMEWORK GATING TESTS\n");
    printf("=================================================================\n\n");

    // Run accuracy gates
    printf("ACCURACY GATES:\n");
    printf("-----------------------------------------------------------------\n");
    int accuracy_passed = 0;
    int accuracy_total = sizeof(accuracy_tests) / sizeof(accuracy_tests[0]);

    for (const auto& test : accuracy_tests) {
        auto fn = (KernelFn)dlsym(handle, test.symbol);
        if (!fn) {
            printf("  [SKIP] %-30s - symbol not found\n", test.name);
            continue;
        }

        int result = fn();
        bool pass = (result == test.expected);
        if (pass) {
            printf("  [PASS] %-30s = %d (expected %d)\n", test.name, result, test.expected);
            accuracy_passed++;
        } else {
            printf("  [FAIL] %-30s = %d (expected %d)\n", test.name, result, test.expected);
        }
    }
    printf("\n");

    // Run performance gates
    printf("PERFORMANCE GATES:\n");
    printf("-----------------------------------------------------------------\n");
    int perf_passed = 0;
    int perf_total = sizeof(perf_tests) / sizeof(perf_tests[0]);

    for (const auto& test : perf_tests) {
        auto fn = (KernelFn)dlsym(handle, test.symbol);
        if (!fn) {
            printf("  [SKIP] %-30s - symbol not found\n", test.name);
            continue;
        }

        // First verify accuracy
        int result = fn();

        // Then measure performance
        double giops = measure_giops(fn, test.M, test.K, test.N);
        bool pass = (giops >= test.min_giops);

        if (pass) {
            printf("  [PASS] %-30s %7.1f GIOP/s (>= %.1f)\n",
                   test.name, giops, test.min_giops);
            perf_passed++;
        } else {
            printf("  [FAIL] %-30s %7.1f GIOP/s (< %.1f threshold)\n",
                   test.name, giops, test.min_giops);
        }
    }
    printf("\n");

    // Summary
    printf("=================================================================\n");
    printf("SUMMARY:\n");
    printf("  Accuracy: %d/%d passed\n", accuracy_passed, accuracy_total);
    printf("  Performance: %d/%d passed\n", perf_passed, perf_total);
    printf("=================================================================\n");

    dlclose(handle);

    // Return 0 only if all gates pass
    bool all_pass = (accuracy_passed == accuracy_total) && (perf_passed == perf_total);
    return all_pass ? 0 : 1;
}
