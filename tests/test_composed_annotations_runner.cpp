// Runner for composed annotations test
// Build: g++ -O3 -march=native -o /tmp/composed_runner tests/test_composed_annotations_runner.cpp -ldl -std=c++17
// Run: /tmp/composed_runner /tmp/composed.so

#include <dlfcn.h>
#include <cstdio>
#include <chrono>

struct TestCase {
    const char* name;
    const char* symbol;
    int expected;  // 0 means no expected value (checksum)
};

struct PerfTest {
    const char* name;
    const char* symbol;
    int M, K, N;
};

TestCase tests[] = {
    {"test_unroll", "test_unroll", 2016},
    {"test_parallel", "test_parallel", 64},
    {"test_vectorize", "test_vectorize", 64},
    {"test_prefetch", "test_prefetch", 64},
    {"test_parallel_tile_lower", "test_parallel_tile_lower", 128},
    {"test_tile_unroll_lower", "test_tile_unroll_lower", 128},
    {"test_full_pipeline", "test_full_pipeline", 256},
    {"test_composed_tiles", "test_composed_tiles", 256},
};

PerfTest perf_tests[] = {
    {"benchmark_256_composed", "benchmark_256_composed", 256, 256, 256},
    {"benchmark_512_composed", "benchmark_512_composed", 512, 512, 512},
};

using KernelFn = int (*)();

double measure_giops(KernelFn fn, int M, int K, int N, int warmup = 2, int iters = 5) {
    for (int i = 0; i < warmup; i++) fn();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) fn();
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double ops_per_iter = 2.0 * M * K * N;
    double total_ops = ops_per_iter * iters;
    return (total_ops / 1e9) / (elapsed_ms / 1000.0);
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
    printf("COMPOSED ANNOTATIONS TEST\n");
    printf("=================================================================\n\n");

    // Run accuracy tests
    printf("ACCURACY TESTS:\n");
    printf("-----------------------------------------------------------------\n");
    int passed = 0;
    int total = sizeof(tests) / sizeof(tests[0]);

    for (const auto& test : tests) {
        auto fn = (KernelFn)dlsym(handle, test.symbol);
        if (!fn) {
            printf("  [SKIP] %-30s - symbol not found\n", test.name);
            continue;
        }

        int result = fn();
        bool pass = (result == test.expected);
        if (pass) {
            printf("  [PASS] %-30s = %d\n", test.name, result);
            passed++;
        } else {
            printf("  [FAIL] %-30s = %d (expected %d)\n",
                   test.name, result, test.expected);
        }
    }
    printf("\n");

    // Run performance tests
    printf("PERFORMANCE TESTS (Composed Annotations):\n");
    printf("-----------------------------------------------------------------\n");
    for (const auto& test : perf_tests) {
        auto fn = (KernelFn)dlsym(handle, test.symbol);
        if (!fn) {
            printf("  [SKIP] %-30s - symbol not found\n", test.name);
            continue;
        }

        // First call to get checksum
        int checksum = fn();

        // Measure performance
        double giops = measure_giops(fn, test.M, test.K, test.N);
        printf("  %-35s %7.1f GIOP/s (checksum: %d)\n",
               test.name, giops, checksum);
    }
    printf("\n");

    // Summary
    printf("=================================================================\n");
    printf("SUMMARY: %d/%d accuracy tests passed\n", passed, total);
    printf("=================================================================\n");

    dlclose(handle);
    return (passed == total) ? 0 : 1;
}
