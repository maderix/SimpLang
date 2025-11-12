// Benchmark runner for memory operations comparison
#include <dlfcn.h>
#include <stdio.h>
#include <chrono>
#include <cmath>

typedef float (*BenchFunc)();

struct BenchmarkCase {
    const char* name;
    const char* simpFuncName;
    const char* nativeFuncName;
};

void runBenchmark(const char* name, BenchFunc simpFunc, BenchFunc nativeFunc, int iterations = 100) {
    printf("\n=== %s ===\n", name);

    // Warmup
    for (int i = 0; i < 5; i++) {
        simpFunc();
        nativeFunc();
    }

    // Benchmark SimpLang
    auto simpStart = std::chrono::high_resolution_clock::now();
    float simpResult = 0.0f;
    for (int i = 0; i < iterations; i++) {
        simpResult = simpFunc();
    }
    auto simpEnd = std::chrono::high_resolution_clock::now();
    auto simpDuration = std::chrono::duration_cast<std::chrono::microseconds>(simpEnd - simpStart);

    // Benchmark Native C++
    auto nativeStart = std::chrono::high_resolution_clock::now();
    float nativeResult = 0.0f;
    for (int i = 0; i < iterations; i++) {
        nativeResult = nativeFunc();
    }
    auto nativeEnd = std::chrono::high_resolution_clock::now();
    auto nativeDuration = std::chrono::duration_cast<std::chrono::microseconds>(nativeEnd - nativeStart);

    // Calculate statistics
    double simpAvg = simpDuration.count() / (double)iterations;
    double nativeAvg = nativeDuration.count() / (double)iterations;
    double slowdown = simpAvg / nativeAvg;

    // Verify results match
    bool resultsMatch = fabs(simpResult - nativeResult) < 0.01;

    printf("SimpLang:  %.2f µs/iter (result: %.2f)\n", simpAvg, simpResult);
    printf("Native C++: %.2f µs/iter (result: %.2f)\n", nativeAvg, nativeResult);
    printf("Slowdown:   %.2fx\n", slowdown);
    printf("Results:    %s\n", resultsMatch ? "MATCH ✓" : "MISMATCH ✗");

    if (!resultsMatch) {
        printf("WARNING: Results don't match! SimpLang=%.2f, Native=%.2f\n", simpResult, nativeResult);
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <simplang_lib.so> <native_lib.so>\n", argv[0]);
        return 1;
    }

    // Load SimpLang library
    void* simpHandle = dlopen(argv[1], RTLD_LAZY);
    if (!simpHandle) {
        printf("Error: Cannot open SimpLang library: %s\n", dlerror());
        return 1;
    }

    // Load native library
    void* nativeHandle = dlopen(argv[2], RTLD_LAZY);
    if (!nativeHandle) {
        printf("Error: Cannot open native library: %s\n", dlerror());
        dlclose(simpHandle);
        return 1;
    }

    printf("======================================\n");
    printf("Memory Operations Benchmark\n");
    printf("======================================\n");

    BenchmarkCase benchmarks[] = {
        {"Reshape 2D → 1D (256x256 → 65536)", "bench_reshape_2d_to_1d", "bench_reshape_2d_to_1d_native"},
        {"Reshape 1D → 2D (65536 → 256x256)", "bench_reshape_1d_to_2d", "bench_reshape_1d_to_2d_native"},
        {"Transpose 2D (256x256)", "bench_transpose_2d", "bench_transpose_2d_native"},
        {"Transpose 3D (64x64x64, permute [2,1,0])", "bench_transpose_3d", "bench_transpose_3d_native"},
        {"Slice 2D (extract 256x256 from 512x512)", "bench_slice_2d", "bench_slice_2d_native"},
        {"Slice 3D (extract 64x64x64 from 128x128x128)", "bench_slice_3d", "bench_slice_3d_native"},
        {"Combined Ops (reshape + transpose + slice)", "bench_combined_ops", "bench_combined_ops_native"},
        {"STRESS: 4D Reshape (32x32x32x32 → 1M)", "stress_reshape_4d_to_1d", "stress_reshape_4d_to_1d_native"},
        {"STRESS: 4D Complex Transpose [3,1,2,0]", "stress_transpose_4d_complex", "stress_transpose_4d_complex_native"},
        {"STRESS: Multi-Slice Chain (3 levels)", "stress_multi_slice", "stress_multi_slice_native"},
        {"STRESS: Reshape+Transpose Chain", "stress_reshape_transpose_chain", "stress_reshape_transpose_chain_native"},
        {"STRESS: Large Slice (256x256 → 8x8)", "stress_large_slice", "stress_large_slice_native"},
        {"CAST: i64 → f32 conversion", "test_cast_i64_to_f32", "test_cast_i64_to_f32_native"},
        {"CAST: f32 → i64 conversion", "test_cast_f32_to_i64", "test_cast_f32_to_i64_native"},
        {"CAST: Mixed arithmetic", "test_cast_mixed_arithmetic", "test_cast_mixed_arithmetic_native"},
        {"CAST: Tensor operations", "test_cast_tensor_ops", "test_cast_tensor_ops_native"},
        {"CAST: Chained conversions", "test_cast_chained", "test_cast_chained_native"},
        {"CAST: With reshape", "test_cast_with_reshape", "test_cast_with_reshape_native"},
        {"CAST: Large values", "test_cast_large_values", "test_cast_large_values_native"},
        {"CAST: Complex operations", "test_cast_complex_ops", "test_cast_complex_ops_native"},
    };

    int numBenchmarks = sizeof(benchmarks) / sizeof(BenchmarkCase);
    double totalSlowdown = 0.0;
    int validBenchmarks = 0;

    for (int i = 0; i < numBenchmarks; i++) {
        BenchFunc simpFunc = (BenchFunc)dlsym(simpHandle, benchmarks[i].simpFuncName);
        BenchFunc nativeFunc = (BenchFunc)dlsym(nativeHandle, benchmarks[i].nativeFuncName);

        if (!simpFunc) {
            printf("\nWarning: SimpLang function '%s' not found\n", benchmarks[i].simpFuncName);
            continue;
        }

        if (!nativeFunc) {
            printf("\nWarning: Native function '%s' not found\n", benchmarks[i].nativeFuncName);
            continue;
        }

        // Run fewer iterations for larger tensors
        int iterations = 100;
        if (i == 3 || i == 5) {  // 3D operations with large tensors
            iterations = 50;
        }
        if (i >= 7) {  // Stress tests
            iterations = 20;
        }

        // Time and warmup separately for accurate measurement
        // Warmup
        for (int w = 0; w < 5; w++) {
            simpFunc();
            nativeFunc();
        }

        // Actual benchmark
        auto simpStart = std::chrono::high_resolution_clock::now();
        float simpResult = 0.0f;
        for (int iter = 0; iter < iterations; iter++) {
            simpResult = simpFunc();
        }
        auto simpEnd = std::chrono::high_resolution_clock::now();
        auto simpDuration = std::chrono::duration_cast<std::chrono::microseconds>(simpEnd - simpStart);

        auto nativeStart = std::chrono::high_resolution_clock::now();
        float nativeResult = 0.0f;
        for (int iter = 0; iter < iterations; iter++) {
            nativeResult = nativeFunc();
        }
        auto nativeEnd = std::chrono::high_resolution_clock::now();
        auto nativeDuration = std::chrono::duration_cast<std::chrono::microseconds>(nativeEnd - nativeStart);

        double simpAvg = simpDuration.count() / (double)iterations;
        double nativeAvg = nativeDuration.count() / (double)iterations;
        double slowdown = simpAvg / nativeAvg;
        bool resultsMatch = fabs(simpResult - nativeResult) < 0.01;

        printf("\n=== %s ===\n", benchmarks[i].name);
        printf("SimpLang:   %.2f µs/iter (result: %.2f)\n", simpAvg, simpResult);
        printf("Native C++: %.2f µs/iter (result: %.2f)\n", nativeAvg, nativeResult);
        printf("Slowdown:   %.2fx\n", slowdown);
        printf("Results:    %s\n", resultsMatch ? "MATCH ✓" : "MISMATCH ✗");

        if (!resultsMatch) {
            printf("WARNING: Results don't match! SimpLang=%.2f, Native=%.2f\n", simpResult, nativeResult);
        }

        totalSlowdown += slowdown;
        validBenchmarks++;
    }

    // Summary
    printf("\n======================================\n");
    printf("Summary\n");
    printf("======================================\n");
    printf("Benchmarks run: %d/%d\n", validBenchmarks, numBenchmarks);
    if (validBenchmarks > 0) {
        double avgSlowdown = totalSlowdown / validBenchmarks;
        printf("Average slowdown: %.2fx\n", avgSlowdown);

        if (avgSlowdown < 1.5) {
            printf("Performance: EXCELLENT (< 1.5x)\n");
        } else if (avgSlowdown < 2.0) {
            printf("Performance: GOOD (< 2.0x)\n");
        } else if (avgSlowdown < 3.0) {
            printf("Performance: ACCEPTABLE (< 3.0x)\n");
        } else {
            printf("Performance: NEEDS IMPROVEMENT (>= 3.0x)\n");
        }
    }

    dlclose(simpHandle);
    dlclose(nativeHandle);
    return 0;
}
