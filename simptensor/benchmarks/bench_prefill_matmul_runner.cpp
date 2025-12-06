/**
 * Prefill Matmul Benchmark Runner
 * Tests tensor_matmul_nt with M > 1 for prompt processing
 */

#include <iostream>
#include <dlfcn.h>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cstring>
#include <immintrin.h>

constexpr int64_t DIM = 2048;
constexpr int64_t HIDDEN_DIM = 8192;

#define MEMREF_I8_PARAMS int8_t*, int8_t*, int64_t, int64_t, int64_t
#define MEMREF_I32_PARAMS int32_t*, int32_t*, int64_t, int64_t, int64_t
#define PASS_MEMREF_I8(ptr, size) ptr, ptr, 0LL, (int64_t)(size), 1LL
#define PASS_MEMREF_I32(ptr, size) ptr, ptr, 0LL, (int64_t)(size), 1LL

typedef int32_t (*PrefillFunc)(MEMREF_I8_PARAMS, MEMREF_I8_PARAMS, MEMREF_I32_PARAMS);

void init_random_i8(int8_t* data, size_t size, int seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-64, 63);
    for (size_t i = 0; i < size; i++) {
        data[i] = (int8_t)dist(rng);
    }
}

// Reference C++ VNNI implementation for comparison
void matmul_vnni_ref(const int8_t* A, const int8_t* B_t, int32_t* C,
                     int M, int N, int K) {
    // B is pre-transposed: B_t[N, K]
    memset(C, 0, M * N * sizeof(int32_t));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t sum = 0;
            int k = 0;

            // VNNI: process 4 elements at a time
            for (; k + 3 < K; k += 4) {
                // Load 4 bytes from A row
                int32_t a_packed = *reinterpret_cast<const int32_t*>(&A[i * K + k]);
                // Load 4 bytes from B_t row (already transposed)
                int32_t b_packed = *reinterpret_cast<const int32_t*>(&B_t[j * K + k]);

                // Signed multiply-add
                int8_t a0 = (a_packed >> 0) & 0xFF;
                int8_t a1 = (a_packed >> 8) & 0xFF;
                int8_t a2 = (a_packed >> 16) & 0xFF;
                int8_t a3 = (a_packed >> 24) & 0xFF;
                int8_t b0 = (b_packed >> 0) & 0xFF;
                int8_t b1 = (b_packed >> 8) & 0xFF;
                int8_t b2 = (b_packed >> 16) & 0xFF;
                int8_t b3 = (b_packed >> 24) & 0xFF;

                sum += (int32_t)a0 * (int32_t)b0;
                sum += (int32_t)a1 * (int32_t)b1;
                sum += (int32_t)a2 * (int32_t)b2;
                sum += (int32_t)a3 * (int32_t)b3;
            }

            // Remainder
            for (; k < K; k++) {
                sum += (int32_t)A[i * K + k] * (int32_t)B_t[j * K + k];
            }

            C[i * N + j] = sum;
        }
    }
}

double benchmark_ref(int M, int N, int K, int iterations) {
    std::vector<int8_t> A(M * K);
    std::vector<int8_t> B_t(N * K);
    std::vector<int32_t> C(M * N);

    init_random_i8(A.data(), A.size(), 1);
    init_random_i8(B_t.data(), B_t.size(), 2);

    // Warmup
    matmul_vnni_ref(A.data(), B_t.data(), C.data(), M, N, K);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        matmul_vnni_ref(A.data(), B_t.data(), C.data(), M, N, K);
    }
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <bench_prefill_matmul.so>" << std::endl;
        return 1;
    }

    void* handle = dlopen(argv[1], RTLD_NOW);
    if (!handle) {
        std::cerr << "Error: " << dlerror() << std::endl;
        return 1;
    }

    std::cout << "================================================================================\n";
    std::cout << "   Prefill Matmul Benchmark (LLaMA 3.2-1B shapes)\n";
    std::cout << "   dim=" << DIM << ", hidden_dim=" << HIDDEN_DIM << "\n";
    std::cout << "================================================================================\n\n";

    const int iterations = 10;

    // Test QKV projection: [M, 2048] @ [2048, 2048]_nt -> [M, 2048]
    std::cout << "=== QKV Projection: [M, 2048] @ [2048, 2048]_nt -> [M, 2048] ===\n";
    std::cout << std::setw(8) << "M"
              << std::setw(12) << "SimpLang"
              << std::setw(12) << "C++ Ref"
              << std::setw(12) << "Ratio"
              << std::setw(15) << "GOPS" << "\n";

    struct TestCase {
        const char* func_name;
        int M;
        int N;
        int K;
        size_t input_size;
        size_t weight_size;
        size_t output_size;
    };

    TestCase qkv_tests[] = {
        {"prefill_qkv_64", 64, 2048, 2048, 64*2048, 2048*2048, 64*2048},
        {"prefill_qkv_128", 128, 2048, 2048, 128*2048, 2048*2048, 128*2048},
        {"prefill_qkv_256", 256, 2048, 2048, 256*2048, 2048*2048, 256*2048},
        {"prefill_qkv_512", 512, 2048, 2048, 512*2048, 2048*2048, 512*2048},
    };

    for (const auto& test : qkv_tests) {
        auto func = (PrefillFunc)dlsym(handle, test.func_name);
        if (!func) {
            std::cerr << "Warning: " << test.func_name << " not found\n";
            continue;
        }

        std::vector<int8_t> input(test.input_size);
        std::vector<int8_t> weight(test.weight_size);
        std::vector<int32_t> output(test.output_size);

        init_random_i8(input.data(), input.size(), 1);
        init_random_i8(weight.data(), weight.size(), 2);

        // Warmup
        func(PASS_MEMREF_I8(input.data(), input.size()),
             PASS_MEMREF_I8(weight.data(), weight.size()),
             PASS_MEMREF_I32(output.data(), output.size()));

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            func(PASS_MEMREF_I8(input.data(), input.size()),
                 PASS_MEMREF_I8(weight.data(), weight.size()),
                 PASS_MEMREF_I32(output.data(), output.size()));
        }
        auto end = std::chrono::high_resolution_clock::now();
        double simp_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

        double ref_ms = benchmark_ref(test.M, test.N, test.K, iterations);
        double ratio = ref_ms / simp_ms * 100.0;

        // GOPS = 2 * M * N * K / time_in_seconds / 1e9
        double gops = 2.0 * test.M * test.N * test.K / (simp_ms / 1000.0) / 1e9;

        std::cout << std::setw(8) << test.M
                  << std::setw(10) << std::fixed << std::setprecision(2) << simp_ms << " ms"
                  << std::setw(10) << ref_ms << " ms"
                  << std::setw(10) << std::setprecision(0) << ratio << "%"
                  << std::setw(12) << std::setprecision(1) << gops << " GOPS\n";
    }

    // Test FFN up: [M, 2048] @ [8192, 2048]_nt -> [M, 8192]
    std::cout << "\n=== FFN Gate/Up: [M, 2048] @ [8192, 2048]_nt -> [M, 8192] ===\n";
    std::cout << std::setw(8) << "M"
              << std::setw(12) << "SimpLang"
              << std::setw(12) << "C++ Ref"
              << std::setw(12) << "Ratio"
              << std::setw(15) << "GOPS" << "\n";

    TestCase ffn_up_tests[] = {
        {"prefill_ffn_up_64", 64, 8192, 2048, 64*2048, 8192*2048, 64*8192},
        {"prefill_ffn_up_128", 128, 8192, 2048, 128*2048, 8192*2048, 128*8192},
    };

    for (const auto& test : ffn_up_tests) {
        auto func = (PrefillFunc)dlsym(handle, test.func_name);
        if (!func) {
            std::cerr << "Warning: " << test.func_name << " not found\n";
            continue;
        }

        std::vector<int8_t> input(test.input_size);
        std::vector<int8_t> weight(test.weight_size);
        std::vector<int32_t> output(test.output_size);

        init_random_i8(input.data(), input.size(), 1);
        init_random_i8(weight.data(), weight.size(), 2);

        // Warmup
        func(PASS_MEMREF_I8(input.data(), input.size()),
             PASS_MEMREF_I8(weight.data(), weight.size()),
             PASS_MEMREF_I32(output.data(), output.size()));

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            func(PASS_MEMREF_I8(input.data(), input.size()),
                 PASS_MEMREF_I8(weight.data(), weight.size()),
                 PASS_MEMREF_I32(output.data(), output.size()));
        }
        auto end = std::chrono::high_resolution_clock::now();
        double simp_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

        double ref_ms = benchmark_ref(test.M, test.N, test.K, iterations);
        double ratio = ref_ms / simp_ms * 100.0;
        double gops = 2.0 * test.M * test.N * test.K / (simp_ms / 1000.0) / 1e9;

        std::cout << std::setw(8) << test.M
                  << std::setw(10) << std::fixed << std::setprecision(2) << simp_ms << " ms"
                  << std::setw(10) << ref_ms << " ms"
                  << std::setw(10) << std::setprecision(0) << ratio << "%"
                  << std::setw(12) << std::setprecision(1) << gops << " GOPS\n";
    }

    // Test FFN down: [M, 8192] @ [2048, 8192]_nt -> [M, 2048]
    std::cout << "\n=== FFN Down: [M, 8192] @ [2048, 8192]_nt -> [M, 2048] ===\n";
    std::cout << std::setw(8) << "M"
              << std::setw(12) << "SimpLang"
              << std::setw(12) << "C++ Ref"
              << std::setw(12) << "Ratio"
              << std::setw(15) << "GOPS" << "\n";

    TestCase ffn_down_tests[] = {
        {"prefill_ffn_down_64", 64, 2048, 8192, 64*8192, 2048*8192, 64*2048},
        {"prefill_ffn_down_128", 128, 2048, 8192, 128*8192, 2048*8192, 128*2048},
    };

    for (const auto& test : ffn_down_tests) {
        auto func = (PrefillFunc)dlsym(handle, test.func_name);
        if (!func) {
            std::cerr << "Warning: " << test.func_name << " not found\n";
            continue;
        }

        std::vector<int8_t> input(test.input_size);
        std::vector<int8_t> weight(test.weight_size);
        std::vector<int32_t> output(test.output_size);

        init_random_i8(input.data(), input.size(), 1);
        init_random_i8(weight.data(), weight.size(), 2);

        // Warmup
        func(PASS_MEMREF_I8(input.data(), input.size()),
             PASS_MEMREF_I8(weight.data(), weight.size()),
             PASS_MEMREF_I32(output.data(), output.size()));

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            func(PASS_MEMREF_I8(input.data(), input.size()),
                 PASS_MEMREF_I8(weight.data(), weight.size()),
                 PASS_MEMREF_I32(output.data(), output.size()));
        }
        auto end = std::chrono::high_resolution_clock::now();
        double simp_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

        double ref_ms = benchmark_ref(test.M, test.N, test.K, iterations);
        double ratio = ref_ms / simp_ms * 100.0;
        double gops = 2.0 * test.M * test.N * test.K / (simp_ms / 1000.0) / 1e9;

        std::cout << std::setw(8) << test.M
                  << std::setw(10) << std::fixed << std::setprecision(2) << simp_ms << " ms"
                  << std::setw(10) << ref_ms << " ms"
                  << std::setw(10) << std::setprecision(0) << ratio << "%"
                  << std::setw(12) << std::setprecision(1) << gops << " GOPS\n";
    }

    std::cout << "\n================================================================================\n";

    dlclose(handle);
    return 0;
}
