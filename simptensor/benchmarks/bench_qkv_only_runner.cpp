// QKV Projection Benchmark Runner - All using pre-transposed weights
#include <iostream>
#include <chrono>
#include <dlfcn.h>
#include <cstring>
#include <immintrin.h>
#include <random>
#include <iomanip>

constexpr int DIM = 2048;
constexpr int WARMUP = 3;
constexpr int ITERATIONS = 10;

#define MEMREF_I8_PARAMS int8_t*, int8_t*, int64_t, int64_t, int64_t
#define PASS_MEMREF_I8(ptr, size) ptr, ptr, 0LL, (int64_t)(size), 1LL

typedef int32_t (*QM1Func)(MEMREF_I8_PARAMS, MEMREF_I8_PARAMS, int64_t);
typedef int32_t (*QBatchedFunc)(MEMREF_I8_PARAMS, MEMREF_I8_PARAMS, int64_t, int64_t);

alignas(64) int8_t g_x[DIM];
alignas(64) int8_t g_X_4[4 * DIM];
alignas(64) int8_t g_X_16[16 * DIM];
alignas(64) int8_t g_X_128[128 * DIM];
alignas(64) int8_t g_Wq_T[DIM * DIM];  // Pre-transposed weights [N, K]

void init_random_i8(int8_t* data, size_t size, int seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-64, 63);
    for (size_t i = 0; i < size; i++) {
        data[i] = (int8_t)dist(rng);
    }
}

void init_data() {
    init_random_i8(g_x, DIM, 1);
    init_random_i8(g_X_4, 4 * DIM, 2);
    init_random_i8(g_X_16, 16 * DIM, 3);
    init_random_i8(g_X_128, 128 * DIM, 4);
    init_random_i8(g_Wq_T, DIM * DIM, 5);  // Already in [N, K] layout
}

// C++ VNNI matmul with I=4 tiling (B_T is already transposed)
void vnni_matmul_i8(int8_t* A, int8_t* B_T, int32_t* C, int M, int N, int K) {
    const __m512i sign_flip = _mm512_set1_epi8((char)0x80);
    const __m512i ones = _mm512_set1_epi8(1);
    memset(C, 0, M * N * sizeof(int32_t));

    for (int i = 0; i < M; i += 4) {
        int i_end = std::min(i + 4, M);
        for (int j = 0; j < N; j++) {
            __m512i acc[4] = {_mm512_setzero_si512()};
            __m512i bias_acc = _mm512_setzero_si512();

            for (int k = 0; k < K; k += 64) {
                __m512i b = _mm512_loadu_si512(&B_T[j * K + k]);
                for (int ii = 0; ii < i_end - i; ii++) {
                    __m512i a = _mm512_xor_si512(
                        _mm512_loadu_si512(&A[(i + ii) * K + k]), sign_flip);
                    acc[ii] = _mm512_dpbusd_epi32(acc[ii], a, b);
                }
                bias_acc = _mm512_dpbusd_epi32(bias_acc, ones, b);
            }

            int32_t bias = _mm512_reduce_add_epi32(bias_acc) * 128;
            for (int ii = 0; ii < i_end - i; ii++) {
                C[(i + ii) * N + j] = _mm512_reduce_add_epi32(acc[ii]) - bias;
            }
        }
    }
}

int32_t vnni_q_m1() {
    static int32_t Q[DIM];
    vnni_matmul_i8(g_x, g_Wq_T, Q, 1, DIM, DIM);
    return Q[0] + Q[1023] + Q[2047];
}

int32_t vnni_q_m4() {
    static int32_t Q[4 * DIM];
    vnni_matmul_i8(g_X_4, g_Wq_T, Q, 4, DIM, DIM);
    return Q[0] + Q[1*DIM + 1023] + Q[3*DIM + 2047];
}

int32_t vnni_q_m16() {
    static int32_t Q[16 * DIM];
    vnni_matmul_i8(g_X_16, g_Wq_T, Q, 16, DIM, DIM);
    return Q[0] + Q[7*DIM + 1023] + Q[15*DIM + 2047];
}

int32_t vnni_q_m128() {
    static int32_t Q[128 * DIM];
    vnni_matmul_i8(g_X_128, g_Wq_T, Q, 128, DIM, DIM);
    return Q[0] + Q[63*DIM + 1023] + Q[127*DIM + 2047];
}

template<typename F>
double benchmark(F func, int iterations) {
    for (int i = 0; i < WARMUP; i++) func();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <kernel.so>" << std::endl;
        return 1;
    }

    void* handle = dlopen(argv[1], RTLD_NOW);
    if (!handle) {
        std::cerr << "Error: " << dlerror() << std::endl;
        return 1;
    }

    auto simp_m1 = (QM1Func)dlsym(handle, "bench_q_m1");
    auto simp_m4 = (QBatchedFunc)dlsym(handle, "bench_q_m4");
    auto simp_m16 = (QBatchedFunc)dlsym(handle, "bench_q_m16");
    auto simp_m128 = (QBatchedFunc)dlsym(handle, "bench_q_m128");

    std::cout << "Functions: m1=" << (simp_m1?"✓":"✗")
              << " m4=" << (simp_m4?"✓":"✗")
              << " m16=" << (simp_m16?"✓":"✗")
              << " m128=" << (simp_m128?"✓":"✗") << std::endl;

    init_data();

    std::cout << "\n========================================================\n";
    std::cout << "  Q Projection [M, 2048] @ [2048, 2048]_T -> [M, 2048]\n";
    std::cout << "  All using PRE-TRANSPOSED weights (fair comparison)\n";
    std::cout << "========================================================\n\n";

    // M=1
    if (simp_m1) {
        int64_t ops = 2LL * 1 * DIM * DIM;
        double vnni_ms = benchmark(vnni_q_m1, ITERATIONS);
        double vnni_gops = (ops / 1e9) / (vnni_ms / 1000.0);
        double simp_ms = benchmark([&]() {
            return simp_m1(PASS_MEMREF_I8(g_x, DIM), PASS_MEMREF_I8(g_Wq_T, DIM*DIM), DIM);
        }, ITERATIONS);
        double simp_gops = (ops / 1e9) / (simp_ms / 1000.0);
        std::cout << "M=1:   VNNI=" << std::fixed << std::setprecision(3) << vnni_ms
                  << "ms (" << std::setprecision(1) << vnni_gops << " GIOP/s)  "
                  << "SimpLang=" << std::setprecision(3) << simp_ms
                  << "ms (" << std::setprecision(1) << simp_gops << " GIOP/s)  "
                  << std::setprecision(1) << (simp_gops/vnni_gops*100) << "%\n";
    }

    // M=4
    if (simp_m4) {
        int64_t ops = 2LL * 4 * DIM * DIM;
        double vnni_ms = benchmark(vnni_q_m4, ITERATIONS);
        double vnni_gops = (ops / 1e9) / (vnni_ms / 1000.0);
        double simp_ms = benchmark([&]() {
            return simp_m4(PASS_MEMREF_I8(g_X_4, 4*DIM), PASS_MEMREF_I8(g_Wq_T, DIM*DIM), 4, DIM);
        }, ITERATIONS);
        double simp_gops = (ops / 1e9) / (simp_ms / 1000.0);
        std::cout << "M=4:   VNNI=" << std::fixed << std::setprecision(3) << vnni_ms
                  << "ms (" << std::setprecision(1) << vnni_gops << " GIOP/s)  "
                  << "SimpLang=" << std::setprecision(3) << simp_ms
                  << "ms (" << std::setprecision(1) << simp_gops << " GIOP/s)  "
                  << std::setprecision(1) << (simp_gops/vnni_gops*100) << "%\n";
    }

    // M=16
    if (simp_m16) {
        int64_t ops = 2LL * 16 * DIM * DIM;
        double vnni_ms = benchmark(vnni_q_m16, ITERATIONS);
        double vnni_gops = (ops / 1e9) / (vnni_ms / 1000.0);
        double simp_ms = benchmark([&]() {
            return simp_m16(PASS_MEMREF_I8(g_X_16, 16*DIM), PASS_MEMREF_I8(g_Wq_T, DIM*DIM), 16, DIM);
        }, ITERATIONS);
        double simp_gops = (ops / 1e9) / (simp_ms / 1000.0);
        std::cout << "M=16:  VNNI=" << std::fixed << std::setprecision(3) << vnni_ms
                  << "ms (" << std::setprecision(1) << vnni_gops << " GIOP/s)  "
                  << "SimpLang=" << std::setprecision(3) << simp_ms
                  << "ms (" << std::setprecision(1) << simp_gops << " GIOP/s)  "
                  << std::setprecision(1) << (simp_gops/vnni_gops*100) << "%\n";
    }

    // M=128
    if (simp_m128) {
        int64_t ops = 2LL * 128 * DIM * DIM;
        double vnni_ms = benchmark(vnni_q_m128, ITERATIONS);
        double vnni_gops = (ops / 1e9) / (vnni_ms / 1000.0);
        double simp_ms = benchmark([&]() {
            return simp_m128(PASS_MEMREF_I8(g_X_128, 128*DIM), PASS_MEMREF_I8(g_Wq_T, DIM*DIM), 128, DIM);
        }, ITERATIONS);
        double simp_gops = (ops / 1e9) / (simp_ms / 1000.0);
        std::cout << "M=128: VNNI=" << std::fixed << std::setprecision(3) << vnni_ms
                  << "ms (" << std::setprecision(1) << vnni_gops << " GIOP/s)  "
                  << "SimpLang=" << std::setprecision(3) << simp_ms
                  << "ms (" << std::setprecision(1) << simp_gops << " GIOP/s)  "
                  << std::setprecision(1) << (simp_gops/vnni_gops*100) << "%\n";
    }

    std::cout << "\n========================================================\n";

    dlclose(handle);
    return 0;
}
