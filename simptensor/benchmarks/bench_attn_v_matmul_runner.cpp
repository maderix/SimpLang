/**
 * Attention × V MatMul Benchmark Runner for LLaMA 3.2 1B
 *
 * Tests INT8 Attn×V computation:
 *   Attn: [seq_len, seq_len] = [1024, 1024]
 *   V: [seq_len, head_dim] = [1024, 64]
 *   output: [seq_len, head_dim] = [1024, 64]
 *
 * Compile:
 *   g++ -O3 -march=native -mavx512vnni -o bench_attn_v_runner \
 *       bench_attn_v_matmul_runner.cpp -ldl
 */

#include <iostream>
#include <chrono>
#include <dlfcn.h>
#include <cmath>
#include <vector>
#include <iomanip>
#include <cstring>
#include <immintrin.h>

typedef int32_t (*KernelFunc)();

constexpr int SEQ_LEN = 1024;
constexpr int HEAD_DIM = 64;

// ============================================================================
// Reference Implementations
// ============================================================================

int32_t scalar_attn_v_1head() {
    std::vector<int8_t> Attn(SEQ_LEN * SEQ_LEN);
    std::vector<int8_t> V(SEQ_LEN * HEAD_DIM);
    std::vector<int32_t> output(SEQ_LEN * HEAD_DIM, 0);

    // Initialize
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < SEQ_LEN; j++) {
            int val = ((i * SEQ_LEN + j) % 127) - 64;
            Attn[i * SEQ_LEN + j] = (int8_t)val;
        }
    }
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < HEAD_DIM; j++) {
            int val = ((i * HEAD_DIM + j) % 127) - 64;
            V[i * HEAD_DIM + j] = (int8_t)val;
        }
    }

    // Attn × V: [1024, 1024] × [1024, 64] → [1024, 64]
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < HEAD_DIM; j++) {
            int32_t sum = 0;
            for (int k = 0; k < SEQ_LEN; k++) {
                sum += (int32_t)Attn[i * SEQ_LEN + k] * (int32_t)V[k * HEAD_DIM + j];
            }
            output[i * HEAD_DIM + j] = sum;
        }
    }

    int32_t checksum = 0;
    for (int i = 0; i < SEQ_LEN * HEAD_DIM; i++) {
        checksum += output[i];
    }
    return checksum;
}

int32_t vnni_attn_v_1head() {
    int8_t* Attn = (int8_t*)aligned_alloc(64, SEQ_LEN * SEQ_LEN);
    int8_t* V = (int8_t*)aligned_alloc(64, SEQ_LEN * HEAD_DIM);
    int8_t* V_T = (int8_t*)aligned_alloc(64, HEAD_DIM * SEQ_LEN); // Transposed V
    int32_t* output = (int32_t*)aligned_alloc(64, SEQ_LEN * HEAD_DIM * sizeof(int32_t));
    memset(output, 0, SEQ_LEN * HEAD_DIM * sizeof(int32_t));

    // Initialize
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < SEQ_LEN; j++) {
            int val = ((i * SEQ_LEN + j) % 127) - 64;
            Attn[i * SEQ_LEN + j] = (int8_t)val;
        }
    }
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < HEAD_DIM; j++) {
            int val = ((i * HEAD_DIM + j) % 127) - 64;
            V[i * HEAD_DIM + j] = (int8_t)val;
        }
    }

    // Transpose V for SIMD: V_T[j][k] = V[k][j]
    for (int j = 0; j < HEAD_DIM; j++) {
        for (int k = 0; k < SEQ_LEN; k++) {
            V_T[j * SEQ_LEN + k] = V[k * HEAD_DIM + j];
        }
    }

    const __m512i sign_flip = _mm512_set1_epi8((char)0x80);
    const __m512i ones = _mm512_set1_epi8(1);

    // I=4 tiled Attn×V using vpdpbusd
    // K dimension = SEQ_LEN = 1024, process 64 bytes at a time
    for (int i = 0; i < SEQ_LEN; i += 4) {
        for (int j = 0; j < HEAD_DIM; j++) {
            __m512i acc0 = _mm512_setzero_si512();
            __m512i acc1 = _mm512_setzero_si512();
            __m512i acc2 = _mm512_setzero_si512();
            __m512i acc3 = _mm512_setzero_si512();
            __m512i bias_acc = _mm512_setzero_si512();

            // Process K in chunks of 64
            for (int k = 0; k < SEQ_LEN; k += 64) {
                // Load V_T[j] row (shared across 4 Attn rows)
                __m512i vv = _mm512_loadu_si512(&V_T[j * SEQ_LEN + k]);

                // Load 4 Attn rows and XOR with 0x80
                __m512i va0 = _mm512_xor_si512(_mm512_loadu_si512(&Attn[(i+0) * SEQ_LEN + k]), sign_flip);
                __m512i va1 = _mm512_xor_si512(_mm512_loadu_si512(&Attn[(i+1) * SEQ_LEN + k]), sign_flip);
                __m512i va2 = _mm512_xor_si512(_mm512_loadu_si512(&Attn[(i+2) * SEQ_LEN + k]), sign_flip);
                __m512i va3 = _mm512_xor_si512(_mm512_loadu_si512(&Attn[(i+3) * SEQ_LEN + k]), sign_flip);

                acc0 = _mm512_dpbusd_epi32(acc0, va0, vv);
                acc1 = _mm512_dpbusd_epi32(acc1, va1, vv);
                acc2 = _mm512_dpbusd_epi32(acc2, va2, vv);
                acc3 = _mm512_dpbusd_epi32(acc3, va3, vv);
                bias_acc = _mm512_dpbusd_epi32(bias_acc, ones, vv);
            }

            int32_t sum0 = _mm512_reduce_add_epi32(acc0);
            int32_t sum1 = _mm512_reduce_add_epi32(acc1);
            int32_t sum2 = _mm512_reduce_add_epi32(acc2);
            int32_t sum3 = _mm512_reduce_add_epi32(acc3);
            int32_t bias = _mm512_reduce_add_epi32(bias_acc);
            int32_t correction = bias * 128;

            output[(i+0) * HEAD_DIM + j] = sum0 - correction;
            output[(i+1) * HEAD_DIM + j] = sum1 - correction;
            output[(i+2) * HEAD_DIM + j] = sum2 - correction;
            output[(i+3) * HEAD_DIM + j] = sum3 - correction;
        }
    }

    int32_t checksum = 0;
    for (int i = 0; i < SEQ_LEN * HEAD_DIM; i++) {
        checksum += output[i];
    }

    free(Attn);
    free(V);
    free(V_T);
    free(output);
    return checksum;
}

template<int NUM_HEADS>
int32_t scalar_attn_v_multihead() {
    std::vector<int8_t> Attn(NUM_HEADS * SEQ_LEN * SEQ_LEN);
    std::vector<int8_t> V(NUM_HEADS * SEQ_LEN * HEAD_DIM);
    std::vector<int32_t> output(NUM_HEADS * SEQ_LEN * HEAD_DIM, 0);

    for (int h = 0; h < NUM_HEADS; h++) {
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int j = 0; j < SEQ_LEN; j++) {
                int idx = h * SEQ_LEN * SEQ_LEN + i * SEQ_LEN + j;
                int val = (idx % 127) - 64;
                Attn[idx] = (int8_t)val;
            }
        }
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int j = 0; j < HEAD_DIM; j++) {
                int idx = h * SEQ_LEN * HEAD_DIM + i * HEAD_DIM + j;
                int val = (idx % 127) - 64;
                V[idx] = (int8_t)val;
            }
        }
    }

    for (int h = 0; h < NUM_HEADS; h++) {
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int j = 0; j < HEAD_DIM; j++) {
                int32_t sum = 0;
                for (int k = 0; k < SEQ_LEN; k++) {
                    int attn_idx = h * SEQ_LEN * SEQ_LEN + i * SEQ_LEN + k;
                    int v_idx = h * SEQ_LEN * HEAD_DIM + k * HEAD_DIM + j;
                    sum += (int32_t)Attn[attn_idx] * (int32_t)V[v_idx];
                }
                output[h * SEQ_LEN * HEAD_DIM + i * HEAD_DIM + j] = sum;
            }
        }
    }

    int32_t checksum = 0;
    for (size_t i = 0; i < output.size(); i++) {
        checksum += output[i];
    }
    return checksum;
}

template<int NUM_HEADS>
int32_t vnni_attn_v_multihead() {
    int8_t* Attn = (int8_t*)aligned_alloc(64, NUM_HEADS * SEQ_LEN * SEQ_LEN);
    int8_t* V = (int8_t*)aligned_alloc(64, NUM_HEADS * SEQ_LEN * HEAD_DIM);
    int8_t* V_T = (int8_t*)aligned_alloc(64, NUM_HEADS * HEAD_DIM * SEQ_LEN);
    int32_t* output = (int32_t*)aligned_alloc(64, NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(int32_t));
    memset(output, 0, NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(int32_t));

    for (int h = 0; h < NUM_HEADS; h++) {
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int j = 0; j < SEQ_LEN; j++) {
                int idx = h * SEQ_LEN * SEQ_LEN + i * SEQ_LEN + j;
                int val = (idx % 127) - 64;
                Attn[idx] = (int8_t)val;
            }
        }
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int j = 0; j < HEAD_DIM; j++) {
                int idx = h * SEQ_LEN * HEAD_DIM + i * HEAD_DIM + j;
                int val = (idx % 127) - 64;
                V[idx] = (int8_t)val;
            }
        }
        // Transpose V
        for (int j = 0; j < HEAD_DIM; j++) {
            for (int k = 0; k < SEQ_LEN; k++) {
                V_T[h * HEAD_DIM * SEQ_LEN + j * SEQ_LEN + k] = V[h * SEQ_LEN * HEAD_DIM + k * HEAD_DIM + j];
            }
        }
    }

    const __m512i sign_flip = _mm512_set1_epi8((char)0x80);
    const __m512i ones = _mm512_set1_epi8(1);

    for (int h = 0; h < NUM_HEADS; h++) {
        int8_t* Attn_h = Attn + h * SEQ_LEN * SEQ_LEN;
        int8_t* V_T_h = V_T + h * HEAD_DIM * SEQ_LEN;
        int32_t* out_h = output + h * SEQ_LEN * HEAD_DIM;

        for (int i = 0; i < SEQ_LEN; i += 4) {
            for (int j = 0; j < HEAD_DIM; j++) {
                __m512i acc0 = _mm512_setzero_si512();
                __m512i acc1 = _mm512_setzero_si512();
                __m512i acc2 = _mm512_setzero_si512();
                __m512i acc3 = _mm512_setzero_si512();
                __m512i bias_acc = _mm512_setzero_si512();

                for (int k = 0; k < SEQ_LEN; k += 64) {
                    __m512i vv = _mm512_loadu_si512(&V_T_h[j * SEQ_LEN + k]);
                    __m512i va0 = _mm512_xor_si512(_mm512_loadu_si512(&Attn_h[(i+0) * SEQ_LEN + k]), sign_flip);
                    __m512i va1 = _mm512_xor_si512(_mm512_loadu_si512(&Attn_h[(i+1) * SEQ_LEN + k]), sign_flip);
                    __m512i va2 = _mm512_xor_si512(_mm512_loadu_si512(&Attn_h[(i+2) * SEQ_LEN + k]), sign_flip);
                    __m512i va3 = _mm512_xor_si512(_mm512_loadu_si512(&Attn_h[(i+3) * SEQ_LEN + k]), sign_flip);

                    acc0 = _mm512_dpbusd_epi32(acc0, va0, vv);
                    acc1 = _mm512_dpbusd_epi32(acc1, va1, vv);
                    acc2 = _mm512_dpbusd_epi32(acc2, va2, vv);
                    acc3 = _mm512_dpbusd_epi32(acc3, va3, vv);
                    bias_acc = _mm512_dpbusd_epi32(bias_acc, ones, vv);
                }

                int32_t sum0 = _mm512_reduce_add_epi32(acc0);
                int32_t sum1 = _mm512_reduce_add_epi32(acc1);
                int32_t sum2 = _mm512_reduce_add_epi32(acc2);
                int32_t sum3 = _mm512_reduce_add_epi32(acc3);
                int32_t bias = _mm512_reduce_add_epi32(bias_acc);
                int32_t correction = bias * 128;

                out_h[(i+0) * HEAD_DIM + j] = sum0 - correction;
                out_h[(i+1) * HEAD_DIM + j] = sum1 - correction;
                out_h[(i+2) * HEAD_DIM + j] = sum2 - correction;
                out_h[(i+3) * HEAD_DIM + j] = sum3 - correction;
            }
        }
    }

    int32_t checksum = 0;
    for (int i = 0; i < NUM_HEADS * SEQ_LEN * HEAD_DIM; i++) {
        checksum += output[i];
    }

    free(Attn);
    free(V);
    free(V_T);
    free(output);
    return checksum;
}

// ============================================================================
// Benchmarking
// ============================================================================
volatile int32_t g_sink = 0;

template<typename T>
inline void DoNotOptimize(T const& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

template<typename Func>
double benchmark(Func func, int iterations) {
    int32_t result;
    for (int w = 0; w < 3; w++) {
        result = func();
        DoNotOptimize(result);
    }

    int32_t accumulator = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        result = func();
        DoNotOptimize(result);
        accumulator ^= result;
    }
    auto end = std::chrono::high_resolution_clock::now();
    g_sink = accumulator;

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return elapsed_ms / iterations;
}

int main(int argc, char* argv[]) {
    const char* so_path = "/tmp/bench_attn_v_matmul.so";
    if (argc > 1) so_path = argv[1];

    std::cout << "═══════════════════════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "   Attn×V MatMul Benchmark (LLaMA 3.2 1B Self-Attention)" << std::endl;
    std::cout << "   SEQ_LEN=" << SEQ_LEN << ", HEAD_DIM=" << HEAD_DIM << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════════════════════════" << std::endl;

    void* handle = dlopen(so_path, RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load " << so_path << ": " << dlerror() << std::endl;
        std::cerr << "Running reference benchmarks only..." << std::endl;
    }

    int32_t ref_1head = scalar_attn_v_1head();
    // Skip 4-head reference computation to avoid memory issues
    // int32_t ref_4head = scalar_attn_v_multihead<4>();

    std::cout << "\n=== Single Head (1024x1024 × 1024x64 → 1024x64) ===" << std::endl;
    std::cout << "Reference checksum: " << ref_1head << std::endl;

    double scalar_ms = benchmark(scalar_attn_v_1head, 5);
    double vnni_ms = benchmark(vnni_attn_v_1head, 5);

    // GIOP/s: 2 * SEQ_LEN * HEAD_DIM * SEQ_LEN ops
    double giops = 2.0 * SEQ_LEN * HEAD_DIM * SEQ_LEN / 1e9;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Scalar:    " << scalar_ms << " ms ("
              << std::setprecision(2) << giops / (scalar_ms / 1000.0) << " GIOP/s)" << std::endl;
    std::cout << "VNNI C++:  " << std::setprecision(3) << vnni_ms << " ms ("
              << std::setprecision(2) << giops / (vnni_ms / 1000.0) << " GIOP/s)" << std::endl;

    int32_t vnni_check = vnni_attn_v_1head();
    std::cout << "VNNI checksum: " << vnni_check << (vnni_check == ref_1head ? " ✓" : " ✗") << std::endl;

    if (handle) {
        KernelFunc sl_1head = (KernelFunc)dlsym(handle, "benchmark_attn_v_matmul_1head");
        if (sl_1head) {
            double sl_ms = benchmark(sl_1head, 5);
            int32_t sl_check = sl_1head();
            std::cout << "SimpLang:  " << std::setprecision(3) << sl_ms << " ms ("
                      << std::setprecision(2) << giops / (sl_ms / 1000.0) << " GIOP/s) "
                      << std::setprecision(1) << (vnni_ms / sl_ms * 100.0) << "% vs VNNI" << std::endl;
            std::cout << "SimpLang checksum: " << sl_check << (sl_check == ref_1head ? " ✓" : " (mismatch)") << std::endl;
        }
    }

    // 4-head benchmark
    std::cout << "\n=== 4 Heads (4× [1024×1024] × [1024×64] → [1024×64]) ===" << std::endl;
    int32_t ref_4head = scalar_attn_v_multihead<4>();
    std::cout << "Reference checksum: " << ref_4head << std::endl;

    double scalar_4h_ms = benchmark(scalar_attn_v_multihead<4>, 3);
    double vnni_4h_ms = benchmark(vnni_attn_v_multihead<4>, 3);
    double giops_4h = 4.0 * 2.0 * SEQ_LEN * HEAD_DIM * SEQ_LEN / 1e9;

    std::cout << "Scalar:    " << scalar_4h_ms << " ms ("
              << std::setprecision(2) << giops_4h / (scalar_4h_ms / 1000.0) << " GIOP/s)" << std::endl;
    std::cout << "VNNI C++:  " << std::setprecision(3) << vnni_4h_ms << " ms ("
              << std::setprecision(2) << giops_4h / (vnni_4h_ms / 1000.0) << " GIOP/s)" << std::endl;

    int32_t vnni_4h_check = vnni_attn_v_multihead<4>();
    std::cout << "VNNI checksum: " << vnni_4h_check << (vnni_4h_check == ref_4head ? " ✓" : " ✗") << std::endl;

    if (handle) {
        KernelFunc sl_4head = (KernelFunc)dlsym(handle, "benchmark_attn_v_matmul_4head");
        if (sl_4head) {
            double sl_4h_ms = benchmark(sl_4head, 3);
            int32_t sl_4h_check = sl_4head();
            std::cout << "SimpLang:  " << std::setprecision(3) << sl_4h_ms << " ms ("
                      << std::setprecision(2) << giops_4h / (sl_4h_ms / 1000.0) << " GIOP/s) "
                      << std::setprecision(1) << (vnni_4h_ms / sl_4h_ms * 100.0) << "% vs VNNI" << std::endl;
            std::cout << "SimpLang checksum: " << sl_4h_check << (sl_4h_check == ref_4head ? " ✓" : " (mismatch)") << std::endl;
        }
    }

    std::cout << "\n═══════════════════════════════════════════════════════════════════════════════" << std::endl;

    if (handle) dlclose(handle);
    return 0;
}
