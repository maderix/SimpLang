/**
 * LLaMA 3.2-1B INT8 Prefill Benchmark Runner
 * Tests prefill performance with various sequence lengths: 8, 16, 32, 64, 128, 256
 */

#include <iostream>
#include <dlfcn.h>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cstring>
#include <cmath>

// LLaMA 3.2-1B config
constexpr int64_t DIM = 2048;
constexpr int64_t N_LAYERS = 16;
constexpr int64_t N_HEADS = 32;
constexpr int64_t N_KV_HEADS = 8;
constexpr int64_t HEAD_DIM = 64;
constexpr int64_t HIDDEN_DIM = 8192;
constexpr int64_t KV_DIM = N_KV_HEADS * HEAD_DIM;  // 512
constexpr int64_t MAX_SEQ_LEN = 256;  // Max for prefill

// MLIR MemRef ABI
#define MEMREF_F32_PARAMS float*, float*, int64_t, int64_t, int64_t
#define MEMREF_I8_PARAMS int8_t*, int8_t*, int64_t, int64_t, int64_t
#define MEMREF_I32_PARAMS int32_t*, int32_t*, int64_t, int64_t, int64_t
#define PASS_MEMREF_F32(ptr, size) ptr, ptr, 0LL, (int64_t)(size), 1LL
#define PASS_MEMREF_I8(ptr, size) ptr, ptr, 0LL, (int64_t)(size), 1LL
#define PASS_MEMREF_I32(ptr, size) ptr, ptr, 0LL, (int64_t)(size), 1LL

typedef float (*LLaMA32PrefillFunc)(
    MEMREF_I8_PARAMS,   // x_i8 [seq_len * dim]
    MEMREF_F32_PARAMS,  // rms_att_w
    MEMREF_F32_PARAMS,  // rms_ffn_w
    MEMREF_F32_PARAMS,  // rms_final_w
    MEMREF_I8_PARAMS,   // wq_t
    MEMREF_I8_PARAMS,   // wk_t
    MEMREF_I8_PARAMS,   // wv_t
    MEMREF_I8_PARAMS,   // wo_t
    MEMREF_I8_PARAMS,   // w1_t
    MEMREF_I8_PARAMS,   // w2_t
    MEMREF_I8_PARAMS,   // w3_t
    MEMREF_F32_PARAMS,  // x_fp32 [seq_len * dim]
    MEMREF_F32_PARAMS,  // xb_fp32 [seq_len * dim]
    MEMREF_I8_PARAMS,   // xb_i8 [seq_len * dim]
    MEMREF_I8_PARAMS,   // k_cache
    MEMREF_I8_PARAMS,   // v_cache
    MEMREF_F32_PARAMS,  // att_scores [n_heads * seq_len * seq_len]
    MEMREF_F32_PARAMS,  // att_probs [n_heads * seq_len * seq_len]
    MEMREF_I32_PARAMS,  // attn_out [seq_len * dim]
    MEMREF_I8_PARAMS,   // ffn_hb [seq_len * hidden_dim]
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t  // config
);

void init_random_i8(int8_t* data, size_t size, int seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-64, 63);
    for (size_t i = 0; i < size; i++) {
        data[i] = (int8_t)dist(rng);
    }
}

void init_random_f32(float* data, size_t size, int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.01f, 0.01f);
    for (size_t i = 0; i < size; i++) {
        data[i] = dist(rng);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <llama32_1b_int8_prefill.so>" << std::endl;
        return 1;
    }

    void* handle = dlopen(argv[1], RTLD_NOW);
    if (!handle) {
        std::cerr << "Error: " << dlerror() << std::endl;
        return 1;
    }

    // Load all prefill variants
    struct PrefillVariant {
        const char* name;
        int seq_len;
        LLaMA32PrefillFunc func;
    };

    PrefillVariant variants[] = {
        {"llama32_prefill_8", 8, nullptr},
        {"llama32_prefill_16", 16, nullptr},
        {"llama32_prefill_32", 32, nullptr},
        {"llama32_prefill_64", 64, nullptr},
        {"llama32_prefill_128", 128, nullptr},
        {"llama32_prefill_256", 256, nullptr},
    };

    for (auto& v : variants) {
        v.func = (LLaMA32PrefillFunc)dlsym(handle, v.name);
        if (!v.func) {
            std::cerr << "Warning: " << v.name << " not found\n";
        }
    }

    std::cout << "================================================================================\n";
    std::cout << "   LLaMA 3.2-1B INT8 Prefill Benchmark\n";
    std::cout << "   dim=" << DIM << ", layers=" << N_LAYERS << ", heads=" << N_HEADS;
    std::cout << ", kv_heads=" << N_KV_HEADS << ", hidden=" << HIDDEN_DIM << "\n";
    std::cout << "================================================================================\n\n";

    std::cout << "Allocating buffers..." << std::flush;

    try {
        // Allocate for max seq_len
        std::vector<int8_t> x_i8(MAX_SEQ_LEN * DIM);

        // RMSNorm weights (FP32)
        std::vector<float> rms_att_w(N_LAYERS * DIM);
        std::vector<float> rms_ffn_w(N_LAYERS * DIM);
        std::vector<float> rms_final_w(DIM);

        // INT8 weights (pre-transposed)
        std::vector<int8_t> wq_t(N_LAYERS * DIM * DIM);
        std::vector<int8_t> wk_t(N_LAYERS * KV_DIM * DIM);
        std::vector<int8_t> wv_t(N_LAYERS * KV_DIM * DIM);
        std::vector<int8_t> wo_t(N_LAYERS * DIM * DIM);
        std::vector<int8_t> w1_t(N_LAYERS * HIDDEN_DIM * DIM);
        std::vector<int8_t> w2_t(N_LAYERS * DIM * HIDDEN_DIM);
        std::vector<int8_t> w3_t(N_LAYERS * HIDDEN_DIM * DIM);

        // Activations [max_seq_len, dim]
        std::vector<float> x_fp32(MAX_SEQ_LEN * DIM);
        std::vector<float> xb_fp32(MAX_SEQ_LEN * DIM);
        std::vector<int8_t> xb_i8(MAX_SEQ_LEN * DIM);
        std::vector<int32_t> attn_out(MAX_SEQ_LEN * DIM);

        // Attention scores [n_heads, max_seq_len, max_seq_len]
        std::vector<float> att_scores(N_HEADS * MAX_SEQ_LEN * MAX_SEQ_LEN);
        std::vector<float> att_probs(N_HEADS * MAX_SEQ_LEN * MAX_SEQ_LEN);

        // KV Cache [n_layers, max_seq_len, kv_dim]
        std::vector<int8_t> k_cache(N_LAYERS * MAX_SEQ_LEN * KV_DIM, 0);
        std::vector<int8_t> v_cache(N_LAYERS * MAX_SEQ_LEN * KV_DIM, 0);

        // FFN hidden buffer [max_seq_len, hidden_dim]
        std::vector<int8_t> ffn_hb(MAX_SEQ_LEN * HIDDEN_DIM);

        std::cout << " OK" << std::endl;

        // Initialize with random data
        std::cout << "Initializing weights..." << std::flush;
        init_random_i8(x_i8.data(), x_i8.size(), 1);
        init_random_f32(rms_att_w.data(), rms_att_w.size(), 2);
        init_random_f32(rms_ffn_w.data(), rms_ffn_w.size(), 3);
        init_random_f32(rms_final_w.data(), rms_final_w.size(), 4);
        init_random_i8(wq_t.data(), wq_t.size(), 5);
        init_random_i8(wk_t.data(), wk_t.size(), 6);
        init_random_i8(wv_t.data(), wv_t.size(), 7);
        init_random_i8(wo_t.data(), wo_t.size(), 8);
        init_random_i8(w1_t.data(), w1_t.size(), 9);
        init_random_i8(w2_t.data(), w2_t.size(), 10);
        init_random_i8(w3_t.data(), w3_t.size(), 11);
        std::cout << " OK" << std::endl;

        std::cout << "\n=== Prefill Performance ===" << std::endl;
        std::cout << std::setw(10) << "SeqLen"
                  << std::setw(15) << "Total (ms)"
                  << std::setw(15) << "ms/token"
                  << std::setw(15) << "tokens/sec"
                  << std::setw(15) << "vs Decode" << std::endl;
        std::cout << std::string(70, '-') << std::endl;

        // Decode reference: ~160ms per token at short context
        const double decode_ms_per_token = 160.0;

        for (const auto& v : variants) {
            if (!v.func) continue;

            int64_t seq_len = v.seq_len;
            const int iterations = 3;

            // Warmup
            std::fill(k_cache.begin(), k_cache.end(), 0);
            std::fill(v_cache.begin(), v_cache.end(), 0);

            v.func(
                PASS_MEMREF_I8(x_i8.data(), seq_len * DIM),
                PASS_MEMREF_F32(rms_att_w.data(), rms_att_w.size()),
                PASS_MEMREF_F32(rms_ffn_w.data(), rms_ffn_w.size()),
                PASS_MEMREF_F32(rms_final_w.data(), rms_final_w.size()),
                PASS_MEMREF_I8(wq_t.data(), wq_t.size()),
                PASS_MEMREF_I8(wk_t.data(), wk_t.size()),
                PASS_MEMREF_I8(wv_t.data(), wv_t.size()),
                PASS_MEMREF_I8(wo_t.data(), wo_t.size()),
                PASS_MEMREF_I8(w1_t.data(), w1_t.size()),
                PASS_MEMREF_I8(w2_t.data(), w2_t.size()),
                PASS_MEMREF_I8(w3_t.data(), w3_t.size()),
                PASS_MEMREF_F32(x_fp32.data(), seq_len * DIM),
                PASS_MEMREF_F32(xb_fp32.data(), seq_len * DIM),
                PASS_MEMREF_I8(xb_i8.data(), seq_len * DIM),
                PASS_MEMREF_I8(k_cache.data(), N_LAYERS * seq_len * KV_DIM),
                PASS_MEMREF_I8(v_cache.data(), N_LAYERS * seq_len * KV_DIM),
                PASS_MEMREF_F32(att_scores.data(), N_HEADS * seq_len * seq_len),
                PASS_MEMREF_F32(att_probs.data(), N_HEADS * seq_len * seq_len),
                PASS_MEMREF_I32(attn_out.data(), seq_len * DIM),
                PASS_MEMREF_I8(ffn_hb.data(), seq_len * HIDDEN_DIM),
                seq_len, DIM, HIDDEN_DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HEAD_DIM
            );

            // Benchmark
            auto start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < iterations; i++) {
                std::fill(k_cache.begin(), k_cache.begin() + N_LAYERS * seq_len * KV_DIM, 0);
                std::fill(v_cache.begin(), v_cache.begin() + N_LAYERS * seq_len * KV_DIM, 0);

                v.func(
                    PASS_MEMREF_I8(x_i8.data(), seq_len * DIM),
                    PASS_MEMREF_F32(rms_att_w.data(), rms_att_w.size()),
                    PASS_MEMREF_F32(rms_ffn_w.data(), rms_ffn_w.size()),
                    PASS_MEMREF_F32(rms_final_w.data(), rms_final_w.size()),
                    PASS_MEMREF_I8(wq_t.data(), wq_t.size()),
                    PASS_MEMREF_I8(wk_t.data(), wk_t.size()),
                    PASS_MEMREF_I8(wv_t.data(), wv_t.size()),
                    PASS_MEMREF_I8(wo_t.data(), wo_t.size()),
                    PASS_MEMREF_I8(w1_t.data(), w1_t.size()),
                    PASS_MEMREF_I8(w2_t.data(), w2_t.size()),
                    PASS_MEMREF_I8(w3_t.data(), w3_t.size()),
                    PASS_MEMREF_F32(x_fp32.data(), seq_len * DIM),
                    PASS_MEMREF_F32(xb_fp32.data(), seq_len * DIM),
                    PASS_MEMREF_I8(xb_i8.data(), seq_len * DIM),
                    PASS_MEMREF_I8(k_cache.data(), N_LAYERS * seq_len * KV_DIM),
                    PASS_MEMREF_I8(v_cache.data(), N_LAYERS * seq_len * KV_DIM),
                    PASS_MEMREF_F32(att_scores.data(), N_HEADS * seq_len * seq_len),
                    PASS_MEMREF_F32(att_probs.data(), N_HEADS * seq_len * seq_len),
                    PASS_MEMREF_I32(attn_out.data(), seq_len * DIM),
                    PASS_MEMREF_I8(ffn_hb.data(), seq_len * HIDDEN_DIM),
                    seq_len, DIM, HIDDEN_DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HEAD_DIM
                );
            }

            auto end = std::chrono::high_resolution_clock::now();
            double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
            double avg_ms = total_ms / iterations;
            double ms_per_token = avg_ms / seq_len;
            double tokens_per_sec = seq_len / (avg_ms / 1000.0);

            // Compare to decode: decode would take seq_len * decode_ms_per_token
            double decode_equivalent = seq_len * decode_ms_per_token;
            double speedup = decode_equivalent / avg_ms;

            std::cout << std::setw(10) << seq_len
                      << std::setw(12) << std::fixed << std::setprecision(1) << avg_ms << " ms"
                      << std::setw(12) << std::setprecision(2) << ms_per_token << " ms"
                      << std::setw(12) << std::setprecision(0) << tokens_per_sec
                      << std::setw(12) << std::setprecision(1) << speedup << "x" << std::endl;
        }

        std::cout << "\n================================================================================\n";

    } catch (const std::bad_alloc& e) {
        std::cerr << "ERROR: Memory allocation failed!" << std::endl;
        return 1;
    }

    dlclose(handle);
    return 0;
}
