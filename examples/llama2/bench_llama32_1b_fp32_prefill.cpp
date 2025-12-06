/**
 * LLaMA 3.2-1B FP32 Prefill Benchmark Runner
 * Tests prefill performance with various sequence lengths: 8, 16, 32, 64, 128
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
constexpr int64_t MAX_SEQ_LEN = 128;  // Max for prefill

// MLIR MemRef ABI
#define MEMREF_F32_PARAMS float*, float*, int64_t, int64_t, int64_t
#define PASS_MEMREF_F32(ptr, size) ptr, ptr, 0LL, (int64_t)(size), 1LL

typedef float (*LLaMA32FP32PrefillFunc)(
    MEMREF_F32_PARAMS,  // x [seq_len * dim]
    MEMREF_F32_PARAMS,  // rms_att_w
    MEMREF_F32_PARAMS,  // rms_ffn_w
    MEMREF_F32_PARAMS,  // rms_final_w
    MEMREF_F32_PARAMS,  // wq
    MEMREF_F32_PARAMS,  // wk
    MEMREF_F32_PARAMS,  // wv
    MEMREF_F32_PARAMS,  // wo
    MEMREF_F32_PARAMS,  // w1
    MEMREF_F32_PARAMS,  // w2
    MEMREF_F32_PARAMS,  // w3
    MEMREF_F32_PARAMS,  // xb [seq_len * dim]
    MEMREF_F32_PARAMS,  // q_buf
    MEMREF_F32_PARAMS,  // k_buf
    MEMREF_F32_PARAMS,  // v_buf
    MEMREF_F32_PARAMS,  // k_cache
    MEMREF_F32_PARAMS,  // v_cache
    MEMREF_F32_PARAMS,  // att_scores [n_heads * seq_len * seq_len]
    MEMREF_F32_PARAMS,  // att_probs [n_heads * seq_len * seq_len]
    MEMREF_F32_PARAMS,  // attn_out [seq_len * dim]
    MEMREF_F32_PARAMS,  // hb [seq_len * hidden_dim]
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t  // config
);

void init_random_f32(float* data, size_t size, int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.01f, 0.01f);
    for (size_t i = 0; i < size; i++) {
        data[i] = dist(rng);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <llama32_1b_fp32_prefill.so>" << std::endl;
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
        LLaMA32FP32PrefillFunc func;
    };

    PrefillVariant variants[] = {
        {"llama32_fp32_prefill_8", 8, nullptr},
        {"llama32_fp32_prefill_16", 16, nullptr},
        {"llama32_fp32_prefill_32", 32, nullptr},
        {"llama32_fp32_prefill_64", 64, nullptr},
        {"llama32_fp32_prefill_128", 128, nullptr},
    };

    for (auto& v : variants) {
        v.func = (LLaMA32FP32PrefillFunc)dlsym(handle, v.name);
        if (!v.func) {
            std::cerr << "Warning: " << v.name << " not found\n";
        }
    }

    std::cout << "================================================================================\n";
    std::cout << "   LLaMA 3.2-1B FP32 Prefill Benchmark\n";
    std::cout << "   dim=" << DIM << ", layers=" << N_LAYERS << ", heads=" << N_HEADS;
    std::cout << ", kv_heads=" << N_KV_HEADS << ", hidden=" << HIDDEN_DIM << "\n";
    std::cout << "================================================================================\n\n";

    std::cout << "Allocating buffers..." << std::flush;

    try {
        // Allocate for max seq_len
        std::vector<float> x(MAX_SEQ_LEN * DIM);

        // RMSNorm weights (FP32)
        std::vector<float> rms_att_w(N_LAYERS * DIM);
        std::vector<float> rms_ffn_w(N_LAYERS * DIM);
        std::vector<float> rms_final_w(DIM);

        // FP32 weights
        std::vector<float> wq(N_LAYERS * DIM * DIM);
        std::vector<float> wk(N_LAYERS * KV_DIM * DIM);
        std::vector<float> wv(N_LAYERS * KV_DIM * DIM);
        std::vector<float> wo(N_LAYERS * DIM * DIM);
        std::vector<float> w1(N_LAYERS * HIDDEN_DIM * DIM);
        std::vector<float> w2(N_LAYERS * DIM * HIDDEN_DIM);
        std::vector<float> w3(N_LAYERS * HIDDEN_DIM * DIM);

        // Activations [max_seq_len, dim]
        std::vector<float> xb(MAX_SEQ_LEN * DIM);
        std::vector<float> q_buf(MAX_SEQ_LEN * DIM);
        std::vector<float> k_buf(MAX_SEQ_LEN * KV_DIM);
        std::vector<float> v_buf(MAX_SEQ_LEN * KV_DIM);
        std::vector<float> attn_out(MAX_SEQ_LEN * DIM);

        // Attention scores [n_heads, max_seq_len, max_seq_len]
        std::vector<float> att_scores(N_HEADS * MAX_SEQ_LEN * MAX_SEQ_LEN);
        std::vector<float> att_probs(N_HEADS * MAX_SEQ_LEN * MAX_SEQ_LEN);

        // KV Cache [n_layers, max_seq_len, kv_dim]
        std::vector<float> k_cache(N_LAYERS * MAX_SEQ_LEN * KV_DIM, 0);
        std::vector<float> v_cache(N_LAYERS * MAX_SEQ_LEN * KV_DIM, 0);

        // FFN hidden buffer [max_seq_len, hidden_dim]
        std::vector<float> hb(MAX_SEQ_LEN * HIDDEN_DIM);

        std::cout << " OK" << std::endl;

        // Initialize with random data
        std::cout << "Initializing weights..." << std::flush;
        init_random_f32(x.data(), x.size(), 1);
        init_random_f32(rms_att_w.data(), rms_att_w.size(), 2);
        init_random_f32(rms_ffn_w.data(), rms_ffn_w.size(), 3);
        init_random_f32(rms_final_w.data(), rms_final_w.size(), 4);
        init_random_f32(wq.data(), wq.size(), 5);
        init_random_f32(wk.data(), wk.size(), 6);
        init_random_f32(wv.data(), wv.size(), 7);
        init_random_f32(wo.data(), wo.size(), 8);
        init_random_f32(w1.data(), w1.size(), 9);
        init_random_f32(w2.data(), w2.size(), 10);
        init_random_f32(w3.data(), w3.size(), 11);
        std::cout << " OK" << std::endl;

        std::cout << "\n=== FP32 Prefill Performance ===\n";
        std::cout << std::setw(10) << "SeqLen"
                  << std::setw(15) << "Total (ms)"
                  << std::setw(15) << "ms/token"
                  << std::setw(15) << "tokens/sec"
                  << std::setw(15) << "vs Decode" << std::endl;
        std::cout << std::string(70, '-') << std::endl;

        // Decode reference: ~160ms per token for FP32 at short context
        const double decode_ms_per_token = 160.0;

        for (const auto& v : variants) {
            if (!v.func) continue;

            int64_t seq_len = v.seq_len;
            const int iterations = 3;

            // Reset x buffer and caches
            init_random_f32(x.data(), seq_len * DIM, 1);
            std::fill(k_cache.begin(), k_cache.end(), 0.0f);
            std::fill(v_cache.begin(), v_cache.end(), 0.0f);

            // Warmup
            v.func(
                PASS_MEMREF_F32(x.data(), seq_len * DIM),
                PASS_MEMREF_F32(rms_att_w.data(), rms_att_w.size()),
                PASS_MEMREF_F32(rms_ffn_w.data(), rms_ffn_w.size()),
                PASS_MEMREF_F32(rms_final_w.data(), rms_final_w.size()),
                PASS_MEMREF_F32(wq.data(), wq.size()),
                PASS_MEMREF_F32(wk.data(), wk.size()),
                PASS_MEMREF_F32(wv.data(), wv.size()),
                PASS_MEMREF_F32(wo.data(), wo.size()),
                PASS_MEMREF_F32(w1.data(), w1.size()),
                PASS_MEMREF_F32(w2.data(), w2.size()),
                PASS_MEMREF_F32(w3.data(), w3.size()),
                PASS_MEMREF_F32(xb.data(), seq_len * DIM),
                PASS_MEMREF_F32(q_buf.data(), seq_len * DIM),
                PASS_MEMREF_F32(k_buf.data(), seq_len * KV_DIM),
                PASS_MEMREF_F32(v_buf.data(), seq_len * KV_DIM),
                PASS_MEMREF_F32(k_cache.data(), N_LAYERS * seq_len * KV_DIM),
                PASS_MEMREF_F32(v_cache.data(), N_LAYERS * seq_len * KV_DIM),
                PASS_MEMREF_F32(att_scores.data(), N_HEADS * seq_len * seq_len),
                PASS_MEMREF_F32(att_probs.data(), N_HEADS * seq_len * seq_len),
                PASS_MEMREF_F32(attn_out.data(), seq_len * DIM),
                PASS_MEMREF_F32(hb.data(), seq_len * HIDDEN_DIM),
                seq_len, DIM, HIDDEN_DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HEAD_DIM
            );

            // Benchmark
            auto start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < iterations; i++) {
                init_random_f32(x.data(), seq_len * DIM, 1);
                std::fill(k_cache.begin(), k_cache.begin() + N_LAYERS * seq_len * KV_DIM, 0.0f);
                std::fill(v_cache.begin(), v_cache.begin() + N_LAYERS * seq_len * KV_DIM, 0.0f);

                v.func(
                    PASS_MEMREF_F32(x.data(), seq_len * DIM),
                    PASS_MEMREF_F32(rms_att_w.data(), rms_att_w.size()),
                    PASS_MEMREF_F32(rms_ffn_w.data(), rms_ffn_w.size()),
                    PASS_MEMREF_F32(rms_final_w.data(), rms_final_w.size()),
                    PASS_MEMREF_F32(wq.data(), wq.size()),
                    PASS_MEMREF_F32(wk.data(), wk.size()),
                    PASS_MEMREF_F32(wv.data(), wv.size()),
                    PASS_MEMREF_F32(wo.data(), wo.size()),
                    PASS_MEMREF_F32(w1.data(), w1.size()),
                    PASS_MEMREF_F32(w2.data(), w2.size()),
                    PASS_MEMREF_F32(w3.data(), w3.size()),
                    PASS_MEMREF_F32(xb.data(), seq_len * DIM),
                    PASS_MEMREF_F32(q_buf.data(), seq_len * DIM),
                    PASS_MEMREF_F32(k_buf.data(), seq_len * KV_DIM),
                    PASS_MEMREF_F32(v_buf.data(), seq_len * KV_DIM),
                    PASS_MEMREF_F32(k_cache.data(), N_LAYERS * seq_len * KV_DIM),
                    PASS_MEMREF_F32(v_cache.data(), N_LAYERS * seq_len * KV_DIM),
                    PASS_MEMREF_F32(att_scores.data(), N_HEADS * seq_len * seq_len),
                    PASS_MEMREF_F32(att_probs.data(), N_HEADS * seq_len * seq_len),
                    PASS_MEMREF_F32(attn_out.data(), seq_len * DIM),
                    PASS_MEMREF_F32(hb.data(), seq_len * HIDDEN_DIM),
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
