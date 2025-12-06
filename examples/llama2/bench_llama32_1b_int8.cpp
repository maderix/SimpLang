/**
 * LLaMA 3.2-1B INT8 Benchmark Runner
 * Tests forward pass performance with 4096 context
 */

#include <iostream>
#include <dlfcn.h>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <immintrin.h>

// LLaMA 3.2-1B config
constexpr int64_t DIM = 2048;
constexpr int64_t N_LAYERS = 16;
constexpr int64_t N_HEADS = 32;
constexpr int64_t N_KV_HEADS = 8;
constexpr int64_t HEAD_DIM = 64;
constexpr int64_t HIDDEN_DIM = 8192;
constexpr int64_t VOCAB_SIZE = 128256;
constexpr int64_t MAX_SEQ_LEN = 4096;
constexpr int64_t KV_DIM = N_KV_HEADS * HEAD_DIM;  // 512

// MLIR MemRef ABI
#define MEMREF_F32_PARAMS float*, float*, int64_t, int64_t, int64_t
#define MEMREF_I8_PARAMS int8_t*, int8_t*, int64_t, int64_t, int64_t
#define MEMREF_I32_PARAMS int32_t*, int32_t*, int64_t, int64_t, int64_t
#define PASS_MEMREF_F32(ptr, size) ptr, ptr, 0LL, (int64_t)(size), 1LL
#define PASS_MEMREF_I8(ptr, size) ptr, ptr, 0LL, (int64_t)(size), 1LL
#define PASS_MEMREF_I32(ptr, size) ptr, ptr, 0LL, (int64_t)(size), 1LL

typedef float (*LLaMA32ForwardFunc)(
    MEMREF_I8_PARAMS,   // x_i8
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
    MEMREF_F32_PARAMS,  // wcls
    MEMREF_F32_PARAMS,  // x_fp32
    MEMREF_F32_PARAMS,  // xb_fp32
    MEMREF_F32_PARAMS,  // logits
    MEMREF_I8_PARAMS,   // xb_i8
    MEMREF_I8_PARAMS,   // k_cache
    MEMREF_I8_PARAMS,   // v_cache
    MEMREF_F32_PARAMS,  // att_scores
    MEMREF_F32_PARAMS,  // att_probs
    MEMREF_I32_PARAMS,  // attn_out
    MEMREF_I8_PARAMS,   // ffn_hb
    int64_t,            // pos
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t  // config
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

size_t calculate_memory_mb() {
    size_t total = 0;

    // Weights
    total += N_LAYERS * DIM;           // rms_att_w (f32)
    total += N_LAYERS * DIM;           // rms_ffn_w (f32)
    total += DIM;                       // rms_final_w (f32)
    total += VOCAB_SIZE * DIM;          // wcls (f32)
    size_t fp32_weights = total * sizeof(float);

    // INT8 weights
    size_t i8_weights = 0;
    i8_weights += N_LAYERS * DIM * DIM;           // wq_t
    i8_weights += N_LAYERS * KV_DIM * DIM;        // wk_t
    i8_weights += N_LAYERS * KV_DIM * DIM;        // wv_t
    i8_weights += N_LAYERS * DIM * DIM;           // wo_t
    i8_weights += N_LAYERS * HIDDEN_DIM * DIM;    // w1_t
    i8_weights += N_LAYERS * DIM * HIDDEN_DIM;    // w2_t
    i8_weights += N_LAYERS * HIDDEN_DIM * DIM;    // w3_t

    // Activations
    size_t activations = 0;
    activations += DIM * 2 * sizeof(float);       // x_fp32, xb_fp32
    activations += VOCAB_SIZE * sizeof(float);    // logits
    activations += DIM;                            // xb_i8
    activations += N_HEADS * MAX_SEQ_LEN * 2 * sizeof(float);  // att_scores, att_probs
    activations += DIM * sizeof(int32_t);         // attn_out

    // KV cache
    size_t kv_cache = 2 * N_LAYERS * MAX_SEQ_LEN * KV_DIM;  // k_cache, v_cache (i8)

    return (fp32_weights + i8_weights + activations + kv_cache) / (1024 * 1024);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <llama32_1b_int8.so>" << std::endl;
        return 1;
    }

    void* handle = dlopen(argv[1], RTLD_NOW);
    if (!handle) {
        std::cerr << "Error: " << dlerror() << std::endl;
        return 1;
    }

    auto forward = (LLaMA32ForwardFunc)dlsym(handle, "llama32_forward");
    if (!forward) {
        std::cerr << "Error: Could not find llama32_forward: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }

    std::cout << "================================================================================\n";
    std::cout << "   LLaMA 3.2-1B INT8 Benchmark\n";
    std::cout << "   dim=" << DIM << ", layers=" << N_LAYERS << ", heads=" << N_HEADS;
    std::cout << ", kv_heads=" << N_KV_HEADS << ", hidden=" << HIDDEN_DIM << "\n";
    std::cout << "   vocab=" << VOCAB_SIZE << ", max_seq_len=" << MAX_SEQ_LEN << "\n";
    std::cout << "================================================================================\n\n";

    size_t mem_mb = calculate_memory_mb();
    std::cout << "Estimated memory: " << mem_mb << " MB" << std::endl;

    std::cout << "Allocating buffers..." << std::flush;

    try {
        // Input
        std::vector<int8_t> x_i8(DIM);

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

        // Classifier (FP32)
        std::vector<float> wcls(VOCAB_SIZE * DIM);

        // Activations
        std::vector<float> x_fp32(DIM);
        std::vector<float> xb_fp32(DIM);
        std::vector<float> logits(VOCAB_SIZE);
        std::vector<int8_t> xb_i8(DIM);
        std::vector<int32_t> attn_out(DIM);
        std::vector<float> att_scores(N_HEADS * MAX_SEQ_LEN);
        std::vector<float> att_probs(N_HEADS * MAX_SEQ_LEN);

        // KV Cache
        std::vector<int8_t> k_cache(N_LAYERS * MAX_SEQ_LEN * KV_DIM, 0);
        std::vector<int8_t> v_cache(N_LAYERS * MAX_SEQ_LEN * KV_DIM, 0);

        // FFN hidden buffer
        std::vector<int8_t> ffn_hb(HIDDEN_DIM);

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
        init_random_f32(wcls.data(), wcls.size(), 12);
        std::cout << " OK" << std::endl;

        // Warmup
        std::cout << "Warming up..." << std::flush;
        for (int i = 0; i < 2; i++) {
            forward(
                PASS_MEMREF_I8(x_i8.data(), x_i8.size()),
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
                PASS_MEMREF_F32(wcls.data(), wcls.size()),
                PASS_MEMREF_F32(x_fp32.data(), x_fp32.size()),
                PASS_MEMREF_F32(xb_fp32.data(), xb_fp32.size()),
                PASS_MEMREF_F32(logits.data(), logits.size()),
                PASS_MEMREF_I8(xb_i8.data(), xb_i8.size()),
                PASS_MEMREF_I8(k_cache.data(), k_cache.size()),
                PASS_MEMREF_I8(v_cache.data(), v_cache.size()),
                PASS_MEMREF_F32(att_scores.data(), att_scores.size()),
                PASS_MEMREF_F32(att_probs.data(), att_probs.size()),
                PASS_MEMREF_I32(attn_out.data(), attn_out.size()),
                PASS_MEMREF_I8(ffn_hb.data(), ffn_hb.size()),
                0,  // pos
                DIM, HIDDEN_DIM, N_LAYERS, N_HEADS, N_KV_HEADS, VOCAB_SIZE, MAX_SEQ_LEN, HEAD_DIM
            );
        }
        std::cout << " OK" << std::endl;

        // Benchmark decode at various positions
        std::cout << "\n=== Decode Performance ===" << std::endl;

        int positions[] = {0, 127, 511, 1023, 2047, 4095};
        for (int pos : positions) {
            if (pos >= MAX_SEQ_LEN) continue;

            // Clear KV cache up to pos
            std::fill(k_cache.begin(), k_cache.begin() + (pos + 1) * N_LAYERS * KV_DIM, 0);
            std::fill(v_cache.begin(), v_cache.begin() + (pos + 1) * N_LAYERS * KV_DIM, 0);

            auto start = std::chrono::high_resolution_clock::now();
            int iterations = 5;

            for (int i = 0; i < iterations; i++) {
                forward(
                    PASS_MEMREF_I8(x_i8.data(), x_i8.size()),
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
                    PASS_MEMREF_F32(wcls.data(), wcls.size()),
                    PASS_MEMREF_F32(x_fp32.data(), x_fp32.size()),
                    PASS_MEMREF_F32(xb_fp32.data(), xb_fp32.size()),
                    PASS_MEMREF_F32(logits.data(), logits.size()),
                    PASS_MEMREF_I8(xb_i8.data(), xb_i8.size()),
                    PASS_MEMREF_I8(k_cache.data(), k_cache.size()),
                    PASS_MEMREF_I8(v_cache.data(), v_cache.size()),
                    PASS_MEMREF_F32(att_scores.data(), att_scores.size()),
                    PASS_MEMREF_F32(att_probs.data(), att_probs.size()),
                    PASS_MEMREF_I32(attn_out.data(), attn_out.size()),
                    PASS_MEMREF_I8(ffn_hb.data(), ffn_hb.size()),
                    pos,
                    DIM, HIDDEN_DIM, N_LAYERS, N_HEADS, N_KV_HEADS, VOCAB_SIZE, MAX_SEQ_LEN, HEAD_DIM
                );
            }

            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
            double tokens_per_sec = 1000.0 / ms;

            std::cout << "pos=" << std::setw(4) << pos << " (seq=" << std::setw(4) << (pos + 1) << "): "
                      << std::fixed << std::setprecision(2) << ms << " ms/token, "
                      << std::setprecision(1) << tokens_per_sec << " tokens/sec" << std::endl;
        }

        // Decode loop simulation
        std::cout << "\n=== Decode Loop (64 tokens from pos=128) ===" << std::endl;
        std::fill(k_cache.begin(), k_cache.end(), 0);
        std::fill(v_cache.begin(), v_cache.end(), 0);

        int start_pos = 128;
        int num_tokens = 64;

        auto loop_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_tokens; i++) {
            forward(
                PASS_MEMREF_I8(x_i8.data(), x_i8.size()),
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
                PASS_MEMREF_F32(wcls.data(), wcls.size()),
                PASS_MEMREF_F32(x_fp32.data(), x_fp32.size()),
                PASS_MEMREF_F32(xb_fp32.data(), xb_fp32.size()),
                PASS_MEMREF_F32(logits.data(), logits.size()),
                PASS_MEMREF_I8(xb_i8.data(), xb_i8.size()),
                PASS_MEMREF_I8(k_cache.data(), k_cache.size()),
                PASS_MEMREF_I8(v_cache.data(), v_cache.size()),
                PASS_MEMREF_F32(att_scores.data(), att_scores.size()),
                PASS_MEMREF_F32(att_probs.data(), att_probs.size()),
                PASS_MEMREF_I32(attn_out.data(), attn_out.size()),
                PASS_MEMREF_I8(ffn_hb.data(), ffn_hb.size()),
                start_pos + i,
                DIM, HIDDEN_DIM, N_LAYERS, N_HEADS, N_KV_HEADS, VOCAB_SIZE, MAX_SEQ_LEN, HEAD_DIM
            );
        }
        auto loop_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(loop_end - loop_start).count();
        double tokens_per_sec = num_tokens / (total_ms / 1000.0);

        std::cout << "Generated " << num_tokens << " tokens in " << std::fixed << std::setprecision(1)
                  << total_ms << " ms" << std::endl;
        std::cout << "Throughput: " << tokens_per_sec << " tokens/sec" << std::endl;

        std::cout << "\n================================================================================\n";

    } catch (const std::bad_alloc& e) {
        std::cerr << "ERROR: Memory allocation failed!" << std::endl;
        return 1;
    }

    dlclose(handle);
    return 0;
}
