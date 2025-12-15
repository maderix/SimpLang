/**
 * LLaMA 3.2-1B INT8 Benchmark Runner
 * Matches llama32_1b_forward() signature in llama32_1b_int8.sl
 *
 * Build:
 *   g++ -O3 -march=native bench_llama32_1b_int8_runner.cpp -o bench_llama32_1b -ldl
 *
 * Run:
 *   ./bench_llama32_1b /tmp/llama32_1b_int8.so
 */

#include <iostream>
#include <iomanip>
#include <dlfcn.h>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <cmath>

// Model config for LLaMA 3.2-1B
constexpr int64_t DIM = 2048;
constexpr int64_t HIDDEN_DIM = 8192;
constexpr int64_t N_LAYERS = 16;
constexpr int64_t N_HEADS = 32;
constexpr int64_t N_KV_HEADS = 8;
constexpr int64_t VOCAB_SIZE = 128256;
constexpr int64_t MAX_SEQ_LEN = 4096;
constexpr int64_t HEAD_DIM = 64;
constexpr int64_t KV_DIM = N_KV_HEADS * HEAD_DIM;  // 512

// MLIR memref descriptor macros
#define MEMREF_F32_PARAMS float*, float*, int64_t, int64_t, int64_t
#define MEMREF_I8_PARAMS int8_t*, int8_t*, int64_t, int64_t, int64_t
#define MEMREF_I32_PARAMS int32_t*, int32_t*, int64_t, int64_t, int64_t

#define MEMREF_F32(ptr, size) (ptr), (ptr), 0, (size), 1
#define MEMREF_I8(ptr, size) (ptr), (ptr), 0, (size), 1
#define MEMREF_I32(ptr, size) (ptr), (ptr), 0, (size), 1

// Function signature matching llama32_1b_forward
typedef float (*LLaMA32ForwardFunc)(
    MEMREF_F32_PARAMS,  // token_embedding [vocab_size, dim]
    MEMREF_F32_PARAMS,  // rms_att_w [n_layers, dim]
    MEMREF_F32_PARAMS,  // rms_ffn_w [n_layers, dim]
    MEMREF_F32_PARAMS,  // rms_final_w [dim]
    MEMREF_I8_PARAMS,   // wq_t [n_layers, dim, dim]
    MEMREF_I8_PARAMS,   // wk_t [n_layers, kv_dim, dim]
    MEMREF_I8_PARAMS,   // wv_t [n_layers, kv_dim, dim]
    MEMREF_I8_PARAMS,   // wo_t [n_layers, dim, dim]
    MEMREF_I8_PARAMS,   // w1_t [n_layers, hidden_dim, dim]
    MEMREF_I8_PARAMS,   // w2_t [n_layers, dim, hidden_dim]
    MEMREF_I8_PARAMS,   // w3_t [n_layers, hidden_dim, dim]
    MEMREF_F32_PARAMS,  // wcls [vocab_size, dim]
    MEMREF_F32_PARAMS,  // x [dim]
    MEMREF_F32_PARAMS,  // xb [dim]
    MEMREF_F32_PARAMS,  // xb2 [dim]
    MEMREF_F32_PARAMS,  // hb [hidden_dim]
    MEMREF_F32_PARAMS,  // hb2 [hidden_dim]
    MEMREF_F32_PARAMS,  // logits [vocab_size]
    MEMREF_I32_PARAMS,  // q_buf [dim]
    MEMREF_I32_PARAMS,  // k_buf [kv_dim]
    MEMREF_I32_PARAMS,  // v_buf [kv_dim]
    MEMREF_F32_PARAMS,  // att_scores [n_heads, max_seq_len]
    MEMREF_F32_PARAMS,  // att_probs [n_heads, max_seq_len]
    MEMREF_I8_PARAMS,   // k_cache [n_layers, max_seq_len, kv_dim]
    MEMREF_I8_PARAMS,   // v_cache [n_layers, max_seq_len, kv_dim]
    int64_t,            // token
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

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <llama32_1b_int8.so>" << std::endl;
        return 1;
    }

    std::cout << "================================================================================\n";
    std::cout << "   LLaMA 3.2-1B INT8 Decode Benchmark\n";
    std::cout << "   dim=" << DIM << ", layers=" << N_LAYERS << ", heads=" << N_HEADS
              << ", kv_heads=" << N_KV_HEADS << ", hidden=" << HIDDEN_DIM << "\n";
    std::cout << "================================================================================\n\n";

    // Load shared library
    void* handle = dlopen(argv[1], RTLD_NOW);
    if (!handle) {
        std::cerr << "Error: " << dlerror() << std::endl;
        return 1;
    }

    auto forward = (LLaMA32ForwardFunc)dlsym(handle, "llama32_1b_forward");
    if (!forward) {
        std::cerr << "Error: Could not find llama32_1b_forward: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }
    std::cout << "Loaded llama32_1b_forward ✓\n\n";

    // Calculate sizes
    size_t embed_size = VOCAB_SIZE * DIM;
    size_t rms_att_size = N_LAYERS * DIM;
    size_t rms_ffn_size = N_LAYERS * DIM;
    size_t rms_final_size = DIM;
    size_t wq_size = N_LAYERS * DIM * DIM;
    size_t wk_size = N_LAYERS * KV_DIM * DIM;
    size_t wv_size = N_LAYERS * KV_DIM * DIM;
    size_t wo_size = N_LAYERS * DIM * DIM;
    size_t w1_size = N_LAYERS * HIDDEN_DIM * DIM;
    size_t w2_size = N_LAYERS * DIM * HIDDEN_DIM;
    size_t w3_size = N_LAYERS * HIDDEN_DIM * DIM;
    size_t wcls_size = VOCAB_SIZE * DIM;
    size_t kv_cache_size = N_LAYERS * MAX_SEQ_LEN * KV_DIM;

    size_t total_mb = (embed_size * 4 + rms_att_size * 4 + rms_ffn_size * 4 + rms_final_size * 4 +
                       wq_size + wk_size + wv_size + wo_size + w1_size + w2_size + w3_size +
                       wcls_size * 4 + kv_cache_size * 2) / (1024 * 1024);
    std::cout << "Allocating ~" << total_mb << " MB...\n";

    // Allocate buffers
    float* token_embedding = (float*)aligned_alloc(64, embed_size * sizeof(float));
    float* rms_att_w = (float*)aligned_alloc(64, rms_att_size * sizeof(float));
    float* rms_ffn_w = (float*)aligned_alloc(64, rms_ffn_size * sizeof(float));
    float* rms_final_w = (float*)aligned_alloc(64, rms_final_size * sizeof(float));
    int8_t* wq_t = (int8_t*)aligned_alloc(64, wq_size);
    int8_t* wk_t = (int8_t*)aligned_alloc(64, wk_size);
    int8_t* wv_t = (int8_t*)aligned_alloc(64, wv_size);
    int8_t* wo_t = (int8_t*)aligned_alloc(64, wo_size);
    int8_t* w1_t = (int8_t*)aligned_alloc(64, w1_size);
    int8_t* w2_t = (int8_t*)aligned_alloc(64, w2_size);
    int8_t* w3_t = (int8_t*)aligned_alloc(64, w3_size);
    float* wcls = (float*)aligned_alloc(64, wcls_size * sizeof(float));
    float* x = (float*)aligned_alloc(64, DIM * sizeof(float));
    float* xb = (float*)aligned_alloc(64, DIM * sizeof(float));
    float* xb2 = (float*)aligned_alloc(64, DIM * sizeof(float));
    float* hb = (float*)aligned_alloc(64, HIDDEN_DIM * sizeof(float));
    float* hb2 = (float*)aligned_alloc(64, HIDDEN_DIM * sizeof(float));
    float* logits = (float*)aligned_alloc(64, VOCAB_SIZE * sizeof(float));
    int32_t* q_buf = (int32_t*)aligned_alloc(64, DIM * sizeof(int32_t));
    int32_t* k_buf = (int32_t*)aligned_alloc(64, KV_DIM * sizeof(int32_t));
    int32_t* v_buf = (int32_t*)aligned_alloc(64, KV_DIM * sizeof(int32_t));
    float* att_scores = (float*)aligned_alloc(64, N_HEADS * MAX_SEQ_LEN * sizeof(float));
    float* att_probs = (float*)aligned_alloc(64, N_HEADS * MAX_SEQ_LEN * sizeof(float));
    int8_t* k_cache = (int8_t*)aligned_alloc(64, kv_cache_size);
    int8_t* v_cache = (int8_t*)aligned_alloc(64, kv_cache_size);

    // Initialize weights
    std::cout << "Initializing weights...\n";
    init_random_f32(token_embedding, embed_size, 1);
    init_random_f32(rms_att_w, rms_att_size, 2);
    init_random_f32(rms_ffn_w, rms_ffn_size, 3);
    init_random_f32(rms_final_w, rms_final_size, 4);
    init_random_i8(wq_t, wq_size, 5);
    init_random_i8(wk_t, wk_size, 6);
    init_random_i8(wv_t, wv_size, 7);
    init_random_i8(wo_t, wo_size, 8);
    init_random_i8(w1_t, w1_size, 9);
    init_random_i8(w2_t, w2_size, 10);
    init_random_i8(w3_t, w3_size, 11);
    init_random_f32(wcls, wcls_size, 12);
    memset(k_cache, 0, kv_cache_size);
    memset(v_cache, 0, kv_cache_size);

    // Warmup
    std::cout << "Warming up...\n";
    for (int i = 0; i < 2; i++) {
        forward(
            MEMREF_F32(token_embedding, embed_size),
            MEMREF_F32(rms_att_w, rms_att_size),
            MEMREF_F32(rms_ffn_w, rms_ffn_size),
            MEMREF_F32(rms_final_w, rms_final_size),
            MEMREF_I8(wq_t, wq_size),
            MEMREF_I8(wk_t, wk_size),
            MEMREF_I8(wv_t, wv_size),
            MEMREF_I8(wo_t, wo_size),
            MEMREF_I8(w1_t, w1_size),
            MEMREF_I8(w2_t, w2_size),
            MEMREF_I8(w3_t, w3_size),
            MEMREF_F32(wcls, wcls_size),
            MEMREF_F32(x, DIM),
            MEMREF_F32(xb, DIM),
            MEMREF_F32(xb2, DIM),
            MEMREF_F32(hb, HIDDEN_DIM),
            MEMREF_F32(hb2, HIDDEN_DIM),
            MEMREF_F32(logits, VOCAB_SIZE),
            MEMREF_I32(q_buf, DIM),
            MEMREF_I32(k_buf, KV_DIM),
            MEMREF_I32(v_buf, KV_DIM),
            MEMREF_F32(att_scores, N_HEADS * MAX_SEQ_LEN),
            MEMREF_F32(att_probs, N_HEADS * MAX_SEQ_LEN),
            MEMREF_I8(k_cache, kv_cache_size),
            MEMREF_I8(v_cache, kv_cache_size),
            1000,  // token
            i,     // pos
            DIM, HIDDEN_DIM, N_LAYERS, N_HEADS, N_KV_HEADS, VOCAB_SIZE, MAX_SEQ_LEN, HEAD_DIM
        );
    }

    std::cout << "\n=== Decode Performance (single token) ===\n";
    std::cout << std::setw(10) << "Position" << std::setw(15) << "Time (ms)"
              << std::setw(15) << "Tokens/sec" << "\n";
    std::cout << std::string(40, '-') << "\n";

    int positions[] = {0, 64, 128, 256, 512, 1024, 2048};
    for (int pos : positions) {
        memset(k_cache, 0, kv_cache_size);
        memset(v_cache, 0, kv_cache_size);

        auto start = std::chrono::high_resolution_clock::now();
        int iters = (pos < 256) ? 10 : 5;
        for (int i = 0; i < iters; i++) {
            forward(
                MEMREF_F32(token_embedding, embed_size),
                MEMREF_F32(rms_att_w, rms_att_size),
                MEMREF_F32(rms_ffn_w, rms_ffn_size),
                MEMREF_F32(rms_final_w, rms_final_size),
                MEMREF_I8(wq_t, wq_size),
                MEMREF_I8(wk_t, wk_size),
                MEMREF_I8(wv_t, wv_size),
                MEMREF_I8(wo_t, wo_size),
                MEMREF_I8(w1_t, w1_size),
                MEMREF_I8(w2_t, w2_size),
                MEMREF_I8(w3_t, w3_size),
                MEMREF_F32(wcls, wcls_size),
                MEMREF_F32(x, DIM),
                MEMREF_F32(xb, DIM),
                MEMREF_F32(xb2, DIM),
                MEMREF_F32(hb, HIDDEN_DIM),
                MEMREF_F32(hb2, HIDDEN_DIM),
                MEMREF_F32(logits, VOCAB_SIZE),
                MEMREF_I32(q_buf, DIM),
                MEMREF_I32(k_buf, KV_DIM),
                MEMREF_I32(v_buf, KV_DIM),
                MEMREF_F32(att_scores, N_HEADS * MAX_SEQ_LEN),
                MEMREF_F32(att_probs, N_HEADS * MAX_SEQ_LEN),
                MEMREF_I8(k_cache, kv_cache_size),
                MEMREF_I8(v_cache, kv_cache_size),
                1000,  // token
                pos,
                DIM, HIDDEN_DIM, N_LAYERS, N_HEADS, N_KV_HEADS, VOCAB_SIZE, MAX_SEQ_LEN, HEAD_DIM
            );
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
        double tok_per_sec = 1000.0 / ms;

        std::cout << std::setw(10) << pos
                  << std::setw(15) << std::fixed << std::setprecision(2) << ms
                  << std::setw(15) << std::setprecision(1) << tok_per_sec << "\n";
    }

    std::cout << "\n=== Decode Loop (generate 32 tokens) ===\n";
    memset(k_cache, 0, kv_cache_size);
    memset(v_cache, 0, kv_cache_size);

    auto start = std::chrono::high_resolution_clock::now();
    for (int pos = 0; pos < 32; pos++) {
        forward(
            MEMREF_F32(token_embedding, embed_size),
            MEMREF_F32(rms_att_w, rms_att_size),
            MEMREF_F32(rms_ffn_w, rms_ffn_size),
            MEMREF_F32(rms_final_w, rms_final_size),
            MEMREF_I8(wq_t, wq_size),
            MEMREF_I8(wk_t, wk_size),
            MEMREF_I8(wv_t, wv_size),
            MEMREF_I8(wo_t, wo_size),
            MEMREF_I8(w1_t, w1_size),
            MEMREF_I8(w2_t, w2_size),
            MEMREF_I8(w3_t, w3_size),
            MEMREF_F32(wcls, wcls_size),
            MEMREF_F32(x, DIM),
            MEMREF_F32(xb, DIM),
            MEMREF_F32(xb2, DIM),
            MEMREF_F32(hb, HIDDEN_DIM),
            MEMREF_F32(hb2, HIDDEN_DIM),
            MEMREF_F32(logits, VOCAB_SIZE),
            MEMREF_I32(q_buf, DIM),
            MEMREF_I32(k_buf, KV_DIM),
            MEMREF_I32(v_buf, KV_DIM),
            MEMREF_F32(att_scores, N_HEADS * MAX_SEQ_LEN),
            MEMREF_F32(att_probs, N_HEADS * MAX_SEQ_LEN),
            MEMREF_I8(k_cache, kv_cache_size),
            MEMREF_I8(v_cache, kv_cache_size),
            1000,
            pos,
            DIM, HIDDEN_DIM, N_LAYERS, N_HEADS, N_KV_HEADS, VOCAB_SIZE, MAX_SEQ_LEN, HEAD_DIM
        );
    }
    auto end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Generated 32 tokens in " << std::fixed << std::setprecision(1) << total_ms << " ms\n";
    std::cout << "Throughput: " << std::setprecision(2) << (32.0 / (total_ms / 1000.0)) << " tokens/sec\n";

    std::cout << "\n================================================================================\n";

    // Cleanup
    free(token_embedding); free(rms_att_w); free(rms_ffn_w); free(rms_final_w);
    free(wq_t); free(wk_t); free(wv_t); free(wo_t);
    free(w1_t); free(w2_t); free(w3_t); free(wcls);
    free(x); free(xb); free(xb2); free(hb); free(hb2); free(logits);
    free(q_buf); free(k_buf); free(v_buf);
    free(att_scores); free(att_probs);
    free(k_cache); free(v_cache);
    dlclose(handle);

    return 0;
}
