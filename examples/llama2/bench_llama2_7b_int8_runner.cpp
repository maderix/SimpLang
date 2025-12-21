/**
 * LLaMA 2-7B INT8 Benchmark Runner
 * Matches llama2_7b_decode_i8() signature in llama2_7b_int8_full.sl
 *
 * Build:
 *   g++ -O3 -march=native bench_llama2_7b_int8_runner.cpp -o bench_llama2_7b -ldl
 *
 * Run:
 *   ./bench_llama2_7b /tmp/llama2_7b_int8_full.so
 */

#include <iostream>
#include <iomanip>
#include <dlfcn.h>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <cmath>

// Model config for LLaMA 2-7B
constexpr int64_t DIM = 4096;
constexpr int64_t HIDDEN_DIM = 11008;
constexpr int64_t N_LAYERS = 32;
constexpr int64_t N_HEADS = 32;
constexpr int64_t N_KV_HEADS = 32;  // No GQA
constexpr int64_t VOCAB_SIZE = 32000;
constexpr int64_t MAX_SEQ_LEN = 2048;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t KV_DIM = N_KV_HEADS * HEAD_DIM;  // 4096

// MLIR memref descriptor: ptr, aligned_ptr, offset, size, stride
#define MEMREF_I8_PARAMS int8_t*, int8_t*, int64_t, int64_t, int64_t
#define MEMREF_I16_PARAMS int16_t*, int16_t*, int64_t, int64_t, int64_t
#define MEMREF_I32_PARAMS int32_t*, int32_t*, int64_t, int64_t, int64_t

#define MEMREF_I8(ptr, size) (ptr), (ptr), 0, (size), 1
#define MEMREF_I16(ptr, size) (ptr), (ptr), 0, (size), 1
#define MEMREF_I32(ptr, size) (ptr), (ptr), 0, (size), 1

// Function signature matching llama2_7b_prefill_32_i8
typedef int32_t (*LLaMA7BPrefillFunc)(
    MEMREF_I8_PARAMS,   // x_i8 [seq_len * dim]
    MEMREF_I16_PARAMS,  // rms_att_w [n_layers * dim]
    MEMREF_I16_PARAMS,  // rms_ffn_w [n_layers * dim]
    MEMREF_I8_PARAMS,   // wq_t
    MEMREF_I8_PARAMS,   // wk_t
    MEMREF_I8_PARAMS,   // wv_t
    MEMREF_I8_PARAMS,   // wo_t
    MEMREF_I8_PARAMS,   // w1_t
    MEMREF_I8_PARAMS,   // w2_t
    MEMREF_I8_PARAMS,   // w3_t
    MEMREF_I16_PARAMS,  // xb_i16 [seq_len * dim]
    MEMREF_I8_PARAMS,   // xb_i8 [seq_len * dim]
    MEMREF_I8_PARAMS,   // k_cache
    MEMREF_I8_PARAMS,   // v_cache
    MEMREF_I32_PARAMS,  // att_scores
    MEMREF_I8_PARAMS,   // att_probs
    MEMREF_I32_PARAMS,  // attn_out [seq_len * dim]
    MEMREF_I8_PARAMS,   // ffn_hb [seq_len * hidden_dim]
    MEMREF_I16_PARAMS,  // cos_tab
    MEMREF_I16_PARAMS,  // sin_tab
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t  // seq_len, dim, hidden_dim, n_layers, n_heads, n_kv_heads, head_dim
);

// Function signature matching llama2_7b_decode_i8
typedef int32_t (*LLaMA7BDecodeFunc)(
    MEMREF_I8_PARAMS,   // x_i8 [dim]
    MEMREF_I16_PARAMS,  // rms_att_w [n_layers * dim]
    MEMREF_I16_PARAMS,  // rms_ffn_w [n_layers * dim]
    MEMREF_I16_PARAMS,  // rms_final_w [dim]
    MEMREF_I8_PARAMS,   // wq_t [n_layers * dim * dim]
    MEMREF_I8_PARAMS,   // wk_t [n_layers * dim * dim]
    MEMREF_I8_PARAMS,   // wv_t [n_layers * dim * dim]
    MEMREF_I8_PARAMS,   // wo_t [n_layers * dim * dim]
    MEMREF_I8_PARAMS,   // w1_t [n_layers * hidden_dim * dim]
    MEMREF_I8_PARAMS,   // w2_t [n_layers * dim * hidden_dim]
    MEMREF_I8_PARAMS,   // w3_t [n_layers * hidden_dim * dim]
    MEMREF_I8_PARAMS,   // wcls_t [vocab_size * dim]
    MEMREF_I16_PARAMS,  // xb_i16 [dim]
    MEMREF_I8_PARAMS,   // xb_i8 [dim]
    MEMREF_I32_PARAMS,  // q_i32 [dim]
    MEMREF_I32_PARAMS,  // k_i32 [dim]
    MEMREF_I32_PARAMS,  // v_i32 [dim]
    MEMREF_I32_PARAMS,  // attn_out [dim]
    MEMREF_I8_PARAMS,   // ffn_hb [hidden_dim]
    MEMREF_I8_PARAMS,   // k_cache [n_layers * max_seq_len * dim]
    MEMREF_I8_PARAMS,   // v_cache [n_layers * max_seq_len * dim]
    MEMREF_I32_PARAMS,  // att_scores [n_heads * max_seq_len]
    MEMREF_I8_PARAMS,   // att_probs [n_heads * max_seq_len]
    MEMREF_I16_PARAMS,  // cos_tab [max_seq_len * head_dim/2]
    MEMREF_I16_PARAMS,  // sin_tab [max_seq_len * head_dim/2]
    MEMREF_I32_PARAMS,  // logits [vocab_size]
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

void init_random_i16(int16_t* data, size_t size, int seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-1000, 1000);
    for (size_t i = 0; i < size; i++) {
        data[i] = (int16_t)dist(rng);
    }
}

void init_rope_tables(int16_t* cos_tab, int16_t* sin_tab, int max_seq_len, int head_dim) {
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(10000.0f, (float)(2 * i) / head_dim);
            float angle = pos * freq;
            cos_tab[pos * (head_dim / 2) + i] = (int16_t)(cosf(angle) * 32767);
            sin_tab[pos * (head_dim / 2) + i] = (int16_t)(sinf(angle) * 32767);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <llama2_7b_int8_full.so>" << std::endl;
        return 1;
    }

    std::cout << "================================================================================\n";
    std::cout << "   LLaMA 2-7B INT8 Decode Benchmark\n";
    std::cout << "   dim=" << DIM << ", layers=" << N_LAYERS << ", heads=" << N_HEADS
              << ", hidden=" << HIDDEN_DIM << "\n";
    std::cout << "================================================================================\n\n";

    // Load shared library
    void* handle = dlopen(argv[1], RTLD_NOW);
    if (!handle) {
        std::cerr << "Error: " << dlerror() << std::endl;
        return 1;
    }

    auto decode = (LLaMA7BDecodeFunc)dlsym(handle, "llama2_7b_decode_i8");
    if (!decode) {
        std::cerr << "Error: Could not find llama2_7b_decode_i8: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }

    auto prefill = (LLaMA7BPrefillFunc)dlsym(handle, "llama2_7b_prefill_32_i8");
    if (!prefill) {
        std::cerr << "Warning: Could not find llama2_7b_prefill_32_i8\n";
    }
    std::cout << "Loaded llama2_7b_decode_i8" << (prefill ? " + prefill" : "") << "\n\n";

    // Calculate sizes
    size_t rms_att_size = N_LAYERS * DIM;
    size_t rms_ffn_size = N_LAYERS * DIM;
    size_t rms_final_size = DIM;
    size_t wq_size = N_LAYERS * DIM * DIM;
    size_t wk_size = N_LAYERS * DIM * DIM;
    size_t wv_size = N_LAYERS * DIM * DIM;
    size_t wo_size = N_LAYERS * DIM * DIM;
    size_t w1_size = N_LAYERS * HIDDEN_DIM * DIM;
    size_t w2_size = N_LAYERS * DIM * HIDDEN_DIM;
    size_t w3_size = N_LAYERS * HIDDEN_DIM * DIM;
    size_t wcls_size = VOCAB_SIZE * DIM;
    size_t kv_cache_size = N_LAYERS * MAX_SEQ_LEN * DIM;
    size_t rope_size = MAX_SEQ_LEN * (HEAD_DIM / 2);

    size_t total_mb = (wq_size + wk_size + wv_size + wo_size + w1_size + w2_size + w3_size +
                       wcls_size + kv_cache_size * 2 +
                       rms_att_size * 2 + rms_ffn_size * 2 + rms_final_size * 2) / (1024 * 1024);
    std::cout << "Allocating ~" << total_mb << " MB...\n";

    // Allocate buffers
    int8_t* x_i8 = (int8_t*)aligned_alloc(64, DIM);
    int16_t* rms_att_w = (int16_t*)aligned_alloc(64, rms_att_size * sizeof(int16_t));
    int16_t* rms_ffn_w = (int16_t*)aligned_alloc(64, rms_ffn_size * sizeof(int16_t));
    int16_t* rms_final_w = (int16_t*)aligned_alloc(64, rms_final_size * sizeof(int16_t));
    int8_t* wq_t = (int8_t*)aligned_alloc(64, wq_size);
    int8_t* wk_t = (int8_t*)aligned_alloc(64, wk_size);
    int8_t* wv_t = (int8_t*)aligned_alloc(64, wv_size);
    int8_t* wo_t = (int8_t*)aligned_alloc(64, wo_size);
    int8_t* w1_t = (int8_t*)aligned_alloc(64, w1_size);
    int8_t* w2_t = (int8_t*)aligned_alloc(64, w2_size);
    int8_t* w3_t = (int8_t*)aligned_alloc(64, w3_size);
    int8_t* wcls_t = (int8_t*)aligned_alloc(64, wcls_size);
    int16_t* xb_i16 = (int16_t*)aligned_alloc(64, DIM * sizeof(int16_t));
    int8_t* xb_i8 = (int8_t*)aligned_alloc(64, DIM);
    int32_t* q_i32 = (int32_t*)aligned_alloc(64, DIM * sizeof(int32_t));
    int32_t* k_i32 = (int32_t*)aligned_alloc(64, DIM * sizeof(int32_t));
    int32_t* v_i32 = (int32_t*)aligned_alloc(64, DIM * sizeof(int32_t));
    int32_t* attn_out = (int32_t*)aligned_alloc(64, DIM * sizeof(int32_t));
    int8_t* ffn_hb = (int8_t*)aligned_alloc(64, HIDDEN_DIM);
    int8_t* k_cache = (int8_t*)aligned_alloc(64, kv_cache_size);
    int8_t* v_cache = (int8_t*)aligned_alloc(64, kv_cache_size);
    int32_t* att_scores = (int32_t*)aligned_alloc(64, N_HEADS * MAX_SEQ_LEN * sizeof(int32_t));
    int8_t* att_probs = (int8_t*)aligned_alloc(64, N_HEADS * MAX_SEQ_LEN);
    int16_t* cos_tab = (int16_t*)aligned_alloc(64, rope_size * sizeof(int16_t));
    int16_t* sin_tab = (int16_t*)aligned_alloc(64, rope_size * sizeof(int16_t));
    int32_t* logits = (int32_t*)aligned_alloc(64, VOCAB_SIZE * sizeof(int32_t));

    // Initialize weights
    std::cout << "Initializing weights...\n";
    init_random_i8(x_i8, DIM, 0);
    init_random_i16(rms_att_w, rms_att_size, 1);
    init_random_i16(rms_ffn_w, rms_ffn_size, 2);
    init_random_i16(rms_final_w, rms_final_size, 3);
    init_random_i8(wq_t, wq_size, 4);
    init_random_i8(wk_t, wk_size, 5);
    init_random_i8(wv_t, wv_size, 6);
    init_random_i8(wo_t, wo_size, 7);
    init_random_i8(w1_t, w1_size, 8);
    init_random_i8(w2_t, w2_size, 9);
    init_random_i8(w3_t, w3_size, 10);
    init_random_i8(wcls_t, wcls_size, 11);
    init_rope_tables(cos_tab, sin_tab, MAX_SEQ_LEN, HEAD_DIM);
    memset(k_cache, 0, kv_cache_size);
    memset(v_cache, 0, kv_cache_size);
    memset(xb_i16, 0, DIM * sizeof(int16_t));
    memset(xb_i8, 0, DIM);

    // Warmup
    std::cout << "Warming up...\n";
    for (int i = 0; i < 2; i++) {
        decode(
            MEMREF_I8(x_i8, DIM),
            MEMREF_I16(rms_att_w, rms_att_size),
            MEMREF_I16(rms_ffn_w, rms_ffn_size),
            MEMREF_I16(rms_final_w, rms_final_size),
            MEMREF_I8(wq_t, wq_size),
            MEMREF_I8(wk_t, wk_size),
            MEMREF_I8(wv_t, wv_size),
            MEMREF_I8(wo_t, wo_size),
            MEMREF_I8(w1_t, w1_size),
            MEMREF_I8(w2_t, w2_size),
            MEMREF_I8(w3_t, w3_size),
            MEMREF_I8(wcls_t, wcls_size),
            MEMREF_I16(xb_i16, DIM),
            MEMREF_I8(xb_i8, DIM),
            MEMREF_I32(q_i32, DIM),
            MEMREF_I32(k_i32, DIM),
            MEMREF_I32(v_i32, DIM),
            MEMREF_I32(attn_out, DIM),
            MEMREF_I8(ffn_hb, HIDDEN_DIM),
            MEMREF_I8(k_cache, kv_cache_size),
            MEMREF_I8(v_cache, kv_cache_size),
            MEMREF_I32(att_scores, N_HEADS * MAX_SEQ_LEN),
            MEMREF_I8(att_probs, N_HEADS * MAX_SEQ_LEN),
            MEMREF_I16(cos_tab, rope_size),
            MEMREF_I16(sin_tab, rope_size),
            MEMREF_I32(logits, VOCAB_SIZE),
            i,  // pos
            DIM, HIDDEN_DIM, N_LAYERS, N_HEADS, N_KV_HEADS, VOCAB_SIZE, MAX_SEQ_LEN, HEAD_DIM
        );
    }

    // Prefill benchmark (32 tokens at once)
    if (prefill) {
        constexpr int64_t PREFILL_SEQ = 32;
        size_t prefill_x_size = PREFILL_SEQ * DIM;
        size_t prefill_xb_size = PREFILL_SEQ * DIM;
        size_t prefill_ffn_size = PREFILL_SEQ * HIDDEN_DIM;
        size_t prefill_attn_size = PREFILL_SEQ * DIM;

        int8_t* prefill_x = (int8_t*)aligned_alloc(64, prefill_x_size);
        int16_t* prefill_xb16 = (int16_t*)aligned_alloc(64, prefill_xb_size * sizeof(int16_t));
        int8_t* prefill_xb8 = (int8_t*)aligned_alloc(64, prefill_xb_size);
        int8_t* prefill_ffn = (int8_t*)aligned_alloc(64, prefill_ffn_size);
        int32_t* prefill_attn = (int32_t*)aligned_alloc(64, prefill_attn_size * sizeof(int32_t));

        init_random_i8(prefill_x, prefill_x_size, 100);
        memset(prefill_xb16, 0, prefill_xb_size * sizeof(int16_t));
        memset(prefill_xb8, 0, prefill_xb_size);
        memset(k_cache, 0, kv_cache_size);
        memset(v_cache, 0, kv_cache_size);

        std::cout << "\n=== Prefill Performance (32 tokens batch) ===\n";

        // Warmup
        prefill(
            MEMREF_I8(prefill_x, prefill_x_size),
            MEMREF_I16(rms_att_w, rms_att_size),
            MEMREF_I16(rms_ffn_w, rms_ffn_size),
            MEMREF_I8(wq_t, wq_size),
            MEMREF_I8(wk_t, wk_size),
            MEMREF_I8(wv_t, wv_size),
            MEMREF_I8(wo_t, wo_size),
            MEMREF_I8(w1_t, w1_size),
            MEMREF_I8(w2_t, w2_size),
            MEMREF_I8(w3_t, w3_size),
            MEMREF_I16(prefill_xb16, prefill_xb_size),
            MEMREF_I8(prefill_xb8, prefill_xb_size),
            MEMREF_I8(k_cache, kv_cache_size),
            MEMREF_I8(v_cache, kv_cache_size),
            MEMREF_I32(att_scores, N_HEADS * MAX_SEQ_LEN),
            MEMREF_I8(att_probs, N_HEADS * MAX_SEQ_LEN),
            MEMREF_I32(prefill_attn, prefill_attn_size),
            MEMREF_I8(prefill_ffn, prefill_ffn_size),
            MEMREF_I16(cos_tab, rope_size),
            MEMREF_I16(sin_tab, rope_size),
            PREFILL_SEQ, DIM, HIDDEN_DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HEAD_DIM
        );

        // Benchmark
        memset(k_cache, 0, kv_cache_size);
        memset(v_cache, 0, kv_cache_size);
        auto start = std::chrono::high_resolution_clock::now();
        int iters = 3;
        for (int i = 0; i < iters; i++) {
            prefill(
                MEMREF_I8(prefill_x, prefill_x_size),
                MEMREF_I16(rms_att_w, rms_att_size),
                MEMREF_I16(rms_ffn_w, rms_ffn_size),
                MEMREF_I8(wq_t, wq_size),
                MEMREF_I8(wk_t, wk_size),
                MEMREF_I8(wv_t, wv_size),
                MEMREF_I8(wo_t, wo_size),
                MEMREF_I8(w1_t, w1_size),
                MEMREF_I8(w2_t, w2_size),
                MEMREF_I8(w3_t, w3_size),
                MEMREF_I16(prefill_xb16, prefill_xb_size),
                MEMREF_I8(prefill_xb8, prefill_xb_size),
                MEMREF_I8(k_cache, kv_cache_size),
                MEMREF_I8(v_cache, kv_cache_size),
                MEMREF_I32(att_scores, N_HEADS * MAX_SEQ_LEN),
                MEMREF_I8(att_probs, N_HEADS * MAX_SEQ_LEN),
                MEMREF_I32(prefill_attn, prefill_attn_size),
                MEMREF_I8(prefill_ffn, prefill_ffn_size),
                MEMREF_I16(cos_tab, rope_size),
                MEMREF_I16(sin_tab, rope_size),
                PREFILL_SEQ, DIM, HIDDEN_DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HEAD_DIM
            );
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
        double tok_per_sec = PREFILL_SEQ * 1000.0 / ms;
        std::cout << "Time for " << PREFILL_SEQ << " tokens: " << std::fixed << std::setprecision(2) << ms << " ms\n";
        std::cout << "Throughput: " << std::setprecision(1) << tok_per_sec << " tokens/sec\n";

        free(prefill_x); free(prefill_xb16); free(prefill_xb8);
        free(prefill_ffn); free(prefill_attn);
    }

    std::cout << "\n=== Decode Performance (single token) ===\n";
    std::cout << std::setw(10) << "Position" << std::setw(15) << "Time (ms)"
              << std::setw(15) << "Tokens/sec" << "\n";
    std::cout << std::string(40, '-') << "\n";

    int positions[] = {0, 64, 128, 256, 512};
    for (int pos : positions) {
        memset(k_cache, 0, kv_cache_size);
        memset(v_cache, 0, kv_cache_size);

        auto start = std::chrono::high_resolution_clock::now();
        int iters = 3;
        for (int i = 0; i < iters; i++) {
            decode(
                MEMREF_I8(x_i8, DIM),
                MEMREF_I16(rms_att_w, rms_att_size),
                MEMREF_I16(rms_ffn_w, rms_ffn_size),
                MEMREF_I16(rms_final_w, rms_final_size),
                MEMREF_I8(wq_t, wq_size),
                MEMREF_I8(wk_t, wk_size),
                MEMREF_I8(wv_t, wv_size),
                MEMREF_I8(wo_t, wo_size),
                MEMREF_I8(w1_t, w1_size),
                MEMREF_I8(w2_t, w2_size),
                MEMREF_I8(w3_t, w3_size),
                MEMREF_I8(wcls_t, wcls_size),
                MEMREF_I16(xb_i16, DIM),
                MEMREF_I8(xb_i8, DIM),
                MEMREF_I32(q_i32, DIM),
                MEMREF_I32(k_i32, DIM),
                MEMREF_I32(v_i32, DIM),
                MEMREF_I32(attn_out, DIM),
                MEMREF_I8(ffn_hb, HIDDEN_DIM),
                MEMREF_I8(k_cache, kv_cache_size),
                MEMREF_I8(v_cache, kv_cache_size),
                MEMREF_I32(att_scores, N_HEADS * MAX_SEQ_LEN),
                MEMREF_I8(att_probs, N_HEADS * MAX_SEQ_LEN),
                MEMREF_I16(cos_tab, rope_size),
                MEMREF_I16(sin_tab, rope_size),
                MEMREF_I32(logits, VOCAB_SIZE),
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

    std::cout << "\n================================================================================\n";

    // Cleanup
    free(x_i8); free(rms_att_w); free(rms_ffn_w); free(rms_final_w);
    free(wq_t); free(wk_t); free(wv_t); free(wo_t);
    free(w1_t); free(w2_t); free(w3_t); free(wcls_t);
    free(xb_i16); free(xb_i8);
    free(q_i32); free(k_i32); free(v_i32); free(attn_out);
    free(ffn_hb);
    free(k_cache); free(v_cache);
    free(att_scores); free(att_probs);
    free(cos_tab); free(sin_tab);
    free(logits);
    dlclose(handle);

    return 0;
}
