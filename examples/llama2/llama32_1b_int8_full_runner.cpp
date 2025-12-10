// LLaMA 3.2-1B Fully Quantized Model Runner
// Benchmarks decode (single token) and prefill (32 tokens)

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>
#include <dlfcn.h>
#include <cmath>

// MLIR memref ABI: (base_ptr, aligned_ptr, offset, size, stride)
#define MEMREF_I8  int8_t*, int8_t*, int64_t, int64_t, int64_t
#define MEMREF_I16 int16_t*, int16_t*, int64_t, int64_t, int64_t
#define MEMREF_I32 int32_t*, int32_t*, int64_t, int64_t, int64_t
#define PASS_I8(p, s)  p, p, 0LL, (int64_t)(s), 1LL
#define PASS_I16(p, s) p, p, 0LL, (int64_t)(s), 1LL
#define PASS_I32(p, s) p, p, 0LL, (int64_t)(s), 1LL

// Model config
constexpr int64_t DIM = 2048;
constexpr int64_t HIDDEN_DIM = 8192;
constexpr int64_t N_LAYERS = 16;
constexpr int64_t N_HEADS = 32;
constexpr int64_t N_KV_HEADS = 8;
constexpr int64_t VOCAB_SIZE = 128256;
constexpr int64_t MAX_SEQ_LEN = 4096;
constexpr int64_t HEAD_DIM = 64;
constexpr int64_t KV_DIM = N_KV_HEADS * HEAD_DIM;  // 512

// Function signatures
using decode_fn = int32_t(*)(
    MEMREF_I8,   // x_i8
    MEMREF_I16,  // rms_att_w
    MEMREF_I16,  // rms_ffn_w
    MEMREF_I16,  // rms_final_w
    MEMREF_I8,   // wq_t
    MEMREF_I8,   // wk_t
    MEMREF_I8,   // wv_t
    MEMREF_I8,   // wo_t
    MEMREF_I8,   // w1_t
    MEMREF_I8,   // w2_t
    MEMREF_I8,   // w3_t
    MEMREF_I8,   // wcls_t
    MEMREF_I16,  // xb_i16
    MEMREF_I8,   // xb_i8
    MEMREF_I32,  // q_i32
    MEMREF_I32,  // k_i32
    MEMREF_I32,  // v_i32
    MEMREF_I32,  // attn_out
    MEMREF_I8,   // ffn_hb
    MEMREF_I8,   // k_cache
    MEMREF_I8,   // v_cache
    MEMREF_I32,  // att_scores
    MEMREF_I8,   // att_probs
    MEMREF_I16,  // cos_tab
    MEMREF_I16,  // sin_tab
    MEMREF_I32,  // logits
    int64_t,     // pos
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t
);

// All prefill functions have same signature
using prefill_fn = int32_t(*)(
    MEMREF_I8,   // x_i8
    MEMREF_I16,  // rms_att_w
    MEMREF_I16,  // rms_ffn_w
    MEMREF_I8,   // wq_t
    MEMREF_I8,   // wk_t
    MEMREF_I8,   // wv_t
    MEMREF_I8,   // wo_t
    MEMREF_I8,   // w1_t
    MEMREF_I8,   // w2_t
    MEMREF_I8,   // w3_t
    MEMREF_I16,  // xb_i16
    MEMREF_I8,   // xb_i8
    MEMREF_I8,   // k_cache
    MEMREF_I8,   // v_cache
    MEMREF_I32,  // att_scores
    MEMREF_I8,   // att_probs
    MEMREF_I32,  // attn_out
    MEMREF_I8,   // ffn_hb
    MEMREF_I16,  // cos_tab
    MEMREF_I16,  // sin_tab
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t
);

// Prefill batch sizes to benchmark
constexpr int BATCH_SIZES[] = {4, 8, 16, 32, 64, 128};
constexpr int NUM_BATCHES = 6;
const char* PREFILL_NAMES[] = {
    "llama32_prefill_4_i8",
    "llama32_prefill_8_i8",
    "llama32_prefill_16_i8",
    "llama32_prefill_32_i8",
    "llama32_prefill_64_i8",
    "llama32_prefill_128_i8"
};

// Aligned allocation helper
template<typename T>
T* alloc_aligned(size_t count) {
    return (T*)aligned_alloc(64, count * sizeof(T));
}

// Initialize with random INT8 values
void init_random_i8(int8_t* ptr, size_t n) {
    for (size_t i = 0; i < n; i++) {
        ptr[i] = (int8_t)((rand() % 256) - 128);
    }
}

void init_random_i16(int16_t* ptr, size_t n) {
    for (size_t i = 0; i < n; i++) {
        ptr[i] = (int16_t)((rand() % 65536) - 32768);
    }
}

// Initialize RoPE tables (Q15 sin/cos)
void init_rope_tables(int16_t* cos_tab, int16_t* sin_tab, int64_t max_seq, int64_t half_head) {
    float theta = 500000.0f;
    for (int64_t pos = 0; pos < max_seq; pos++) {
        for (int64_t i = 0; i < half_head; i++) {
            float freq = 1.0f / powf(theta, (float)(i * 2) / (float)(HEAD_DIM));
            float angle = (float)pos * freq;
            cos_tab[pos * half_head + i] = (int16_t)(cosf(angle) * 32767.0f);
            sin_tab[pos * half_head + i] = (int16_t)(sinf(angle) * 32767.0f);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.so>\n";
        return 1;
    }

    void* handle = dlopen(argv[1], RTLD_NOW);
    if (!handle) {
        std::cerr << "Error loading: " << dlerror() << "\n";
        return 1;
    }

    auto decode = (decode_fn)dlsym(handle, "llama32_decode_i8");

    // Load all prefill functions
    prefill_fn prefill_fns[NUM_BATCHES];
    for (int i = 0; i < NUM_BATCHES; i++) {
        prefill_fns[i] = (prefill_fn)dlsym(handle, PREFILL_NAMES[i]);
        if (!prefill_fns[i]) {
            std::cerr << "Warning: " << PREFILL_NAMES[i] << " not found\n";
        }
    }

    if (!decode) std::cerr << "Warning: decode function not found\n";

    std::cout << "=== LLaMA 3.2-1B Fully Quantized (INT8/INT16/INT32) ===\n\n";
    std::cout << "Model config:\n";
    std::cout << "  dim=" << DIM << ", hidden_dim=" << HIDDEN_DIM << "\n";
    std::cout << "  n_layers=" << N_LAYERS << ", n_heads=" << N_HEADS << "\n";
    std::cout << "  n_kv_heads=" << N_KV_HEADS << ", vocab_size=" << VOCAB_SIZE << "\n\n";

    srand(42);

    // Allocate weights
    std::cout << "Allocating weights...\n";
    auto rms_att_w = alloc_aligned<int16_t>(N_LAYERS * DIM);
    auto rms_ffn_w = alloc_aligned<int16_t>(N_LAYERS * DIM);
    auto rms_final_w = alloc_aligned<int16_t>(DIM);
    auto wq_t = alloc_aligned<int8_t>(N_LAYERS * DIM * DIM);
    auto wk_t = alloc_aligned<int8_t>(N_LAYERS * KV_DIM * DIM);
    auto wv_t = alloc_aligned<int8_t>(N_LAYERS * KV_DIM * DIM);
    auto wo_t = alloc_aligned<int8_t>(N_LAYERS * DIM * DIM);
    auto w1_t = alloc_aligned<int8_t>(N_LAYERS * HIDDEN_DIM * DIM);
    auto w2_t = alloc_aligned<int8_t>(N_LAYERS * DIM * HIDDEN_DIM);
    auto w3_t = alloc_aligned<int8_t>(N_LAYERS * HIDDEN_DIM * DIM);
    auto wcls_t = alloc_aligned<int8_t>(VOCAB_SIZE * DIM);

    // Initialize weights
    std::cout << "Initializing weights...\n";
    init_random_i16(rms_att_w, N_LAYERS * DIM);
    init_random_i16(rms_ffn_w, N_LAYERS * DIM);
    init_random_i16(rms_final_w, DIM);
    init_random_i8(wq_t, N_LAYERS * DIM * DIM);
    init_random_i8(wk_t, N_LAYERS * KV_DIM * DIM);
    init_random_i8(wv_t, N_LAYERS * KV_DIM * DIM);
    init_random_i8(wo_t, N_LAYERS * DIM * DIM);
    init_random_i8(w1_t, N_LAYERS * HIDDEN_DIM * DIM);
    init_random_i8(w2_t, N_LAYERS * DIM * HIDDEN_DIM);
    init_random_i8(w3_t, N_LAYERS * HIDDEN_DIM * DIM);
    init_random_i8(wcls_t, VOCAB_SIZE * DIM);

    // Allocate buffers (sized for max batch = 128)
    constexpr int MAX_BATCH = 128;
    std::cout << "Allocating buffers...\n";
    auto x_i8 = alloc_aligned<int8_t>(MAX_BATCH * DIM);  // For prefill
    auto xb_i16 = alloc_aligned<int16_t>(MAX_BATCH * DIM);
    auto xb_i8 = alloc_aligned<int8_t>(MAX_BATCH * DIM);
    auto q_i32 = alloc_aligned<int32_t>(DIM);
    auto k_i32 = alloc_aligned<int32_t>(KV_DIM);
    auto v_i32 = alloc_aligned<int32_t>(KV_DIM);
    auto attn_out = alloc_aligned<int32_t>(MAX_BATCH * DIM);
    auto ffn_hb = alloc_aligned<int8_t>(MAX_BATCH * HIDDEN_DIM);
    auto k_cache = alloc_aligned<int8_t>(N_LAYERS * MAX_SEQ_LEN * KV_DIM);
    auto v_cache = alloc_aligned<int8_t>(N_LAYERS * MAX_SEQ_LEN * KV_DIM);
    auto att_scores = alloc_aligned<int32_t>(N_HEADS * MAX_SEQ_LEN);  // For decode (max sized)
    auto att_probs = alloc_aligned<int8_t>(N_HEADS * MAX_SEQ_LEN);
    auto cos_tab = alloc_aligned<int16_t>(MAX_SEQ_LEN * HEAD_DIM / 2);
    auto sin_tab = alloc_aligned<int16_t>(MAX_SEQ_LEN * HEAD_DIM / 2);
    auto logits = alloc_aligned<int32_t>(VOCAB_SIZE);

    // Initialize
    init_random_i8(x_i8, MAX_BATCH * DIM);
    init_rope_tables(cos_tab, sin_tab, MAX_SEQ_LEN, HEAD_DIM / 2);
    memset(k_cache, 0, N_LAYERS * MAX_SEQ_LEN * KV_DIM);
    memset(v_cache, 0, N_LAYERS * MAX_SEQ_LEN * KV_DIM);

    // ========================================
    // Benchmark Decode (single token)
    // ========================================
    if (decode) {
        std::cout << "\n--- Decode Benchmark (single token) ---\n";

        // Warmup
        for (int i = 0; i < 3; i++) {
            decode(
                PASS_I8(x_i8, DIM),
                PASS_I16(rms_att_w, N_LAYERS * DIM),
                PASS_I16(rms_ffn_w, N_LAYERS * DIM),
                PASS_I16(rms_final_w, DIM),
                PASS_I8(wq_t, N_LAYERS * DIM * DIM),
                PASS_I8(wk_t, N_LAYERS * KV_DIM * DIM),
                PASS_I8(wv_t, N_LAYERS * KV_DIM * DIM),
                PASS_I8(wo_t, N_LAYERS * DIM * DIM),
                PASS_I8(w1_t, N_LAYERS * HIDDEN_DIM * DIM),
                PASS_I8(w2_t, N_LAYERS * DIM * HIDDEN_DIM),
                PASS_I8(w3_t, N_LAYERS * HIDDEN_DIM * DIM),
                PASS_I8(wcls_t, VOCAB_SIZE * DIM),
                PASS_I16(xb_i16, DIM),
                PASS_I8(xb_i8, DIM),
                PASS_I32(q_i32, DIM),
                PASS_I32(k_i32, KV_DIM),
                PASS_I32(v_i32, KV_DIM),
                PASS_I32(attn_out, DIM),
                PASS_I8(ffn_hb, HIDDEN_DIM),
                PASS_I8(k_cache, N_LAYERS * MAX_SEQ_LEN * KV_DIM),
                PASS_I8(v_cache, N_LAYERS * MAX_SEQ_LEN * KV_DIM),
                PASS_I32(att_scores, N_HEADS * MAX_SEQ_LEN),
                PASS_I8(att_probs, N_HEADS * MAX_SEQ_LEN),
                PASS_I16(cos_tab, MAX_SEQ_LEN * HEAD_DIM / 2),
                PASS_I16(sin_tab, MAX_SEQ_LEN * HEAD_DIM / 2),
                PASS_I32(logits, VOCAB_SIZE),
                0,  // pos
                DIM, HIDDEN_DIM, N_LAYERS, N_HEADS, N_KV_HEADS, VOCAB_SIZE, MAX_SEQ_LEN, HEAD_DIM
            );
        }

        // Benchmark at different positions
        int positions[] = {0, 10, 100, 500};
        for (int pos : positions) {
            const int ITERS = 10;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < ITERS; i++) {
                decode(
                    PASS_I8(x_i8, DIM),
                    PASS_I16(rms_att_w, N_LAYERS * DIM),
                    PASS_I16(rms_ffn_w, N_LAYERS * DIM),
                    PASS_I16(rms_final_w, DIM),
                    PASS_I8(wq_t, N_LAYERS * DIM * DIM),
                    PASS_I8(wk_t, N_LAYERS * KV_DIM * DIM),
                    PASS_I8(wv_t, N_LAYERS * KV_DIM * DIM),
                    PASS_I8(wo_t, N_LAYERS * DIM * DIM),
                    PASS_I8(w1_t, N_LAYERS * HIDDEN_DIM * DIM),
                    PASS_I8(w2_t, N_LAYERS * DIM * HIDDEN_DIM),
                    PASS_I8(w3_t, N_LAYERS * HIDDEN_DIM * DIM),
                    PASS_I8(wcls_t, VOCAB_SIZE * DIM),
                    PASS_I16(xb_i16, DIM),
                    PASS_I8(xb_i8, DIM),
                    PASS_I32(q_i32, DIM),
                    PASS_I32(k_i32, KV_DIM),
                    PASS_I32(v_i32, KV_DIM),
                    PASS_I32(attn_out, DIM),
                    PASS_I8(ffn_hb, HIDDEN_DIM),
                    PASS_I8(k_cache, N_LAYERS * MAX_SEQ_LEN * KV_DIM),
                    PASS_I8(v_cache, N_LAYERS * MAX_SEQ_LEN * KV_DIM),
                    PASS_I32(att_scores, N_HEADS * MAX_SEQ_LEN),
                    PASS_I8(att_probs, N_HEADS * MAX_SEQ_LEN),
                    PASS_I16(cos_tab, MAX_SEQ_LEN * HEAD_DIM / 2),
                    PASS_I16(sin_tab, MAX_SEQ_LEN * HEAD_DIM / 2),
                    PASS_I32(logits, VOCAB_SIZE),
                    pos,
                    DIM, HIDDEN_DIM, N_LAYERS, N_HEADS, N_KV_HEADS, VOCAB_SIZE, MAX_SEQ_LEN, HEAD_DIM
                );
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;
            printf("  pos=%3d: %.2f ms (%.1f tok/s)\n", pos, ms, 1000.0 / ms);
        }
    }

    // ========================================
    // Benchmark Prefill (all batch sizes)
    // ========================================
    std::cout << "\n--- Prefill Benchmark (all batch sizes) ---\n";
    std::cout << "+---------+----------+------------+\n";
    std::cout << "| Tokens  | Time(ms) | Tok/s      |\n";
    std::cout << "+---------+----------+------------+\n";

    double prefill_results[NUM_BATCHES][2];  // [ms, tok/s]

    for (int b = 0; b < NUM_BATCHES; b++) {
        int seq_len = BATCH_SIZES[b];
        auto prefill = prefill_fns[b];

        if (!prefill) {
            printf("|  %4d   |   N/A    |    N/A     |\n", seq_len);
            prefill_results[b][0] = 0;
            prefill_results[b][1] = 0;
            continue;
        }

        // Allocate attention buffers for this batch size
        auto att_scores_pf = alloc_aligned<int32_t>(N_HEADS * seq_len * seq_len);
        auto att_probs_pf = alloc_aligned<int8_t>(N_HEADS * seq_len * seq_len);
        auto k_cache_pf = alloc_aligned<int8_t>(N_LAYERS * seq_len * KV_DIM);
        auto v_cache_pf = alloc_aligned<int8_t>(N_LAYERS * seq_len * KV_DIM);
        memset(k_cache_pf, 0, N_LAYERS * seq_len * KV_DIM);
        memset(v_cache_pf, 0, N_LAYERS * seq_len * KV_DIM);

        // Warmup
        for (int i = 0; i < 2; i++) {
            prefill(
                PASS_I8(x_i8, seq_len * DIM),
                PASS_I16(rms_att_w, N_LAYERS * DIM),
                PASS_I16(rms_ffn_w, N_LAYERS * DIM),
                PASS_I8(wq_t, N_LAYERS * DIM * DIM),
                PASS_I8(wk_t, N_LAYERS * KV_DIM * DIM),
                PASS_I8(wv_t, N_LAYERS * KV_DIM * DIM),
                PASS_I8(wo_t, N_LAYERS * DIM * DIM),
                PASS_I8(w1_t, N_LAYERS * HIDDEN_DIM * DIM),
                PASS_I8(w2_t, N_LAYERS * DIM * HIDDEN_DIM),
                PASS_I8(w3_t, N_LAYERS * HIDDEN_DIM * DIM),
                PASS_I16(xb_i16, seq_len * DIM),
                PASS_I8(xb_i8, seq_len * DIM),
                PASS_I8(k_cache_pf, N_LAYERS * seq_len * KV_DIM),
                PASS_I8(v_cache_pf, N_LAYERS * seq_len * KV_DIM),
                PASS_I32(att_scores_pf, N_HEADS * seq_len * seq_len),
                PASS_I8(att_probs_pf, N_HEADS * seq_len * seq_len),
                PASS_I32(attn_out, seq_len * DIM),
                PASS_I8(ffn_hb, seq_len * HIDDEN_DIM),
                PASS_I16(cos_tab, MAX_SEQ_LEN * HEAD_DIM / 2),
                PASS_I16(sin_tab, MAX_SEQ_LEN * HEAD_DIM / 2),
                seq_len, DIM, HIDDEN_DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HEAD_DIM
            );
        }

        // Benchmark
        const int ITERS = 5;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) {
            prefill(
                PASS_I8(x_i8, seq_len * DIM),
                PASS_I16(rms_att_w, N_LAYERS * DIM),
                PASS_I16(rms_ffn_w, N_LAYERS * DIM),
                PASS_I8(wq_t, N_LAYERS * DIM * DIM),
                PASS_I8(wk_t, N_LAYERS * KV_DIM * DIM),
                PASS_I8(wv_t, N_LAYERS * KV_DIM * DIM),
                PASS_I8(wo_t, N_LAYERS * DIM * DIM),
                PASS_I8(w1_t, N_LAYERS * HIDDEN_DIM * DIM),
                PASS_I8(w2_t, N_LAYERS * DIM * HIDDEN_DIM),
                PASS_I8(w3_t, N_LAYERS * HIDDEN_DIM * DIM),
                PASS_I16(xb_i16, seq_len * DIM),
                PASS_I8(xb_i8, seq_len * DIM),
                PASS_I8(k_cache_pf, N_LAYERS * seq_len * KV_DIM),
                PASS_I8(v_cache_pf, N_LAYERS * seq_len * KV_DIM),
                PASS_I32(att_scores_pf, N_HEADS * seq_len * seq_len),
                PASS_I8(att_probs_pf, N_HEADS * seq_len * seq_len),
                PASS_I32(attn_out, seq_len * DIM),
                PASS_I8(ffn_hb, seq_len * HIDDEN_DIM),
                PASS_I16(cos_tab, MAX_SEQ_LEN * HEAD_DIM / 2),
                PASS_I16(sin_tab, MAX_SEQ_LEN * HEAD_DIM / 2),
                seq_len, DIM, HIDDEN_DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HEAD_DIM
            );
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;
        double tok_per_sec = (double)seq_len * 1000.0 / ms;

        prefill_results[b][0] = ms;
        prefill_results[b][1] = tok_per_sec;

        printf("|  %4d   | %8.2f | %10.1f |\n", seq_len, ms, tok_per_sec);

        free(att_scores_pf);
        free(att_probs_pf);
        free(k_cache_pf);
        free(v_cache_pf);
    }
    std::cout << "+---------+----------+------------+\n";

    // Memory usage summary
    size_t weight_bytes =
        N_LAYERS * DIM * sizeof(int16_t) * 2 +  // rms weights
        DIM * sizeof(int16_t) +                  // final rms
        N_LAYERS * DIM * DIM +                   // wq
        N_LAYERS * KV_DIM * DIM * 2 +           // wk, wv
        N_LAYERS * DIM * DIM +                   // wo
        N_LAYERS * HIDDEN_DIM * DIM * 2 +       // w1, w3
        N_LAYERS * DIM * HIDDEN_DIM +           // w2
        VOCAB_SIZE * DIM;                        // wcls

    printf("\n--- Memory Usage ---\n");
    printf("  Weights: %.1f MB (INT8)\n", weight_bytes / 1e6);
    printf("  KV Cache (max): %.1f MB (INT8)\n",
           (N_LAYERS * MAX_SEQ_LEN * KV_DIM * 2) / 1e6);

    // Cleanup
    free(rms_att_w); free(rms_ffn_w); free(rms_final_w);
    free(wq_t); free(wk_t); free(wv_t); free(wo_t);
    free(w1_t); free(w2_t); free(w3_t); free(wcls_t);
    free(x_i8); free(xb_i16); free(xb_i8);
    free(q_i32); free(k_i32); free(v_i32);
    free(attn_out); free(ffn_hb);
    free(k_cache); free(v_cache);
    free(att_scores); free(att_probs);
    free(cos_tab); free(sin_tab); free(logits);

    dlclose(handle);
    return 0;
}
