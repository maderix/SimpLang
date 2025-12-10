// LLaMA 3.2-1B FP32+INT8 Model Runner
// Benchmarks decode (tensor version) and prefill (multiple batch sizes)
// FP32 for RMSNorm, Softmax, SiLU; INT8 matmuls with tensor_matmul_nt

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>
#include <dlfcn.h>
#include <cmath>

// MLIR memref ABI: (base_ptr, aligned_ptr, offset, size, stride)
#define MEMREF_I8  int8_t*, int8_t*, int64_t, int64_t, int64_t
#define MEMREF_F32 float*, float*, int64_t, int64_t, int64_t
#define MEMREF_I32 int32_t*, int32_t*, int64_t, int64_t, int64_t
#define PASS_I8(p, s)  p, p, 0LL, (int64_t)(s), 1LL
#define PASS_F32(p, s) p, p, 0LL, (int64_t)(s), 1LL
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

// Decode function signature (from llama32_1b_int8_tensor.sl)
using decode_fn = float(*)(
    MEMREF_I8,   // x_i8
    MEMREF_F32,  // rms_att_w
    MEMREF_F32,  // rms_ffn_w
    MEMREF_F32,  // rms_final_w
    MEMREF_I8,   // wq_t
    MEMREF_I8,   // wk_t
    MEMREF_I8,   // wv_t
    MEMREF_I8,   // wo_t
    MEMREF_I8,   // w1_t
    MEMREF_I8,   // w2_t
    MEMREF_I8,   // w3_t
    MEMREF_F32,  // wcls
    MEMREF_F32,  // x_fp32
    MEMREF_F32,  // xb_fp32
    MEMREF_F32,  // logits
    MEMREF_I8,   // xb_i8
    MEMREF_I8,   // k_cache
    MEMREF_I8,   // v_cache
    MEMREF_F32,  // att_scores
    MEMREF_F32,  // att_probs
    MEMREF_I32,  // attn_out
    MEMREF_I8,   // ffn_hb
    int64_t,     // pos
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t
);

// Prefill function signature (from llama32_1b_int8_prefill.sl)
using prefill_fn = float(*)(
    MEMREF_I8,   // x_i8
    MEMREF_F32,  // rms_att_w
    MEMREF_F32,  // rms_ffn_w
    MEMREF_F32,  // rms_final_w
    MEMREF_I8,   // wq_t
    MEMREF_I8,   // wk_t
    MEMREF_I8,   // wv_t
    MEMREF_I8,   // wo_t
    MEMREF_I8,   // w1_t
    MEMREF_I8,   // w2_t
    MEMREF_I8,   // w3_t
    MEMREF_F32,  // x_fp32
    MEMREF_F32,  // xb_fp32
    MEMREF_I8,   // xb_i8
    MEMREF_I8,   // k_cache
    MEMREF_I8,   // v_cache
    MEMREF_F32,  // att_scores
    MEMREF_F32,  // att_probs
    MEMREF_I32,  // attn_out
    MEMREF_I8,   // ffn_hb
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t
);

// Prefill batch sizes
constexpr int BATCH_SIZES[] = {8, 16, 32, 64, 128, 256};
constexpr int NUM_BATCHES = 6;
const char* PREFILL_NAMES[] = {
    "llama32_prefill_8", "llama32_prefill_16", "llama32_prefill_32",
    "llama32_prefill_64", "llama32_prefill_128", "llama32_prefill_256"
};

template<typename T>
T* alloc_aligned(size_t count) {
    return (T*)aligned_alloc(64, count * sizeof(T));
}

void init_random_i8(int8_t* ptr, size_t n) {
    for (size_t i = 0; i < n; i++) {
        ptr[i] = (int8_t)((rand() % 256) - 128);
    }
}

void init_random_f32(float* ptr, size_t n, float scale = 0.01f) {
    for (size_t i = 0; i < n; i++) {
        ptr[i] = ((float)(rand() % 1000) / 1000.0f - 0.5f) * scale;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <decode.so> [prefill.so]\n";
        std::cerr << "  decode.so  - compiled from llama32_1b_int8_tensor.sl\n";
        std::cerr << "  prefill.so - compiled from llama32_1b_int8_prefill.sl (optional)\n";
        return 1;
    }

    // Load decode library
    void* decode_handle = dlopen(argv[1], RTLD_NOW);
    if (!decode_handle) {
        std::cerr << "Error loading decode: " << dlerror() << "\n";
        return 1;
    }

    auto decode = (decode_fn)dlsym(decode_handle, "llama32_forward");
    if (!decode) std::cerr << "Warning: llama32_forward not found in decode library\n";

    // Load prefill library (optional, can be same or different .so)
    void* prefill_handle = nullptr;
    prefill_fn prefill_fns[NUM_BATCHES] = {nullptr};

    if (argc >= 3) {
        prefill_handle = dlopen(argv[2], RTLD_NOW);
    } else {
        prefill_handle = dlopen(argv[1], RTLD_NOW);  // Try same library
    }

    if (prefill_handle) {
        for (int i = 0; i < NUM_BATCHES; i++) {
            prefill_fns[i] = (prefill_fn)dlsym(prefill_handle, PREFILL_NAMES[i]);
        }
    }

    std::cout << "=== LLaMA 3.2-1B FP32+INT8 Benchmark ===\n";
    std::cout << "(FP32: RMSNorm, Softmax, SiLU | INT8: Matmuls via tensor_matmul_nt)\n\n";
    std::cout << "Model config:\n";
    std::cout << "  dim=" << DIM << ", hidden_dim=" << HIDDEN_DIM << "\n";
    std::cout << "  n_layers=" << N_LAYERS << ", n_heads=" << N_HEADS << "\n";
    std::cout << "  n_kv_heads=" << N_KV_HEADS << ", vocab_size=" << VOCAB_SIZE << "\n\n";

    srand(42);

    // Allocate weights (shared between decode and prefill)
    std::cout << "Allocating weights...\n";
    auto rms_att_w = alloc_aligned<float>(N_LAYERS * DIM);
    auto rms_ffn_w = alloc_aligned<float>(N_LAYERS * DIM);
    auto rms_final_w = alloc_aligned<float>(DIM);
    auto wq_t = alloc_aligned<int8_t>(N_LAYERS * DIM * DIM);
    auto wk_t = alloc_aligned<int8_t>(N_LAYERS * KV_DIM * DIM);
    auto wv_t = alloc_aligned<int8_t>(N_LAYERS * KV_DIM * DIM);
    auto wo_t = alloc_aligned<int8_t>(N_LAYERS * DIM * DIM);
    auto w1_t = alloc_aligned<int8_t>(N_LAYERS * HIDDEN_DIM * DIM);
    auto w2_t = alloc_aligned<int8_t>(N_LAYERS * DIM * HIDDEN_DIM);
    auto w3_t = alloc_aligned<int8_t>(N_LAYERS * HIDDEN_DIM * DIM);
    auto wcls = alloc_aligned<float>(VOCAB_SIZE * DIM);  // FP32 classifier for decode

    // Initialize weights
    std::cout << "Initializing weights...\n";
    init_random_f32(rms_att_w, N_LAYERS * DIM);
    init_random_f32(rms_ffn_w, N_LAYERS * DIM);
    init_random_f32(rms_final_w, DIM);
    init_random_i8(wq_t, N_LAYERS * DIM * DIM);
    init_random_i8(wk_t, N_LAYERS * KV_DIM * DIM);
    init_random_i8(wv_t, N_LAYERS * KV_DIM * DIM);
    init_random_i8(wo_t, N_LAYERS * DIM * DIM);
    init_random_i8(w1_t, N_LAYERS * HIDDEN_DIM * DIM);
    init_random_i8(w2_t, N_LAYERS * DIM * HIDDEN_DIM);
    init_random_i8(w3_t, N_LAYERS * HIDDEN_DIM * DIM);
    init_random_f32(wcls, VOCAB_SIZE * DIM, 0.001f);

    // ========================================
    // Benchmark Decode (single token)
    // ========================================
    if (decode) {
        std::cout << "\n--- Decode Benchmark (single token) ---\n";

        // Allocate decode buffers
        auto x_i8 = alloc_aligned<int8_t>(DIM);
        auto x_fp32 = alloc_aligned<float>(DIM);
        auto xb_fp32 = alloc_aligned<float>(DIM);
        auto logits = alloc_aligned<float>(VOCAB_SIZE);
        auto xb_i8 = alloc_aligned<int8_t>(DIM);
        auto k_cache = alloc_aligned<int8_t>(N_LAYERS * MAX_SEQ_LEN * KV_DIM);
        auto v_cache = alloc_aligned<int8_t>(N_LAYERS * MAX_SEQ_LEN * KV_DIM);
        auto att_scores = alloc_aligned<float>(N_HEADS * MAX_SEQ_LEN);
        auto att_probs = alloc_aligned<float>(N_HEADS * MAX_SEQ_LEN);
        auto attn_out = alloc_aligned<int32_t>(DIM);
        auto ffn_hb = alloc_aligned<int8_t>(HIDDEN_DIM);

        init_random_i8(x_i8, DIM);
        memset(k_cache, 0, N_LAYERS * MAX_SEQ_LEN * KV_DIM);
        memset(v_cache, 0, N_LAYERS * MAX_SEQ_LEN * KV_DIM);

        // Warmup
        for (int i = 0; i < 2; i++) {
            decode(
                PASS_I8(x_i8, DIM), PASS_F32(rms_att_w, N_LAYERS * DIM),
                PASS_F32(rms_ffn_w, N_LAYERS * DIM), PASS_F32(rms_final_w, DIM),
                PASS_I8(wq_t, N_LAYERS * DIM * DIM), PASS_I8(wk_t, N_LAYERS * KV_DIM * DIM),
                PASS_I8(wv_t, N_LAYERS * KV_DIM * DIM), PASS_I8(wo_t, N_LAYERS * DIM * DIM),
                PASS_I8(w1_t, N_LAYERS * HIDDEN_DIM * DIM), PASS_I8(w2_t, N_LAYERS * DIM * HIDDEN_DIM),
                PASS_I8(w3_t, N_LAYERS * HIDDEN_DIM * DIM), PASS_F32(wcls, VOCAB_SIZE * DIM),
                PASS_F32(x_fp32, DIM), PASS_F32(xb_fp32, DIM), PASS_F32(logits, VOCAB_SIZE),
                PASS_I8(xb_i8, DIM), PASS_I8(k_cache, N_LAYERS * MAX_SEQ_LEN * KV_DIM),
                PASS_I8(v_cache, N_LAYERS * MAX_SEQ_LEN * KV_DIM),
                PASS_F32(att_scores, N_HEADS * MAX_SEQ_LEN), PASS_F32(att_probs, N_HEADS * MAX_SEQ_LEN),
                PASS_I32(attn_out, DIM), PASS_I8(ffn_hb, HIDDEN_DIM),
                i, DIM, HIDDEN_DIM, N_LAYERS, N_HEADS, N_KV_HEADS, VOCAB_SIZE, MAX_SEQ_LEN, HEAD_DIM
            );
        }

        // Benchmark at different positions
        int positions[] = {0, 10, 100, 500};
        for (int pos : positions) {
            const int ITERS = 5;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < ITERS; i++) {
                decode(
                    PASS_I8(x_i8, DIM), PASS_F32(rms_att_w, N_LAYERS * DIM),
                    PASS_F32(rms_ffn_w, N_LAYERS * DIM), PASS_F32(rms_final_w, DIM),
                    PASS_I8(wq_t, N_LAYERS * DIM * DIM), PASS_I8(wk_t, N_LAYERS * KV_DIM * DIM),
                    PASS_I8(wv_t, N_LAYERS * KV_DIM * DIM), PASS_I8(wo_t, N_LAYERS * DIM * DIM),
                    PASS_I8(w1_t, N_LAYERS * HIDDEN_DIM * DIM), PASS_I8(w2_t, N_LAYERS * DIM * HIDDEN_DIM),
                    PASS_I8(w3_t, N_LAYERS * HIDDEN_DIM * DIM), PASS_F32(wcls, VOCAB_SIZE * DIM),
                    PASS_F32(x_fp32, DIM), PASS_F32(xb_fp32, DIM), PASS_F32(logits, VOCAB_SIZE),
                    PASS_I8(xb_i8, DIM), PASS_I8(k_cache, N_LAYERS * MAX_SEQ_LEN * KV_DIM),
                    PASS_I8(v_cache, N_LAYERS * MAX_SEQ_LEN * KV_DIM),
                    PASS_F32(att_scores, N_HEADS * MAX_SEQ_LEN), PASS_F32(att_probs, N_HEADS * MAX_SEQ_LEN),
                    PASS_I32(attn_out, DIM), PASS_I8(ffn_hb, HIDDEN_DIM),
                    pos, DIM, HIDDEN_DIM, N_LAYERS, N_HEADS, N_KV_HEADS, VOCAB_SIZE, MAX_SEQ_LEN, HEAD_DIM
                );
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;
            printf("  pos=%3d: %.2f ms (%.1f tok/s)\n", pos, ms, 1000.0 / ms);
        }

        free(x_i8); free(x_fp32); free(xb_fp32); free(logits); free(xb_i8);
        free(k_cache); free(v_cache); free(att_scores); free(att_probs);
        free(attn_out); free(ffn_hb);
    }

    // ========================================
    // Benchmark Prefill (all batch sizes)
    // ========================================
    bool has_any_prefill = false;
    for (int i = 0; i < NUM_BATCHES; i++) {
        if (prefill_fns[i]) has_any_prefill = true;
    }

    if (has_any_prefill) {
        std::cout << "\n--- Prefill Benchmark (all batch sizes) ---\n";
        std::cout << "+---------+----------+------------+\n";
        std::cout << "| Tokens  | Time(ms) | Tok/s      |\n";
        std::cout << "+---------+----------+------------+\n";

        for (int b = 0; b < NUM_BATCHES; b++) {
            int seq_len = BATCH_SIZES[b];
            auto prefill = prefill_fns[b];

            if (!prefill) {
                printf("|  %4d   |   N/A    |    N/A     |\n", seq_len);
                continue;
            }

            // Allocate buffers for this batch size
            auto x_i8 = alloc_aligned<int8_t>(seq_len * DIM);
            auto x_fp32 = alloc_aligned<float>(seq_len * DIM);
            auto xb_fp32 = alloc_aligned<float>(seq_len * DIM);
            auto xb_i8 = alloc_aligned<int8_t>(seq_len * DIM);
            auto k_cache_pf = alloc_aligned<int8_t>(N_LAYERS * seq_len * KV_DIM);
            auto v_cache_pf = alloc_aligned<int8_t>(N_LAYERS * seq_len * KV_DIM);
            auto att_scores_pf = alloc_aligned<float>(N_HEADS * seq_len * seq_len);
            auto att_probs_pf = alloc_aligned<float>(N_HEADS * seq_len * seq_len);
            auto attn_out = alloc_aligned<int32_t>(seq_len * DIM);
            auto ffn_hb = alloc_aligned<int8_t>(seq_len * HIDDEN_DIM);

            init_random_i8(x_i8, seq_len * DIM);
            memset(k_cache_pf, 0, N_LAYERS * seq_len * KV_DIM);
            memset(v_cache_pf, 0, N_LAYERS * seq_len * KV_DIM);

            // Warmup
            for (int i = 0; i < 2; i++) {
                prefill(
                    PASS_I8(x_i8, seq_len * DIM), PASS_F32(rms_att_w, N_LAYERS * DIM),
                    PASS_F32(rms_ffn_w, N_LAYERS * DIM), PASS_F32(rms_final_w, DIM),
                    PASS_I8(wq_t, N_LAYERS * DIM * DIM), PASS_I8(wk_t, N_LAYERS * KV_DIM * DIM),
                    PASS_I8(wv_t, N_LAYERS * KV_DIM * DIM), PASS_I8(wo_t, N_LAYERS * DIM * DIM),
                    PASS_I8(w1_t, N_LAYERS * HIDDEN_DIM * DIM), PASS_I8(w2_t, N_LAYERS * DIM * HIDDEN_DIM),
                    PASS_I8(w3_t, N_LAYERS * HIDDEN_DIM * DIM),
                    PASS_F32(x_fp32, seq_len * DIM), PASS_F32(xb_fp32, seq_len * DIM),
                    PASS_I8(xb_i8, seq_len * DIM),
                    PASS_I8(k_cache_pf, N_LAYERS * seq_len * KV_DIM),
                    PASS_I8(v_cache_pf, N_LAYERS * seq_len * KV_DIM),
                    PASS_F32(att_scores_pf, N_HEADS * seq_len * seq_len),
                    PASS_F32(att_probs_pf, N_HEADS * seq_len * seq_len),
                    PASS_I32(attn_out, seq_len * DIM), PASS_I8(ffn_hb, seq_len * HIDDEN_DIM),
                    seq_len, DIM, HIDDEN_DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HEAD_DIM
                );
            }

            // Benchmark
            const int ITERS = 5;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < ITERS; i++) {
                prefill(
                    PASS_I8(x_i8, seq_len * DIM), PASS_F32(rms_att_w, N_LAYERS * DIM),
                    PASS_F32(rms_ffn_w, N_LAYERS * DIM), PASS_F32(rms_final_w, DIM),
                    PASS_I8(wq_t, N_LAYERS * DIM * DIM), PASS_I8(wk_t, N_LAYERS * KV_DIM * DIM),
                    PASS_I8(wv_t, N_LAYERS * KV_DIM * DIM), PASS_I8(wo_t, N_LAYERS * DIM * DIM),
                    PASS_I8(w1_t, N_LAYERS * HIDDEN_DIM * DIM), PASS_I8(w2_t, N_LAYERS * DIM * HIDDEN_DIM),
                    PASS_I8(w3_t, N_LAYERS * HIDDEN_DIM * DIM),
                    PASS_F32(x_fp32, seq_len * DIM), PASS_F32(xb_fp32, seq_len * DIM),
                    PASS_I8(xb_i8, seq_len * DIM),
                    PASS_I8(k_cache_pf, N_LAYERS * seq_len * KV_DIM),
                    PASS_I8(v_cache_pf, N_LAYERS * seq_len * KV_DIM),
                    PASS_F32(att_scores_pf, N_HEADS * seq_len * seq_len),
                    PASS_F32(att_probs_pf, N_HEADS * seq_len * seq_len),
                    PASS_I32(attn_out, seq_len * DIM), PASS_I8(ffn_hb, seq_len * HIDDEN_DIM),
                    seq_len, DIM, HIDDEN_DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HEAD_DIM
                );
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;
            double tok_per_sec = (double)seq_len * 1000.0 / ms;

            printf("|  %4d   | %8.2f | %10.1f |\n", seq_len, ms, tok_per_sec);

            free(x_i8); free(x_fp32); free(xb_fp32); free(xb_i8);
            free(k_cache_pf); free(v_cache_pf);
            free(att_scores_pf); free(att_probs_pf);
            free(attn_out); free(ffn_hb);
        }
        std::cout << "+---------+----------+------------+\n";
    }

    // Memory usage summary
    size_t weight_bytes =
        N_LAYERS * DIM * sizeof(float) * 2 +  // rms weights (FP32)
        DIM * sizeof(float) +                  // final rms (FP32)
        N_LAYERS * DIM * DIM +                 // wq (INT8)
        N_LAYERS * KV_DIM * DIM * 2 +         // wk, wv (INT8)
        N_LAYERS * DIM * DIM +                 // wo (INT8)
        N_LAYERS * HIDDEN_DIM * DIM * 2 +     // w1, w3 (INT8)
        N_LAYERS * DIM * HIDDEN_DIM +         // w2 (INT8)
        VOCAB_SIZE * DIM * sizeof(float);     // wcls (FP32)

    printf("\n--- Memory Usage ---\n");
    printf("  Weights: %.1f MB (INT8 matmul + FP32 norm/cls)\n", weight_bytes / 1e6);
    printf("  KV Cache (max): %.1f MB (INT8)\n",
           (N_LAYERS * MAX_SEQ_LEN * KV_DIM * 2) / 1e6);

    // Cleanup
    free(rms_att_w); free(rms_ffn_w); free(rms_final_w);
    free(wq_t); free(wk_t); free(wv_t); free(wo_t);
    free(w1_t); free(w2_t); free(w3_t); free(wcls);

    dlclose(decode_handle);
    if (prefill_handle && prefill_handle != decode_handle) {
        dlclose(prefill_handle);
    }

    return 0;
}
