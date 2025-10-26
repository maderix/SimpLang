#include <iostream>
#include <dlfcn.h>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>

#define MEMREF_PARAMS float*, float*, int64_t, int64_t, int64_t
#define PASS_MEMREF(vec) vec.data(), vec.data(), 0, (int64_t)vec.size(), 1

typedef float (*Stories110MFunc)(
    MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS,
    MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS,
    MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS,
    MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS,
    MEMREF_PARAMS,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t
);

void initialize_weights(std::vector<float>& weights, int seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    for (auto& w : weights) {
        w = dist(gen);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <stories110M.so>" << std::endl;
        return 1;
    }

    // Load shared library
    void* handle = dlopen(argv[1], RTLD_LAZY);
    if (!handle) {
        std::cerr << "Error loading library: " << dlerror() << std::endl;
        return 1;
    }

    // Load function
    Stories110MFunc stories110M = (Stories110MFunc)dlsym(handle, "stories110M_forward");
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
        std::cerr << "Error loading symbol: " << dlsym_error << std::endl;
        dlclose(handle);
        return 1;
    }

    std::cout << "=== stories110M Test (Karpathy's llama2.c) ===" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Parameters: 110M" << std::endl;
    std::cout << "  dim: 768" << std::endl;
    std::cout << "  n_layers: 12" << std::endl;
    std::cout << "  n_heads: 12" << std::endl;
    std::cout << "  hidden_dim: 3072 (4 * dim)" << std::endl;
    std::cout << "  vocab_size: 32000" << std::endl;
    std::cout << "  seq_len: 1024" << std::endl;
    std::cout << std::endl;

    // Model config
    const int64_t dim = 768;
    const int64_t n_layers = 12;
    const int64_t n_heads = 12;
    const int64_t n_kv_heads = 12;
    const int64_t hidden_dim = 3072;  // 4 * dim
    const int64_t vocab_size = 32000;
    const int64_t seq_len = 1024;
    const int64_t head_size = dim / n_heads;  // 64

    // Calculate memory requirements
    size_t token_emb_size = vocab_size * dim;  // 24.6M
    size_t layer_weights = n_layers * (
        dim +                    // rms_att_w
        4 * dim * dim +         // wq, wk, wv, wo
        dim +                    // rms_ffn_w
        2 * dim * hidden_dim +  // w1, w3
        hidden_dim * dim        // w2
    );
    size_t classifier_size = vocab_size * dim;  // 24.6M
    size_t total_params = token_emb_size + layer_weights + dim + classifier_size;
    size_t kv_cache_size = 2 * n_layers * seq_len * dim;

    std::cout << "Memory Requirements:" << std::endl;
    std::cout << "  Parameters: " << (total_params * 4 / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  KV Cache: " << (kv_cache_size * 4 / 1024 / 1024) << " MB" << std::endl;
    std::cout << std::endl;

    // Allocate weights
    std::cout << "Allocating weights..." << std::flush;
    std::vector<float> token_embedding_table(token_emb_size);
    std::vector<float> rms_att_w(n_layers * dim);
    std::vector<float> wq(n_layers * dim * dim);
    std::vector<float> wk(n_layers * dim * dim);
    std::vector<float> wv(n_layers * dim * dim);
    std::vector<float> wo(n_layers * dim * dim);
    std::vector<float> rms_ffn_w(n_layers * dim);
    std::vector<float> w1(n_layers * dim * hidden_dim);
    std::vector<float> w2(n_layers * hidden_dim * dim);
    std::vector<float> w3(n_layers * dim * hidden_dim);
    std::vector<float> rms_final_w(dim);
    std::vector<float> wcls(vocab_size * dim);
    std::cout << " OK" << std::endl;

    // Allocate activations
    std::vector<float> x(dim);
    std::vector<float> xb(dim);
    std::vector<float> xb2(std::max(dim, hidden_dim));
    std::vector<float> hb(hidden_dim);
    std::vector<float> hb_silu(hidden_dim);
    std::vector<float> q(dim);
    std::vector<float> k(dim);
    std::vector<float> v(dim);
    std::vector<float> att(n_heads * seq_len);
    std::vector<float> att_soft(n_heads * seq_len);
    std::vector<float> logits(vocab_size);
    std::vector<float> key_cache(n_layers * seq_len * dim);
    std::vector<float> value_cache(n_layers * seq_len * dim);

    // Initialize weights with small random values
    std::cout << "Initializing weights..." << std::flush;
    initialize_weights(token_embedding_table, 1);
    initialize_weights(rms_att_w, 2);
    initialize_weights(wq, 3);
    initialize_weights(wk, 4);
    initialize_weights(wv, 5);
    initialize_weights(wo, 6);
    initialize_weights(rms_ffn_w, 7);
    initialize_weights(w1, 8);
    initialize_weights(w2, 9);
    initialize_weights(w3, 10);
    initialize_weights(rms_final_w, 11);
    initialize_weights(wcls, 12);
    std::cout << " OK" << std::endl;

    // Warm-up run
    std::cout << "Warming up..." << std::flush;
    int64_t warmup_token = 50;
    int64_t warmup_pos = 0;
    stories110M(
        PASS_MEMREF(token_embedding_table),
        PASS_MEMREF(rms_att_w), PASS_MEMREF(wq), PASS_MEMREF(wk),
        PASS_MEMREF(wv), PASS_MEMREF(wo),
        PASS_MEMREF(rms_ffn_w), PASS_MEMREF(w1), PASS_MEMREF(w2), PASS_MEMREF(w3),
        PASS_MEMREF(rms_final_w), PASS_MEMREF(wcls),
        PASS_MEMREF(x), PASS_MEMREF(xb), PASS_MEMREF(xb2),
        PASS_MEMREF(hb), PASS_MEMREF(hb_silu),
        PASS_MEMREF(q), PASS_MEMREF(k), PASS_MEMREF(v),
        PASS_MEMREF(att), PASS_MEMREF(att_soft), PASS_MEMREF(logits),
        PASS_MEMREF(key_cache), PASS_MEMREF(value_cache),
        warmup_token, warmup_pos, dim, hidden_dim, n_layers,
        n_heads, n_kv_heads, vocab_size, seq_len
    );
    std::cout << " OK" << std::endl;

    // Run autoregressive sequence
    std::cout << "\nRunning autoregressive sequence (10 tokens)..." << std::endl;
    int64_t num_tokens = 10;

    auto start = std::chrono::high_resolution_clock::now();

    for (int64_t pos = 0; pos < num_tokens; pos++) {
        int64_t token = 50 + pos * 10;  // tokens: 50, 60, 70, ...

        auto token_start = std::chrono::high_resolution_clock::now();

        float result = stories110M(
            PASS_MEMREF(token_embedding_table),
            PASS_MEMREF(rms_att_w), PASS_MEMREF(wq), PASS_MEMREF(wk),
            PASS_MEMREF(wv), PASS_MEMREF(wo),
            PASS_MEMREF(rms_ffn_w), PASS_MEMREF(w1), PASS_MEMREF(w2), PASS_MEMREF(w3),
            PASS_MEMREF(rms_final_w), PASS_MEMREF(wcls),
            PASS_MEMREF(x), PASS_MEMREF(xb), PASS_MEMREF(xb2),
            PASS_MEMREF(hb), PASS_MEMREF(hb_silu),
            PASS_MEMREF(q), PASS_MEMREF(k), PASS_MEMREF(v),
            PASS_MEMREF(att), PASS_MEMREF(att_soft), PASS_MEMREF(logits),
            PASS_MEMREF(key_cache), PASS_MEMREF(value_cache),
            token, pos, dim, hidden_dim, n_layers,
            n_heads, n_kv_heads, vocab_size, seq_len
        );

        auto token_end = std::chrono::high_resolution_clock::now();
        auto token_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            token_end - token_start).count();

        std::cout << "  pos " << pos << " (token " << token << "): "
                  << result << " (" << token_time << " ms)" << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "\n--- Performance Metrics ---" << std::endl;
    double avg_time = duration / (double)num_tokens;
    double throughput = 1000.0 / avg_time;

    // Estimate FLOPs (approximate for transformer)
    double flops_per_token = 2.0 * total_params;  // rough estimate
    double gflops = (flops_per_token * throughput) / 1e9;

    std::cout << "Average time/token: " << avg_time << " ms" << std::endl;
    std::cout << "Throughput: " << throughput << " tokens/s" << std::endl;
    std::cout << "Compute: " << gflops << " GFLOPS" << std::endl;
    std::cout << "Total time: " << duration << " ms" << std::endl;

    std::cout << "\n=== Test Complete ===" << std::endl;

    dlclose(handle);
    return 0;
}
