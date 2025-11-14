#include <iostream>
#include <dlfcn.h>
#include <cmath>
#include <cstring>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>

// Helper macro for memref descriptor (5 params per array)
#define MEMREF_PARAMS float*, float*, int64_t, int64_t, int64_t

typedef float (*LLaMA2Func)(
    // 26 arrays × 5 params each = 130 params
    MEMREF_PARAMS,  // token_emb
    MEMREF_PARAMS,  // rms_att_w
    MEMREF_PARAMS,  // wq
    MEMREF_PARAMS,  // wk
    MEMREF_PARAMS,  // wv
    MEMREF_PARAMS,  // wo
    MEMREF_PARAMS,  // rms_ffn_w
    MEMREF_PARAMS,  // w1
    MEMREF_PARAMS,  // w2
    MEMREF_PARAMS,  // w3
    MEMREF_PARAMS,  // rms_final_w
    MEMREF_PARAMS,  // wcls
    MEMREF_PARAMS,  // x
    MEMREF_PARAMS,  // xb
    MEMREF_PARAMS,  // xb2
    MEMREF_PARAMS,  // hb
    MEMREF_PARAMS,  // hb_silu
    MEMREF_PARAMS,  // q
    MEMREF_PARAMS,  // k
    MEMREF_PARAMS,  // v
    MEMREF_PARAMS,  // att
    MEMREF_PARAMS,  // att_soft
    MEMREF_PARAMS,  // logits
    MEMREF_PARAMS,  // key_cache
    MEMREF_PARAMS,  // value_cache
    // 9 scalar params
    int64_t, int64_t,  // token, pos
    int64_t, int64_t,  // dim, hidden_dim
    int64_t, int64_t, int64_t,  // n_layers, n_heads, n_kv_heads
    int64_t, int64_t  // vocab_size, seq_len
);

// Helper macro to pass array as memref descriptor
#define PASS_MEMREF(vec) vec.data(), vec.data(), 0, (int64_t)vec.size(), 1

void init_random(std::vector<float>& arr) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-0.1f, 0.1f);
    for (auto& val : arr) {
        val = dis(gen);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model.so> <model_size: 1B|3B|7B>" << std::endl;
        return 1;
    }

    const char* so_path = argv[1];
    const char* model_size = argv[2];

    // Model configurations
    int64_t dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;

    if (strcmp(model_size, "1B") == 0) {
        // LLaMA 2 1B: dim=1536, hidden_dim=6144, n_layers=24, n_heads=16, vocab=32000
        dim = 1536;
        hidden_dim = 6144;
        n_layers = 24;
        n_heads = 16;
        n_kv_heads = 16;
        vocab_size = 32000;
        seq_len = 512;
    } else if (strcmp(model_size, "3B") == 0) {
        // LLaMA 2 3B: dim=2048, hidden_dim=8192, n_layers=32, n_heads=32, vocab=32000
        dim = 2048;
        hidden_dim = 8192;
        n_layers = 32;
        n_heads = 32;
        n_kv_heads = 32;
        vocab_size = 32000;
        seq_len = 512;
    } else if (strcmp(model_size, "7B") == 0) {
        // LLaMA 2 7B: dim=4096, hidden_dim=11008, n_layers=32, n_heads=32, vocab=32000
        dim = 4096;
        hidden_dim = 11008;
        n_layers = 32;
        n_heads = 32;
        n_kv_heads = 32;
        vocab_size = 32000;
        seq_len = 512;
    } else {
        std::cerr << "Unknown model size: " << model_size << ". Use 1B, 3B, or 7B" << std::endl;
        return 1;
    }

    int64_t kv_dim = (dim * n_kv_heads) / n_heads;

    std::cout << "=== LLaMA2 Forward Pass Benchmark ===" << std::endl;
    std::cout << "Config: dim=" << dim << ", n_layers=" << n_layers
              << ", n_heads=" << n_heads << ", vocab=" << vocab_size << std::endl;

    // Allocate weights
    std::vector<float> token_emb(vocab_size * dim);
    std::vector<float> rms_att_w(n_layers * dim);
    std::vector<float> rms_ffn_w(n_layers * dim);
    std::vector<float> wq(n_layers * dim * dim);
    std::vector<float> wk(n_layers * dim * kv_dim);
    std::vector<float> wv(n_layers * dim * kv_dim);
    std::vector<float> wo(n_layers * dim * dim);
    std::vector<float> w1(n_layers * hidden_dim * dim);
    std::vector<float> w2(n_layers * dim * hidden_dim);
    std::vector<float> w3(n_layers * hidden_dim * dim);
    std::vector<float> rms_final_w(dim);
    std::vector<float> wcls(vocab_size * dim);

    // Allocate activations
    std::vector<float> x(dim);
    std::vector<float> xb(dim);
    std::vector<float> xb2(dim);
    std::vector<float> hb(hidden_dim);
    std::vector<float> hb_silu(hidden_dim);
    std::vector<float> q(dim);
    std::vector<float> k(kv_dim);
    std::vector<float> v(kv_dim);
    std::vector<float> att(n_heads * seq_len);
    std::vector<float> att_soft(n_heads * seq_len);
    std::vector<float> logits(vocab_size);

    // KV cache
    std::vector<float> key_cache(n_layers * seq_len * kv_dim, 0.0f);
    std::vector<float> value_cache(n_layers * seq_len * kv_dim, 0.0f);

    std::cout << "Initializing weights..." << std::endl;
    init_random(token_emb);
    init_random(rms_att_w);
    init_random(rms_ffn_w);
    init_random(wq);
    init_random(wk);
    init_random(wv);
    init_random(wo);
    init_random(w1);
    init_random(w2);
    init_random(w3);
    init_random(rms_final_w);
    init_random(wcls);

    // Load library
    void* handle = dlopen(so_path, RTLD_LAZY);
    if (!handle) {
        std::cerr << "Error loading library: " << dlerror() << std::endl;
        return 1;
    }

    LLaMA2Func llama2_forward = (LLaMA2Func)dlsym(handle, "llama2_forward");
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
        std::cerr << "Error loading symbol: " << dlsym_error << std::endl;
        dlclose(handle);
        return 1;
    }

    std::cout << "Running inference..." << std::endl;

    // Warmup
    for (int i = 0; i < 3; i++) {
        llama2_forward(
            PASS_MEMREF(token_emb),
            PASS_MEMREF(rms_att_w),
            PASS_MEMREF(wq),
            PASS_MEMREF(wk),
            PASS_MEMREF(wv),
            PASS_MEMREF(wo),
            PASS_MEMREF(rms_ffn_w),
            PASS_MEMREF(w1),
            PASS_MEMREF(w2),
            PASS_MEMREF(w3),
            PASS_MEMREF(rms_final_w),
            PASS_MEMREF(wcls),
            PASS_MEMREF(x),
            PASS_MEMREF(xb),
            PASS_MEMREF(xb2),
            PASS_MEMREF(hb),
            PASS_MEMREF(hb_silu),
            PASS_MEMREF(q),
            PASS_MEMREF(k),
            PASS_MEMREF(v),
            PASS_MEMREF(att),
            PASS_MEMREF(att_soft),
            PASS_MEMREF(logits),
            PASS_MEMREF(key_cache),
            PASS_MEMREF(value_cache),
            100, 0,  // token, pos
            dim, hidden_dim,
            n_layers, n_heads, n_kv_heads,
            vocab_size, seq_len
        );
    }

    // Test autoregressive generation (different tokens at different positions)
    std::cout << "\nTesting autoregressive sequence (tokens 50, 60, 70 at pos 0, 1, 2):" << std::endl;

    // Clear KV cache
    std::fill(key_cache.begin(), key_cache.end(), 0.0f);
    std::fill(value_cache.begin(), value_cache.end(), 0.0f);

    int64_t tokens[] = {50, 60, 70};
    for (int p = 0; p < 3; p++) {
        float res = llama2_forward(
            PASS_MEMREF(token_emb),
            PASS_MEMREF(rms_att_w),
            PASS_MEMREF(wq),
            PASS_MEMREF(wk),
            PASS_MEMREF(wv),
            PASS_MEMREF(wo),
            PASS_MEMREF(rms_ffn_w),
            PASS_MEMREF(w1),
            PASS_MEMREF(w2),
            PASS_MEMREF(w3),
            PASS_MEMREF(rms_final_w),
            PASS_MEMREF(wcls),
            PASS_MEMREF(x),
            PASS_MEMREF(xb),
            PASS_MEMREF(xb2),
            PASS_MEMREF(hb),
            PASS_MEMREF(hb_silu),
            PASS_MEMREF(q),
            PASS_MEMREF(k),
            PASS_MEMREF(v),
            PASS_MEMREF(att),
            PASS_MEMREF(att_soft),
            PASS_MEMREF(logits),
            PASS_MEMREF(key_cache),
            PASS_MEMREF(value_cache),
            tokens[p], p,
            dim, hidden_dim,
            n_layers, n_heads, n_kv_heads,
            vocab_size, seq_len
        );
        std::cout << "  pos " << p << " (token " << tokens[p] << "): " << res << std::endl;
    }

    // Benchmark: Generate 10 tokens sequentially
    std::cout << "\nBenchmark: Generating 10 tokens..." << std::endl;
    const int num_tokens = 10;

    // Clear KV cache
    std::fill(key_cache.begin(), key_cache.end(), 0.0f);
    std::fill(value_cache.begin(), value_cache.end(), 0.0f);

    auto start = std::chrono::high_resolution_clock::now();

    for (int pos = 0; pos < num_tokens; pos++) {
        float result = llama2_forward(
            PASS_MEMREF(token_emb),
            PASS_MEMREF(rms_att_w),
            PASS_MEMREF(wq),
            PASS_MEMREF(wk),
            PASS_MEMREF(wv),
            PASS_MEMREF(wo),
            PASS_MEMREF(rms_ffn_w),
            PASS_MEMREF(w1),
            PASS_MEMREF(w2),
            PASS_MEMREF(w3),
            PASS_MEMREF(rms_final_w),
            PASS_MEMREF(wcls),
            PASS_MEMREF(x),
            PASS_MEMREF(xb),
            PASS_MEMREF(xb2),
            PASS_MEMREF(hb),
            PASS_MEMREF(hb_silu),
            PASS_MEMREF(q),
            PASS_MEMREF(k),
            PASS_MEMREF(v),
            PASS_MEMREF(att),
            PASS_MEMREF(att_soft),
            PASS_MEMREF(logits),
            PASS_MEMREF(key_cache),
            PASS_MEMREF(value_cache),
            100 + pos, pos,  // token, pos
            dim, hidden_dim,
            n_layers, n_heads, n_kv_heads,
            vocab_size, seq_len
        );

        if (!std::isfinite(result)) {
            std::cerr << "ERROR: Output is not finite at position " << pos << std::endl;
            dlclose(handle);
            return 1;
        }
        std::cout << "  Token " << pos << ": logit=" << result << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double total_time_ms = duration.count() / 1000.0;
    double avg_time_ms = total_time_ms / num_tokens;
    double tokens_per_sec = 1000.0 / avg_time_ms;

    // Calculate GFLOPS
    // FLOPs per token = 2 * n_params (forward pass approximation)
    // For transformer: ~2 * (12 * n_layers * dim^2 + vocab * dim)
    double flops_per_token = 2.0 * (12.0 * n_layers * dim * dim + vocab_size * dim);
    double gflops = (flops_per_token * tokens_per_sec) / 1e9;

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Total time: " << total_time_ms << " ms" << std::endl;
    std::cout << "Average time per token: " << avg_time_ms << " ms" << std::endl;
    std::cout << "Throughput: " << tokens_per_sec << " tokens/s" << std::endl;
    std::cout << "Compute: " << gflops << " GFLOPS" << std::endl;
    std::cout << "SUCCESS!" << std::endl;

    dlclose(handle);
    return 0;
}
