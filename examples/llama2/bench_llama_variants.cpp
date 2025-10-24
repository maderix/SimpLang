#include <iostream>
#include <dlfcn.h>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <cstring>

// Helper macro for memref descriptor (5 params per array)
#define MEMREF_PARAMS float*, float*, int64_t, int64_t, int64_t
#define PASS_MEMREF(vec) vec.data(), vec.data(), 0, (int64_t)vec.size(), 1

typedef float (*LLaMA2Func)(
    MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS,
    MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS,
    MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS,
    MEMREF_PARAMS, MEMREF_PARAMS,
    MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS,
    MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS,
    MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS,
    MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS,
    int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t
);

struct ModelConfig {
    std::string name;
    int64_t params_M;
    int64_t dim;
    int64_t n_layers;
    int64_t n_heads;
    int64_t n_kv_heads;
    int64_t hidden_dim;
    int64_t vocab_size;
    int64_t seq_len;
};

void init_random(std::vector<float>& arr) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-0.01f, 0.01f);
    for (auto& val : arr) {
        val = dis(gen);
    }
}

size_t calculate_memory_mb(const ModelConfig& cfg) {
    size_t kv_dim = (cfg.dim * cfg.n_kv_heads) / cfg.n_heads;

    // Weights
    size_t token_emb = cfg.vocab_size * cfg.dim;
    size_t rms = 2 * cfg.n_layers * cfg.dim;
    size_t wqkv = cfg.n_layers * (cfg.dim * cfg.dim + 2 * cfg.dim * kv_dim);
    size_t wo = cfg.n_layers * cfg.dim * cfg.dim;
    size_t ffn = cfg.n_layers * (2 * cfg.hidden_dim * cfg.dim + cfg.dim * cfg.hidden_dim);
    size_t wcls = cfg.vocab_size * cfg.dim;
    size_t rms_final = cfg.dim;

    // Activations
    size_t activations = cfg.dim * 5 + kv_dim * 3 + cfg.n_heads * cfg.seq_len + cfg.hidden_dim * 2 + cfg.vocab_size;

    // KV cache
    size_t kv_cache = 2 * cfg.n_layers * cfg.seq_len * kv_dim;

    size_t total_floats = token_emb + rms + wqkv + wo + ffn + wcls + rms_final + activations + kv_cache;
    return (total_floats * sizeof(float)) / (1024 * 1024);
}

void benchmark_model(LLaMA2Func kernel, const ModelConfig& cfg) {
    std::cout << "\n=== " << cfg.name << " ===" << std::endl;
    std::cout << "Parameters: " << cfg.params_M << "M" << std::endl;
    std::cout << "Config: dim=" << cfg.dim << ", n_layers=" << cfg.n_layers
              << ", n_heads=" << cfg.n_heads << ", vocab=" << cfg.vocab_size << std::endl;

    int64_t kv_dim = (cfg.dim * cfg.n_kv_heads) / cfg.n_heads;

    size_t memory_mb = calculate_memory_mb(cfg);
    std::cout << "Estimated memory: " << memory_mb << " MB" << std::endl;

    if (memory_mb > 16000) {
        std::cout << "WARNING: May exceed system memory, skipping..." << std::endl;
        return;
    }

    std::cout << "Allocating memory..." << std::flush;

    try {
        // Allocate weights
        std::vector<float> token_emb(cfg.vocab_size * cfg.dim);
        std::vector<float> rms_att_w(cfg.n_layers * cfg.dim);
        std::vector<float> rms_ffn_w(cfg.n_layers * cfg.dim);
        std::vector<float> wq(cfg.n_layers * cfg.dim * cfg.dim);
        std::vector<float> wk(cfg.n_layers * cfg.dim * kv_dim);
        std::vector<float> wv(cfg.n_layers * cfg.dim * kv_dim);
        std::vector<float> wo(cfg.n_layers * cfg.dim * cfg.dim);
        std::vector<float> w1(cfg.n_layers * cfg.hidden_dim * cfg.dim);
        std::vector<float> w2(cfg.n_layers * cfg.dim * cfg.hidden_dim);
        std::vector<float> w3(cfg.n_layers * cfg.hidden_dim * cfg.dim);
        std::vector<float> rms_final_w(cfg.dim);
        std::vector<float> wcls(cfg.vocab_size * cfg.dim);

        // Allocate activations
        std::vector<float> x(cfg.dim);
        std::vector<float> xb(cfg.dim);
        std::vector<float> xb2(cfg.dim);
        std::vector<float> q(cfg.dim);
        std::vector<float> k(kv_dim);
        std::vector<float> v(kv_dim);
        std::vector<float> att(cfg.n_heads * cfg.seq_len);
        std::vector<float> hb(cfg.hidden_dim);
        std::vector<float> hb2(cfg.hidden_dim);
        std::vector<float> logits(cfg.vocab_size);

        // KV cache
        std::vector<float> key_cache(cfg.n_layers * cfg.seq_len * kv_dim, 0.0f);
        std::vector<float> value_cache(cfg.n_layers * cfg.seq_len * kv_dim, 0.0f);

        std::cout << " OK" << std::endl;
        std::cout << "Initializing weights..." << std::flush;

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

        std::cout << " OK" << std::endl;

        // Warmup
        std::cout << "Warming up..." << std::flush;
        for (int i = 0; i < 3; i++) {
            kernel(
                PASS_MEMREF(token_emb), PASS_MEMREF(rms_att_w), PASS_MEMREF(rms_ffn_w),
                PASS_MEMREF(wq), PASS_MEMREF(wk), PASS_MEMREF(wv), PASS_MEMREF(wo),
                PASS_MEMREF(w1), PASS_MEMREF(w2), PASS_MEMREF(w3),
                PASS_MEMREF(rms_final_w), PASS_MEMREF(wcls),
                PASS_MEMREF(x), PASS_MEMREF(xb), PASS_MEMREF(xb2),
                PASS_MEMREF(q), PASS_MEMREF(k), PASS_MEMREF(v),
                PASS_MEMREF(att), PASS_MEMREF(hb), PASS_MEMREF(hb2),
                PASS_MEMREF(logits), PASS_MEMREF(key_cache), PASS_MEMREF(value_cache),
                100, 0, cfg.dim, cfg.hidden_dim,
                cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size, cfg.seq_len
            );
        }
        std::cout << " OK" << std::endl;

        // Benchmark autoregressive sequence
        std::cout << "Running autoregressive sequence (10 tokens)..." << std::endl;
        std::fill(key_cache.begin(), key_cache.end(), 0.0f);
        std::fill(value_cache.begin(), value_cache.end(), 0.0f);

        auto start = std::chrono::high_resolution_clock::now();

        for (int pos = 0; pos < 10; pos++) {
            int64_t token = 50 + pos * 10;
            float result = kernel(
                PASS_MEMREF(token_emb), PASS_MEMREF(rms_att_w), PASS_MEMREF(rms_ffn_w),
                PASS_MEMREF(wq), PASS_MEMREF(wk), PASS_MEMREF(wv), PASS_MEMREF(wo),
                PASS_MEMREF(w1), PASS_MEMREF(w2), PASS_MEMREF(w3),
                PASS_MEMREF(rms_final_w), PASS_MEMREF(wcls),
                PASS_MEMREF(x), PASS_MEMREF(xb), PASS_MEMREF(xb2),
                PASS_MEMREF(q), PASS_MEMREF(k), PASS_MEMREF(v),
                PASS_MEMREF(att), PASS_MEMREF(hb), PASS_MEMREF(hb2),
                PASS_MEMREF(logits), PASS_MEMREF(key_cache), PASS_MEMREF(value_cache),
                token, pos, cfg.dim, cfg.hidden_dim,
                cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size, cfg.seq_len
            );

            if (pos < 3) {
                std::cout << "  pos " << pos << " (token " << token << "): " << result << std::endl;
            }

            if (!std::isfinite(result)) {
                std::cerr << "ERROR: Non-finite output at pos " << pos << std::endl;
                return;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        double total_time_ms = duration.count() / 1000.0;
        double avg_time_ms = total_time_ms / 10.0;
        double tokens_per_sec = 1000.0 / avg_time_ms;

        // Calculate FLOPs and bandwidth
        // Rough FLOP estimate per token: 2 * params + attention overhead
        double flops_per_token = 2.0 * cfg.params_M * 1e6;
        double gflops = (flops_per_token * 10.0) / (total_time_ms * 1e-3) / 1e9;

        // Memory bandwidth (weights + KV cache reads/writes)
        size_t bytes_per_token = memory_mb * 1024 * 1024;  // Rough estimate
        double bandwidth_gbps = (bytes_per_token * 10.0) / (total_time_ms * 1e-3) / 1e9;

        // KV cache size
        size_t kv_cache_mb = (2 * cfg.n_layers * cfg.seq_len * kv_dim * sizeof(float)) / (1024 * 1024);

        std::cout << "\n--- Performance Metrics ---" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Average time/token: " << avg_time_ms << " ms" << std::endl;
        std::cout << "Throughput: " << tokens_per_sec << " tokens/s" << std::endl;
        std::cout << "Compute: " << gflops << " GFLOPS" << std::endl;
        std::cout << "Memory bandwidth: " << bandwidth_gbps << " GB/s (estimate)" << std::endl;
        std::cout << "KV cache size: " << kv_cache_mb << " MB" << std::endl;
        std::cout << "Total memory used: " << memory_mb << " MB" << std::endl;

    } catch (const std::bad_alloc& e) {
        std::cerr << "ERROR: Failed to allocate memory!" << std::endl;
        return;
    }
}

int main() {
    std::cout << "=== LLaMA Variants Benchmark ===" << std::endl;
    std::cout << "Testing different model sizes with full transformer implementation" << std::endl;

    void* handle = dlopen("/tmp/llama2_forward.so", RTLD_LAZY);
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

    std::vector<ModelConfig> configs = {
        {"LLaMA 125M", 125, 768, 12, 12, 12, 3072, 32000, 256},
        {"LLaMA 500M", 500, 1024, 24, 16, 16, 4096, 32000, 512},
        {"LLaMA 1B", 1000, 1536, 24, 16, 16, 6144, 32000, 512},
        {"LLaMA 3B", 3000, 2048, 32, 32, 32, 8192, 32000, 1024},
        {"LLaMA 7B", 7000, 4096, 32, 32, 32, 11008, 32000, 2048},
        {"LLaMA 8B", 8000, 4096, 32, 32, 8, 14336, 128256, 2048},  // LLaMA 3.1 8B with GQA
        {"LLaMA 30B", 30000, 6656, 60, 52, 52, 17920, 32000, 2048},
    };

    for (const auto& cfg : configs) {
        benchmark_model(llama2_forward, cfg);
    }

    std::cout << "\n=== Benchmark Complete ===" << std::endl;

    dlclose(handle);
    return 0;
}
