#include <iostream>
#include <dlfcn.h>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <cstring>

#define MEMREF_PARAMS float*, float*, int64_t, int64_t, int64_t
#define MEMREF_PARAMS_I8 int8_t*, int8_t*, int64_t, int64_t, int64_t
#define PASS_MEMREF(vec) vec.data(), vec.data(), 0, (int64_t)vec.size(), 1
#define PASS_MEMREF_I8(vec) vec.data(), vec.data(), 0, (int64_t)vec.size(), 1

typedef float (*LLaMA2QuantFunc)(
    MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS,  // FP32 weights
    MEMREF_PARAMS_I8, MEMREF_PARAMS, MEMREF_PARAMS,  // Quantized weights
    MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS,     // Activations 1
    MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS,     // Activations 2
    MEMREF_PARAMS, MEMREF_PARAMS,                     // Attention
    MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS,     // FFN
    MEMREF_PARAMS, MEMREF_PARAMS, MEMREF_PARAMS,     // Logits + KV cache
    int64_t, int64_t, int64_t, int64_t,               // token, pos, dim, hidden_dim
    int64_t, int64_t, int64_t, int64_t, int64_t,     // n_layers, n_heads, n_kv_heads, vocab_size, seq_len
    int64_t,                                           // group_size
    int64_t, int64_t, int64_t, int64_t,               // wq_off, wk_off, wv_off, wo_off
    int64_t, int64_t, int64_t                         // w1_off, w2_off, w3_off
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
    int64_t group_size = 128;
};

void init_random(std::vector<float>& arr) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-0.01f, 0.01f);
    for (auto& val : arr) {
        val = dis(gen);
    }
}

void quantize_w4(const std::vector<float>& weights, std::vector<int8_t>& qweights,
                 std::vector<float>& scales, std::vector<float>& zeros,
                 int64_t group_size) {
    int64_t total = weights.size();
    int64_t num_groups = (total + group_size - 1) / group_size;

    int64_t qweight_start = qweights.size();
    int64_t scale_start = scales.size();

    qweights.resize(qweight_start + (total + 1) / 2);
    scales.resize(scale_start + num_groups);
    zeros.resize(scale_start + num_groups);

    for (int64_t g = 0; g < num_groups; g++) {
        int64_t start = g * group_size;
        int64_t end = std::min(start + group_size, total);

        float min_val = weights[start];
        float max_val = weights[start];
        for (int64_t i = start; i < end; i++) {
            min_val = std::min(min_val, weights[i]);
            max_val = std::max(max_val, weights[i]);
        }

        float scale = (max_val - min_val) / 15.0f;
        if (scale < 1e-8f) scale = 1e-8f;

        scales[scale_start + g] = scale;
        zeros[scale_start + g] = min_val;

        for (int64_t i = start; i < end; i++) {
            int qval = std::round((weights[i] - min_val) / scale);
            qval = std::max(0, std::min(15, qval));

            int64_t byte_idx = qweight_start + i / 2;
            if (i % 2 == 0) {
                qweights[byte_idx] = qval;
            } else {
                qweights[byte_idx] |= (qval << 4);
            }
        }
    }
}

size_t calculate_memory_mb(const ModelConfig& cfg, bool quantized) {
    size_t kv_dim = (cfg.dim * cfg.n_kv_heads) / cfg.n_heads;

    // FP32 weights (kept in FP32)
    size_t token_emb = cfg.vocab_size * cfg.dim;
    size_t rms = 2 * cfg.n_layers * cfg.dim;
    size_t wcls = cfg.vocab_size * cfg.dim;
    size_t rms_final = cfg.dim;

    size_t fp32_total = (token_emb + rms + wcls + rms_final) * sizeof(float);

    // Quantized weights
    size_t wqkv = cfg.n_layers * (cfg.dim * cfg.dim + 2 * cfg.dim * kv_dim);
    size_t wo = cfg.n_layers * cfg.dim * cfg.dim;
    size_t ffn = cfg.n_layers * (2 * cfg.hidden_dim * cfg.dim + cfg.dim * cfg.hidden_dim);

    size_t quant_weights = wqkv + wo + ffn;
    size_t quant_total;

    if (quantized) {
        // W4: 4 bits per weight + scales/zeros
        size_t num_groups = (quant_weights + cfg.group_size - 1) / cfg.group_size;
        quant_total = (quant_weights / 2) + num_groups * 2 * sizeof(float);
    } else {
        quant_total = quant_weights * sizeof(float);
    }

    // Activations + KV cache
    size_t activations = (cfg.dim * 5 + kv_dim * 3 + cfg.n_heads * cfg.seq_len + cfg.hidden_dim * 2 + cfg.vocab_size) * sizeof(float);
    size_t kv_cache = 2 * cfg.n_layers * cfg.seq_len * kv_dim * sizeof(float);

    return (fp32_total + quant_total + activations + kv_cache) / (1024 * 1024);
}

void benchmark_model_quantized(LLaMA2QuantFunc kernel, const ModelConfig& cfg) {
    std::cout << "\n=== " << cfg.name << " (W4 Quantized) ===" << std::endl;
    std::cout << "Parameters: " << cfg.params_M << "M" << std::endl;
    std::cout << "Config: dim=" << cfg.dim << ", n_layers=" << cfg.n_layers
              << ", n_heads=" << cfg.n_heads << ", vocab=" << cfg.vocab_size << std::endl;

    int64_t kv_dim = (cfg.dim * cfg.n_kv_heads) / cfg.n_heads;

    size_t memory_mb = calculate_memory_mb(cfg, true);
    size_t fp32_memory_mb = calculate_memory_mb(cfg, false);

    std::cout << "Memory: FP32=" << fp32_memory_mb << " MB, W4=" << memory_mb
              << " MB (" << (float)fp32_memory_mb / memory_mb << "x compression)" << std::endl;

    if (memory_mb > 16000) {
        std::cout << "WARNING: May exceed system memory, skipping..." << std::endl;
        return;
    }

    std::cout << "Allocating memory..." << std::flush;

    try {
        // FP32 weights
        std::vector<float> token_emb(cfg.vocab_size * cfg.dim);
        std::vector<float> rms_att_w(cfg.n_layers * cfg.dim);
        std::vector<float> rms_ffn_w(cfg.n_layers * cfg.dim);
        std::vector<float> rms_final_w(cfg.dim);
        std::vector<float> wcls(cfg.vocab_size * cfg.dim);

        // Quantized weights (concatenated)
        std::vector<int8_t> qweights;
        std::vector<float> scales;
        std::vector<float> zeros;

        // Temporary for quantization
        std::vector<float> wq_temp(cfg.n_layers * cfg.dim * cfg.dim);
        std::vector<float> wk_temp(cfg.n_layers * cfg.dim * kv_dim);
        std::vector<float> wv_temp(cfg.n_layers * cfg.dim * kv_dim);
        std::vector<float> wo_temp(cfg.n_layers * cfg.dim * cfg.dim);
        std::vector<float> w1_temp(cfg.n_layers * cfg.hidden_dim * cfg.dim);
        std::vector<float> w2_temp(cfg.n_layers * cfg.dim * cfg.hidden_dim);
        std::vector<float> w3_temp(cfg.n_layers * cfg.hidden_dim * cfg.dim);

        std::cout << " OK" << std::endl;
        std::cout << "Initializing and quantizing weights..." << std::flush;

        init_random(token_emb);
        init_random(rms_att_w);
        init_random(rms_ffn_w);
        init_random(rms_final_w);
        init_random(wcls);

        init_random(wq_temp);
        init_random(wk_temp);
        init_random(wv_temp);
        init_random(wo_temp);
        init_random(w1_temp);
        init_random(w2_temp);
        init_random(w3_temp);

        // Quantize and track offsets
        int64_t wq_off = 0;
        quantize_w4(wq_temp, qweights, scales, zeros, cfg.group_size);

        int64_t wk_off = wq_temp.size();
        quantize_w4(wk_temp, qweights, scales, zeros, cfg.group_size);

        int64_t wv_off = wk_off + wk_temp.size();
        quantize_w4(wv_temp, qweights, scales, zeros, cfg.group_size);

        int64_t wo_off = wv_off + wv_temp.size();
        quantize_w4(wo_temp, qweights, scales, zeros, cfg.group_size);

        int64_t w1_off = wo_off + wo_temp.size();
        quantize_w4(w1_temp, qweights, scales, zeros, cfg.group_size);

        int64_t w2_off = w1_off + w1_temp.size();
        quantize_w4(w2_temp, qweights, scales, zeros, cfg.group_size);

        int64_t w3_off = w2_off + w2_temp.size();
        quantize_w4(w3_temp, qweights, scales, zeros, cfg.group_size);

        // Activations
        std::vector<float> x(cfg.dim);
        std::vector<float> xb(cfg.dim);
        std::vector<float> xb2(cfg.dim);
        std::vector<float> q(cfg.dim);
        std::vector<float> k(kv_dim);
        std::vector<float> v(kv_dim);
        std::vector<float> att(cfg.n_heads * cfg.seq_len);
        std::vector<float> att_soft(cfg.n_heads * cfg.seq_len);
        std::vector<float> hb(cfg.hidden_dim);
        std::vector<float> hb_silu(cfg.hidden_dim);
        std::vector<float> hb2(cfg.hidden_dim);
        std::vector<float> logits(cfg.vocab_size);
        std::vector<float> key_cache(cfg.n_layers * cfg.seq_len * kv_dim, 0.0f);
        std::vector<float> value_cache(cfg.n_layers * cfg.seq_len * kv_dim, 0.0f);

        std::cout << " OK" << std::endl;
        std::cout << "Quantized weights: " << qweights.size() / 1024.0 << " KB" << std::endl;

        // Warmup
        std::cout << "Warming up..." << std::flush;
        for (int i = 0; i < 2; i++) {
            kernel(
                PASS_MEMREF(token_emb), PASS_MEMREF(rms_att_w), PASS_MEMREF(rms_ffn_w),
                PASS_MEMREF(rms_final_w), PASS_MEMREF(wcls),
                PASS_MEMREF_I8(qweights), PASS_MEMREF(scales), PASS_MEMREF(zeros),
                PASS_MEMREF(x), PASS_MEMREF(xb), PASS_MEMREF(xb2),
                PASS_MEMREF(q), PASS_MEMREF(k), PASS_MEMREF(v),
                PASS_MEMREF(att), PASS_MEMREF(att_soft),
                PASS_MEMREF(hb), PASS_MEMREF(hb_silu), PASS_MEMREF(hb2),
                PASS_MEMREF(logits), PASS_MEMREF(key_cache), PASS_MEMREF(value_cache),
                100, 0, cfg.dim, cfg.hidden_dim,
                cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size, cfg.seq_len,
                cfg.group_size,
                wq_off, wk_off, wv_off, wo_off, w1_off, w2_off, w3_off
            );
        }
        std::cout << " OK" << std::endl;

        // Benchmark
        std::cout << "Running autoregressive sequence (10 tokens)..." << std::endl;
        std::fill(key_cache.begin(), key_cache.end(), 0.0f);
        std::fill(value_cache.begin(), value_cache.end(), 0.0f);

        auto start = std::chrono::high_resolution_clock::now();

        for (int pos = 0; pos < 10; pos++) {
            int64_t token = 50 + pos * 10;
            float result = kernel(
                PASS_MEMREF(token_emb), PASS_MEMREF(rms_att_w), PASS_MEMREF(rms_ffn_w),
                PASS_MEMREF(rms_final_w), PASS_MEMREF(wcls),
                PASS_MEMREF_I8(qweights), PASS_MEMREF(scales), PASS_MEMREF(zeros),
                PASS_MEMREF(x), PASS_MEMREF(xb), PASS_MEMREF(xb2),
                PASS_MEMREF(q), PASS_MEMREF(k), PASS_MEMREF(v),
                PASS_MEMREF(att), PASS_MEMREF(att_soft),
                PASS_MEMREF(hb), PASS_MEMREF(hb_silu), PASS_MEMREF(hb2),
                PASS_MEMREF(logits), PASS_MEMREF(key_cache), PASS_MEMREF(value_cache),
                token, pos, cfg.dim, cfg.hidden_dim,
                cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size, cfg.seq_len,
                cfg.group_size,
                wq_off, wk_off, wv_off, wo_off, w1_off, w2_off, w3_off
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

        double flops_per_token = 2.0 * cfg.params_M * 1e6;
        double gflops = (flops_per_token * 10.0) / (total_time_ms * 1e-3) / 1e9;

        size_t kv_cache_mb = (2 * cfg.n_layers * cfg.seq_len * kv_dim * sizeof(float)) / (1024 * 1024);

        std::cout << "\n--- Performance Metrics ---" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Average time/token: " << avg_time_ms << " ms" << std::endl;
        std::cout << "Throughput: " << tokens_per_sec << " tokens/s" << std::endl;
        std::cout << "Compute: " << gflops << " GFLOPS" << std::endl;
        std::cout << "KV cache size: " << kv_cache_mb << " MB" << std::endl;
        std::cout << "Total memory used: " << memory_mb << " MB" << std::endl;
        std::cout << "Compression vs FP32: " << (float)fp32_memory_mb / memory_mb << "x" << std::endl;

    } catch (const std::bad_alloc& e) {
        std::cerr << "ERROR: Failed to allocate memory!" << std::endl;
        return;
    }
}

int main() {
    std::cout << "=== LLaMA W4 Quantized Benchmark ===" << std::endl;

    void* handle = dlopen("/tmp/llama2_quant.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Error loading library: " << dlerror() << std::endl;
        return 1;
    }

    LLaMA2QuantFunc llama2_quant = (LLaMA2QuantFunc)dlsym(handle, "llama2_quant_forward");
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
        std::cerr << "Error loading symbol: " << dlsym_error << std::endl;
        dlclose(handle);
        return 1;
    }

    std::vector<ModelConfig> configs = {
        {"LLaMA 1B", 1000, 1536, 24, 16, 16, 6144, 32000, 512},
        {"LLaMA 3B", 3000, 2048, 32, 32, 32, 8192, 32000, 1024},
        {"LLaMA 7B", 7000, 4096, 32, 32, 32, 11008, 32000, 2048},
    };

    for (const auto& cfg : configs) {
        benchmark_model_quantized(llama2_quant, cfg);
    }

    std::cout << "\n=== Benchmark Complete ===" << std::endl;

    dlclose(handle);
    return 0;
}
