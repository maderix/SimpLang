// Test llama2_quant.sl with fake weights to measure roofline performance
#include <iostream>
#include <vector>
#include <dlfcn.h>
#include <chrono>
#include <cstring>

#define PASS_MEMREF(vec) vec.data(), vec.data(), 0, vec.size(), 1
#define PASS_MEMREF_I8(vec) (int8_t*)vec.data(), (int8_t*)vec.data(), 0, vec.size(), 1

typedef float (*KernelFunc)(
    float*, float*, int64_t, int64_t, int64_t,  // token_emb
    float*, float*, int64_t, int64_t, int64_t,  // rms_att_w
    float*, float*, int64_t, int64_t, int64_t,  // rms_ffn_w
    float*, float*, int64_t, int64_t, int64_t,  // rms_final_w
    float*, float*, int64_t, int64_t, int64_t,  // wcls
    int8_t*, int8_t*, int64_t, int64_t, int64_t,  // qweights
    float*, float*, int64_t, int64_t, int64_t,  // scales
    float*, float*, int64_t, int64_t, int64_t,  // zeros
    float*, float*, int64_t, int64_t, int64_t,  // x
    float*, float*, int64_t, int64_t, int64_t,  // xb
    float*, float*, int64_t, int64_t, int64_t,  // xb2
    float*, float*, int64_t, int64_t, int64_t,  // q
    float*, float*, int64_t, int64_t, int64_t,  // k
    float*, float*, int64_t, int64_t, int64_t,  // v
    float*, float*, int64_t, int64_t, int64_t,  // att
    float*, float*, int64_t, int64_t, int64_t,  // att_soft
    float*, float*, int64_t, int64_t, int64_t,  // hb
    float*, float*, int64_t, int64_t, int64_t,  // hb_silu
    float*, float*, int64_t, int64_t, int64_t,  // hb2
    float*, float*, int64_t, int64_t, int64_t,  // logits
    float*, float*, int64_t, int64_t, int64_t,  // key_cache
    float*, float*, int64_t, int64_t, int64_t,  // value_cache
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t
);

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <kernel.so>" << std::endl;
        return 1;
    }

    // Load kernel
    void* handle = dlopen(argv[1], RTLD_NOW);
    if (!handle) {
        std::cerr << "Failed to load kernel: " << dlerror() << std::endl;
        return 1;
    }

    auto kernel = (KernelFunc)dlsym(handle, "llama2_quant_forward");
    if (!kernel) {
        std::cerr << "Failed to find kernel function: " << dlerror() << std::endl;
        return 1;
    }

    // LLaMA 1.5B config
    const int64_t dim = 1536;
    const int64_t hidden_dim = 6144;
    const int64_t n_layers = 28;
    const int64_t n_heads = 12;
    const int64_t n_kv_heads = 12;
    const int64_t vocab_size = 32000;
    const int64_t seq_len = 256;
    const int64_t group_size = 128;
    const int64_t kv_dim = (dim * n_kv_heads) / n_heads;

    std::cout << "Allocating buffers for LLaMA 1.5B..." << std::flush;

    // FP32 weights
    std::vector<float> token_emb(vocab_size * dim, 1.0f);
    std::vector<float> rms_att_w(n_layers * dim, 1.0f);
    std::vector<float> rms_ffn_w(n_layers * dim, 1.0f);
    std::vector<float> rms_final_w(dim, 1.0f);
    std::vector<float> wcls(vocab_size * dim, 1.0f);

    // Calculate quantized weight sizes
    int64_t wq_size = n_layers * dim * dim;
    int64_t wk_size = n_layers * dim * kv_dim;
    int64_t wv_size = n_layers * dim * kv_dim;
    int64_t wo_size = n_layers * dim * dim;
    int64_t w1_size = n_layers * hidden_dim * dim;
    int64_t w2_size = n_layers * dim * hidden_dim;
    int64_t w3_size = n_layers * hidden_dim * dim;

    int64_t total_weights = wq_size + wk_size + wv_size + wo_size + w1_size + w2_size + w3_size;
    int64_t total_bytes = total_weights / 2;  // 4-bit = 0.5 bytes per weight
    int64_t total_groups = total_weights / group_size;

    std::vector<uint8_t> qweights(total_bytes, 0x55);  // Fake quantized weights
    std::vector<float> scales(total_groups, 0.01f);
    std::vector<float> zeros(total_groups, 0.0f);

    // Weight offsets
    int64_t wq_off = 0;
    int64_t wk_off = wq_off + wq_size;
    int64_t wv_off = wk_off + wk_size;
    int64_t wo_off = wv_off + wv_size;
    int64_t w1_off = wo_off + wo_size;
    int64_t w2_off = w1_off + w1_size;
    int64_t w3_off = w2_off + w2_size;

    // Activations
    std::vector<float> x(dim, 0.5f);
    std::vector<float> xb(dim);
    std::vector<float> xb2(dim);
    std::vector<float> q(dim);
    std::vector<float> k(kv_dim);
    std::vector<float> v(kv_dim);
    std::vector<float> att(n_heads * seq_len);
    std::vector<float> att_soft(n_heads * seq_len);
    std::vector<float> hb(hidden_dim);
    std::vector<float> hb_silu(hidden_dim);
    std::vector<float> hb2(hidden_dim);
    std::vector<float> logits(vocab_size);
    std::vector<float> key_cache(n_layers * seq_len * kv_dim, 0.0f);
    std::vector<float> value_cache(n_layers * seq_len * kv_dim, 0.0f);

    std::cout << " OK" << std::endl;
    std::cout << "Quantized weights: " << qweights.size() / (1024.0*1024.0) << " MB" << std::endl;

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
            100, 0, dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, group_size,
            wq_off, wk_off, wv_off, wo_off, w1_off, w2_off, w3_off
        );
    }
    std::cout << " OK" << std::endl;

    // Benchmark
    std::cout << "Benchmarking..." << std::endl;
    for (int run = 0; run < 3; run++) {
        auto start = std::chrono::high_resolution_clock::now();

        for (int pos = 0; pos < 10; pos++) {
            kernel(
                PASS_MEMREF(token_emb), PASS_MEMREF(rms_att_w), PASS_MEMREF(rms_ffn_w),
                PASS_MEMREF(rms_final_w), PASS_MEMREF(wcls),
                PASS_MEMREF_I8(qweights), PASS_MEMREF(scales), PASS_MEMREF(zeros),
                PASS_MEMREF(x), PASS_MEMREF(xb), PASS_MEMREF(xb2),
                PASS_MEMREF(q), PASS_MEMREF(k), PASS_MEMREF(v),
                PASS_MEMREF(att), PASS_MEMREF(att_soft),
                PASS_MEMREF(hb), PASS_MEMREF(hb_silu), PASS_MEMREF(hb2),
                PASS_MEMREF(logits), PASS_MEMREF(key_cache), PASS_MEMREF(value_cache),
                100, pos, dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, group_size,
                wq_off, wk_off, wv_off, wo_off, w1_off, w2_off, w3_off
            );
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        double tok_per_sec = 10.0 / elapsed;

        std::cout << "--- Performance Metrics ---" << std::endl;
        std::cout << "Throughput: " << tok_per_sec << " tokens/s" << std::endl;
    }

    dlclose(handle);
    return 0;
}
