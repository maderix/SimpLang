#include <iostream>
#include <dlfcn.h>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <random>
#include <cmath>
#include <algorithm>

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

// Config structure matching llama2.c format
struct Config {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
};

// Tokenizer structure
struct Tokenizer {
    std::vector<std::string> vocab;
    std::vector<float> vocab_scores;
    int vocab_size;
    int max_token_length;

    void load(const char* path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            std::cerr << "Error: Could not open " << path << std::endl;
            exit(1);
        }

        file.read((char*)&max_token_length, sizeof(int));

        int32_t len;
        for (int i = 0; i < vocab_size; i++) {
            float score;
            file.read((char*)&score, sizeof(float));
            vocab_scores.push_back(score);

            file.read((char*)&len, sizeof(int));
            std::string token(len, '\0');
            file.read(&token[0], len);
            vocab.push_back(token);
        }
    }

    std::string decode(int token_id) {
        if (token_id < 0 || token_id >= vocab_size) {
            return "?";
        }
        return vocab[token_id];
    }
};

// Argmax sampling - returns token with highest logit
int sample_argmax(const std::vector<float>& logits, int vocab_size) {
    int max_idx = 0;
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// Temperature sampling - applies temperature and samples from distribution
int sample_temperature(std::vector<float>& logits, int vocab_size, float temperature, std::mt19937& rng) {
    // Apply temperature
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= temperature;
    }

    // Softmax
    float max_logit = *std::max_element(logits.begin(), logits.begin() + vocab_size);
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] = expf(logits[i] - max_logit);
        sum += logits[i];
    }
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= sum;
    }

    // Sample from distribution
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);

    float cdf = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cdf += logits[i];
        if (r < cdf) {
            return i;
        }
    }
    return vocab_size - 1;
}

int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 5) {
        std::cerr << "Usage: " << argv[0] << " <stories110M.bin> <tokenizer.bin> <kernel.so> [temperature]" << std::endl;
        std::cerr << "  temperature: 0.0 = greedy, 0.8 = creative (default: 0.9)" << std::endl;
        return 1;
    }

    const char* checkpoint_path = argv[1];
    const char* tokenizer_path = argv[2];
    const char* kernel_path = argv[3];
    float temperature = argc == 5 ? std::atof(argv[4]) : 0.9f;

    // Load kernel
    void* handle = dlopen(kernel_path, RTLD_LAZY);
    if (!handle) {
        std::cerr << "Error loading kernel: " << dlerror() << std::endl;
        return 1;
    }

    Stories110MFunc forward = (Stories110MFunc)dlsym(handle, "stories110M_forward");
    if (!forward) {
        std::cerr << "Error loading function: " << dlerror() << std::endl;
        return 1;
    }

    // Load config
    std::ifstream checkpoint(checkpoint_path, std::ios::binary);
    if (!checkpoint) {
        std::cerr << "Error: Could not open " << checkpoint_path << std::endl;
        return 1;
    }

    Config config;
    checkpoint.read((char*)&config, sizeof(Config));

    // Check if shared weights (negative vocab_size means unshared)
    bool shared_weights = config.vocab_size > 0;
    int vocab_size = abs(config.vocab_size);

    // Calculate weight sizes
    size_t token_emb_size = vocab_size * config.dim;
    size_t layer_size = config.dim + 4 * config.dim * config.dim + config.dim +
                        2 * config.dim * config.hidden_dim + config.hidden_dim * config.dim;
    size_t total_layer_size = config.n_layers * layer_size;
    size_t final_norm_size = config.dim;
    size_t classifier_size = shared_weights ? 0 : (vocab_size * config.dim);

    size_t total_weights = token_emb_size + total_layer_size + final_norm_size + classifier_size;

    // Allocate and load all weights
    std::vector<float> token_embedding_table(token_emb_size);
    std::vector<float> rms_att_w(config.n_layers * config.dim);
    std::vector<float> wq(config.n_layers * config.dim * config.dim);
    std::vector<float> wk(config.n_layers * config.dim * config.dim);
    std::vector<float> wv(config.n_layers * config.dim * config.dim);
    std::vector<float> wo(config.n_layers * config.dim * config.dim);
    std::vector<float> rms_ffn_w(config.n_layers * config.dim);
    std::vector<float> w1(config.n_layers * config.dim * config.hidden_dim);
    std::vector<float> w2(config.n_layers * config.hidden_dim * config.dim);
    std::vector<float> w3(config.n_layers * config.dim * config.hidden_dim);
    std::vector<float> rms_final_w(config.dim);
    std::vector<float> wcls(vocab_size * config.dim);

    // Read weights from checkpoint in order
    checkpoint.read((char*)token_embedding_table.data(), token_emb_size * sizeof(float));
    checkpoint.read((char*)rms_att_w.data(), config.n_layers * config.dim * sizeof(float));
    checkpoint.read((char*)wq.data(), config.n_layers * config.dim * config.dim * sizeof(float));
    checkpoint.read((char*)wk.data(), config.n_layers * config.dim * config.dim * sizeof(float));
    checkpoint.read((char*)wv.data(), config.n_layers * config.dim * config.dim * sizeof(float));
    checkpoint.read((char*)wo.data(), config.n_layers * config.dim * config.dim * sizeof(float));
    checkpoint.read((char*)rms_ffn_w.data(), config.n_layers * config.dim * sizeof(float));
    checkpoint.read((char*)w1.data(), config.n_layers * config.dim * config.hidden_dim * sizeof(float));
    checkpoint.read((char*)w2.data(), config.n_layers * config.hidden_dim * config.dim * sizeof(float));
    checkpoint.read((char*)w3.data(), config.n_layers * config.dim * config.hidden_dim * sizeof(float));
    checkpoint.read((char*)rms_final_w.data(), config.dim * sizeof(float));

    // Classifier weights (wcls)
    if (shared_weights) {
        // Shared: wcls points to token_embedding_table
        wcls = token_embedding_table;
    } else {
        // Unshared: read separate wcls weights
        checkpoint.read((char*)wcls.data(), vocab_size * config.dim * sizeof(float));
    }

    checkpoint.close();

    // Load tokenizer
    Tokenizer tokenizer;
    tokenizer.vocab_size = vocab_size;
    tokenizer.load(tokenizer_path);

    // Allocate activations
    std::vector<float> x(config.dim);
    std::vector<float> xb(config.dim);
    std::vector<float> xb2(std::max(config.dim, config.hidden_dim));
    std::vector<float> hb(config.hidden_dim);
    std::vector<float> hb_silu(config.hidden_dim);
    std::vector<float> q(config.dim);
    std::vector<float> k(config.dim);
    std::vector<float> v(config.dim);
    std::vector<float> att(config.n_heads * config.seq_len);
    std::vector<float> att_soft(config.n_heads * config.seq_len);
    std::vector<float> logits(vocab_size);
    std::vector<float> key_cache(config.n_layers * config.seq_len * config.dim);
    std::vector<float> value_cache(config.n_layers * config.seq_len * config.dim);
    int64_t num_steps = 256;  // Generate up to 256 tokens
    int64_t start_token = 1;  // BOS token

    // Initialize RNG for sampling
    std::random_device rd;
    std::mt19937 rng(rd());

    int64_t token = start_token;
    std::cout << tokenizer.decode(token) << std::flush;

    auto gen_start = std::chrono::high_resolution_clock::now();
    int64_t tokens_generated = 0;

    for (int64_t pos = 0; pos < num_steps; pos++) {
        // Run forward pass
        forward(
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
            token, pos, config.dim, config.hidden_dim, config.n_layers,
            config.n_heads, config.n_kv_heads, vocab_size, config.seq_len
        );

        // Sample next token
        if (temperature == 0.0f) {
            token = sample_argmax(logits, vocab_size);
        } else {
            token = sample_temperature(logits, vocab_size, temperature, rng);
        }

        // Decode and print
        std::string piece = tokenizer.decode(token);
        std::cout << piece << std::flush;

        tokens_generated++;

        // Stop at EOS
        if (token == 2) break;
    }

    auto gen_end = std::chrono::high_resolution_clock::now();
    auto gen_time = std::chrono::duration_cast<std::chrono::milliseconds>(gen_end - gen_start).count();

    double tokens_per_sec = 1000.0 * tokens_generated / gen_time;
    std::cout << "\n[" << tokens_generated << " tokens, " << tokens_per_sec << " tokens/s]" << std::endl;

    dlclose(handle);
    return 0;
}
