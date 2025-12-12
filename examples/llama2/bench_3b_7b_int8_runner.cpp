// Quick benchmark for 3B and 7B INT8 models
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>
#include <dlfcn.h>

#define MEMREF_I8  int8_t*, int8_t*, int64_t, int64_t, int64_t
#define MEMREF_I16 int16_t*, int16_t*, int64_t, int64_t, int64_t
#define MEMREF_I32 int32_t*, int32_t*, int64_t, int64_t, int64_t
#define PASS(p, s) p, p, 0LL, (int64_t)(s), 1LL

template<typename T> T* alloc(size_t n) { return (T*)aligned_alloc(64, n * sizeof(T)); }

void bench_3b(void* handle) {
    // 3B config
    const int64_t DIM=3072, HIDDEN=8192, LAYERS=28, HEADS=24, KV_HEADS=8, VOCAB=128256, SEQ=4096, HEAD=128;
    const int64_t KV_DIM = KV_HEADS * HEAD;
    
    using decode_fn = int32_t(*)(
        MEMREF_I8, MEMREF_I16, MEMREF_I16, MEMREF_I16,
        MEMREF_I8, MEMREF_I8, MEMREF_I8, MEMREF_I8,
        MEMREF_I8, MEMREF_I8, MEMREF_I8, MEMREF_I8,
        MEMREF_I16, MEMREF_I8, MEMREF_I32, MEMREF_I32, MEMREF_I32,
        MEMREF_I32, MEMREF_I8, MEMREF_I8, MEMREF_I8,
        MEMREF_I32, MEMREF_I8, MEMREF_I16, MEMREF_I16, MEMREF_I32,
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t
    );
    
    auto decode = (decode_fn)dlsym(handle, "llama32_3b_decode_i8");
    if (!decode) { std::cout << "3B decode not found\n"; return; }
    
    std::cout << "\n=== LLaMA 3.2-3B INT8 Benchmark ===\n";
    std::cout << "dim=" << DIM << ", layers=" << LAYERS << ", heads=" << HEADS << ", kv_heads=" << KV_HEADS << "\n";
    
    // Allocate
    auto x_i8 = alloc<int8_t>(DIM);
    auto rms_att = alloc<int16_t>(LAYERS * DIM);
    auto rms_ffn = alloc<int16_t>(LAYERS * DIM);
    auto rms_final = alloc<int16_t>(DIM);
    auto wq = alloc<int8_t>(LAYERS * DIM * DIM);
    auto wk = alloc<int8_t>(LAYERS * KV_DIM * DIM);
    auto wv = alloc<int8_t>(LAYERS * KV_DIM * DIM);
    auto wo = alloc<int8_t>(LAYERS * DIM * DIM);
    auto w1 = alloc<int8_t>(LAYERS * HIDDEN * DIM);
    auto w2 = alloc<int8_t>(LAYERS * DIM * HIDDEN);
    auto w3 = alloc<int8_t>(LAYERS * HIDDEN * DIM);
    auto wcls = alloc<int8_t>(VOCAB * DIM);
    auto xb_i16 = alloc<int16_t>(DIM);
    auto xb_i8 = alloc<int8_t>(DIM);
    auto q_i32 = alloc<int32_t>(DIM);
    auto k_i32 = alloc<int32_t>(KV_DIM);
    auto v_i32 = alloc<int32_t>(KV_DIM);
    auto attn_out = alloc<int32_t>(DIM);
    auto ffn_hb = alloc<int8_t>(HIDDEN);
    auto k_cache = alloc<int8_t>(LAYERS * SEQ * KV_DIM);
    auto v_cache = alloc<int8_t>(LAYERS * SEQ * KV_DIM);
    auto att_scores = alloc<int32_t>(HEADS * SEQ);
    auto att_probs = alloc<int8_t>(HEADS * SEQ);
    auto cos_tab = alloc<int16_t>(SEQ * HEAD / 2);
    auto sin_tab = alloc<int16_t>(SEQ * HEAD / 2);
    auto logits = alloc<int32_t>(VOCAB);
    
    memset(x_i8, 1, DIM);
    memset(k_cache, 0, LAYERS * SEQ * KV_DIM);
    memset(v_cache, 0, LAYERS * SEQ * KV_DIM);
    
    // Warmup
    decode(PASS(x_i8,DIM), PASS(rms_att,LAYERS*DIM), PASS(rms_ffn,LAYERS*DIM), PASS(rms_final,DIM),
           PASS(wq,LAYERS*DIM*DIM), PASS(wk,LAYERS*KV_DIM*DIM), PASS(wv,LAYERS*KV_DIM*DIM), PASS(wo,LAYERS*DIM*DIM),
           PASS(w1,LAYERS*HIDDEN*DIM), PASS(w2,LAYERS*DIM*HIDDEN), PASS(w3,LAYERS*HIDDEN*DIM), PASS(wcls,VOCAB*DIM),
           PASS(xb_i16,DIM), PASS(xb_i8,DIM), PASS(q_i32,DIM), PASS(k_i32,KV_DIM), PASS(v_i32,KV_DIM),
           PASS(attn_out,DIM), PASS(ffn_hb,HIDDEN), PASS(k_cache,LAYERS*SEQ*KV_DIM), PASS(v_cache,LAYERS*SEQ*KV_DIM),
           PASS(att_scores,HEADS*SEQ), PASS(att_probs,HEADS*SEQ), PASS(cos_tab,SEQ*HEAD/2), PASS(sin_tab,SEQ*HEAD/2),
           PASS(logits,VOCAB), 0, DIM, HIDDEN, LAYERS, HEADS, KV_HEADS, VOCAB, SEQ, HEAD);
    
    // Benchmark
    int positions[] = {0, 100, 500};
    for (int pos : positions) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 3; i++) {
            decode(PASS(x_i8,DIM), PASS(rms_att,LAYERS*DIM), PASS(rms_ffn,LAYERS*DIM), PASS(rms_final,DIM),
                   PASS(wq,LAYERS*DIM*DIM), PASS(wk,LAYERS*KV_DIM*DIM), PASS(wv,LAYERS*KV_DIM*DIM), PASS(wo,LAYERS*DIM*DIM),
                   PASS(w1,LAYERS*HIDDEN*DIM), PASS(w2,LAYERS*DIM*HIDDEN), PASS(w3,LAYERS*HIDDEN*DIM), PASS(wcls,VOCAB*DIM),
                   PASS(xb_i16,DIM), PASS(xb_i8,DIM), PASS(q_i32,DIM), PASS(k_i32,KV_DIM), PASS(v_i32,KV_DIM),
                   PASS(attn_out,DIM), PASS(ffn_hb,HIDDEN), PASS(k_cache,LAYERS*SEQ*KV_DIM), PASS(v_cache,LAYERS*SEQ*KV_DIM),
                   PASS(att_scores,HEADS*SEQ), PASS(att_probs,HEADS*SEQ), PASS(cos_tab,SEQ*HEAD/2), PASS(sin_tab,SEQ*HEAD/2),
                   PASS(logits,VOCAB), pos, DIM, HIDDEN, LAYERS, HEADS, KV_HEADS, VOCAB, SEQ, HEAD);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 3;
        printf("  pos=%3d: %.2f ms (%.1f tok/s)\n", pos, ms, 1000.0/ms);
    }
    
    printf("  Weights: %.1f GB\n", (LAYERS*DIM*DIM*4.0 + LAYERS*KV_DIM*DIM*2 + LAYERS*HIDDEN*DIM*3 + VOCAB*DIM) / 1e9);
}

void bench_7b(void* handle) {
    // 7B config
    const int64_t DIM=4096, HIDDEN=11008, LAYERS=32, HEADS=32, KV_HEADS=32, VOCAB=32000, SEQ=2048, HEAD=128;
    
    using decode_fn = int32_t(*)(
        MEMREF_I8, MEMREF_I16, MEMREF_I16, MEMREF_I16,
        MEMREF_I8, MEMREF_I8, MEMREF_I8, MEMREF_I8,
        MEMREF_I8, MEMREF_I8, MEMREF_I8, MEMREF_I8,
        MEMREF_I16, MEMREF_I8, MEMREF_I32, MEMREF_I32, MEMREF_I32,
        MEMREF_I32, MEMREF_I8, MEMREF_I8, MEMREF_I8,
        MEMREF_I32, MEMREF_I8, MEMREF_I16, MEMREF_I16, MEMREF_I32,
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t
    );
    
    auto decode = (decode_fn)dlsym(handle, "llama2_7b_decode_i8");
    if (!decode) { std::cout << "7B decode not found\n"; return; }
    
    std::cout << "\n=== LLaMA 2-7B INT8 Benchmark ===\n";
    std::cout << "dim=" << DIM << ", layers=" << LAYERS << ", heads=" << HEADS << " (no GQA)\n";
    
    // Allocate
    auto x_i8 = alloc<int8_t>(DIM);
    auto rms_att = alloc<int16_t>(LAYERS * DIM);
    auto rms_ffn = alloc<int16_t>(LAYERS * DIM);
    auto rms_final = alloc<int16_t>(DIM);
    auto wq = alloc<int8_t>(LAYERS * DIM * DIM);
    auto wk = alloc<int8_t>(LAYERS * DIM * DIM);
    auto wv = alloc<int8_t>(LAYERS * DIM * DIM);
    auto wo = alloc<int8_t>(LAYERS * DIM * DIM);
    auto w1 = alloc<int8_t>(LAYERS * HIDDEN * DIM);
    auto w2 = alloc<int8_t>(LAYERS * DIM * HIDDEN);
    auto w3 = alloc<int8_t>(LAYERS * HIDDEN * DIM);
    auto wcls = alloc<int8_t>(VOCAB * DIM);
    auto xb_i16 = alloc<int16_t>(DIM);
    auto xb_i8 = alloc<int8_t>(DIM);
    auto q_i32 = alloc<int32_t>(DIM);
    auto k_i32 = alloc<int32_t>(DIM);
    auto v_i32 = alloc<int32_t>(DIM);
    auto attn_out = alloc<int32_t>(DIM);
    auto ffn_hb = alloc<int8_t>(HIDDEN);
    auto k_cache = alloc<int8_t>(LAYERS * SEQ * DIM);
    auto v_cache = alloc<int8_t>(LAYERS * SEQ * DIM);
    auto att_scores = alloc<int32_t>(HEADS * SEQ);
    auto att_probs = alloc<int8_t>(HEADS * SEQ);
    auto cos_tab = alloc<int16_t>(SEQ * HEAD / 2);
    auto sin_tab = alloc<int16_t>(SEQ * HEAD / 2);
    auto logits = alloc<int32_t>(VOCAB);
    
    memset(x_i8, 1, DIM);
    memset(k_cache, 0, LAYERS * SEQ * DIM);
    memset(v_cache, 0, LAYERS * SEQ * DIM);
    
    // Warmup
    decode(PASS(x_i8,DIM), PASS(rms_att,LAYERS*DIM), PASS(rms_ffn,LAYERS*DIM), PASS(rms_final,DIM),
           PASS(wq,LAYERS*DIM*DIM), PASS(wk,LAYERS*DIM*DIM), PASS(wv,LAYERS*DIM*DIM), PASS(wo,LAYERS*DIM*DIM),
           PASS(w1,LAYERS*HIDDEN*DIM), PASS(w2,LAYERS*DIM*HIDDEN), PASS(w3,LAYERS*HIDDEN*DIM), PASS(wcls,VOCAB*DIM),
           PASS(xb_i16,DIM), PASS(xb_i8,DIM), PASS(q_i32,DIM), PASS(k_i32,DIM), PASS(v_i32,DIM),
           PASS(attn_out,DIM), PASS(ffn_hb,HIDDEN), PASS(k_cache,LAYERS*SEQ*DIM), PASS(v_cache,LAYERS*SEQ*DIM),
           PASS(att_scores,HEADS*SEQ), PASS(att_probs,HEADS*SEQ), PASS(cos_tab,SEQ*HEAD/2), PASS(sin_tab,SEQ*HEAD/2),
           PASS(logits,VOCAB), 0, DIM, HIDDEN, LAYERS, HEADS, KV_HEADS, VOCAB, SEQ, HEAD);
    
    // Benchmark
    int positions[] = {0, 100, 500};
    for (int pos : positions) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 3; i++) {
            decode(PASS(x_i8,DIM), PASS(rms_att,LAYERS*DIM), PASS(rms_ffn,LAYERS*DIM), PASS(rms_final,DIM),
                   PASS(wq,LAYERS*DIM*DIM), PASS(wk,LAYERS*DIM*DIM), PASS(wv,LAYERS*DIM*DIM), PASS(wo,LAYERS*DIM*DIM),
                   PASS(w1,LAYERS*HIDDEN*DIM), PASS(w2,LAYERS*DIM*HIDDEN), PASS(w3,LAYERS*HIDDEN*DIM), PASS(wcls,VOCAB*DIM),
                   PASS(xb_i16,DIM), PASS(xb_i8,DIM), PASS(q_i32,DIM), PASS(k_i32,DIM), PASS(v_i32,DIM),
                   PASS(attn_out,DIM), PASS(ffn_hb,HIDDEN), PASS(k_cache,LAYERS*SEQ*DIM), PASS(v_cache,LAYERS*SEQ*DIM),
                   PASS(att_scores,HEADS*SEQ), PASS(att_probs,HEADS*SEQ), PASS(cos_tab,SEQ*HEAD/2), PASS(sin_tab,SEQ*HEAD/2),
                   PASS(logits,VOCAB), pos, DIM, HIDDEN, LAYERS, HEADS, KV_HEADS, VOCAB, SEQ, HEAD);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 3;
        printf("  pos=%3d: %.2f ms (%.1f tok/s)\n", pos, ms, 1000.0/ms);
    }
    
    printf("  Weights: %.1f GB\n", (LAYERS*DIM*DIM*4.0 + LAYERS*HIDDEN*DIM*3 + VOCAB*DIM) / 1e9);
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "Usage: " << argv[0] << " <3b.so> [7b.so]\n"; return 1; }
    
    void* h3b = dlopen(argv[1], RTLD_NOW);
    if (h3b) bench_3b(h3b);
    
    if (argc >= 3) {
        void* h7b = dlopen(argv[2], RTLD_NOW);
        if (h7b) bench_7b(h7b);
    }
    
    return 0;
}
