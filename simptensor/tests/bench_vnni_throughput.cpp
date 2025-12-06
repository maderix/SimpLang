/**
 * VNNI Throughput Benchmark
 *
 * Measures raw throughput of AVX-512 VNNI vpdpbusd instruction.
 * This establishes a performance baseline for INT8 matmul optimization.
 *
 * The vpdpbusd instruction performs:
 * - 16 lanes of 4-way i8xi8->i32 dot product + accumulation
 * - Total: 64 i8 multiplications + 48 additions + 16 accumulations per instruction
 * - We count this as 64 "integer operations" (mult-adds)
 *
 * Compile: g++ -O3 -march=native -mavx512vnni bench_vnni_throughput.cpp -o bench_vnni_throughput
 * Run: ./bench_vnni_throughput
 */

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <immintrin.h>

// Number of iterations for each benchmark
constexpr int64_t WARMUP_ITERS = 1000000;
constexpr int64_t BENCH_ITERS = 100000000;  // 100M iterations

// Prevent compiler from optimizing away the computation
__attribute__((noinline))
void bench_vpdpbusd_latency(
    __m512i* acc,
    const __m512i va,
    const __m512i vb,
    int64_t iterations
) {
    __m512i local_acc = *acc;
    for (int64_t i = 0; i < iterations; i++) {
        // Latency-bound: each instruction depends on previous result
        local_acc = _mm512_dpbusd_epi32(local_acc, va, vb);
    }
    *acc = local_acc;
}

// Throughput benchmark: multiple independent chains
__attribute__((noinline))
void bench_vpdpbusd_throughput(
    __m512i* acc0, __m512i* acc1, __m512i* acc2, __m512i* acc3,
    __m512i* acc4, __m512i* acc5, __m512i* acc6, __m512i* acc7,
    const __m512i va,
    const __m512i vb,
    int64_t iterations
) {
    __m512i local_acc0 = *acc0, local_acc1 = *acc1;
    __m512i local_acc2 = *acc2, local_acc3 = *acc3;
    __m512i local_acc4 = *acc4, local_acc5 = *acc5;
    __m512i local_acc6 = *acc6, local_acc7 = *acc7;

    for (int64_t i = 0; i < iterations; i++) {
        // 8 independent chains to saturate throughput
        local_acc0 = _mm512_dpbusd_epi32(local_acc0, va, vb);
        local_acc1 = _mm512_dpbusd_epi32(local_acc1, va, vb);
        local_acc2 = _mm512_dpbusd_epi32(local_acc2, va, vb);
        local_acc3 = _mm512_dpbusd_epi32(local_acc3, va, vb);
        local_acc4 = _mm512_dpbusd_epi32(local_acc4, va, vb);
        local_acc5 = _mm512_dpbusd_epi32(local_acc5, va, vb);
        local_acc6 = _mm512_dpbusd_epi32(local_acc6, va, vb);
        local_acc7 = _mm512_dpbusd_epi32(local_acc7, va, vb);
    }

    *acc0 = local_acc0; *acc1 = local_acc1;
    *acc2 = local_acc2; *acc3 = local_acc3;
    *acc4 = local_acc4; *acc5 = local_acc5;
    *acc6 = local_acc6; *acc7 = local_acc7;
}

// Memory-bound benchmark: load from memory each iteration
__attribute__((noinline))
void bench_vpdpbusd_memory(
    __m512i* acc,
    const uint8_t* a_data,
    const int8_t* b_data,
    int64_t data_size,
    int64_t iterations
) {
    __m512i local_acc = *acc;
    int64_t idx = 0;

    for (int64_t i = 0; i < iterations; i++) {
        __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a_data + idx));
        __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b_data + idx));
        local_acc = _mm512_dpbusd_epi32(local_acc, va, vb);

        idx += 64;
        if (idx >= data_size) idx = 0;
    }

    *acc = local_acc;
}

double get_cpu_freq_ghz() {
    // Try to read from /proc/cpuinfo
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            double freq;
            if (sscanf(line, "cpu MHz : %lf", &freq) == 1) {
                fclose(f);
                return freq / 1000.0;  // Convert to GHz
            }
        }
        fclose(f);
    }
    return 4.0;  // Default estimate
}

int main() {
    printf("=== VNNI Throughput Benchmark ===\n\n");

    // Estimate CPU frequency
    double cpu_freq_ghz = get_cpu_freq_ghz();
    printf("Estimated CPU frequency: %.2f GHz\n\n", cpu_freq_ghz);

    // Initialize data
    alignas(64) uint8_t a_data[64];
    alignas(64) int8_t b_data[64];
    alignas(64) int32_t acc_data[16] = {0};

    for (int i = 0; i < 64; i++) {
        a_data[i] = (uint8_t)(i + 1);
        b_data[i] = (int8_t)((i % 2 == 0) ? (i + 1) : -(i + 1));
    }

    __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a_data));
    __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b_data));

    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();
    __m512i acc2 = _mm512_setzero_si512();
    __m512i acc3 = _mm512_setzero_si512();
    __m512i acc4 = _mm512_setzero_si512();
    __m512i acc5 = _mm512_setzero_si512();
    __m512i acc6 = _mm512_setzero_si512();
    __m512i acc7 = _mm512_setzero_si512();

    // ===== Latency Benchmark =====
    printf("1. Latency Benchmark (single chain)\n");
    printf("   Measures instruction latency (data dependency)\n");

    // Warmup
    bench_vpdpbusd_latency(&acc0, va, vb, WARMUP_ITERS);

    acc0 = _mm512_setzero_si512();
    auto start = std::chrono::high_resolution_clock::now();
    bench_vpdpbusd_latency(&acc0, va, vb, BENCH_ITERS);
    auto end = std::chrono::high_resolution_clock::now();

    double latency_ns = std::chrono::duration<double, std::nano>(end - start).count() / BENCH_ITERS;
    double latency_cycles = latency_ns * cpu_freq_ghz;

    printf("   Instructions: %ld\n", BENCH_ITERS);
    printf("   Latency: %.2f ns (%.1f cycles)\n", latency_ns, latency_cycles);
    printf("\n");

    // ===== Throughput Benchmark =====
    printf("2. Throughput Benchmark (8 independent chains)\n");
    printf("   Measures sustained throughput with ILP\n");

    // Warmup
    bench_vpdpbusd_throughput(&acc0, &acc1, &acc2, &acc3, &acc4, &acc5, &acc6, &acc7,
                              va, vb, WARMUP_ITERS);

    acc0 = acc1 = acc2 = acc3 = _mm512_setzero_si512();
    acc4 = acc5 = acc6 = acc7 = _mm512_setzero_si512();

    start = std::chrono::high_resolution_clock::now();
    bench_vpdpbusd_throughput(&acc0, &acc1, &acc2, &acc3, &acc4, &acc5, &acc6, &acc7,
                              va, vb, BENCH_ITERS);
    end = std::chrono::high_resolution_clock::now();

    double elapsed_s = std::chrono::duration<double>(end - start).count();
    int64_t total_instructions = BENCH_ITERS * 8;  // 8 chains
    int64_t total_int_ops = total_instructions * 64;  // 64 i8 multiplies per instruction

    double instr_per_sec = total_instructions / elapsed_s;
    double giop_s = total_int_ops / elapsed_s / 1e9;
    double instr_per_cycle = instr_per_sec / (cpu_freq_ghz * 1e9);

    printf("   Instructions: %ld (8 chains x %ld iters)\n", total_instructions, BENCH_ITERS);
    printf("   Time: %.3f s\n", elapsed_s);
    printf("   Throughput: %.2f M instructions/s\n", instr_per_sec / 1e6);
    printf("   Throughput: %.2f instructions/cycle\n", instr_per_cycle);
    printf("   Integer ops: %.2f GIOP/s (counting 64 i8 mult-adds per instruction)\n", giop_s);
    printf("\n");

    // ===== Memory-bound Benchmark =====
    printf("3. Memory-bound Benchmark (with loads)\n");
    printf("   Measures throughput when data comes from memory\n");

    // Allocate larger buffer for memory benchmark
    constexpr int64_t MEM_SIZE = 64 * 1024 * 1024;  // 64 MB
    uint8_t* mem_a = (uint8_t*)aligned_alloc(64, MEM_SIZE);
    int8_t* mem_b = (int8_t*)aligned_alloc(64, MEM_SIZE);

    if (!mem_a || !mem_b) {
        printf("   ERROR: Failed to allocate memory\n");
        return 1;
    }

    // Initialize memory
    for (int64_t i = 0; i < MEM_SIZE; i++) {
        mem_a[i] = (uint8_t)(i & 0xFF);
        mem_b[i] = (int8_t)((i & 0xFF) - 128);
    }

    // Warmup
    __m512i mem_acc = _mm512_setzero_si512();
    bench_vpdpbusd_memory(&mem_acc, mem_a, mem_b, MEM_SIZE, WARMUP_ITERS);

    int64_t mem_iters = BENCH_ITERS / 10;  // Fewer iterations since memory-bound
    mem_acc = _mm512_setzero_si512();

    start = std::chrono::high_resolution_clock::now();
    bench_vpdpbusd_memory(&mem_acc, mem_a, mem_b, MEM_SIZE, mem_iters);
    end = std::chrono::high_resolution_clock::now();

    elapsed_s = std::chrono::duration<double>(end - start).count();
    total_int_ops = mem_iters * 64;
    int64_t bytes_loaded = mem_iters * 128;  // 64 bytes each for a and b

    double mem_giop_s = total_int_ops / elapsed_s / 1e9;
    double bandwidth_gbps = bytes_loaded / elapsed_s / 1e9;

    printf("   Instructions: %ld\n", mem_iters);
    printf("   Time: %.3f s\n", elapsed_s);
    printf("   Integer ops: %.2f GIOP/s\n", mem_giop_s);
    printf("   Memory bandwidth: %.2f GB/s\n", bandwidth_gbps);
    printf("\n");

    free(mem_a);
    free(mem_b);

    // ===== Summary =====
    printf("=== Summary ===\n");
    printf("Instruction latency:     %.1f cycles\n", latency_cycles);
    printf("Sustained throughput:    %.2f instructions/cycle\n", instr_per_cycle);
    printf("Peak integer ops:        %.2f GIOP/s\n", giop_s);
    printf("Memory-bound ops:        %.2f GIOP/s\n", mem_giop_s);
    printf("\n");

    // Performance assessment
    if (giop_s >= 100.0) {
        printf("\033[32mEXCELLENT: Peak throughput >= 100 GIOP/s\033[0m\n");
    } else if (giop_s >= 50.0) {
        printf("\033[32mGOOD: Peak throughput >= 50 GIOP/s\033[0m\n");
    } else if (giop_s >= 20.0) {
        printf("\033[33mACCEPTABLE: Peak throughput >= 20 GIOP/s\033[0m\n");
    } else {
        printf("\033[31mWARNING: Peak throughput < 20 GIOP/s (unexpectedly low)\033[0m\n");
    }

    // Theoretical analysis
    printf("\nTheoretical Analysis:\n");
    printf("  vpdpbusd @ %.1f GHz:\n", cpu_freq_ghz);
    printf("  - If throughput = 2/cycle: %.0f GIOP/s theoretical peak\n",
           cpu_freq_ghz * 2 * 64);  // 2 instructions/cycle * 64 ops/instruction
    printf("  - If throughput = 1/cycle: %.0f GIOP/s theoretical peak\n",
           cpu_freq_ghz * 1 * 64);
    printf("  - Measured efficiency: %.1f%%\n",
           (giop_s / (cpu_freq_ghz * 2 * 64)) * 100);

    return 0;
}
