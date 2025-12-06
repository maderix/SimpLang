/**
 * VNNI Availability Test
 *
 * Tests for AVX-512 VNNI support on the target CPU.
 * This is a pre-requisite for INT8 quantized matmul benchmarks.
 *
 * Compile: g++ -O3 -march=native test_vnni_availability.cpp -o test_vnni_availability
 * Run: ./test_vnni_availability
 */

#include <cstdio>
#include <cstdint>
#include <cstring>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

struct CPUFeatures {
    bool sse;
    bool sse2;
    bool sse3;
    bool ssse3;
    bool sse4_1;
    bool sse4_2;
    bool avx;
    bool avx2;
    bool avx512f;
    bool avx512bw;
    bool avx512dq;
    bool avx512vl;
    bool avx512vnni;
    bool avx512_bf16;
    char vendor[13];
    char brand[49];
};

void get_cpuid(uint32_t leaf, uint32_t subleaf, uint32_t* eax, uint32_t* ebx, uint32_t* ecx, uint32_t* edx) {
#ifdef _MSC_VER
    int regs[4];
    __cpuidex(regs, leaf, subleaf);
    *eax = regs[0];
    *ebx = regs[1];
    *ecx = regs[2];
    *edx = regs[3];
#else
    __cpuid_count(leaf, subleaf, *eax, *ebx, *ecx, *edx);
#endif
}

CPUFeatures detect_cpu_features() {
    CPUFeatures features = {};
    uint32_t eax, ebx, ecx, edx;

    // Get vendor string
    get_cpuid(0, 0, &eax, &ebx, &ecx, &edx);
    memcpy(features.vendor + 0, &ebx, 4);
    memcpy(features.vendor + 4, &edx, 4);
    memcpy(features.vendor + 8, &ecx, 4);
    features.vendor[12] = '\0';

    // Get brand string (leaves 0x80000002-0x80000004)
    uint32_t max_extended;
    get_cpuid(0x80000000, 0, &max_extended, &ebx, &ecx, &edx);
    if (max_extended >= 0x80000004) {
        for (int i = 0; i < 3; i++) {
            get_cpuid(0x80000002 + i, 0, &eax, &ebx, &ecx, &edx);
            memcpy(features.brand + i * 16 + 0, &eax, 4);
            memcpy(features.brand + i * 16 + 4, &ebx, 4);
            memcpy(features.brand + i * 16 + 8, &ecx, 4);
            memcpy(features.brand + i * 16 + 12, &edx, 4);
        }
        features.brand[48] = '\0';
    }

    // Get feature flags (leaf 1)
    get_cpuid(1, 0, &eax, &ebx, &ecx, &edx);
    features.sse = (edx >> 25) & 1;
    features.sse2 = (edx >> 26) & 1;
    features.sse3 = (ecx >> 0) & 1;
    features.ssse3 = (ecx >> 9) & 1;
    features.sse4_1 = (ecx >> 19) & 1;
    features.sse4_2 = (ecx >> 20) & 1;
    features.avx = (ecx >> 28) & 1;

    // Get extended feature flags (leaf 7, subleaf 0)
    get_cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    features.avx2 = (ebx >> 5) & 1;
    features.avx512f = (ebx >> 16) & 1;
    features.avx512dq = (ebx >> 17) & 1;
    features.avx512bw = (ebx >> 30) & 1;
    features.avx512vl = (ebx >> 31) & 1;
    features.avx512vnni = (ecx >> 11) & 1;  // AVX-512 VNNI
    features.avx512_bf16 = (eax >> 5) & 1;  // From leaf 7, subleaf 1, but approximate

    return features;
}

void print_feature(const char* name, bool supported) {
    printf("  %-20s %s\n", name, supported ? "\033[32m[YES]\033[0m" : "\033[31m[NO]\033[0m");
}

int main() {
    printf("=== VNNI Availability Test ===\n\n");

    CPUFeatures features = detect_cpu_features();

    printf("CPU Vendor: %s\n", features.vendor);
    printf("CPU Brand:  %s\n\n", features.brand);

    printf("Feature Detection:\n");
    printf("------------------\n");

    printf("\nBasic SIMD:\n");
    print_feature("SSE", features.sse);
    print_feature("SSE2", features.sse2);
    print_feature("SSE3", features.sse3);
    print_feature("SSSE3", features.ssse3);
    print_feature("SSE4.1", features.sse4_1);
    print_feature("SSE4.2", features.sse4_2);
    print_feature("AVX", features.avx);
    print_feature("AVX2", features.avx2);

    printf("\nAVX-512:\n");
    print_feature("AVX-512F", features.avx512f);
    print_feature("AVX-512BW", features.avx512bw);
    print_feature("AVX-512DQ", features.avx512dq);
    print_feature("AVX-512VL", features.avx512vl);
    print_feature("AVX-512 VNNI", features.avx512vnni);

    printf("\n");
    printf("=== Summary ===\n");

    if (features.avx512vnni) {
        printf("\033[32mSUCCESS: AVX-512 VNNI is supported!\033[0m\n");
        printf("The vpdpbusd instruction is available for INT8 matmul acceleration.\n");
        printf("Expected throughput: ~16 i8xi8->i32 ops per instruction\n");
        return 0;
    } else if (features.avx512f) {
        printf("\033[33mWARNING: AVX-512 is supported but VNNI is NOT available.\033[0m\n");
        printf("Fallback: Generic AVX-512 INT8 path will be used (slower).\n");
        return 1;
    } else if (features.avx2) {
        printf("\033[33mWARNING: Only AVX2 is supported (no AVX-512).\033[0m\n");
        printf("Fallback: AVX2 INT8 path will be used.\n");
        return 1;
    } else {
        printf("\033[31mERROR: Insufficient SIMD support for INT8 matmul benchmarks.\033[0m\n");
        return 2;
    }
}
