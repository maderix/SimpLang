#include "cpuid.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _MSC_VER
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

#ifdef __aarch64__
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

int sb_detect_cpu_features(sb_cpu_features_t* features) {
    if (!features) {
        return -1;
    }
    
    // Initialize all features to false
    memset(features, 0, sizeof(sb_cpu_features_t));
    
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    
    // Check for CPUID support
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return -1;
    }
    
    // Check AVX2 support (requires CPUID leaf 7)
    if (__get_cpuid_max(0, NULL) >= 7) {
        if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
            features->avx2 = (ebx & (1 << 5)) != 0;  // AVX2
            features->avx512f = (ebx & (1 << 16)) != 0;  // AVX-512 Foundation
            features->avx512bw = (ebx & (1 << 30)) != 0;  // AVX-512 Byte and Word
            features->avx512vl = (ebx & (1 << 31)) != 0;  // AVX-512 Vector Length
        }
    }
    
    // Check FMA support
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        features->fma = (ecx & (1 << 12)) != 0;  // FMA
    }
    
#elif defined(__aarch64__)
    // ARM64 feature detection using auxiliary vector
    unsigned long hwcap = getauxval(AT_HWCAP);
    unsigned long hwcap2 = getauxval(AT_HWCAP2);
    
    features->neon = (hwcap & HWCAP_ASIMD) != 0;  // Advanced SIMD (NEON)
    
    // Check for dot product support (ARMv8.2-A)
    #ifdef HWCAP_ASIMDDP
    features->dotprod = (hwcap & HWCAP_ASIMDDP) != 0;
    #endif
    
#elif defined(__arm__)
    // ARM32 NEON detection
    #ifdef __ARM_NEON
    features->neon = true;  // Compile-time detection
    #endif
#endif
    
    return 0;
}

char* sb_cpu_features_to_string(const sb_cpu_features_t* features) {
    if (!features) {
        return NULL;
    }
    
    char* result = malloc(256);
    if (!result) {
        return NULL;
    }
    
    int pos = 0;
    pos += snprintf(result + pos, 256 - pos, "CPU Features: ");
    
    if (features->avx2) pos += snprintf(result + pos, 256 - pos, "AVX2 ");
    if (features->avx512f) pos += snprintf(result + pos, 256 - pos, "AVX512F ");
    if (features->avx512bw) pos += snprintf(result + pos, 256 - pos, "AVX512BW ");
    if (features->avx512vl) pos += snprintf(result + pos, 256 - pos, "AVX512VL ");
    if (features->fma) pos += snprintf(result + pos, 256 - pos, "FMA ");
    if (features->neon) pos += snprintf(result + pos, 256 - pos, "NEON ");
    if (features->dotprod) pos += snprintf(result + pos, 256 - pos, "DOTPROD ");
    
    if (pos == strlen("CPU Features: ")) {
        pos += snprintf(result + pos, 256 - pos, "SCALAR_ONLY");
    }
    
    return result;
}