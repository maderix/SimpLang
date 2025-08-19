/**
 * @file cpuid.h
 * @brief CPU feature detection for runtime dispatch
 */

#ifndef SIMPBLAS_CPUID_H
#define SIMPBLAS_CPUID_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief CPU feature flags
 */
typedef struct {
    bool avx2;
    bool avx512f;
    bool avx512bw;
    bool avx512vl;
    bool fma;
    bool neon;      // ARM NEON
    bool dotprod;   // ARM dot product
} sb_cpu_features_t;

/**
 * @brief Detect CPU features
 * @param features Output structure to fill
 * @return 0 on success, -1 on error
 */
int sb_detect_cpu_features(sb_cpu_features_t* features);

/**
 * @brief Get CPU feature string for debugging
 * @param features CPU features structure
 * @return String describing features (caller must free)
 */
char* sb_cpu_features_to_string(const sb_cpu_features_t* features);

#ifdef __cplusplus
}
#endif

#endif /* SIMPBLAS_CPUID_H */