#ifndef KERNEL_H
#define KERNEL_H

#include "sl_main.h"
#include <immintrin.h>
#include <cstddef>
#include <cstdio>

#ifdef __cplusplus
extern "C" {
#endif

// Debug printing configuration
#define DEBUG 1
#define DEBUG_PRINT(msg) if (DEBUG) printf("%s", msg)

// Debug vector printing - separate functions for SSE and AVX
#if DEBUG
static inline void print_debug_vector_sse(const char* msg, __m256d vec) {
    alignas(32) double values[4];
    _mm256_store_pd(values, vec);
    printf("%s: [%.2f, %.2f, %.2f, %.2f]\n", msg,
           values[0], values[1], values[2], values[3]);
}

static inline void print_debug_vector_avx(const char* msg, __m512d vec) {
    alignas(64) double values[8];
    _mm512_store_pd(values, vec);
    printf("%s: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n", msg,
           values[0], values[1], values[2], values[3],
           values[4], values[5], values[6], values[7]);
}
#else
#define print_debug_vector_sse(msg, vec)
#define print_debug_vector_avx(msg, vec)
#endif

// Slice structures for SSE and AVX
typedef struct {
    __m256d* data;
    size_t len;
    size_t cap;
} sse_slice_t;

typedef struct {
    __m512d* data;
    size_t len;
    size_t cap;
} avx_slice_t;

// Vector creation functions
__m256d sse(double a, double b, double c, double d);
__m512d avx(double a, double b, double c, double d, 
            double e, double f, double g, double h);

// SIMD arithmetic operations
__m256d simd_add(__m256d a, __m256d b);
__m512d simd_add_avx(__m512d a, __m512d b);
__m256d simd_mul(__m256d a, __m256d b);
__m512d simd_mul_avx(__m512d a, __m512d b);

// Slice management functions
sse_slice_t* make_sse_slice(size_t len);
avx_slice_t* make_avx_slice(size_t len);
void free_slice(void* slice);

// Slice access operations
void slice_set_sse(sse_slice_t* slice, size_t idx, __m256d value);
__m256d slice_get_sse(sse_slice_t* slice, size_t idx);
void slice_set_avx(avx_slice_t* slice, size_t idx, __m512d value);
__m512d slice_get_avx(avx_slice_t* slice, size_t idx);

// Vector printing functions
void print_sse_vector(__m256d vec);
void print_avx_vector(__m512d vec);

// Slice printing functions
void print_sse_slice(sse_slice_t* slice);
void print_avx_slice(avx_slice_t* slice);

// Kernel entry point - handles both test and regular cases
#ifdef TEST_SIMD
void kernel_main(sse_slice_t* out_sse, avx_slice_t* out_avx);
#else
double kernel_main();
#endif

#ifdef __cplusplus
}
#endif

#endif // KERNEL_H