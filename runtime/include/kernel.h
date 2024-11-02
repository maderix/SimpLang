#pragma once
#include <cstddef>
#include <immintrin.h>

// Debug printing macros
#ifdef SIMD_DEBUG
    #include <stdio.h>
    #define DEBUG_PRINT(...) printf(__VA_ARGS__)
    #define DEBUG_VECTOR(msg, vec) print_debug_vector(msg, vec)
#else
    #define DEBUG_PRINT(...)
    #define DEBUG_VECTOR(msg, vec)
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Debug helper functions
#ifdef SIMD_DEBUG
void print_debug_vector(const char* msg, __m256d vec) {
    alignas(32) double values[4];
    _mm256_store_pd(values, vec);
    printf("%s: [%.2f, %.2f, %.2f, %.2f]\n", 
           msg, values[0], values[1], values[2], values[3]);
}

void print_debug_vector(const char* msg, __m512d vec) {
    alignas(64) double values[8];
    _mm512_store_pd(values, vec);
    printf("%s: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n",
           msg, values[0], values[1], values[2], values[3],
           values[4], values[5], values[6], values[7]);
}
#endif

// SSE slice type (4-wide double vectors)
typedef struct {
    __m256d* data;  // Using AVX for SSE operations (4 doubles)
    size_t len;
    size_t cap;
} sse_slice_t;

// AVX slice type (8-wide double vectors)
typedef struct {
    __m512d* data;  // Using AVX-512 for AVX operations (8 doubles)
    size_t len;
    size_t cap;
} avx_slice_t;

// Vector creation
__m256d sse(double a, double b, double c, double d);
__m512d avx(double a, double b, double c, double d,
            double e, double f, double g, double h);

// SIMD operations
__m256d simd_add(__m256d a, __m256d b);
__m256d simd_mul(__m256d a, __m256d b);
__m512d simd_add_avx(__m512d a, __m512d b);
__m512d simd_mul_avx(__m512d a, __m512d b);

// Slice operations
sse_slice_t* make_sse_slice(size_t len);
void slice_set_sse(sse_slice_t* slice, size_t idx, __m256d value);
__m256d slice_get_sse(sse_slice_t* slice, size_t idx);

avx_slice_t* make_avx_slice(size_t len);
void slice_set_avx(avx_slice_t* slice, size_t idx, __m512d value);
__m512d slice_get_avx(avx_slice_t* slice, size_t idx);

// Memory management
void free_vectors(void* ptr);
void free_slice(void* slice);

// Printing functions
void print_sse_vector(__m256d vec);
void print_avx_vector(__m512d vec);
void print_sse_slice(sse_slice_t* slice);
void print_avx_slice(avx_slice_t* slice);

// Main kernel function - implemented by the generated code
void kernel_main(sse_slice_t* out_sse, avx_slice_t* out_avx);

#ifdef __cplusplus
}
#endif