#ifndef KERNEL_H
#define KERNEL_H

#include <immintrin.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// Vector types
typedef __m256d sse_vector_t;  // 4 x double for SSE
typedef __m512d avx_vector_t;  // 8 x double for AVX

// Slice types
typedef struct {
    sse_vector_t* data;
    size_t len;
    size_t cap;
} sse_slice_t;

typedef struct {
    avx_vector_t* data;
    size_t len;
    size_t cap;
} avx_slice_t;

// Updated kernel prototype to match SimpleLang
void kernel_main(sse_slice_t* out_sse, avx_slice_t* out_avx);

// Memory management
sse_vector_t* allocate_sse_vectors(size_t count);
avx_vector_t* allocate_avx_vectors(size_t count);
void free_vectors(void* ptr);

// Slice operations
sse_slice_t* make_sse_slice(size_t len);
avx_slice_t* make_avx_slice(size_t len);
void free_slice(void* slice);

// Vector operations
static inline sse_vector_t sse(double x0, double x1, double x2, double x3) {
    double values[4] = {x0, x1, x2, x3};
    return _mm256_loadu_pd(values);
}

static inline avx_vector_t avx(double x0, double x1, double x2, double x3,
                             double x4, double x5, double x6, double x7) {
    double values[8] = {x0, x1, x2, x3, x4, x5, x6, x7};
    return _mm512_loadu_pd(values);
}

// SIMD operations
static inline sse_vector_t simd_add_sse(sse_vector_t a, sse_vector_t b) {
    return _mm256_add_pd(a, b);
}

static inline avx_vector_t simd_add_avx(avx_vector_t a, avx_vector_t b) {
    return _mm512_add_pd(a, b);
}

static inline sse_vector_t simd_mul_sse(sse_vector_t a, sse_vector_t b) {
    return _mm256_mul_pd(a, b);
}

static inline avx_vector_t simd_mul_avx(avx_vector_t a, avx_vector_t b) {
    return _mm512_mul_pd(a, b);
}

// Slice access with bounds checking
[[noreturn]] static inline void slice_bounds_error(const char* msg) {
    fprintf(stderr, "Slice bounds error: %s\n", msg);
    abort();
}

static inline sse_vector_t slice_get_sse(sse_slice_t* slice, size_t idx) {
    if (idx >= slice->len) {
        slice_bounds_error("SSE slice index out of bounds");
    }
    return slice->data[idx];
}

static inline void slice_set_sse(sse_slice_t* slice, size_t idx, sse_vector_t value) {
    if (idx >= slice->len) {
        slice_bounds_error("SSE slice index out of bounds");
    }
    slice->data[idx] = value;
}

static inline avx_vector_t slice_get_avx(avx_slice_t* slice, size_t idx) {
    if (idx >= slice->len) {
        slice_bounds_error("AVX slice index out of bounds");
    }
    return slice->data[idx];
}

static inline void slice_set_avx(avx_slice_t* slice, size_t idx, avx_vector_t value) {
    if (idx >= slice->len) {
        slice_bounds_error("AVX slice index out of bounds");
    }
    slice->data[idx] = value;
}

// Debug utilities
void print_sse_vector(sse_vector_t vec);
void print_avx_vector(avx_vector_t vec);
void print_sse_slice(sse_slice_t* slice);
void print_avx_slice(avx_slice_t* slice);

#ifdef __cplusplus
}
#endif

#endif // KERNEL_H