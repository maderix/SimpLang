#pragma once

#include <cstddef>
#include <immintrin.h>

// SSE slice type (128-bit vectors, 2 doubles)
typedef struct {
    __m128d* data;
    size_t len;
    size_t cap;
} sse_slice_t;

// AVX slice type (512-bit vectors, 8 doubles)
typedef struct {
    __m512d* data;
    size_t len;
    size_t cap;
} avx_slice_t;

// SIMD vector creation functions
extern "C" {
    __m128d sse(double a, double b, double c, double d);
    __m512d avx(double a, double b, double c, double d,
                double e, double f, double g, double h);
                
    // Slice management functions
    sse_slice_t* make_sse_slice(size_t len);
    avx_slice_t* make_avx_slice(size_t len);
    void free_slice(void* slice);
    
    // Slice operations
    __m128d slice_get_sse(sse_slice_t* slice, size_t idx);
    __m512d slice_get_avx(avx_slice_t* slice, size_t idx);
    void slice_set_sse(sse_slice_t* slice, size_t idx, __m128d value);
    void slice_set_avx(avx_slice_t* slice, size_t idx, __m512d value);
}