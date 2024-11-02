#include "kernel.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <immintrin.h>

// Platform-specific aligned allocation
#ifdef _WIN32
    #include <malloc.h>
    #define aligned_alloc(align, size) _aligned_malloc(size, align)
    #define aligned_free(ptr) _aligned_free(ptr)
#else
    #define aligned_free(ptr) free(ptr)
#endif

extern "C" {

// Vector creation operations
__m256d sse(double a, double b, double c, double d) {
    __m256d result = _mm256_set_pd(d, c, b, a);  // Note: set_pd loads in reverse order
    //DEBUG_VECTOR("SSE vector created", result);
    return result;
}

__m512d avx(double a, double b, double c, double d,
            double e, double f, double g, double h) {
    __m512d result = _mm512_set_pd(h, g, f, e, d, c, b, a);  // Note: set_pd loads in reverse order
    //DEBUG_VECTOR("AVX vector created", result);
    return result;
}

// SIMD arithmetic operations
__m256d simd_add(__m256d a, __m256d b) {
    DEBUG_PRINT("\nSSE Addition:\n");
    //DEBUG_VECTOR("  Input a", a);
    //DEBUG_VECTOR("  Input b", b);
    
    __m256d result = _mm256_add_pd(a, b);
    //DEBUG_VECTOR("  Result", result);
    return result;
}

__m512d simd_add_avx(__m512d a, __m512d b) {
    DEBUG_PRINT("\nAVX Addition:\n");
    //DEBUG_VECTOR("  Input a", a);
    //DEBUG_VECTOR("  Input b", b);
    
    __m512d result = _mm512_add_pd(a, b);
    //DEBUG_VECTOR("  Result", result);
    return result;
}

__m256d simd_mul(__m256d a, __m256d b) {
    DEBUG_PRINT("\nSSE Multiplication:\n");
    //DEBUG_VECTOR("  Input a", a);
    //DEBUG_VECTOR("  Input b", b);
    
    __m256d result = _mm256_mul_pd(a, b);
    //DEBUG_VECTOR("  Result", result);
    return result;
}

__m512d simd_mul_avx(__m512d a, __m512d b) {
    DEBUG_PRINT("\nAVX Multiplication:\n");
    //DEBUG_VECTOR("  Input a", a);
    //DEBUG_VECTOR("  Input b", b);
    
    __m512d result = _mm512_mul_pd(a, b);
    //DEBUG_VECTOR("  Result", result);
    return result;
}

// Vector allocation functions
__m256d* allocate_sse_vectors(size_t count) {
    void* ptr = aligned_alloc(32, count * sizeof(__m256d));
    return static_cast<__m256d*>(ptr);
}

__m512d* allocate_avx_vectors(size_t count) {
    void* ptr = aligned_alloc(64, count * sizeof(__m512d));
    return static_cast<__m512d*>(ptr);
}

void free_vectors(void* ptr) {
    aligned_free(ptr);
}

// Slice management functions
sse_slice_t* make_sse_slice(size_t len) {
    sse_slice_t* slice = (sse_slice_t*)malloc(sizeof(sse_slice_t));
    if (!slice) return nullptr;

    slice->data = allocate_sse_vectors(len);
    if (!slice->data) {
        free(slice);
        return nullptr;
    }

    // Initialize to zero
    for (size_t i = 0; i < len; i++) {
        slice->data[i] = _mm256_setzero_pd();
    }

    slice->len = len;
    slice->cap = len;
    return slice;
}

avx_slice_t* make_avx_slice(size_t len) {
    avx_slice_t* slice = (avx_slice_t*)malloc(sizeof(avx_slice_t));
    if (!slice) return nullptr;

    slice->data = allocate_avx_vectors(len);
    if (!slice->data) {
        free(slice);
        return nullptr;
    }

    // Initialize to zero
    for (size_t i = 0; i < len; i++) {
        slice->data[i] = _mm512_setzero_pd();
    }

    slice->len = len;
    slice->cap = len;
    return slice;
}

// Slice access operations
void slice_set_sse(sse_slice_t* slice, size_t idx, __m256d value) {
    if (idx >= slice->len) {
        fprintf(stderr, "SSE slice index out of bounds\n");
        abort();
    }
    slice->data[idx] = value;
}

__m256d slice_get_sse(sse_slice_t* slice, size_t idx) {
    if (idx >= slice->len) {
        fprintf(stderr, "SSE slice index out of bounds\n");
        abort();
    }
    return slice->data[idx];
}

void slice_set_avx(avx_slice_t* slice, size_t idx, __m512d value) {
    if (idx >= slice->len) {
        fprintf(stderr, "AVX slice index out of bounds\n");
        abort();
    }
    slice->data[idx] = value;
}

__m512d slice_get_avx(avx_slice_t* slice, size_t idx) {
    if (idx >= slice->len) {
        fprintf(stderr, "AVX slice index out of bounds\n");
        abort();
    }
    return slice->data[idx];
}

void free_slice(void* slice) {
    if (!slice) return;
    void* data_ptr = *(void**)slice;  // Get data pointer
    if (data_ptr) {
        aligned_free(data_ptr);
    }
    free(slice);
}

// Vector printing functions
void print_sse_vector(__m256d vec) {
    alignas(32) double values[4];
    _mm256_store_pd(values, vec);
    printf("[%.2f, %.2f, %.2f, %.2f]\n", 
           values[0], values[1], values[2], values[3]);
}

void print_avx_vector(__m512d vec) {
    alignas(64) double values[8];
    _mm512_store_pd(values, vec);
    printf("[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n",
           values[0], values[1], values[2], values[3],
           values[4], values[5], values[6], values[7]);
}

// Slice printing functions
void print_sse_slice(sse_slice_t* slice) {
    if (!slice) {
        printf("(null slice)\n");
        return;
    }

    printf("SSESlice(len=%zu, cap=%zu) [\n", slice->len, slice->cap);
    for (size_t i = 0; i < slice->len; i++) {
        printf("  %zu: ", i);
        print_sse_vector(slice->data[i]);
    }
    printf("]\n");
}

void print_avx_slice(avx_slice_t* slice) {
    if (!slice) {
        printf("(null slice)\n");
        return;
    }

    printf("AVXSlice(len=%zu, cap=%zu) [\n", slice->len, slice->cap);
    for (size_t i = 0; i < slice->len; i++) {
        printf("  %zu: ", i);
        print_avx_vector(slice->data[i]);
    }
    printf("]\n");
}

} // extern "C"