#include "kernel.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <immintrin.h>

#ifdef _WIN32
    #include <malloc.h>
    #define aligned_alloc(align, size) _aligned_malloc(size, align)
    #define aligned_free(ptr) _aligned_free(ptr)
#else
    #define aligned_free(ptr) free(ptr)
#endif

// Vector creation operations
__m128d sse(double a, double b, double c, double d) {
    printf("\nSSE vector creation debug:\n");
    printf("  Input values: %.2f, %.2f, %.2f, %.2f\n", a, b, c, d);
    
    alignas(16) double values[2] = {a, b};  // SSE only uses first two values
    __m128d result = _mm_load_pd(values);
    
    // Debug the created vector
    alignas(16) double check[2];
    _mm_store_pd(check, result);
    printf("  Created vector: [%.2f, %.2f]\n", check[0], check[1]);
    
    return result;
}

#ifdef _MSC_VER
__m512d __vectorcall avx(double a, double b, double c, double d,
                        double e, double f, double g, double h)
#else
__m512d avx(double a, double b, double c, double d,
            double e, double f, double g, double h)
#endif
{
    // Immediately store all XMM registers to prevent them from being clobbered
    alignas(64) double values[8];
    __asm__ volatile(
        "vmovsd %%xmm0, %0\n\t"
        "vmovsd %%xmm1, %1\n\t"
        "vmovsd %%xmm2, %2\n\t"
        "vmovsd %%xmm3, %3\n\t"
        "vmovsd %%xmm4, %4\n\t"
        "vmovsd %%xmm5, %5\n\t"
        "vmovsd %%xmm6, %6\n\t"
        "vmovsd %%xmm7, %7\n\t"
        : "=m"(values[0]), "=m"(values[1]), "=m"(values[2]), "=m"(values[3]),
          "=m"(values[4]), "=m"(values[5]), "=m"(values[6]), "=m"(values[7])
        :
        : "memory"
    );

    printf("\nAVX vector creation debug:\n");
    printf("  Input values: %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n",
           values[0], values[1], values[2], values[3],
           values[4], values[5], values[6], values[7]);
           
    __m512d result = _mm512_load_pd(values);
    
    // Debug the created vector
    alignas(64) double check[8];
    _mm512_store_pd(check, result);
    printf("  Created vector: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n",
           check[0], check[1], check[2], check[3],
           check[4], check[5], check[6], check[7]);
           
    return result;
}

// SIMD arithmetic operations
__m128d simd_add(__m128d a, __m128d b) {
    printf("\nSSE Addition:\n");
    print_debug_vector_sse("  Input a", a);
    print_debug_vector_sse("  Input b", b);
    __m128d result = _mm_add_pd(a, b);
    print_debug_vector_sse("  Result", result);
    return result;
}

__m512d simd_add_avx(__m512d a, __m512d b) {
    printf("\nAVX Addition:\n");
    print_debug_vector_avx("  Input a", a);
    print_debug_vector_avx("  Input b", b);
    __m512d result = _mm512_add_pd(a, b);
    print_debug_vector_avx("  Result", result);
    return result;
}

__m128d simd_mul(__m128d a, __m128d b) {
    printf("\nSSE Multiplication:\n");
    print_debug_vector_sse("  Input a", a);
    print_debug_vector_sse("  Input b", b);
    __m128d result = _mm_mul_pd(a, b);
    print_debug_vector_sse("  Result", result);
    return result;
}

__m512d simd_mul_avx(__m512d a, __m512d b) {
    printf("\nAVX Multiplication:\n");
    print_debug_vector_avx("  Input a", a);
    print_debug_vector_avx("  Input b", b);
    __m512d result = _mm512_mul_pd(a, b);
    print_debug_vector_avx("  Result", result);
    return result;
}

// Memory management
__m128d* allocate_sse_vectors(size_t count) {
    return (__m128d*)aligned_alloc(16, count * sizeof(__m128d));
}

__m512d* allocate_avx_vectors(size_t count) {
    return (__m512d*)aligned_alloc(64, count * sizeof(__m512d));
}

void free_vectors(void* ptr) {
    aligned_free(ptr);
}

// Slice management
sse_slice_t* make_sse_slice(size_t len) {
    sse_slice_t* slice = (sse_slice_t*)malloc(sizeof(sse_slice_t));
    if (!slice) {
        printf("Failed to allocate SSE slice\n");
        abort();
    }
    
    size_t cap = len > 0 ? len : 1;
    slice->data = allocate_sse_vectors(cap);
    if (!slice->data) {
        printf("Failed to allocate SSE vectors\n");
        free(slice);
        abort();
    }
    
    slice->len = 0;
    slice->cap = cap;
    return slice;
}

avx_slice_t* make_avx_slice(size_t len) {
    avx_slice_t* slice = (avx_slice_t*)malloc(sizeof(avx_slice_t));
    if (!slice) {
        printf("Failed to allocate AVX slice\n");
        abort();
    }
    
    size_t cap = len > 0 ? len : 1;
    slice->data = allocate_avx_vectors(cap);
    if (!slice->data) {
        printf("Failed to allocate AVX vectors\n");
        free(slice);
        abort();
    }
    
    slice->len = 0;
    slice->cap = cap;
    return slice;
}

// Slice operations
__m128d slice_get_sse(sse_slice_t* slice, size_t idx) {
    if (idx >= slice->cap) {
        printf("SSE slice index out of bounds\n");
        abort();
    }
    
    __m128d result = slice->data[idx];
    print_debug_vector_sse("Retrieved values", result);
    return result;
}

__m512d slice_get_avx(avx_slice_t* slice, size_t idx) {
    if (!slice) {
        printf("ERROR: Null slice pointer in slice_get_avx\n");
        abort();
    }
    
    if (!slice->data) {
        printf("ERROR: Null data pointer in slice_get_avx\n");
        abort();
    }
    
    if (idx >= slice->cap) {
        printf("AVX slice index out of bounds\n");
        abort();
    }
    
    printf("\nAVX slice_get_avx debug:\n");
    printf("  Slice pointer: %p\n", (void*)slice);
    printf("  Data pointer: %p\n", (void*)slice->data);
    printf("  Getting index: %zu\n", idx);
    printf("  Slice length: %zu\n", slice->len);
    printf("  Slice capacity: %zu\n", slice->cap);
    
    __m512d result = slice->data[idx];
    
    // Debug the retrieved value
    alignas(64) double values[8];
    _mm512_store_pd(values, result);
    printf("  Retrieved vector: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n",
           values[0], values[1], values[2], values[3],
           values[4], values[5], values[6], values[7]);
           
    return result;
}

void slice_set_sse(sse_slice_t* slice, size_t idx, __m128d value) {
    if (idx >= slice->cap) {
        printf("SSE slice index out of bounds\n");
        abort();
    }
    
    printf("Setting SSE slice at index %zu\n", idx);
    print_debug_vector_sse("With values", value);
    slice->data[idx] = value;
    if (idx >= slice->len) {
        slice->len = idx + 1;
    }
}

void slice_set_avx(avx_slice_t* slice, size_t idx, __m512d value) {
    if (!slice || !slice->data || idx >= slice->cap) {
        printf("Invalid slice_set_avx parameters\n");
        abort();
    }
    
    printf("\nAVX slice_set_avx debug:\n");
    printf("  Slice pointer: %p\n", (void*)slice);
    printf("  Data pointer: %p\n", (void*)slice->data);
    printf("  Setting index: %zu\n", idx);
    
    // Safely extract values from the vector
    alignas(64) double values[8];
    _mm512_store_pd(values, value);
    
    printf("  Input values: %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n",
           values[0], values[1], values[2], values[3],
           values[4], values[5], values[6], values[7]);
           
    // Store back using aligned load
    __m512d aligned_value = _mm512_load_pd(values);
    slice->data[idx] = aligned_value;
    
    // Verify stored values
    alignas(64) double stored[8];
    _mm512_store_pd(stored, slice->data[idx]);
    printf("  Stored vector: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n",
           stored[0], stored[1], stored[2], stored[3],
           stored[4], stored[5], stored[6], stored[7]);
           
    if (idx >= slice->len) {
        slice->len = idx + 1;
    }
}

void slice_set_avx_values(avx_slice_t* slice, size_t idx,
                         double v0, double v1, double v2, double v3,
                         double v4, double v5, double v6, double v7) {
    printf("\nAVX slice_set_avx_values debug:\n");
    printf("  Input values: %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n",
           v0, v1, v2, v3, v4, v5, v6, v7);
           
    alignas(64) double values[8] = {v0, v1, v2, v3, v4, v5, v6, v7};
    __m512d vec = _mm512_load_pd(values);
    slice_set_avx(slice, idx, vec);
}

void free_slice(void* slice) {
    if (!slice) return;
    
    // Try as SSE slice first
    sse_slice_t* sse_slice = (sse_slice_t*)slice;
    if (sse_slice->data) {
        free_vectors(sse_slice->data);
        free(sse_slice);
        return;
    }
    
    // Try as AVX slice next
    avx_slice_t* avx_slice = (avx_slice_t*)slice;
    if (avx_slice->data) {
        free_vectors(avx_slice->data);
        free(avx_slice);
        return;
    }
    
    // If neither, just free
    free(slice);
}

// Vector printing functions (these need to be defined)
void print_sse_vector(__m128d vec) {
    alignas(16) double values[2];
    _mm_store_pd(values, vec);
    printf("[%.2f, %.2f]\n", values[0], values[1]);
}

void print_avx_vector(__m512d vec) {
    alignas(64) double values[8];
    _mm512_store_pd(values, vec);
    printf("[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n",
           values[0], values[1], values[2], values[3],
           values[4], values[5], values[6], values[7]);
}