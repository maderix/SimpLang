#include "kernel.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>

// Platform-specific aligned allocation
#ifdef _WIN32
    #include <malloc.h>
    #define aligned_alloc(align, size) _aligned_malloc(size, align)
    #define aligned_free(ptr) _aligned_free(ptr)
#else
    #define aligned_free(ptr) free(ptr)
#endif

// Vector allocation functions
sse_vector_t* allocate_sse_vectors(size_t count) {
    void* ptr = aligned_alloc(32, count * sizeof(sse_vector_t));
    return static_cast<sse_vector_t*>(ptr);
}

avx_vector_t* allocate_avx_vectors(size_t count) {
    void* ptr = aligned_alloc(64, count * sizeof(avx_vector_t));
    return static_cast<avx_vector_t*>(ptr);
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

void free_slice(void* slice) {
    if (!slice) return;

    // Try to determine slice type by examining size of first element
    void* data_ptr = *(void**)slice;  // Get data pointer
    if (data_ptr) {
        aligned_free(data_ptr);
    }
    free(slice);
}

// Vector printing functions
void print_sse_vector(sse_vector_t vec) {
    double values[4];
    _mm256_storeu_pd(values, vec);
    printf("[%.2f, %.2f, %.2f, %.2f]\n", 
           values[0], values[1], values[2], values[3]);
}

void print_avx_vector(avx_vector_t vec) {
    double values[8];
    _mm512_storeu_pd(values, vec);
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

// Error handling for bounds checking
[[noreturn]] static void bounds_check_failed(size_t index, size_t len) {
    fprintf(stderr, "Index out of bounds: %zu >= %zu\n", index, len);
    abort();
}

void check_bounds(size_t index, size_t len) {
    if (index >= len) {
        bounds_check_failed(index, len);
    }
}