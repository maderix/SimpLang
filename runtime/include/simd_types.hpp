#pragma once

#include <cstddef>

// Vector types for SIMD operations
typedef double __attribute__((vector_size(16))) sse_vector_t;  // 2 doubles
typedef double __attribute__((vector_size(64))) avx_vector_t;  // 8 doubles

struct SSESlice {
    sse_vector_t* data;
    size_t size;  // Size in vectors
    size_t capacity;
    static constexpr size_t VECTOR_SIZE = 2;  // 128-bit SSE = 2 doubles
};

struct AVXSlice {
    avx_vector_t* data;
    size_t size;  // Size in vectors
    size_t capacity;
    static constexpr size_t VECTOR_SIZE = 8;  // 512-bit AVX = 8 doubles
}; 