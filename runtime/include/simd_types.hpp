#pragma once

#include <cstddef>

// Align with SIMDWidth enum in simd_ops.hpp
struct SSESlice {
    float* data;
    size_t size;  // Size in vectors
    static constexpr size_t VECTOR_SIZE = 2;  // 128-bit SSE = 2 doubles
};

struct AVXSlice {
    float* data;
    size_t size;  // Size in vectors
    static constexpr size_t VECTOR_SIZE = 8;  // 512-bit AVX = 8 doubles
}; 