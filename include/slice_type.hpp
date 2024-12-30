#ifndef SLICE_TYPE_HPP
#define SLICE_TYPE_HPP

enum class SliceType {
    SSE_SLICE,  // For SSE vectors (2 doubles)
    AVX_SLICE   // For AVX vectors (8 doubles)
};

#endif // SLICE_TYPE_HPP 