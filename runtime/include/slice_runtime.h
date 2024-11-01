#ifndef SLICE_RUNTIME_H
#define SLICE_RUNTIME_H

#include "kernel.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Note: Slice types are now defined in kernel.h
// Using the names from kernel.h: sse_slice_t and avx_slice_t

// Runtime initialization and cleanup
void init_runtime(void);
void cleanup_runtime(void);

// Slice operations with error handling
int sse_slice_store_checked(sse_slice_t* slice, size_t index, sse_vector_t value);
int avx_slice_store_checked(avx_slice_t* slice, size_t index, avx_vector_t value);
int sse_slice_load_checked(sse_slice_t* slice, size_t index, sse_vector_t* out);
int avx_slice_load_checked(avx_slice_t* slice, size_t index, avx_vector_t* out);

// Extended operations for runtime
size_t sse_slice_len(sse_slice_t* slice);
size_t avx_slice_len(avx_slice_t* slice);
size_t sse_slice_cap(sse_slice_t* slice);
size_t avx_slice_cap(avx_slice_t* slice);

// Slice manipulation
int sse_slice_resize(sse_slice_t* slice, size_t new_len);
int avx_slice_resize(avx_slice_t* slice, size_t new_len);

// Copy operations
sse_slice_t* sse_slice_copy(const sse_slice_t* src);
avx_slice_t* avx_slice_copy(const avx_slice_t* src);

// Slice views (non-owning references)
typedef struct {
    const sse_vector_t* data;
    size_t len;
} sse_slice_view_t;

typedef struct {
    const avx_vector_t* data;
    size_t len;
} avx_slice_view_t;

// View operations
sse_slice_view_t sse_slice_view(const sse_slice_t* slice);
avx_slice_view_t avx_slice_view(const avx_slice_t* slice);

// Error handling
typedef enum {
    SLICE_OK = 0,
    SLICE_INDEX_OUT_OF_BOUNDS,
    SLICE_ALLOCATION_FAILED,
    SLICE_INVALID_ARGUMENT
} slice_error_t;

const char* slice_error_string(slice_error_t error);

#ifdef __cplusplus
}
#endif

#endif // SLICE_RUNTIME_H