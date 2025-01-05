#pragma once
#include <immintrin.h>
#include <llvm/IR/IRBuilder.h>
#include <vector>

// Forward declarations
class CodeGenContext;

#ifdef __cplusplus
extern "C" {
#endif

// SSE Operations
__m128d simd_add(__m128d a, __m128d b);
__m128d simd_sub(__m128d a, __m128d b);
__m128d simd_mul(__m128d a, __m128d b);
__m128d simd_div(__m128d a, __m128d b);

// AVX Operations
__m256d simd_add_avx(__m256d a, __m256d b);
__m256d simd_sub_avx(__m256d a, __m256d b);
__m256d simd_mul_avx(__m256d a, __m256d b);
__m256d simd_div_avx(__m256d a, __m256d b);

#ifdef __cplusplus
}
#endif

enum class SIMDOp {
    ADD,
    SUB,
    MUL,
    DIV
};

enum class SIMDWidth {
    SSE = 2,
    AVX = 8
};

struct SliceStruct {
    llvm::Value* data;    // Pointer to data
    llvm::Value* length;  // Length of slice
    llvm::Value* capacity;  // Capacity of slice
};

class SIMDHelper {
public:
    static llvm::Value* performOp(
        CodeGenContext& context,
        llvm::Value* lhs,
        llvm::Value* rhs,
        SIMDOp op,
        SIMDWidth width
    );

    static llvm::Value* createVector(
        CodeGenContext& context,
        const std::vector<llvm::Value*>& elements,
        SIMDWidth width
    );

    static llvm::Value* broadcastScalar(
        CodeGenContext& context,
        llvm::Value* scalar,
        llvm::Type* vecType
    );
};

// SSE slice operations
llvm::Value* make_sse_slice(llvm::IRBuilder<>& builder, unsigned size);
void slice_set_sse(llvm::IRBuilder<>& builder, SliceStruct& slice, 
                   unsigned index, llvm::Value* value);

// AVX slice operations
llvm::Value* make_avx_slice(llvm::IRBuilder<>& builder, unsigned size);
void slice_set_avx(llvm::IRBuilder<>& builder, SliceStruct& slice, 
                   unsigned index, llvm::Value* value);
