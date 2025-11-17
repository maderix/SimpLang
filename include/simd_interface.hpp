#ifndef SIMD_INTERFACE_HPP
#define SIMD_INTERFACE_HPP

#include <llvm/IR/Value.h>
#include <llvm/IR/IRBuilder.h>
#include <vector>
#include <string>

// Architecture-specific intrinsics
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    #include <immintrin.h>
#elif defined(__aarch64__) || defined(__arm__)
    #include <arm_neon.h>
#endif

#include <iostream>

// On MSVC use __forceinline; otherwise use GCC/Clang attribute
#ifdef _MSC_VER
  #define FORCE_INLINE __forceinline
#else
  #define FORCE_INLINE __attribute__((always_inline)) inline
#endif

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
//
// Fallback: if __AVX512F__ not defined, create our own emulation type "SimdEmu512d".
// We do *not* redefine __m512d nor _mm512_* symbols to avoid conflicts.
//
#if !defined(__AVX512F__)

/**
 * Emulated 512-bit type using two 256-bit __m256d.
 * This avoids naming it __m512d, which conflicts with real AVX-512 definitions.
 */
typedef struct {
    __m256d low;
    __m256d high;
} SimdEmu512d;

// Emulated load
static FORCE_INLINE SimdEmu512d simd_emu_mm512_load_pd(const double* mem_addr) {
    SimdEmu512d result;
    result.low  = _mm256_load_pd(mem_addr);
    result.high = _mm256_load_pd(mem_addr + 4);
    return result;
}

// Template for all binary operations
template<typename BinaryOp>
static FORCE_INLINE SimdEmu512d simd_emu_mm512_binary_op(
    SimdEmu512d a, SimdEmu512d b, BinaryOp op) {
    SimdEmu512d result;
    result.low  = op(a.low,  b.low);
    result.high = op(a.high, b.high);
    return result;
}

// Operation wrappers using the template
static FORCE_INLINE SimdEmu512d simd_emu_mm512_add_pd(SimdEmu512d a, SimdEmu512d b) {
    return simd_emu_mm512_binary_op(a, b, _mm256_add_pd);
}

static FORCE_INLINE SimdEmu512d simd_emu_mm512_sub_pd(SimdEmu512d a, SimdEmu512d b) {
    return simd_emu_mm512_binary_op(a, b, _mm256_sub_pd);
}

static FORCE_INLINE SimdEmu512d simd_emu_mm512_mul_pd(SimdEmu512d a, SimdEmu512d b) {
    return simd_emu_mm512_binary_op(a, b, _mm256_mul_pd);
}

static FORCE_INLINE SimdEmu512d simd_emu_mm512_div_pd(SimdEmu512d a, SimdEmu512d b) {
    return simd_emu_mm512_binary_op(a, b, _mm256_div_pd);
}

#endif // !__AVX512F__
#endif  // x86

// Platform-agnostic enums and interface
enum class ArithOp { Add, Sub, Mul, Div };
enum class CmpOp { EQ, LT, GT, LE, GE, NE };

/**
 * Main SIMDInterface classes.
 * We do not define or rename __m512d or _mm512_* here if AVX-512 is found.
 */
class SIMDInterface {
public:
    virtual ~SIMDInterface() = default;

    // Move arithOp to public and make it a public interface method
    virtual llvm::Value* arithOp(llvm::IRBuilder<>& builder,
                                llvm::Value* lhs, llvm::Value* rhs,
                                ArithOp op) {
        switch (op) {
            case ArithOp::Add: return builder.CreateFAdd(lhs, rhs, "simd.add");
            case ArithOp::Sub: return builder.CreateFSub(lhs, rhs, "simd.sub");
            case ArithOp::Mul: return builder.CreateFMul(lhs, rhs, "simd.mul");
            case ArithOp::Div: return builder.CreateFDiv(lhs, rhs, "simd.div");
            default: return nullptr;
        }
    }

    // Helper for comparison operations
    virtual llvm::Value* compareOp(llvm::IRBuilder<>& builder,
                                  llvm::Value* lhs, llvm::Value* rhs,
                                  CmpOp op) {
        switch (op) {
            case CmpOp::EQ: return builder.CreateFCmpOEQ(lhs, rhs);
            case CmpOp::LT: return builder.CreateFCmpOLT(lhs, rhs);
            case CmpOp::GT: return builder.CreateFCmpOGT(lhs, rhs);
            case CmpOp::LE: return builder.CreateFCmpOLE(lhs, rhs);
            case CmpOp::GE: return builder.CreateFCmpOGE(lhs, rhs);
            case CmpOp::NE: return builder.CreateFCmpONE(lhs, rhs);
        }
        return nullptr;
    }

    // Vector creation
    virtual llvm::Value* createVector(llvm::IRBuilder<>& builder,
                                      std::vector<llvm::Value*>& elements) = 0;

    // Basic operations
    virtual llvm::Value* add(llvm::IRBuilder<>& builder,
                             llvm::Value* lhs, llvm::Value* rhs) {
        return arithOp(builder, lhs, rhs, ArithOp::Add);
    }
    virtual llvm::Value* sub(llvm::IRBuilder<>& builder,
                             llvm::Value* lhs, llvm::Value* rhs) {
        return arithOp(builder, lhs, rhs, ArithOp::Sub);
    }
    virtual llvm::Value* mul(llvm::IRBuilder<>& builder,
                             llvm::Value* lhs, llvm::Value* rhs) {
        return arithOp(builder, lhs, rhs, ArithOp::Mul);
    }
    virtual llvm::Value* div(llvm::IRBuilder<>& builder,
                             llvm::Value* lhs, llvm::Value* rhs) {
        return arithOp(builder, lhs, rhs, ArithOp::Div);
    }

    // Comparison operations
    virtual llvm::Value* cmp_eq(llvm::IRBuilder<>& builder,
                                llvm::Value* lhs, llvm::Value* rhs) {
        return compareOp(builder, lhs, rhs, CmpOp::EQ);
    }
    virtual llvm::Value* cmp_lt(llvm::IRBuilder<>& builder,
                                llvm::Value* lhs, llvm::Value* rhs) {
        return compareOp(builder, lhs, rhs, CmpOp::LT);
    }

    // Vector manipulation
    virtual llvm::Value* shuffle(llvm::IRBuilder<>& builder,
                                 llvm::Value* vec, std::vector<int> mask) = 0;
    virtual llvm::Value* broadcast(llvm::IRBuilder<>& builder,
                                   llvm::Value* scalar) = 0;

    // Intrinsic operations
    virtual llvm::Value* call_intrinsic(llvm::IRBuilder<>& builder,
                                        const std::string& name,
                                        std::vector<llvm::Value*>& args) = 0;

    // Type information
    virtual unsigned getVectorWidth() const = 0;
    virtual llvm::Type* getVectorType(llvm::LLVMContext& context) = 0;
};

class BaseSIMDImpl : public SIMDInterface {
protected:
    const unsigned vectorWidth_;

    BaseSIMDImpl(unsigned width) : vectorWidth_(width) {}

public:
    llvm::Value* createVector(llvm::IRBuilder<>& builder,
                             std::vector<llvm::Value*>& elements) override {
        if (elements.size() > vectorWidth_) {
            std::cerr << "Warning: Vector creation truncated to " 
                      << vectorWidth_ << " elements" << std::endl;
        }
        
        llvm::Type* doubleType = builder.getDoubleTy();
        llvm::VectorType* vectorType = llvm::VectorType::get(doubleType, vectorWidth_, false);
        llvm::Value* vector = llvm::UndefValue::get(vectorType);
        
        for (size_t i = 0; i < std::min(elements.size(), size_t(vectorWidth_)); i++) {
            vector = builder.CreateInsertElement(vector, elements[i], 
                                              builder.getInt32(i));
        }
        
        // Fill remaining elements with zeros if needed
        if (elements.size() < vectorWidth_) {
            auto zero = llvm::ConstantFP::get(doubleType, 0.0);
            for (size_t i = elements.size(); i < vectorWidth_; i++) {
                vector = builder.CreateInsertElement(vector, zero, 
                                                  builder.getInt32(i));
            }
        }
        
        return vector;
    }

    // Reuse base class implementations for arithmetic and comparison
    using SIMDInterface::add;
    using SIMDInterface::sub;
    using SIMDInterface::mul;
    using SIMDInterface::div;
    using SIMDInterface::cmp_eq;
    using SIMDInterface::cmp_lt;

    llvm::Value* shuffle(llvm::IRBuilder<>& builder,
                        llvm::Value* vec, std::vector<int> mask) override {
        std::vector<llvm::Constant*> maskConstants;
        for (int idx : mask) {
            maskConstants.push_back(llvm::ConstantInt::get(builder.getInt32Ty(), idx));
        }
        llvm::Value* maskValue = llvm::ConstantVector::get(maskConstants);
        return builder.CreateShuffleVector(vec, vec, maskValue);
    }

    unsigned getVectorWidth() const override { return vectorWidth_; }
    
    llvm::Type* getVectorType(llvm::LLVMContext& context) override {
        return llvm::VectorType::get(llvm::Type::getDoubleTy(context), vectorWidth_, false);
    }
};

// SSE = 2 doubles
class SSEInterface : public BaseSIMDImpl {
public:
    SSEInterface() : BaseSIMDImpl(2) {}
    
    llvm::Value* broadcast(llvm::IRBuilder<>& builder,
                          llvm::Value* scalar) override;
    llvm::Value* call_intrinsic(llvm::IRBuilder<>& builder,
                               const std::string& name,
                               std::vector<llvm::Value*>& args) override;
};

// AVX "interface" = 8 doubles (often labeled AVX-512 in code)
class AVXInterface : public BaseSIMDImpl {
public:
    AVXInterface() : BaseSIMDImpl(8) {}
    
    llvm::Value* broadcast(llvm::IRBuilder<>& builder,
                          llvm::Value* scalar) override;
    llvm::Value* call_intrinsic(llvm::IRBuilder<>& builder,
                               const std::string& name,
                               std::vector<llvm::Value*>& args) override;
};

// Factory function
SIMDInterface* createSIMDInterface(const std::string& arch);

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
// Helpers for building SSE / AVX vectors in user code:

// SSE expects exactly 2 doubles:
template<typename... Args>
constexpr bool ValidateSSEArgs() {
    static_assert(sizeof...(Args) == 2,
        "SSE vectors must contain exactly 2 doubles");
    return true;
}

FORCE_INLINE __m128d sse(double a, double b) {
    alignas(16) double values[2] = { a, b };
    return _mm_load_pd(values);
}

#define SSE_VECTOR(a, b) \
    ( ValidateSSEArgs<decltype(a), decltype(b)>(), sse(a, b) )

// AVX (in your code) actually expects 8 doubles (emulating 512 bits):
template<typename... Args>
constexpr bool ValidateAVXArgs() {
    static_assert(sizeof...(Args) == 8,
        "AVX vectors must contain exactly 8 doubles");
    return true;
}

/**
 * If AVX-512 is available, this returns real __m512d.
 * Otherwise returns the fallback SimdEmu512d.
 */
FORCE_INLINE
#if defined(__AVX512F__)
__m512d
#else
SimdEmu512d
#endif
avx(double a, double b, double c, double d,
    double e, double f, double g, double h)
{
    alignas(64) double values[8] = { a, b, c, d, e, f, g, h };

#if defined(__AVX512F__)
    // Real AVX-512
    return _mm512_load_pd(values);
#else
    // Fallback emulate
    return simd_emu_mm512_load_pd(values);
#endif
}

#define AVX_VECTOR(a,b,c,d,e,f,g,h) \
    ( ValidateAVXArgs<decltype(a), decltype(b), decltype(c), decltype(d), \
                     decltype(e), decltype(f), decltype(g), decltype(h)>(), \
      avx(a,b,c,d,e,f,g,h) )

#endif  // x86

#endif // SIMD_INTERFACE_HPP
