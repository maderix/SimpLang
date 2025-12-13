#ifndef SIMD_BACKEND_HPP
#define SIMD_BACKEND_HPP

#include <llvm/IR/Value.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Intrinsics.h>
#include <string>
#include <memory>

// Forward declarations
class CodeGenContext;
namespace llvm {
    class LLVMContext;
}

enum class SIMDType {
    None,
    Auto,
    AVX512,
    AVX,
    SSE,
    NEON,
    SVE
};

class SIMDBackend {
public:
    virtual ~SIMDBackend() = default;
    
    // Backend identification
    virtual std::string getName() const = 0;
    virtual SIMDType getType() const = 0;
    
    // Capability queries
    virtual bool supportsTarget() const = 0;
    virtual int getVectorWidth(llvm::Type* elementType) const = 0;  // f32->16, f64->8 for AVX-512
    virtual int getAlignment() const = 0;                           // 64 for AVX-512
    virtual bool supportsFMA() const = 0;                          // Fused multiply-add
    
    // Memory operations
    virtual llvm::Value* createAlignedAlloc(llvm::IRBuilder<>& builder, 
                                           llvm::Type* elementType, 
                                           llvm::Value* count) = 0;
    virtual llvm::Value* createVectorLoad(llvm::IRBuilder<>& builder, 
                                        llvm::Value* ptr, 
                                        llvm::Type* vectorType) = 0;
    virtual void createVectorStore(llvm::IRBuilder<>& builder, 
                                 llvm::Value* vector, 
                                 llvm::Value* ptr) = 0;
    
    // Arithmetic operations
    virtual llvm::Value* createVectorAdd(llvm::IRBuilder<>& builder, 
                                       llvm::Value* lhs, 
                                       llvm::Value* rhs) = 0;
    virtual llvm::Value* createVectorSub(llvm::IRBuilder<>& builder, 
                                       llvm::Value* lhs, 
                                       llvm::Value* rhs) = 0;
    virtual llvm::Value* createVectorMul(llvm::IRBuilder<>& builder, 
                                       llvm::Value* lhs, 
                                       llvm::Value* rhs) = 0;
    virtual llvm::Value* createVectorDiv(llvm::IRBuilder<>& builder, 
                                       llvm::Value* lhs, 
                                       llvm::Value* rhs) = 0;
    virtual llvm::Value* createVectorFMA(llvm::IRBuilder<>& builder, 
                                       llvm::Value* a, 
                                       llvm::Value* b, 
                                       llvm::Value* c) = 0;
    
    // Advanced operations
    virtual llvm::Value* createDotProduct(llvm::IRBuilder<>& builder,
                                        llvm::Value* lhs,
                                        llvm::Value* rhs,
                                        int elementCount) = 0;
    virtual llvm::Value* createHorizontalSum(llvm::IRBuilder<>& builder,
                                           llvm::Value* vector) = 0;
    
    // Comparison operations
    virtual llvm::Value* createVectorCmp(llvm::IRBuilder<>& builder,
                                       llvm::CmpInst::Predicate pred,
                                       llvm::Value* lhs,
                                       llvm::Value* rhs) = 0;
    
    // Type helpers
    virtual llvm::Type* getVectorType(llvm::Type* elementType, llvm::LLVMContext& context) const = 0;
    virtual llvm::Type* getMaskType(llvm::LLVMContext& context) const = 0;  // For AVX-512 masks
    
    // Vector slice operations
    // LLVM 21: elementType must be passed explicitly due to opaque pointers
    virtual llvm::Value* createVectorSliceLoad(llvm::IRBuilder<>& builder,
                                             llvm::Value* basePtr,
                                             llvm::Type* elementType,
                                             llvm::Value* startIndex,
                                             int sliceWidth) = 0;
    virtual void createVectorSliceStore(llvm::IRBuilder<>& builder,
                                      llvm::Value* vector,
                                      llvm::Value* basePtr,
                                      llvm::Type* elementType,
                                      llvm::Value* startIndex) = 0;
};

// Backend factory
class SIMDBackendFactory {
public:
    static std::unique_ptr<SIMDBackend> createBackend(SIMDType type);
    static SIMDType detectBestBackend();
    static bool isBackendAvailable(SIMDType type);
};

#endif // SIMD_BACKEND_HPP