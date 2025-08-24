#include "simd_backend.hpp"
#include <llvm/IR/IntrinsicsX86.h>
#include <llvm/IR/Constants.h>
#include <immintrin.h>

class AVX512Backend : public SIMDBackend {
public:
    std::string getName() const override {
        return "AVX-512";
    }
    
    SIMDType getType() const override {
        return SIMDType::AVX512;
    }
    
    bool supportsTarget() const override {
        // Runtime detection of AVX-512 support
        return __builtin_cpu_supports("avx512f");
    }
    
    int getVectorWidth(llvm::Type* elementType) const override {
        if (elementType->isFloatTy()) {
            return 16;  // 16 x f32 in 512-bit vector
        } else if (elementType->isDoubleTy()) {
            return 8;   // 8 x f64 in 512-bit vector
        } else if (elementType->isIntegerTy(32)) {
            return 16;  // 16 x i32 in 512-bit vector
        } else if (elementType->isIntegerTy(64)) {
            return 8;   // 8 x i64 in 512-bit vector
        }
        return 1;  // Fallback to scalar
    }
    
    int getAlignment() const override {
        return 64;  // 64-byte alignment for AVX-512
    }
    
    bool supportsFMA() const override {
        return true;  // AVX-512 includes FMA
    }
    
    llvm::Value* createAlignedAlloc(llvm::IRBuilder<>& builder, 
                                   llvm::Type* elementType, 
                                   llvm::Value* count) override {
        auto& context = builder.getContext();
        auto* module = builder.GetInsertBlock()->getModule();
        
        // Calculate total size: count * sizeof(elementType)
        auto* elementSize = llvm::ConstantInt::get(
            llvm::Type::getInt64Ty(context), 
            elementType->getPrimitiveSizeInBits() / 8
        );
        auto* totalSize = builder.CreateMul(count, elementSize, "total_size");
        
        // Call aligned_alloc(64, totalSize)
        auto aligned_alloc = module->getOrInsertFunction(
            "aligned_alloc",
            llvm::FunctionType::get(
                llvm::Type::getInt8PtrTy(context),
                {llvm::Type::getInt64Ty(context), llvm::Type::getInt64Ty(context)},
                false
            )
        );
        
        auto* alignment = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 64);
        auto* ptr = builder.CreateCall(aligned_alloc, {alignment, totalSize});
        
        // Cast to appropriate pointer type
        return builder.CreateBitCast(ptr, elementType->getPointerTo(), "aligned_ptr");
    }
    
    llvm::Value* createVectorLoad(llvm::IRBuilder<>& builder, 
                                 llvm::Value* ptr, 
                                 llvm::Type* vectorType) override {
        // Use AVX-512 aligned load
        auto* module = builder.GetInsertBlock()->getModule();
        
        // For now, use regular LLVM vector load and let LLVM optimize to AVX-512
        // LLVM will automatically use the best available instructions
        
        // Fallback to regular load
        return builder.CreateLoad(vectorType, ptr);
    }
    
    void createVectorStore(llvm::IRBuilder<>& builder, 
                          llvm::Value* vector, 
                          llvm::Value* ptr) override {
        auto* module = builder.GetInsertBlock()->getModule();
        auto* vectorType = vector->getType();
        
        // Use regular LLVM vector store - LLVM will optimize to AVX-512
        builder.CreateStore(vector, ptr);
    }
    
    llvm::Value* createVectorAdd(llvm::IRBuilder<>& builder, 
                                llvm::Value* lhs, 
                                llvm::Value* rhs) override {
        auto* vectorType = lhs->getType();
        
        if (llvm::cast<llvm::VectorType>(vectorType)->getElementType()->isFloatTy()) {
            // Use LLVM's built-in vector add - LLVM will generate optimal AVX-512
            return builder.CreateFAdd(lhs, rhs, "vadd");
        } else if (llvm::cast<llvm::VectorType>(vectorType)->getElementType()->isDoubleTy()) {
            return builder.CreateFAdd(lhs, rhs, "vadd");
        } else if (llvm::cast<llvm::VectorType>(vectorType)->getElementType()->isIntegerTy()) {
            return builder.CreateAdd(lhs, rhs, "vadd");
        }
        
        return nullptr;
    }
    
    llvm::Value* createVectorSub(llvm::IRBuilder<>& builder, 
                                llvm::Value* lhs, 
                                llvm::Value* rhs) override {
        auto* vectorType = lhs->getType();
        
        if (llvm::cast<llvm::VectorType>(vectorType)->getElementType()->isFloatTy() || 
            llvm::cast<llvm::VectorType>(vectorType)->getElementType()->isDoubleTy()) {
            return builder.CreateFSub(lhs, rhs, "vsub");
        } else if (llvm::cast<llvm::VectorType>(vectorType)->getElementType()->isIntegerTy()) {
            return builder.CreateSub(lhs, rhs, "vsub");
        }
        
        return nullptr;
    }
    
    llvm::Value* createVectorMul(llvm::IRBuilder<>& builder, 
                                llvm::Value* lhs, 
                                llvm::Value* rhs) override {
        auto* vectorType = lhs->getType();
        
        if (llvm::cast<llvm::VectorType>(vectorType)->getElementType()->isFloatTy() || 
            llvm::cast<llvm::VectorType>(vectorType)->getElementType()->isDoubleTy()) {
            return builder.CreateFMul(lhs, rhs, "vmul");
        } else if (llvm::cast<llvm::VectorType>(vectorType)->getElementType()->isIntegerTy()) {
            return builder.CreateMul(lhs, rhs, "vmul");
        }
        
        return nullptr;
    }
    
    llvm::Value* createVectorDiv(llvm::IRBuilder<>& builder, 
                                llvm::Value* lhs, 
                                llvm::Value* rhs) override {
        auto* vectorType = lhs->getType();
        
        if (llvm::cast<llvm::VectorType>(vectorType)->getElementType()->isFloatTy() || 
            llvm::cast<llvm::VectorType>(vectorType)->getElementType()->isDoubleTy()) {
            return builder.CreateFDiv(lhs, rhs, "vdiv");
        } else if (llvm::cast<llvm::VectorType>(vectorType)->getElementType()->isIntegerTy()) {
            return builder.CreateSDiv(lhs, rhs, "vdiv");  // Signed division
        }
        
        return nullptr;
    }
    
    llvm::Value* createVectorFMA(llvm::IRBuilder<>& builder, 
                                llvm::Value* a, 
                                llvm::Value* b, 
                                llvm::Value* c) override {
        // FMA: a * b + c using LLVM intrinsic (LLVM will use AVX-512 FMA)
        auto* module = builder.GetInsertBlock()->getModule();
        auto* vectorType = a->getType();
        
        if (llvm::cast<llvm::VectorType>(vectorType)->getElementType()->isFloatTy()) {
            auto* fma_intrinsic = llvm::Intrinsic::getDeclaration(
                module, llvm::Intrinsic::fma, {vectorType});
            return builder.CreateCall(fma_intrinsic, {a, b, c}, "vfma");
        } else if (llvm::cast<llvm::VectorType>(vectorType)->getElementType()->isDoubleTy()) {
            auto* fma_intrinsic = llvm::Intrinsic::getDeclaration(
                module, llvm::Intrinsic::fma, {vectorType});
            return builder.CreateCall(fma_intrinsic, {a, b, c}, "vfma");
        }
        
        // Fallback: a * b + c
        auto* mul = createVectorMul(builder, a, b);
        return createVectorAdd(builder, mul, c);
    }
    
    llvm::Value* createDotProduct(llvm::IRBuilder<>& builder,
                                 llvm::Value* lhs,
                                 llvm::Value* rhs,
                                 int elementCount) override {
        // Multiply vectors element-wise
        auto* product = createVectorMul(builder, lhs, rhs);
        
        // Horizontal sum of the result
        return createHorizontalSum(builder, product);
    }
    
    llvm::Value* createHorizontalSum(llvm::IRBuilder<>& builder,
                                    llvm::Value* vector) override {
        auto* module = builder.GetInsertBlock()->getModule();
        auto* vectorType = vector->getType();
        
        if (llvm::cast<llvm::VectorType>(vectorType)->getElementType()->isFloatTy()) {
            // Use AVX-512 horizontal sum reduction
            auto* intrinsic = llvm::Intrinsic::getDeclaration(
                module, llvm::Intrinsic::vector_reduce_fadd,
                llvm::cast<llvm::VectorType>(vectorType)->getElementType());
            auto* zero = llvm::ConstantFP::get(llvm::cast<llvm::VectorType>(vectorType)->getElementType(), 0.0);
            return builder.CreateCall(intrinsic, {zero, vector}, "hsum");
        }
        
        // Fallback: extract and sum manually (LLVM will optimize)
        auto* elementType = llvm::cast<llvm::VectorType>(vectorType)->getElementType();
        llvm::Value* sum = llvm::Constant::getNullValue(elementType);
        
        for (unsigned i = 0; i < llvm::cast<llvm::VectorType>(vectorType)->getElementCount().getFixedValue(); ++i) {
            auto* idx = llvm::ConstantInt::get(llvm::Type::getInt32Ty(builder.getContext()), i);
            auto* element = builder.CreateExtractElement(vector, idx);
            sum = builder.CreateFAdd(sum, element);
        }
        
        return sum;
    }
    
    llvm::Value* createVectorCmp(llvm::IRBuilder<>& builder,
                                llvm::CmpInst::Predicate pred,
                                llvm::Value* lhs,
                                llvm::Value* rhs) override {
        auto* vectorType = lhs->getType();
        
        if (llvm::cast<llvm::VectorType>(vectorType)->getElementType()->isFloatTy() || 
            llvm::cast<llvm::VectorType>(vectorType)->getElementType()->isDoubleTy()) {
            return builder.CreateFCmp(pred, lhs, rhs, "vcmp");
        } else if (llvm::cast<llvm::VectorType>(vectorType)->getElementType()->isIntegerTy()) {
            return builder.CreateICmp(pred, lhs, rhs, "vcmp");
        }
        
        return nullptr;
    }
    
    llvm::Type* getVectorType(llvm::Type* elementType, llvm::LLVMContext& context) const override {
        int width = getVectorWidth(elementType);
        return llvm::VectorType::get(elementType, width, false);
    }
    
    llvm::Type* getMaskType(llvm::LLVMContext& context) const override {
        return llvm::Type::getInt16Ty(context);  // 16-bit mask for AVX-512
    }
    
    llvm::Value* createVectorSliceLoad(llvm::IRBuilder<>& builder,
                                      llvm::Value* basePtr,
                                      llvm::Value* startIndex,
                                      int sliceWidth) override {
        // Calculate offset: basePtr + startIndex
        auto* offset = builder.CreateGEP(
            basePtr->getType()->getPointerElementType(),
            basePtr, startIndex, "slice_ptr");
        
        // Create vector type for slice
        auto* elementType = basePtr->getType()->getPointerElementType();
        auto* sliceVectorType = llvm::VectorType::get(elementType, sliceWidth, false);
        
        return createVectorLoad(builder, offset, sliceVectorType);
    }
    
    void createVectorSliceStore(llvm::IRBuilder<>& builder,
                               llvm::Value* vector,
                               llvm::Value* basePtr,
                               llvm::Value* startIndex) override {
        // Calculate offset: basePtr + startIndex  
        auto* offset = builder.CreateGEP(
            basePtr->getType()->getPointerElementType(),
            basePtr, startIndex, "slice_ptr");
        
        createVectorStore(builder, vector, offset);
    }
};

// Backend factory implementation
std::unique_ptr<SIMDBackend> SIMDBackendFactory::createBackend(SIMDType type) {
    switch (type) {
        case SIMDType::AVX512:
            return std::make_unique<AVX512Backend>();
        case SIMDType::Auto:
            return createBackend(detectBestBackend());
        default:
            return nullptr;
    }
}

SIMDType SIMDBackendFactory::detectBestBackend() {
    if (__builtin_cpu_supports("avx512f")) {
        return SIMDType::AVX512;
    }
    // For now, fallback to None if AVX-512 not available
    return SIMDType::None;
}

bool SIMDBackendFactory::isBackendAvailable(SIMDType type) {
    switch (type) {
        case SIMDType::AVX512:
            return __builtin_cpu_supports("avx512f");
        default:
            return false;
    }
}