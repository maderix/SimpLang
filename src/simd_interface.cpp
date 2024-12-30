#include "simd_interface.hpp"
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IntrinsicsX86.h>
#include <iostream>

// SSE Implementation
llvm::Value* SSEInterface::createVector(llvm::IRBuilder<>& builder,
                                      std::vector<llvm::Value*>& elements) {
    llvm::Type* doubleType = builder.getDoubleTy();
    llvm::VectorType* vectorType = llvm::VectorType::get(doubleType, 2, false);
    llvm::Value* vector = llvm::UndefValue::get(vectorType);
    
    for (size_t i = 0; i < elements.size() && i < 2; i++) {
        vector = builder.CreateInsertElement(vector, elements[i], 
                                          builder.getInt32(i));
    }
    
    return vector;
}

llvm::Value* SSEInterface::add(llvm::IRBuilder<>& builder,
                             llvm::Value* lhs, llvm::Value* rhs) {
    return builder.CreateFAdd(lhs, rhs);
}

llvm::Value* SSEInterface::sub(llvm::IRBuilder<>& builder,
                             llvm::Value* lhs, llvm::Value* rhs) {
    return builder.CreateFSub(lhs, rhs);
}

llvm::Value* SSEInterface::mul(llvm::IRBuilder<>& builder,
                             llvm::Value* lhs, llvm::Value* rhs) {
    return builder.CreateFMul(lhs, rhs);
}

llvm::Value* SSEInterface::div(llvm::IRBuilder<>& builder,
                             llvm::Value* lhs, llvm::Value* rhs) {
    return builder.CreateFDiv(lhs, rhs);
}

llvm::Value* SSEInterface::cmp_eq(llvm::IRBuilder<>& builder,
                                llvm::Value* lhs, llvm::Value* rhs) {
    return builder.CreateFCmpOEQ(lhs, rhs);
}

llvm::Value* SSEInterface::cmp_lt(llvm::IRBuilder<>& builder,
                                llvm::Value* lhs, llvm::Value* rhs) {
    return builder.CreateFCmpOLT(lhs, rhs);
}

llvm::Value* SSEInterface::shuffle(llvm::IRBuilder<>& builder,
                                 llvm::Value* vec, std::vector<int> mask) {
    // Convert mask to LLVM constant vector
    std::vector<llvm::Constant*> maskConstants;
    for (int idx : mask) {
        maskConstants.push_back(llvm::ConstantInt::get(builder.getInt32Ty(), idx));
    }
    llvm::Value* maskValue = llvm::ConstantVector::get(maskConstants);
    return builder.CreateShuffleVector(vec, vec, maskValue);
}

llvm::Value* SSEInterface::broadcast(llvm::IRBuilder<>& builder,
                                   llvm::Value* scalar) {
    llvm::Type* doubleType = builder.getDoubleTy();
    llvm::VectorType* vectorType = llvm::VectorType::get(doubleType, 2, false);
    llvm::Value* vector = llvm::UndefValue::get(vectorType);
    vector = builder.CreateInsertElement(vector, scalar, builder.getInt32(0));
    return builder.CreateShuffleVector(vector, vector, 
                                     llvm::ArrayRef<int>{0, 0});
}

llvm::Value* SSEInterface::call_intrinsic(llvm::IRBuilder<>& builder,
                                        const std::string& name,
                                        std::vector<llvm::Value*>& args) {
    llvm::Module* module = builder.GetInsertBlock()->getModule();
    llvm::Function* intrinsic = module->getFunction(name);
    
    if (!intrinsic) {
        std::cerr << "Intrinsic not found: " << name << std::endl;
        return nullptr;
    }
    
    return builder.CreateCall(intrinsic, args);
}

llvm::Type* SSEInterface::getVectorType(llvm::LLVMContext& context) {
    return llvm::VectorType::get(llvm::Type::getDoubleTy(context), 2, false);
}

// AVX Implementation
llvm::Value* AVXInterface::createVector(llvm::IRBuilder<>& builder,
                                     std::vector<llvm::Value*>& elements) {
    llvm::Type* doubleType = builder.getDoubleTy();
    llvm::VectorType* vectorType = llvm::VectorType::get(doubleType, 8, false);
    llvm::Value* vector = llvm::UndefValue::get(vectorType);
    
    for (size_t i = 0; i < elements.size() && i < 8; i++) {
        vector = builder.CreateInsertElement(vector, elements[i], 
                                          builder.getInt32(i));
    }
    
    return vector;
}

llvm::Value* AVXInterface::add(llvm::IRBuilder<>& builder,
                            llvm::Value* lhs, llvm::Value* rhs) {
    return builder.CreateFAdd(lhs, rhs);
}

llvm::Value* AVXInterface::sub(llvm::IRBuilder<>& builder,
                            llvm::Value* lhs, llvm::Value* rhs) {
    return builder.CreateFSub(lhs, rhs);
}

llvm::Value* AVXInterface::mul(llvm::IRBuilder<>& builder,
                            llvm::Value* lhs, llvm::Value* rhs) {
    return builder.CreateFMul(lhs, rhs);
}

llvm::Value* AVXInterface::div(llvm::IRBuilder<>& builder,
                            llvm::Value* lhs, llvm::Value* rhs) {
    return builder.CreateFDiv(lhs, rhs);
}

llvm::Value* AVXInterface::cmp_eq(llvm::IRBuilder<>& builder,
                               llvm::Value* lhs, llvm::Value* rhs) {
    return builder.CreateFCmpOEQ(lhs, rhs);
}

llvm::Value* AVXInterface::cmp_lt(llvm::IRBuilder<>& builder,
                               llvm::Value* lhs, llvm::Value* rhs) {
    return builder.CreateFCmpOLT(lhs, rhs);
}

llvm::Value* AVXInterface::shuffle(llvm::IRBuilder<>& builder,
                                llvm::Value* vec, std::vector<int> mask) {
    std::vector<llvm::Constant*> maskConstants;
    for (int idx : mask) {
        maskConstants.push_back(llvm::ConstantInt::get(builder.getInt32Ty(), idx));
    }
    llvm::Value* maskValue = llvm::ConstantVector::get(maskConstants);
    return builder.CreateShuffleVector(vec, vec, maskValue);
}

llvm::Value* AVXInterface::broadcast(llvm::IRBuilder<>& builder,
                                  llvm::Value* scalar) {
    llvm::Type* doubleType = builder.getDoubleTy();
    llvm::VectorType* vectorType = llvm::VectorType::get(doubleType, 8, false);
    llvm::Value* vector = llvm::UndefValue::get(vectorType);
    vector = builder.CreateInsertElement(vector, scalar, builder.getInt32(0));
    return builder.CreateShuffleVector(vector, vector, 
                                     llvm::ArrayRef<int>{0, 0, 0, 0, 0, 0, 0, 0});
}

llvm::Value* AVXInterface::call_intrinsic(llvm::IRBuilder<>& builder,
                                       const std::string& name,
                                       std::vector<llvm::Value*>& args) {
    llvm::Module* module = builder.GetInsertBlock()->getModule();
    llvm::Function* intrinsic = module->getFunction(name);
    
    if (!intrinsic) {
        std::cerr << "Intrinsic not found: " << name << std::endl;
        return nullptr;
    }
    
    return builder.CreateCall(intrinsic, args);
}

llvm::Type* AVXInterface::getVectorType(llvm::LLVMContext& context) {
    return llvm::VectorType::get(llvm::Type::getDoubleTy(context), 8, false);
}

// Factory function implementation
SIMDInterface* createSIMDInterface(const std::string& arch) {
    if (arch == "avx") {
        return new AVXInterface();
    } else {
        return new SSEInterface();
    }
} 