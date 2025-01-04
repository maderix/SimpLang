#include "simd_interface.hpp"
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IntrinsicsX86.h>
#include <iostream>

// SSE Implementation - only unique operations remain
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

// AVX Implementation - only unique operations remain
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

// Factory function implementation
SIMDInterface* createSIMDInterface(const std::string& arch) {
    if (arch == "avx") {
        return new AVXInterface();
    } else {
        return new SSEInterface();
    }
} 