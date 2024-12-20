#include "simd_ops.hpp"
#include "codegen.hpp"
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Type.h>
#include <iostream>

llvm::Value* SIMDHelper::performOp(
    CodeGenContext& context,
    llvm::Value* lhs,
    llvm::Value* rhs,
    SIMDOp op,
    SIMDWidth width
) {
    auto& builder = context.getBuilder();
    
    // Get vector type
    unsigned numElements = static_cast<unsigned>(width);
    llvm::Type* vecType = llvm::VectorType::get(
        builder.getDoubleTy(),
        numElements,
        false
    );
    
    // Load pointers if needed
    if (lhs->getType()->isPointerTy()) {
        lhs = builder.CreateLoad(lhs->getType()->getPointerElementType(), lhs);
    }
    if (rhs->getType()->isPointerTy()) {
        rhs = builder.CreateLoad(rhs->getType()->getPointerElementType(), rhs);
    }
    
    // Broadcast scalar to vector if needed
    if (!lhs->getType()->isVectorTy()) {
        lhs = broadcastScalar(context, lhs, vecType);
    }
    if (!rhs->getType()->isVectorTy()) {
        rhs = broadcastScalar(context, rhs, vecType);
    }
    
    // Perform operation
    switch (op) {
        case SIMDOp::ADD:
            return builder.CreateFAdd(lhs, rhs, "vec.add");
        case SIMDOp::SUB:
            return builder.CreateFSub(lhs, rhs, "vec.sub");
        case SIMDOp::MUL:
            return builder.CreateFMul(lhs, rhs, "vec.mul");
        case SIMDOp::DIV:
            return builder.CreateFDiv(lhs, rhs, "vec.div");
        default:
            std::cerr << "Unknown SIMD operation" << std::endl;
            return nullptr;
    }
}

llvm::Value* SIMDHelper::createVector(
    CodeGenContext& context,
    const std::vector<llvm::Value*>& elements,
    SIMDWidth width
) {
    auto& builder = context.getBuilder();
    unsigned numElements = static_cast<unsigned>(width);
    
    // Create vector type
    llvm::Type* vecType = llvm::VectorType::get(
        builder.getDoubleTy(),
        numElements,
        false
    );
    
    // If we have exactly one element, broadcast it
    if (elements.size() == 1) {
        llvm::Value* elem = elements[0];
        if (elem->getType()->isPointerTy()) {
            elem = builder.CreateLoad(elem->getType()->getPointerElementType(), elem);
        }
        return broadcastScalar(context, elem, vecType);
    }
    
    // Create undefined vector
    llvm::Value* vec = llvm::UndefValue::get(vecType);
    
    // Insert elements
    for (size_t i = 0; i < elements.size() && i < numElements; ++i) {
        vec = builder.CreateInsertElement(
            vec,
            elements[i],
            builder.getInt32(i),
            "vec.insert"
        );
    }
    
    // Fill remaining elements with zeros if needed
    if (elements.size() < numElements) {
        llvm::Value* zero = llvm::ConstantFP::get(
            context.getContext(),
            llvm::APFloat(0.0)
        );
        for (size_t i = elements.size(); i < numElements; ++i) {
            vec = builder.CreateInsertElement(
                vec,
                zero,
                builder.getInt32(i),
                "vec.insert.zero"
            );
        }
    }
    
    return vec;
}

llvm::Value* SIMDHelper::broadcastScalar(
    CodeGenContext& context,
    llvm::Value* scalar,
    llvm::Type* vecType
) {
    auto& builder = context.getBuilder();
    
    // Create undefined vector
    llvm::Value* vec = llvm::UndefValue::get(vecType);
    
    // Get number of elements
    unsigned numElements = llvm::cast<llvm::VectorType>(vecType)
        ->getElementCount().getKnownMinValue();
    
    // Insert scalar into all elements
    for (unsigned i = 0; i < numElements; ++i) {
        vec = builder.CreateInsertElement(
            vec,
            scalar,
            builder.getInt32(i),
            "vec.broadcast"
        );
    }
    
    return vec;
} 