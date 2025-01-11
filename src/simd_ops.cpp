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
    auto* simd = context.getSIMDInterface();
    
    // Load pointers if needed
    if (lhs->getType()->isPointerTy()) {
        lhs = builder.CreateLoad(lhs->getType()->getPointerElementType(), lhs);
    }
    if (rhs->getType()->isPointerTy()) {
        rhs = builder.CreateLoad(rhs->getType()->getPointerElementType(), rhs);
    }
    
    // Convert SIMDOp to ArithOp
    ArithOp arithOp;
    switch (op) {
        case SIMDOp::ADD: arithOp = ArithOp::Add; break;
        case SIMDOp::SUB: arithOp = ArithOp::Sub; break;
        case SIMDOp::MUL: arithOp = ArithOp::Mul; break;
        case SIMDOp::DIV: arithOp = ArithOp::Div; break;
        default:
            std::cerr << "Unknown SIMD operation" << std::endl;
            return nullptr;
    }
    
    // Use SIMD interface for operation
    return simd->arithOp(builder, lhs, rhs, arithOp);
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

llvm::Value* make_sse_slice(llvm::IRBuilder<>& builder, unsigned size) {
    std::vector<llvm::Type*> members = {
        llvm::PointerType::get(llvm::VectorType::get(
            builder.getDoubleTy(), 2, false), 0),
        builder.getInt64Ty(),  // size
        builder.getInt64Ty()   // capacity
    };
    
    llvm::StructType* sliceType = llvm::StructType::create(
        builder.getContext(), members, "SSESlice");
    
    // Allocate memory for the vector data
    llvm::Value* dataSize = builder.getInt64(size * sizeof(double) * 2);
    llvm::Value* data = builder.CreateCall(
        builder.GetInsertBlock()->getModule()->getFunction("malloc"), 
        {dataSize});
    data = builder.CreateBitCast(data, members[0]);
    
    // Create and initialize the slice struct
    llvm::Value* slice = llvm::UndefValue::get(sliceType);
    slice = builder.CreateInsertValue(slice, data, 0);
    slice = builder.CreateInsertValue(slice, builder.getInt64(size), 1);
    slice = builder.CreateInsertValue(slice, builder.getInt64(size), 2);
    
    return slice;
}

void slice_set_sse(llvm::IRBuilder<>& builder, SliceStruct& slice, 
                   unsigned index, llvm::Value* value) {
    llvm::Value* ptr = builder.CreateGEP(
        llvm::VectorType::get(builder.getDoubleTy(), 2, false),
        slice.data, 
        builder.getInt32(index));
    builder.CreateStore(value, ptr);
}

llvm::Value* make_avx_slice(llvm::IRBuilder<>& builder, unsigned size) {
    // Make AVX slice consistent with SSE slice structure
    std::vector<llvm::Type*> members = {
        llvm::PointerType::get(llvm::VectorType::get(
            builder.getDoubleTy(), 8, false), 0),  // data pointer to 8-wide vectors
        builder.getInt64Ty(),  // size
        builder.getInt64Ty()   // capacity
    };
    
    llvm::StructType* sliceType = llvm::StructType::create(
        builder.getContext(), members, "AVXSlice");
    
    // Allocate memory for the vector data (8 doubles per vector)
    llvm::Value* dataSize = builder.getInt64(size * sizeof(double) * 8);
    llvm::Value* data = builder.CreateCall(
        builder.GetInsertBlock()->getModule()->getFunction("malloc"), 
        {dataSize});
    data = builder.CreateBitCast(data, members[0]);
    
    // Create and initialize the slice struct (same pattern as SSE)
    llvm::Value* slice = llvm::UndefValue::get(sliceType);
    slice = builder.CreateInsertValue(slice, data, 0);
    slice = builder.CreateInsertValue(slice, builder.getInt64(size), 1);
    slice = builder.CreateInsertValue(slice, builder.getInt64(size), 2);
    
    std::cout << "Created AVX slice with size " << size 
              << " and data pointer " << data << std::endl;
    
    return slice;
}

void slice_set_avx(llvm::IRBuilder<>& builder, SliceStruct& slice, 
                   unsigned index, llvm::Value* value) {
    //std::cout << "\nAVX slice_set_avx debug:" << std::endl;
    
    // Calculate proper byte offset for AVX vectors (8 doubles = 64 bytes)
    llvm::Value* offset = builder.CreateMul(
        builder.getInt64(index),
        builder.getInt64(8),  // 8 doubles per AVX vector
        "avx.offset"
    );
    
    // Create GEP with explicit AVX vector type (8 doubles)
    auto avxType = llvm::VectorType::get(builder.getDoubleTy(), 8, false);
    llvm::Value* ptr = builder.CreateGEP(avxType, slice.data, offset, "avx.ptr");
    
    // Store with proper alignment
    auto store = builder.CreateStore(value, ptr);
    store->setAlignment(llvm::Align(64));  // AVX-512 needs 64-byte alignment
    
    //std::cout << "  Vector width: " << 
    //    llvm::dyn_cast<llvm::VectorType>(avxType)->getElementCount().getFixedValue() << std::endl;
    //std::cout << "  Offset in doubles: " << index * 8 << std::endl;
    //std::cout << "  Store operation completed with 64-byte alignment" << std::endl;
} 