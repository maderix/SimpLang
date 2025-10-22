#include "ast/expr/array_expr.hpp"
#include "ast/expr/variable_expr.hpp"
#include "ast/type/type_info.hpp"
#include "../ast_utils.hpp"
#include "logger.hpp"
#include "codegen.hpp"
#include <llvm/IR/Value.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/DataLayout.h>

// ArrayCreateExprAST implementation
llvm::Value* ArrayCreateExprAST::codeGen(CodeGenContext& context) {
    LOG_DEBUG("Creating array with element type: ", elementType->toString());

    // Check if this is a SIMD array
    auto* arrayTypeInfo = dynamic_cast<ArrayTypeInfo*>(elementType.get());
    if (arrayTypeInfo && arrayTypeInfo->vectorizable) {
        LOG_DEBUG("Creating SIMD array with hint: ", static_cast<int>(arrayTypeInfo->simdHint));

        // Get the SIMD backend
        auto* backend = context.getSIMDBackend(arrayTypeInfo->simdHint);
        if (!backend) {
            LOG_ERROR("SIMD backend not available for requested type");
            // Fall back to regular array
        } else {
            // Use SIMD backend to create aligned array
            llvm::Type* elemType = arrayTypeInfo->elementType->getLLVMType(context.getContext());
            if (!elemType) {
                LOG_ERROR("Invalid SIMD array element type");
                return nullptr;
            }

            // Calculate total size
            llvm::Value* totalSize = llvm::ConstantInt::get(context.getBuilder().getInt64Ty(), 1);
            for (auto& dimExpr : dimensionExprs) {
                llvm::Value* dim = dimExpr->codeGen(context);
                if (!dim) {
                    LOG_ERROR("Failed to generate array dimension expression");
                    return nullptr;
                }
                if (dim->getType() != context.getBuilder().getInt64Ty()) {
                    dim = convertType(dim, context.getBuilder().getInt64Ty(), context, "dim_conv");
                }
                totalSize = context.getBuilder().CreateMul(totalSize, dim, "simd_array_size");
            }

            // Use SIMD backend for aligned allocation
            llvm::Value* simdArrayPtr = backend->createAlignedAlloc(context.getBuilder(), elemType, totalSize);
            LOG_DEBUG("SIMD array created with alignment: ", backend->getAlignment());
            return simdArrayPtr;
        }
    }

    // Regular array creation (fallback or non-SIMD)
    llvm::Type* elemType = elementType->getLLVMType(context.getContext());
    if (!elemType) {
        LOG_ERROR("Invalid array element type");
        return nullptr;
    }

    // Calculate total size by multiplying all dimensions
    llvm::Value* totalSize = llvm::ConstantInt::get(context.getBuilder().getInt64Ty(), 1);

    for (auto& dimExpr : dimensionExprs) {
        llvm::Value* dim = dimExpr->codeGen(context);
        if (!dim) {
            LOG_ERROR("Failed to generate array dimension expression");
            return nullptr;
        }
        // Convert dimension to i64 if needed
        if (dim->getType() != context.getBuilder().getInt64Ty()) {
            dim = convertType(dim, context.getBuilder().getInt64Ty(), context, "dim_conv");
        }
        totalSize = context.getBuilder().CreateMul(totalSize, dim, "array_size_mul");
    }

    LOG_DEBUG("Array total size calculation generated");

    // Allocate memory for array data (aligned for SIMD operations)
    llvm::Value* elementSize = llvm::ConstantInt::get(context.getBuilder().getInt64Ty(),
        context.getModule()->getDataLayout().getTypeAllocSize(elemType));
    llvm::Value* totalBytes = context.getBuilder().CreateMul(totalSize, elementSize, "total_bytes");

    // Use malloc for memory allocation
    llvm::Function* mallocFunc = context.getModule()->getFunction("malloc");
    if (!mallocFunc) {
        LOG_ERROR("malloc function not found");
        return nullptr;
    }

    llvm::Value* rawPtr = context.getBuilder().CreateCall(mallocFunc, {totalBytes}, "array_raw_ptr");
    llvm::Value* typedPtr = context.getBuilder().CreateBitCast(rawPtr,
        llvm::PointerType::get(elemType, 0), "array_data_ptr");

    LOG_DEBUG("Array memory allocated successfully");
    return typedPtr;
}

// SIMDArrayCreateExprAST implementation
llvm::Value* SIMDArrayCreateExprAST::codeGen(CodeGenContext& context) {
    LOG_DEBUG("Creating SIMD array with hint: ", static_cast<int>(simdHint));

    // Get the SIMD backend
    auto* backend = context.getSIMDBackend(simdHint);
    if (!backend) {
        LOG_ERROR("SIMD backend not available for requested type");
        return nullptr;
    }

    // Get element type
    llvm::Type* elemType = elementType->getLLVMType(context.getContext());
    if (!elemType) {
        LOG_ERROR("Invalid SIMD array element type");
        return nullptr;
    }

    // Calculate total size
    llvm::Value* totalSize = llvm::ConstantInt::get(context.getBuilder().getInt64Ty(), 1);
    for (auto& dimExpr : dimensionExprs) {
        llvm::Value* dim = dimExpr->codeGen(context);
        if (!dim) {
            LOG_ERROR("Failed to generate SIMD array dimension expression");
            return nullptr;
        }
        if (dim->getType() != context.getBuilder().getInt64Ty()) {
            dim = convertType(dim, context.getBuilder().getInt64Ty(), context, "dim_conv");
        }
        totalSize = context.getBuilder().CreateMul(totalSize, dim, "simd_array_size");
    }

    // Use SIMD backend for aligned allocation
    llvm::Value* simdArrayPtr = backend->createAlignedAlloc(context.getBuilder(), elemType, totalSize);
    if (!simdArrayPtr) {
        LOG_DEBUG("SIMD backend failed (likely global context) - falling back to regular allocation");

        // Fallback to regular malloc allocation
        llvm::Value* elementSize = llvm::ConstantInt::get(context.getBuilder().getInt64Ty(),
            context.getModule()->getDataLayout().getTypeAllocSize(elemType));
        llvm::Value* totalBytes = context.getBuilder().CreateMul(totalSize, elementSize, "total_bytes");

        // Use malloc for memory allocation
        llvm::Function* mallocFunc = context.getModule()->getFunction("malloc");
        if (!mallocFunc) {
            LOG_ERROR("malloc function not found");
            return nullptr;
        }

        llvm::Value* rawPtr = context.getBuilder().CreateCall(mallocFunc, {totalBytes}, "simd_array_raw_ptr");
        simdArrayPtr = context.getBuilder().CreateBitCast(rawPtr,
            llvm::PointerType::get(elemType, 0), "simd_array_data_ptr");
    }

    LOG_DEBUG("SIMD array created with alignment: ", backend->getAlignment());
    return simdArrayPtr;
}

// ArrayAccessExprAST implementation
llvm::Value* ArrayAccessExprAST::codeGen(CodeGenContext& context) {
    LOG_DEBUG("Generating array element access");

    // Generate array pointer
    llvm::Value* arrayPtr = array->codeGen(context);
    if (!arrayPtr) {
        LOG_ERROR("Failed to generate array pointer");
        return nullptr;
    }

    // For now, do simple linear indexing (assumes row-major order)
    // Real implementation would need to know the array dimensions

    if (indices.size() == 1) {
        // Single dimension - straightforward
        llvm::Value* index = indices[0]->codeGen(context);
        if (!index) {
            LOG_ERROR("Failed to generate array index");
            return nullptr;
        }

        // Convert to i64 if needed
        if (index->getType() != context.getBuilder().getInt64Ty()) {
            index = convertType(index, context.getBuilder().getInt64Ty(), context, "idx_conv");
        }

        // Create GEP and load
        // For opaque pointers, we need to get element type from type system
        llvm::Type* elemType = nullptr;

        // Try to get the variable name from the array expression
        if (auto* varExpr = dynamic_cast<VariableExprAST*>(array.get())) {
            elemType = context.getArrayElementType(varExpr->getName());
        }

        if (!elemType) {
            // Fallback to f32 if we can't determine the type
            elemType = llvm::Type::getFloatTy(context.getContext());
            LOG_DEBUG("Using fallback f32 element type for array access");
        }
        llvm::Value* elementPtr = context.getBuilder().CreateGEP(
            elemType, arrayPtr, index, "element_ptr");

        llvm::Value* element = context.getBuilder().CreateLoad(
            elemType, elementPtr, "array_element");

        LOG_DEBUG("Single-dimension array element access generated");
        return element;
    } else {
        // Multi-dimensional - for now, just use first index
        // Real implementation would need array metadata for proper indexing
        LOG_DEBUG("Multi-dimensional array access (simplified to first index)");
        llvm::Value* index = indices[0]->codeGen(context);
        if (!index) {
            LOG_ERROR("Failed to generate multi-dimensional array index");
            return nullptr;
        }

        if (index->getType() != context.getBuilder().getInt64Ty()) {
            index = convertType(index, context.getBuilder().getInt64Ty(), context, "multi_idx_conv");
        }

        // For opaque pointers, we need to get element type from type system
        llvm::Type* elemType = nullptr;

        // Try to get the variable name from the array expression
        if (auto* varExpr = dynamic_cast<VariableExprAST*>(array.get())) {
            elemType = context.getArrayElementType(varExpr->getName());
        }

        if (!elemType) {
            // Fallback to f32 if we can't determine the type
            elemType = llvm::Type::getFloatTy(context.getContext());
            LOG_DEBUG("Using fallback f32 element type for multi-dimensional array access");
        }
        llvm::Value* elementPtr = context.getBuilder().CreateGEP(
            elemType, arrayPtr, index, "multi_element_ptr");

        llvm::Value* element = context.getBuilder().CreateLoad(
            elemType, elementPtr, "multi_array_element");

        return element;
    }
}

// ArrayAccessExprAST::hasVectorSlice implementation
bool ArrayAccessExprAST::hasVectorSlice() const {
    for (const auto& index : indices) {
        if (dynamic_cast<const VectorSliceExprAST*>(index.get())) {
            return true;
        }
    }
    return false;
}

// ArrayStoreExprAST implementation
llvm::Value* ArrayStoreExprAST::codeGen(CodeGenContext& context) {
    LOG_DEBUG("Generating array element store");

    // Generate array pointer
    llvm::Value* arrayPtr = array->codeGen(context);
    if (!arrayPtr) {
        LOG_ERROR("Failed to generate array pointer for store");
        return nullptr;
    }

    // Generate value to store
    llvm::Value* storeValue = value->codeGen(context);
    if (!storeValue) {
        LOG_ERROR("Failed to generate value for array store");
        return nullptr;
    }

    // Generate index (simplified to first index for now)
    llvm::Value* index;
    if (indices.size() >= 1) {
        index = indices[0]->codeGen(context);
        if (!index) {
            LOG_ERROR("Failed to generate array store index");
            return nullptr;
        }
        if (index->getType() != context.getBuilder().getInt64Ty()) {
            index = convertType(index, context.getBuilder().getInt64Ty(), context, "store_idx_conv");
        }
    } else {
        LOG_ERROR("Array store requires at least one index");
        return nullptr;
    }

    // Create GEP and store
    // For opaque pointers, we need to get element type from type system
    llvm::Type* elemType = nullptr;

    // Try to get the variable name from the array expression
    if (auto* varExpr = dynamic_cast<VariableExprAST*>(array.get())) {
        elemType = context.getArrayElementType(varExpr->getName());
    }

    if (!elemType) {
        // Fallback to f32 if we can't determine the type
        elemType = llvm::Type::getFloatTy(context.getContext());
        LOG_DEBUG("Using fallback f32 element type for array store");
    }
    llvm::Value* elementPtr = context.getBuilder().CreateGEP(
        elemType, arrayPtr, index, "store_element_ptr");

    // Convert value type if needed
    llvm::Type* targetType = elemType;
    if (storeValue->getType() != targetType) {
        storeValue = convertType(storeValue, targetType, context, "array_store_conv");
    }

    context.getBuilder().CreateStore(storeValue, elementPtr);

    LOG_DEBUG("Array element store generated");
    return storeValue; // Return stored value
}
