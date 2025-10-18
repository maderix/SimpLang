#include "logger.hpp"
#include "codegen.hpp"
#include <llvm/IR/Value.h>

// Generic type conversion utility
llvm::Value* convertType(llvm::Value* value, llvm::Type* targetType,
                                CodeGenContext& context, const std::string& name) {
    if (!value || !targetType) return nullptr;

    llvm::Type* sourceType = value->getType();
    if (sourceType == targetType) return value; // No conversion needed

    auto& builder = context.getBuilder();

    // Both are integers
    if (sourceType->isIntegerTy() && targetType->isIntegerTy()) {
        unsigned sourceBits = sourceType->getIntegerBitWidth();
        unsigned targetBits = targetType->getIntegerBitWidth();

        if (targetBits > sourceBits) {
            // Sign extend to larger integer
            return builder.CreateSExt(value, targetType, name);
        } else if (targetBits < sourceBits) {
            // Truncate to smaller integer
            return builder.CreateTrunc(value, targetType, name);
        }
        return value; // Same size
    }

    // Both are floating point
    if (sourceType->isFloatingPointTy() && targetType->isFloatingPointTy()) {
        if (targetType->isDoubleTy() && sourceType->isFloatTy()) {
            // Float to double
            return builder.CreateFPExt(value, targetType, name);
        } else if (targetType->isFloatTy() && sourceType->isDoubleTy()) {
            // Double to float
            return builder.CreateFPTrunc(value, targetType, name);
        }
        return value; // Same type
    }

    // Integer to floating point
    if (sourceType->isIntegerTy() && targetType->isFloatingPointTy()) {
        return builder.CreateSIToFP(value, targetType, name);
    }

    // Floating point to integer
    if (sourceType->isFloatingPointTy() && targetType->isIntegerTy()) {
        return builder.CreateFPToSI(value, targetType, name);
    }

    // Pointer types - no conversion
    if (sourceType->isPointerTy() && targetType->isPointerTy()) {
        // Could add pointer cast if needed
        return value;
    }

    // Default: return unchanged if we can't convert
    LOG_WARNING("Unable to convert between types");
    return value;
}