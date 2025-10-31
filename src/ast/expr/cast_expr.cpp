#include "ast/expr/cast_expr.hpp"
#include "../ast_utils.hpp"
#include "logger.hpp"
#include "codegen.hpp"
#include <llvm/IR/Value.h>

llvm::Value* CastExprAST::codeGen(CodeGenContext& context) {
    // Lower the expression being cast
    llvm::Value* value = expr_->codeGen(context);
    if (!value) {
        LOG_ERROR("Failed to generate code for cast expression");
        return nullptr;
    }

    // Get target LLVM type
    llvm::Type* targetType = targetType_->getLLVMType(context.getContext());
    llvm::Type* sourceType = value->getType();

    // If types are already the same, no cast needed
    if (sourceType == targetType) {
        return value;
    }

    // Handle conversions
    bool sourceIsInt = sourceType->isIntegerTy();
    bool targetIsInt = targetType->isIntegerTy();
    bool sourceIsFloat = sourceType->isFloatingPointTy();
    bool targetIsFloat = targetType->isFloatingPointTy();

    // int -> float
    if (sourceIsInt && targetIsFloat) {
        return context.getBuilder().CreateSIToFP(value, targetType, "intToFloat");
    }

    // float -> int
    if (sourceIsFloat && targetIsInt) {
        return context.getBuilder().CreateFPToSI(value, targetType, "floatToInt");
    }

    // float -> float (different precision)
    if (sourceIsFloat && targetIsFloat) {
        if (sourceType->getPrimitiveSizeInBits() < targetType->getPrimitiveSizeInBits()) {
            // Extend
            return context.getBuilder().CreateFPExt(value, targetType, "fpext");
        } else {
            // Truncate
            return context.getBuilder().CreateFPTrunc(value, targetType, "fptrunc");
        }
    }

    // int -> int (different width)
    if (sourceIsInt && targetIsInt) {
        return context.getBuilder().CreateSExtOrTrunc(value, targetType, "intcast");
    }

    LOG_ERROR("Unsupported cast operation");
    return nullptr;
}
