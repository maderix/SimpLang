#include "ast/expr/operator_expr.hpp"
#include "../ast_utils.hpp"
#include "logger.hpp"
#include "codegen.hpp"
#include <llvm/IR/Value.h>

// UnaryExprAST implementation
llvm::Value* UnaryExprAST::codeGen(CodeGenContext& context) {
    llvm::Value* operandV = operand_->codeGen(context);
    if (!operandV) return nullptr;

    switch (op_) {
        case OpNeg:
            // Use integer negation for integer types, float negation for float types
            if (operandV->getType()->isIntegerTy()) {
                return context.getBuilder().CreateNeg(operandV, "negtmp");
            } else {
                return context.getBuilder().CreateFNeg(operandV, "negtmp");
            }
        default:
            LOG_ERROR("Invalid unary operator");
            return nullptr;
    }
}

// BinaryExprAST implementation
llvm::Value* BinaryExprAST::codeGen(CodeGenContext& context) {
    llvm::Value* lhs = left_->codeGen(context);
    llvm::Value* rhs = right_->codeGen(context);
    if (!lhs || !rhs) return nullptr;

    // Check operand types
    bool lhsIsInt = lhs->getType()->isIntegerTy();
    bool rhsIsInt = rhs->getType()->isIntegerTy();

    // Handle type conversions
    if (lhsIsInt && rhsIsInt) {
        // Both integers - ensure same integer type
        if (lhs->getType() != rhs->getType()) {
            // Different integer types - promote to larger type
            llvm::Type* targetType;
            if (lhs->getType()->getIntegerBitWidth() > rhs->getType()->getIntegerBitWidth()) {
                targetType = lhs->getType();
                rhs = context.getBuilder().CreateSExtOrTrunc(rhs, targetType, "conv");
            } else {
                targetType = rhs->getType();
                lhs = context.getBuilder().CreateSExtOrTrunc(lhs, targetType, "conv");
            }
        }
    } else if (lhsIsInt && !rhsIsInt) {
        // LHS is integer, RHS is float - convert LHS to float
        lhs = context.getBuilder().CreateSIToFP(lhs, rhs->getType(), "intToFloat");
        lhsIsInt = false;  // Now both are floats
    } else if (!lhsIsInt && rhsIsInt) {
        // LHS is float, RHS is integer - convert RHS to float
        rhs = context.getBuilder().CreateSIToFP(rhs, lhs->getType(), "intToFloat");
        rhsIsInt = false;  // Now both are floats
    } else {
        // Both floating point - ensure same floating point type
        if (lhs->getType() != rhs->getType()) {
            // Promote to float for mixed float operations (better for vectorization)
            llvm::Type* floatType = llvm::Type::getFloatTy(context.getContext());
            if (lhs->getType()->isDoubleTy()) {
                lhs = context.getBuilder().CreateFPTrunc(lhs, floatType, "conv");
            }
            if (rhs->getType()->isDoubleTy()) {
                rhs = context.getBuilder().CreateFPTrunc(rhs, floatType, "conv");
            }
        }
    }

    // Generate operation based on final types (after conversion)
    bool useIntegerOps = lhs->getType()->isIntegerTy() && rhs->getType()->isIntegerTy();

    switch (op_) {
        case BinaryOp::OpAdd:
            return useIntegerOps ?
                context.getBuilder().CreateAdd(lhs, rhs, "addtmp") :
                context.getBuilder().CreateFAdd(lhs, rhs, "addtmp");
        case BinaryOp::OpSub:
            return useIntegerOps ?
                context.getBuilder().CreateSub(lhs, rhs, "subtmp") :
                context.getBuilder().CreateFSub(lhs, rhs, "subtmp");
        case BinaryOp::OpMul:
            return useIntegerOps ?
                context.getBuilder().CreateMul(lhs, rhs, "multmp") :
                context.getBuilder().CreateFMul(lhs, rhs, "multmp");
        case BinaryOp::OpDiv:
            return useIntegerOps ?
                context.getBuilder().CreateSDiv(lhs, rhs, "divtmp") :
                context.getBuilder().CreateFDiv(lhs, rhs, "divtmp");
        case BinaryOp::OpMod:
            return useIntegerOps ?
                context.getBuilder().CreateSRem(lhs, rhs, "modtmp") :
                context.getBuilder().CreateFRem(lhs, rhs, "modtmp");
        case BinaryOp::OpLT:
            return useIntegerOps ?
                context.getBuilder().CreateICmpSLT(lhs, rhs, "cmptmp") :
                context.getBuilder().CreateFCmpULT(lhs, rhs, "cmptmp");
        case BinaryOp::OpGT:
            return useIntegerOps ?
                context.getBuilder().CreateICmpSGT(lhs, rhs, "cmptmp") :
                context.getBuilder().CreateFCmpUGT(lhs, rhs, "cmptmp");
        case BinaryOp::OpLE:
            return useIntegerOps ?
                context.getBuilder().CreateICmpSLE(lhs, rhs, "cmptmp") :
                context.getBuilder().CreateFCmpULE(lhs, rhs, "cmptmp");
        case BinaryOp::OpGE:
            return useIntegerOps ?
                context.getBuilder().CreateICmpSGE(lhs, rhs, "cmptmp") :
                context.getBuilder().CreateFCmpUGE(lhs, rhs, "cmptmp");
        case BinaryOp::OpEQ:
            return useIntegerOps ?
                context.getBuilder().CreateICmpEQ(lhs, rhs, "cmptmp") :
                context.getBuilder().CreateFCmpUEQ(lhs, rhs, "cmptmp");
        case BinaryOp::OpNE:
            return useIntegerOps ?
                context.getBuilder().CreateICmpNE(lhs, rhs, "cmptmp") :
                context.getBuilder().CreateFCmpUNE(lhs, rhs, "cmptmp");
        default:
            LOG_ERROR("Invalid binary operator");
            return nullptr;
    }
}
