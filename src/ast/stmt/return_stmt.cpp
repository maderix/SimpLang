#include "ast/stmt/return_stmt.hpp"
#include "../ast_utils.hpp"
#include "logger.hpp"
#include "codegen.hpp"
#include <llvm/IR/Value.h>
#include <llvm/IR/Constants.h>

llvm::Value* ReturnAST::codeGen(CodeGenContext& context) {
    LOG_TRACE("Generating return statement");

    // Set debug location for return statement
    if (getLine() > 0) {
        context.setCurrentDebugLocation(getLine());
    }

    llvm::Value* returnValue = nullptr;
    if (expression) {
        returnValue = expression->codeGen(context);
        if (!returnValue) {
            return nullptr;
        }
    } else {
        returnValue = llvm::ConstantFP::get(context.getContext(), llvm::APFloat(0.0));
    }

    // Get the current function's return type and convert if needed
    llvm::Function* currentFunction = context.getBuilder().GetInsertBlock()->getParent();
    llvm::Type* expectedReturnType = currentFunction->getReturnType();
    returnValue = convertType(returnValue, expectedReturnType, context, "retconv");

    // Only create return instruction if we don't already have a terminator
    if (!context.getBuilder().GetInsertBlock()->getTerminator()) {
        context.getBuilder().CreateRet(returnValue);
    }

    return returnValue;
}
