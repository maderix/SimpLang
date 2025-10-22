#include "ast/expr/call_expr.hpp"
#include "../ast_utils.hpp"
#include "logger.hpp"
#include "codegen.hpp"
#include <llvm/IR/Value.h>
#include <llvm/IR/Function.h>

llvm::Value* CallExprAST::codeGen(CodeGenContext& context) {
    llvm::Function* calleeF = context.getModule()->getFunction(callee);
    if (!calleeF) {
        LOG_ERROR("Unknown function: ", callee);
        return nullptr;
    }

    if (calleeF->arg_size() != arguments.size()) {
        LOG_ERROR("Incorrect number of arguments passed");
        return nullptr;
    }

    std::vector<llvm::Value*> argsV;
    for (unsigned i = 0, e = arguments.size(); i != e; ++i) {
        llvm::Value* argVal = arguments[i]->codeGen(context);
        if (!argVal)
            return nullptr;

        // Convert argument to match function parameter type
        llvm::Type* expectedType = calleeF->getFunctionType()->getParamType(i);
        argVal = convertType(argVal, expectedType, context, "argconv");

        argsV.push_back(argVal);
    }

    // Check if function returns void
    if (calleeF->getReturnType()->isVoidTy()) {
        // Create void call without assigning a name
        context.getBuilder().CreateCall(calleeF, argsV);
        return nullptr;
    } else {
        // For non-void functions, create call with name
        return context.getBuilder().CreateCall(calleeF, argsV, callee + "_ret");
    }
}
