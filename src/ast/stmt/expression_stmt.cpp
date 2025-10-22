#include "ast/stmt/expression_stmt.hpp"
#include "ast/expr/slice_expr.hpp"
#include "ast/expr/call_expr.hpp"
#include "logger.hpp"
#include "codegen.hpp"
#include <llvm/IR/Value.h>

llvm::Value* ExpressionStmtAST::codeGen(CodeGenContext& context) {
    LOG_TRACE("Generating expression statement...");

    if (!expression) {
        LOG_ERROR("Null expression");
        return nullptr;
    }

    llvm::Value* exprVal = expression->codeGen(context);

    // For void expressions (like slice store), nullptr is expected
    if (!exprVal && (
        dynamic_cast<SliceStoreExprAST*>(expression) ||
        dynamic_cast<CallExprAST*>(expression)  // Add this for void function calls
    )) {
        LOG_TRACE("Void expression completed successfully");
        return llvm::ConstantInt::get(context.getBuilder().getInt32Ty(), 0);
    }

    if (!exprVal) {
        LOG_ERROR("Expression generation failed");
        return nullptr;
    }

    return exprVal;
}
