#include "ast/expr/literal_expr.hpp"
#include "logger.hpp"
#include "codegen.hpp"
#include <llvm/IR/Value.h>
#include <llvm/IR/Constants.h>
#include <climits>

llvm::Value* NumberExprAST::codeGen(CodeGenContext& context) {
    // Debug output
    LOG_TRACE("Generating number: ", value, (isInteger ? " (integer)" : " (double)"));

    if (isInteger) {
        auto intVal = static_cast<int64_t>(value);

        // Use i32 for small integers (common in loops), i64 for larger ones
        if (intVal >= INT32_MIN && intVal <= INT32_MAX) {
            LOG_TRACE("Creating i32 constant: ", intVal);
            auto* type = llvm::Type::getInt32Ty(context.getContext());
            return llvm::ConstantInt::get(type, static_cast<int32_t>(intVal), true);
        } else {
            LOG_TRACE("Creating i64 constant: ", intVal);
            auto* type = llvm::Type::getInt64Ty(context.getContext());
            return llvm::ConstantInt::get(type, intVal, true);
        }
    }

    // Float literal (for better vectorization performance)
    LOG_TRACE("Creating float constant: ", value);
    return llvm::ConstantFP::get(llvm::Type::getFloatTy(context.getContext()), static_cast<float>(value));
}