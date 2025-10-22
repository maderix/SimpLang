#include "ast/expr/vector_slice_expr.hpp"
#include "ast/expr/literal_expr.hpp"
#include "logger.hpp"
#include "codegen.hpp"
#include <llvm/IR/Value.h>

// VectorSliceExprAST implementation
llvm::Value* VectorSliceExprAST::codeGen(CodeGenContext& context) {
    // This should not be called directly - handled by ArrayAccessExprAST
    LOG_ERROR("VectorSliceExprAST::codeGen called directly");
    return nullptr;
}

int VectorSliceExprAST::getSliceWidth(CodeGenContext& context) const {
    // Try to evaluate start and end as constants
    auto* startConst = dynamic_cast<const NumberExprAST*>(start.get());
    auto* endConst = dynamic_cast<const NumberExprAST*>(end.get());

    if (startConst && endConst) {
        return static_cast<int>(endConst->getValue()) - static_cast<int>(startConst->getValue());
    }

    // Default to AVX-512 width for f32 (16 elements)
    return 16;
}
