#ifndef AST_EXPR_VECTOR_SLICE_EXPR_HPP
#define AST_EXPR_VECTOR_SLICE_EXPR_HPP

#include <memory>
#include "../base/ast_base.hpp"

class CodeGenContext;

class VectorSliceExprAST : public ExprAST {
    std::unique_ptr<ExprAST> start;
    std::unique_ptr<ExprAST> end;

public:
    VectorSliceExprAST(std::unique_ptr<ExprAST> startExpr, std::unique_ptr<ExprAST> endExpr)
        : start(std::move(startExpr)), end(std::move(endExpr)) {}

    virtual llvm::Value* codeGen(CodeGenContext& context) override;

    ExprAST* getStart() const { return start.get(); }
    ExprAST* getEnd() const { return end.get(); }
    int getSliceWidth(CodeGenContext& context) const;
};

#endif // AST_EXPR_VECTOR_SLICE_EXPR_HPP