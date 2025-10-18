#ifndef AST_EXPR_OPERATOR_EXPR_HPP
#define AST_EXPR_OPERATOR_EXPR_HPP

#include <memory>
#include "../base/ast_base.hpp"

class UnaryExprAST : public ExprAST {
    UnaryOp op_;
    std::unique_ptr<ExprAST> operand_;
public:
    UnaryExprAST(UnaryOp op, std::unique_ptr<ExprAST> operand)
        : op_(op), operand_(std::move(operand)) {}
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

class BinaryExprAST : public ExprAST {
    BinaryOp op_;
    std::unique_ptr<ExprAST> left_;
    std::unique_ptr<ExprAST> right_;

public:
    BinaryExprAST(BinaryOp op, std::unique_ptr<ExprAST> left,
                  std::unique_ptr<ExprAST> right)
        : op_(op), left_(std::move(left)), right_(std::move(right)) {}

    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

#endif // AST_EXPR_OPERATOR_EXPR_HPP