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

    // Accessors for MLIR lowering
    UnaryOp getOp() const { return op_; }
    ExprAST* getOperand() const { return operand_.get(); }

    virtual llvm::Value* codeGen(CodeGenContext& context) override;
    virtual ASTKind getKind() const override { return ASTKind::UnaryExpr; }
};

class BinaryExprAST : public ExprAST {
    BinaryOp op_;
    std::unique_ptr<ExprAST> left_;
    std::unique_ptr<ExprAST> right_;

public:
    BinaryExprAST(BinaryOp op, std::unique_ptr<ExprAST> left,
                  std::unique_ptr<ExprAST> right)
        : op_(op), left_(std::move(left)), right_(std::move(right)) {}

    // Accessors for MLIR lowering
    BinaryOp getOp() const { return op_; }
    ExprAST* getLeft() const { return left_.get(); }
    ExprAST* getRight() const { return right_.get(); }

    virtual llvm::Value* codeGen(CodeGenContext& context) override;
    virtual ASTKind getKind() const override { return ASTKind::BinaryExpr; }
};

#endif // AST_EXPR_OPERATOR_EXPR_HPP