#ifndef AST_EXPR_LITERAL_EXPR_HPP
#define AST_EXPR_LITERAL_EXPR_HPP

#include "../base/ast_base.hpp"

class NumberExprAST : public ExprAST {
    double value;
    bool isInteger;
public:
    NumberExprAST(double value, bool isInt = false)
        : value(value), isInteger(isInt) {}
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
    virtual ASTKind getKind() const override { return ASTKind::NumberExpr; }
    bool isIntegerLiteral() const { return isInteger; }
    double getValue() const { return value; }
};

#endif // AST_EXPR_LITERAL_EXPR_HPP