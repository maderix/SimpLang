#ifndef AST_STMT_EXPRESSION_STMT_HPP
#define AST_STMT_EXPRESSION_STMT_HPP

#include "../base/ast_base.hpp"

class ExpressionStmtAST : public StmtAST {
    ExprAST* expression;
public:
    ExpressionStmtAST(ExprAST* expr) : expression(expr) {}
    ExprAST* getExpression() { return expression; }
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
    virtual ASTKind getKind() const override { return ASTKind::ExpressionStmt; }
};

#endif // AST_STMT_EXPRESSION_STMT_HPP