#ifndef AST_STMT_RETURN_STMT_HPP
#define AST_STMT_RETURN_STMT_HPP

#include "../base/ast_base.hpp"

class ReturnAST : public StmtAST {
    ExprAST* expression;
public:
    ReturnAST(ExprAST* expr) : expression(expr) {}
    ExprAST* getExpression() const { return expression; }
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
    virtual ASTKind getKind() const override { return ASTKind::ReturnStmt; }
};

#endif // AST_STMT_RETURN_STMT_HPP