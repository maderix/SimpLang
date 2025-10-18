#ifndef AST_STMT_RETURN_STMT_HPP
#define AST_STMT_RETURN_STMT_HPP

#include "../base/ast_base.hpp"

class ReturnAST : public StmtAST {
    ExprAST* expression;
public:
    ReturnAST(ExprAST* expr) : expression(expr) {}
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

#endif // AST_STMT_RETURN_STMT_HPP