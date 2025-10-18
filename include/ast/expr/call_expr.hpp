#ifndef AST_EXPR_CALL_EXPR_HPP
#define AST_EXPR_CALL_EXPR_HPP

#include <string>
#include <vector>
#include "../base/ast_base.hpp"

class CallExprAST : public ExprAST {
    std::string callee;
    std::vector<ExprAST*> arguments;
public:
    CallExprAST(const std::string& callee, const std::vector<ExprAST*>& args)
        : callee(callee), arguments(args) {}
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

#endif // AST_EXPR_CALL_EXPR_HPP