#ifndef AST_EXPR_VARIABLE_EXPR_HPP
#define AST_EXPR_VARIABLE_EXPR_HPP

#include <string>
#include <memory>
#include "../base/ast_base.hpp"

class VariableExprAST : public ExprAST {
    std::string name;
    bool isWrite;  // Tracks if this is a write access
    unsigned lineNo;  // Source line number for debugging

public:
    VariableExprAST(const std::string& name, bool write = false, unsigned line = 0)
        : name(name), isWrite(write), lineNo(line) {}

    const std::string& getName() const { return name; }
    bool isWriteAccess() const { return isWrite; }
    unsigned getLine() const { return lineNo; }

    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

class AssignmentExprAST : public ExprAST {
    VariableExprAST* lhs_;  // Keep as raw pointer since we don't own it
    std::unique_ptr<ExprAST> rhs_;
public:
    AssignmentExprAST(VariableExprAST* lhs, std::unique_ptr<ExprAST> rhs)
        : lhs_(lhs), rhs_(std::move(rhs)) {}
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

#endif // AST_EXPR_VARIABLE_EXPR_HPP