#ifndef AST_STMT_CONTROL_FLOW_STMT_HPP
#define AST_STMT_CONTROL_FLOW_STMT_HPP

#include "../base/ast_base.hpp"

class BlockAST;

class IfAST : public StmtAST {
    ExprAST* condition;
    BlockAST *thenBlock, *elseBlock;
public:
    IfAST(ExprAST* condition, BlockAST* thenBlock, BlockAST* elseBlock = nullptr)
        : condition(condition), thenBlock(thenBlock), elseBlock(elseBlock) {}
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

class WhileAST : public StmtAST {
    ExprAST* condition;
    BlockAST* body;
public:
    WhileAST(ExprAST* condition, BlockAST* body)
        : condition(condition), body(body) {}
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

#endif // AST_STMT_CONTROL_FLOW_STMT_HPP