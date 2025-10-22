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
    ExprAST* getCondition() const { return condition; }
    BlockAST* getThenBlock() const { return thenBlock; }
    BlockAST* getElseBlock() const { return elseBlock; }
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
    virtual ASTKind getKind() const override { return ASTKind::IfStmt; }
};

class WhileAST : public StmtAST {
    ExprAST* condition;
    BlockAST* body;
public:
    WhileAST(ExprAST* condition, BlockAST* body)
        : condition(condition), body(body) {}
    ExprAST* getCondition() const { return condition; }
    BlockAST* getBody() const { return body; }
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
    virtual ASTKind getKind() const override { return ASTKind::WhileStmt; }
};

#endif // AST_STMT_CONTROL_FLOW_STMT_HPP