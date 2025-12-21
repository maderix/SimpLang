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

class VariableDeclarationAST;
class VariableExprAST;

class ForAST : public StmtAST {
    VariableDeclarationAST* init;    // var i = 0i
    ExprAST* condition;               // i < N
    VariableExprAST* updateVar;       // i (left side of update)
    ExprAST* updateExpr;              // i + 1i (right side of update)
    BlockAST* body;
public:
    ForAST(VariableDeclarationAST* init, ExprAST* condition,
           VariableExprAST* updateVar, ExprAST* updateExpr, BlockAST* body)
        : init(init), condition(condition), updateVar(updateVar),
          updateExpr(updateExpr), body(body) {}
    VariableDeclarationAST* getInit() const { return init; }
    ExprAST* getCondition() const { return condition; }
    VariableExprAST* getUpdateVar() const { return updateVar; }
    ExprAST* getUpdateExpr() const { return updateExpr; }
    BlockAST* getBody() const { return body; }
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
    virtual ASTKind getKind() const override { return ASTKind::ForStmt; }
};

#endif // AST_STMT_CONTROL_FLOW_STMT_HPP