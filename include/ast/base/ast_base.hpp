#ifndef AST_BASE_AST_BASE_HPP
#define AST_BASE_AST_BASE_HPP

#include <llvm/IR/Value.h>

class CodeGenContext;

// AST Kind enumeration for type identification without RTTI
enum class ASTKind {
    // Expression kinds
    NumberExpr,
    VariableExpr,
    AssignmentExpr,
    BinaryExpr,
    UnaryExpr,
    CallExpr,
    ArrayCreateExpr,
    ArrayAccessExpr,
    ArrayStoreExpr,
    SSEExpr,
    AVXExpr,
    SSESliceExpr,
    AVXSliceExpr,
    SliceGetExpr,
    SliceSetExpr,
    VectorSliceExpr,

    // Statement kinds
    VariableDecl,
    FunctionDecl,
    ReturnStmt,
    ExpressionStmt,
    BlockStmt,
    IfStmt,
    WhileStmt,
    IncludeStmt
};

// Base AST classes
class AST {
public:
    virtual ~AST() {}
    virtual llvm::Value* codeGen(CodeGenContext& context) = 0;
    virtual ASTKind getKind() const = 0;
};

class ExprAST : public AST {
public:
    virtual ~ExprAST() {}
    virtual ASTKind getKind() const = 0;
};

class StmtAST : public AST {
public:
    virtual ~StmtAST() {}
    virtual ASTKind getKind() const = 0;
};

// Binary operators
enum BinaryOp {
    OpAdd = '+',
    OpSub = '-',
    OpMul = '*',
    OpDiv = '/',
    OpLT  = '<',
    OpGT  = '>',
    OpLE  = 256,
    OpGE  = 257,
    OpEQ  = 258,
    OpNE  = 259,
    OpMod = '%'  // Modulo operator
};

// Unary operators
enum UnaryOp {
    OpNeg = '-'  // Unary minus
};

#endif // AST_BASE_AST_BASE_HPP