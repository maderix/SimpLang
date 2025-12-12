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
    MatMulExpr,
    SSEExpr,
    AVXExpr,
    SSESliceExpr,
    AVXSliceExpr,
    SliceGetExpr,
    SliceSetExpr,
    VectorSliceExpr,
    CastExpr,

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

// Source location tracking for debug info
struct SourceLocation {
    unsigned line = 0;
    unsigned column = 0;

    SourceLocation() = default;
    SourceLocation(unsigned l, unsigned c = 0) : line(l), column(c) {}
};

// Base AST classes
class AST {
protected:
    SourceLocation loc;
public:
    virtual ~AST() {}
    virtual llvm::Value* codeGen(CodeGenContext& context) = 0;
    virtual ASTKind getKind() const = 0;

    // Source location accessors
    void setLocation(unsigned line, unsigned col = 0) { loc = SourceLocation(line, col); }
    void setLocation(const SourceLocation& l) { loc = l; }
    unsigned getLine() const { return loc.line; }
    unsigned getColumn() const { return loc.column; }
    const SourceLocation& getLocation() const { return loc; }
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
    OpMod = '%',  // Modulo operator
    OpAnd = 260,  // Bitwise AND
    OpOr  = 261,  // Bitwise OR
    OpXor = 262,  // Bitwise XOR
    OpLShift = 263,  // Left shift
    OpRShift = 264   // Right shift
};

// Unary operators
enum UnaryOp {
    OpNeg = '-'  // Unary minus
};

#endif // AST_BASE_AST_BASE_HPP