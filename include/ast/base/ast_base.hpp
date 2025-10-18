#ifndef AST_BASE_AST_BASE_HPP
#define AST_BASE_AST_BASE_HPP

#include <llvm/IR/Value.h>

class CodeGenContext;

// Base AST classes
class AST {
public:
    virtual ~AST() {}
    virtual llvm::Value* codeGen(CodeGenContext& context) = 0;
};

class ExprAST : public AST {
public:
    virtual ~ExprAST() {}
};

class StmtAST : public AST {
public:
    virtual ~StmtAST() {}
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