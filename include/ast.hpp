#ifndef AST_HPP
#define AST_HPP

#include <string>
#include <vector>
#include <memory>
#include <llvm/IR/Value.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

class CodeGenContext;

// Define an enumeration for binary operators
enum BinaryOp {
    OpAdd = '+',  // ASCII value of '+'
    OpSub = '-',  // ASCII value of '-'
    OpMul = '*',  // ASCII value of '*'
    OpDiv = '/',  // ASCII value of '/'
    OpLT  = '<',  // ASCII value of '<'
    OpGT  = '>',  // ASCII value of '>'
    OpAssign = '=', // ASCII value of '='
    OpLE  = 256,   // Start from 256 to avoid conflict with ASCII codes
    OpGE,
    OpEQ,
    OpNE
};

// Base AST node class
class AST {
public:
    virtual ~AST() {}
    virtual llvm::Value* codeGen(CodeGenContext& context) = 0;
};

// Expression class
class ExprAST : public AST {
public:
    virtual ~ExprAST() {}
};

// Number literal expression
class NumberExprAST : public ExprAST {
    double value;
public:
    NumberExprAST(double value) : value(value) {}
    virtual llvm::Value* codeGen(CodeGenContext& context);
};

// Variable expression
class VariableExprAST : public ExprAST {
    std::string name;
public:
    VariableExprAST(const std::string& name) : name(name) {}
    const std::string& getName() const { return name; }
    virtual llvm::Value* codeGen(CodeGenContext& context);
};

// Binary operator expression
class BinaryExprAST : public ExprAST {
    BinaryOp op;
    ExprAST *lhs, *rhs;
public:
    BinaryExprAST(BinaryOp op, ExprAST* lhs, ExprAST* rhs) :
        op(op), lhs(lhs), rhs(rhs) {}
    virtual llvm::Value* codeGen(CodeGenContext& context);
};

// Assignment expression
class AssignmentExprAST : public ExprAST {
    VariableExprAST* lhs;
    ExprAST* rhs;
public:
    AssignmentExprAST(ExprAST* lhs, ExprAST* rhs) :
        lhs(dynamic_cast<VariableExprAST*>(lhs)), rhs(rhs) {}
    virtual llvm::Value* codeGen(CodeGenContext& context);
};

// Function call expression
class CallExprAST : public ExprAST {
    std::string callee;
    std::vector<ExprAST*> arguments;
public:
    CallExprAST(const std::string& callee, const std::vector<ExprAST*>& args) :
        callee(callee), arguments(args) {}
    virtual llvm::Value* codeGen(CodeGenContext& context);
};

// Statement class
class StmtAST : public AST {
public:
    virtual ~StmtAST() {}
};

// Expression statement
class ExpressionStmtAST : public StmtAST {
    ExprAST* expression;
public:
    ExpressionStmtAST(ExprAST* expr) : expression(expr) {}
    virtual llvm::Value* codeGen(CodeGenContext& context);
};

// Block statement
class BlockAST : public StmtAST {
public:
    std::vector<StmtAST*> statements;
    virtual llvm::Value* codeGen(CodeGenContext& context);
};

// Variable declaration
class VariableDeclarationAST : public StmtAST {
    std::string name;
    ExprAST* assignmentExpr;
public:
    VariableDeclarationAST(const std::string& name, ExprAST* expr = nullptr) :
        name(name), assignmentExpr(expr) {}
    const std::string& getName() const { return name; }
    virtual llvm::Value* codeGen(CodeGenContext& context);
};

// Function declaration
class FunctionAST : public StmtAST {
    std::string name;
    std::vector<VariableDeclarationAST*> arguments;
    BlockAST* body;
public:
    FunctionAST(const std::string& name,
                std::vector<VariableDeclarationAST*>* arguments,
                BlockAST* body) :
        name(name), arguments(*arguments), body(body) {}
    virtual llvm::Value* codeGen(CodeGenContext& context);
};

// If statement
class IfAST : public StmtAST {
    ExprAST* condition;
    BlockAST *thenBlock, *elseBlock;
public:
    IfAST(ExprAST* condition, BlockAST* thenBlock, BlockAST* elseBlock = nullptr) :
        condition(condition), thenBlock(thenBlock), elseBlock(elseBlock) {}
    virtual llvm::Value* codeGen(CodeGenContext& context);
};

// While statement
class WhileAST : public StmtAST {
    ExprAST* condition;
    BlockAST* body;
public:
    WhileAST(ExprAST* condition, BlockAST* body) :
        condition(condition), body(body) {}
    virtual llvm::Value* codeGen(CodeGenContext& context);
};

// Return statement
class ReturnAST : public StmtAST {
    ExprAST* expression;
public:
    ReturnAST(ExprAST* expr) : expression(expr) {}
    virtual llvm::Value* codeGen(CodeGenContext& context);
};

#endif // AST_HPP
