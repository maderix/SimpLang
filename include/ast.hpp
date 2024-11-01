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

// Binary operators
enum BinaryOp {
    OpAdd = '+',
    OpSub = '-',
    OpMul = '*',
    OpDiv = '/',
    OpLT  = '<',
    OpGT  = '>',
    OpAssign = '=',
    OpLE  = 256,
    OpGE,
    OpEQ,
    OpNE
};

// SIMD Slice types
enum SliceType {
    SSE_SLICE,  // 4 x double
    AVX_SLICE   // 8 x double
};

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

// Basic expressions
class NumberExprAST : public ExprAST {
    double value;
public:
    NumberExprAST(double value) : value(value) {}
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

class VariableExprAST : public ExprAST {
    std::string name;
public:
    VariableExprAST(const std::string& name) : name(name) {}
    const std::string& getName() const { return name; }
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

class BinaryExprAST : public ExprAST {
    BinaryOp op;
    ExprAST *lhs, *rhs;
public:
    BinaryExprAST(BinaryOp op, ExprAST* lhs, ExprAST* rhs) 
        : op(op), lhs(lhs), rhs(rhs) {}
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

class ExpressionStmtAST : public StmtAST {
    ExprAST* expression;
public:
    ExpressionStmtAST(ExprAST* expr) : expression(expr) {}
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

// SIMD expressions
class SIMDTypeExprAST : public ExprAST {
    std::vector<ExprAST*> elements;
    bool isAVX;
public:
    SIMDTypeExprAST(const std::vector<ExprAST*>& elems, bool avx = false) 
        : elements(elems), isAVX(avx) {}
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

class SIMDIntrinsicExprAST : public ExprAST {
    std::string intrinsic;
    std::vector<ExprAST*> args;
public:
    SIMDIntrinsicExprAST(const std::string& name, std::vector<ExprAST*>& arguments)
        : intrinsic(name), args(arguments) {}
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

// Slice-related classes
class SliceTypeAST : public ExprAST {
    SliceType type;
public:
    SliceTypeAST(SliceType t) : type(t) {}
    virtual ~SliceTypeAST() {}
    
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
    SliceType getType() const { return type; }
};

class SliceExprAST : public ExprAST {
    SliceType type;
    ExprAST* length;
    ExprAST* capacity;  // Optional
public:
    SliceExprAST(SliceType t, ExprAST* len, ExprAST* cap = nullptr)
        : type(t), length(len), capacity(cap) {}
    
    virtual ~SliceExprAST() {
        delete length;
        if (capacity) delete capacity;
    }
    
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
    
    SliceType getType() const { return type; }
    ExprAST* getLength() const { return length; }
    ExprAST* getCapacity() const { return capacity; }
};


class SliceAccessExprAST : public ExprAST {
    std::string slice_name;
    ExprAST* index;
public:
    SliceAccessExprAST(const std::string& name, ExprAST* idx)
        : slice_name(name), index(idx) {}
        
    virtual ~SliceAccessExprAST() {
        delete index;
    }
    
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
    
    const std::string& getName() const { return slice_name; }
    ExprAST* getIndex() const { return index; }
};

class SliceStoreExprAST : public ExprAST {
    std::string slice_name;
    ExprAST* index;
    ExprAST* value;
public:
    SliceStoreExprAST(const std::string& name, ExprAST* idx, ExprAST* val)
        : slice_name(name), index(idx), value(val) {}
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

// Control flow and function definitions
class BlockAST : public StmtAST {
public:
    std::vector<StmtAST*> statements;
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

class VariableDeclarationAST : public StmtAST {
    std::string name;
    ExprAST* assignmentExpr;
    SliceTypeAST* sliceType;
public:
    VariableDeclarationAST(const std::string& name, 
                          ExprAST* expr = nullptr,
                          SliceTypeAST* slice = nullptr)
        : name(name), assignmentExpr(expr), sliceType(slice) {}
    const std::string& getName() const { return name; }
    bool isSlice() const { return sliceType != nullptr; }
    SliceType getSliceType() const { return sliceType ? sliceType->getType() : SSE_SLICE; }
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

class AssignmentExprAST : public ExprAST {
    VariableExprAST* lhs;
    ExprAST* rhs;
public:
    AssignmentExprAST(ExprAST* lhs, ExprAST* rhs)
        : lhs(dynamic_cast<VariableExprAST*>(lhs)), rhs(rhs) {}
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

class FunctionAST : public StmtAST {
    std::string name;
    std::vector<VariableDeclarationAST*> arguments;
    BlockAST* body;
public:
    FunctionAST(const std::string& name,
                std::vector<VariableDeclarationAST*>* arguments,
                BlockAST* body)
        : name(name), arguments(*arguments), body(body) {}
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

class CallExprAST : public ExprAST {
    std::string callee;
    std::vector<ExprAST*> arguments;
public:
    CallExprAST(const std::string& callee, const std::vector<ExprAST*>& args)
        : callee(callee), arguments(args) {}
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

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

class ReturnAST : public StmtAST {
    ExprAST* expression;
public:
    ReturnAST(ExprAST* expr) : expression(expr) {}
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

#endif // AST_HPP