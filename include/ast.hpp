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

enum UnaryOp {
    OpNeg = '-'  // Unary minus
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
    OpMod = '%'  // Add modulo operator
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

class NumberExprAST : public ExprAST {
    double value;
    bool isInteger;
public:
    NumberExprAST(double value, bool isInt = false) 
        : value(value), isInteger(isInt) {}
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
    bool isIntegerLiteral() const { return isInteger; }
    double getValue() const { return value; }
};

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


// Add this new AST class:
class UnaryExprAST : public ExprAST {
    UnaryOp op_;
    std::unique_ptr<ExprAST> operand_;
public:
    UnaryExprAST(UnaryOp op, std::unique_ptr<ExprAST> operand) 
        : op_(op), operand_(std::move(operand)) {}
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

class BinaryExprAST : public ExprAST {
    BinaryOp op_;
    std::unique_ptr<ExprAST> left_;
    std::unique_ptr<ExprAST> right_;
    
public:
    BinaryExprAST(BinaryOp op, std::unique_ptr<ExprAST> left,
                  std::unique_ptr<ExprAST> right)
        : op_(op), left_(std::move(left)), right_(std::move(right)) {}
        
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

class ExpressionStmtAST : public StmtAST {
    ExprAST* expression;
public:
    ExpressionStmtAST(ExprAST* expr) : expression(expr) {}
    ExprAST* getExpression() { return expression; }  // Add this
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
    SliceType type;  // This holds the actual SliceType enum value
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
    std::string slice_name_;
    std::unique_ptr<ExprAST> index_;
    std::unique_ptr<ExprAST> value_;
public:
    SliceStoreExprAST(const std::string& name, ExprAST* idx, std::unique_ptr<ExprAST> val)
        : slice_name_(name), index_(idx), value_(std::move(val)) {}
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

class VariableDeclarationAST : public StmtAST {
    std::string name;
    ExprAST* assignmentExpr;
    SliceTypeAST* sliceType;
    unsigned lineNo;
    bool isGlobal;
    std::string typeName;  // For debug info

public:
    VariableDeclarationAST(const std::string& name, 
                          ExprAST* expr = nullptr,
                          SliceTypeAST* slice = nullptr,
                          unsigned line = 0,
                          bool global = false,
                          const std::string& type = "double")
        : name(name), assignmentExpr(expr), sliceType(slice), 
          lineNo(line), isGlobal(global), typeName(type) {}
          
    virtual ~VariableDeclarationAST() {
        delete assignmentExpr;
        delete sliceType;
    }

    const std::string& getName() const { return name; }
    bool isSlice() const { return sliceType != nullptr; }
    SliceType getSliceType() const { return sliceType ? sliceType->getType() : SSE_SLICE; }
    unsigned getLine() const { return lineNo; }
    bool isGlobalVariable() const { return isGlobal; }
    const std::string& getTypeName() const { return typeName; }
    
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

// Control flow and function definitions
class BlockAST : public StmtAST {
public:
    std::vector<StmtAST*> statements;
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

class VectorCreationExprAST : public ExprAST {
    std::vector<std::unique_ptr<ExprAST>> elements_;
    bool isAVX_;
    
public:
    VectorCreationExprAST(std::vector<std::unique_ptr<ExprAST>> elements, bool isAVX)
        : elements_(std::move(elements)), isAVX_(isAVX) {}
        
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

#endif // AST_HPP