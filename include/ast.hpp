#ifndef AST_HPP
#define AST_HPP

#include <string>
#include <vector>
#include <memory>
#include <llvm/IR/Value.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include "slice_type.hpp"
#include "simd_backend.hpp"

class CodeGenContext;

// Type system for static typing
enum class TypeKind {
    Dynamic,    // Current 'var' behavior
    F32, F64,   // Floating point
    I8, I16, I32, I64,   // Signed integers
    U8, U16, U32, U64,   // Unsigned integers  
    Bool,       // Boolean
    Void,       // Function returns
    Array       // Multi-dimensional array types
};

class TypeInfo {
public:
    TypeKind kind;
    
    TypeInfo(TypeKind k) : kind(k) {}
    virtual ~TypeInfo() = default;
    
    bool isStaticallyTyped() const { return kind != TypeKind::Dynamic; }
    bool isInteger() const { 
        return kind >= TypeKind::I8 && kind <= TypeKind::U64; 
    }
    bool isFloat() const { 
        return kind == TypeKind::F32 || kind == TypeKind::F64; 
    }
    bool isArray() const {
        return kind == TypeKind::Array;
    }
    bool isSigned() const {
        return kind >= TypeKind::I8 && kind <= TypeKind::I64;
    }
    bool isUnsigned() const {
        return kind >= TypeKind::U8 && kind <= TypeKind::U64;
    }
    
    llvm::Type* getLLVMType(llvm::LLVMContext& ctx) const;
    std::string toString() const;
};

class ArrayTypeInfo : public TypeInfo {
public:
    std::unique_ptr<TypeInfo> elementType;
    int size; // -1 for dynamic size
    std::vector<int> dimensions; // For multi-dim arrays
    
    // SIMD Extensions
    SIMDType simdHint;
    int alignment;        // 16=SSE, 32=AVX, 64=AVX512
    bool vectorizable;    // Can use vector operations
    
    ArrayTypeInfo(std::unique_ptr<TypeInfo> elemType, int sz = -1, SIMDType simd = SIMDType::None) 
        : TypeInfo(TypeKind::Array), 
          elementType(std::move(elemType)), 
          size(sz),
          simdHint(simd) {
        
        vectorizable = (simd != SIMDType::None) && 
                      (elementType->isFloat() || elementType->isInteger());
        alignment = getSIMDAlignment(simd);
    }
    
private:
    int getSIMDAlignment(SIMDType simd) const {
        switch (simd) {
            case SIMDType::AVX512: return 64;
            case SIMDType::AVX:    return 32;
            case SIMDType::SSE:    return 16;
            case SIMDType::NEON:   return 16;
            default:               return 8;  // Regular alignment
        }
    }
};

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
    std::unique_ptr<TypeInfo> staticType;  // NEW: Optional static type
    unsigned lineNo;
    bool isGlobal;
    std::string typeName;  // For debug info

public:
    // Constructor for static typing
    VariableDeclarationAST(const std::string& name, 
                          ExprAST* expr = nullptr,
                          std::unique_ptr<TypeInfo> type = nullptr,
                          SliceTypeAST* slice = nullptr,
                          unsigned line = 0,
                          bool global = false)
        : name(name), assignmentExpr(expr), sliceType(slice), 
          staticType(std::move(type)), lineNo(line), isGlobal(global), 
          typeName(staticType ? staticType->toString() : "double") {}

    // Legacy constructor for backward compatibility  
    VariableDeclarationAST(const std::string& name, 
                          ExprAST* expr,
                          SliceTypeAST* slice,
                          unsigned line,
                          bool global,
                          const std::string& type)
        : name(name), assignmentExpr(expr), sliceType(slice), 
          staticType(nullptr), lineNo(line), isGlobal(global), typeName(type) {}
          
    virtual ~VariableDeclarationAST() {
        delete assignmentExpr;
        delete sliceType;
    }

    const std::string& getName() const { return name; }
    bool isSlice() const { return sliceType != nullptr; }
    SliceType getSliceType() const { 
        return sliceType ? sliceType->getType() : SliceType::SSE_SLICE; 
    }
    unsigned getLine() const { return lineNo; }
    bool isGlobalVariable() const { return isGlobal; }
    const std::string& getTypeName() const { return typeName; }
    ExprAST* getAssignmentExpr() const { return assignmentExpr; }
    
    // New methods for static typing
    bool isStaticallyTyped() const { return staticType && staticType->isStaticallyTyped(); }
    TypeKind getTypeKind() const { return staticType ? staticType->kind : TypeKind::Dynamic; }
    const TypeInfo* getStaticType() const { return staticType.get(); }
    
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
    std::unique_ptr<TypeInfo> returnType;  // NEW: Optional static return type
public:
    FunctionAST(const std::string& name,
                std::vector<VariableDeclarationAST*>* arguments,
                BlockAST* body,
                std::unique_ptr<TypeInfo> retType = nullptr)
        : name(name), arguments(*arguments), body(body), returnType(std::move(retType)) {}
    
    bool hasStaticReturnType() const { return returnType && returnType->isStaticallyTyped(); }
    TypeKind getReturnTypeKind() const { return returnType ? returnType->kind : TypeKind::Dynamic; }
    const TypeInfo* getReturnType() const { return returnType.get(); }
    
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

class IncludeStmtAST : public StmtAST {
    std::string filename;
public:
    IncludeStmtAST(const std::string& file) : filename(file) {}
    const std::string& getFilename() const { return filename; }
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

class VectorCreationExprAST : public ExprAST {
    std::vector<std::unique_ptr<ExprAST>> elements_;
    bool isAVX_;
    
public:
    VectorCreationExprAST(std::vector<std::unique_ptr<ExprAST>> elements, bool isAVX)
        : elements_(std::move(elements)), isAVX_(isAVX) {
        // Validate vector size at construction
        size_t expected = isAVX ? 8 : 2;
        if (elements_.size() != expected) {
            std::string msg = "Vector size mismatch. Got " + 
                            std::to_string(elements_.size()) + 
                            " elements but expected " + 
                            std::to_string(expected);
            throw std::runtime_error(msg);
        }
    }
        
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

// ================== Array Operations ==================

// Array creation: array<f32>([10, 20, 30])
class ArrayCreateExprAST : public ExprAST {
    std::unique_ptr<TypeInfo> elementType;
    std::vector<std::unique_ptr<ExprAST>> dimensionExprs; // Runtime dimensions
    
public:
    ArrayCreateExprAST(std::unique_ptr<TypeInfo> elemType, 
                      std::vector<std::unique_ptr<ExprAST>> dimensions)
        : elementType(std::move(elemType)), dimensionExprs(std::move(dimensions)) {}
        
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
    
    TypeInfo* getElementType() const { return elementType.get(); }
    size_t getDimensionCount() const { return dimensionExprs.size(); }
};

class SIMDArrayCreateExprAST : public ExprAST {
    std::unique_ptr<TypeInfo> elementType;
    SIMDType simdHint;
    std::vector<std::unique_ptr<ExprAST>> dimensionExprs;
    
public:
    SIMDArrayCreateExprAST(std::unique_ptr<TypeInfo> elemType, 
                          SIMDType simd,
                          std::vector<std::unique_ptr<ExprAST>> dimensions)
        : elementType(std::move(elemType)), simdHint(simd), dimensionExprs(std::move(dimensions)) {}
        
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
    
    TypeInfo* getElementType() const { return elementType.get(); }
    SIMDType getSIMDHint() const { return simdHint; }
    size_t getDimensionCount() const { return dimensionExprs.size(); }
};

// Multi-dimensional array access: arr[i, j, k]
class ArrayAccessExprAST : public ExprAST {
    std::unique_ptr<ExprAST> array;
    std::vector<std::unique_ptr<ExprAST>> indices;
    
public:
    ArrayAccessExprAST(std::unique_ptr<ExprAST> arrayExpr,
                      std::vector<std::unique_ptr<ExprAST>> idxExprs)
        : array(std::move(arrayExpr)), indices(std::move(idxExprs)) {}
        
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
    
    bool hasVectorSlice() const;
};

class VectorSliceExprAST : public ExprAST {
    std::unique_ptr<ExprAST> start;
    std::unique_ptr<ExprAST> end;
    
public:
    VectorSliceExprAST(std::unique_ptr<ExprAST> startExpr, std::unique_ptr<ExprAST> endExpr)
        : start(std::move(startExpr)), end(std::move(endExpr)) {}
        
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
    
    ExprAST* getStart() const { return start.get(); }
    ExprAST* getEnd() const { return end.get(); }
    int getSliceWidth(CodeGenContext& context) const;
};

// Array element assignment: arr[i, j, k] = value
class ArrayStoreExprAST : public ExprAST {
    std::unique_ptr<ExprAST> array;
    std::vector<std::unique_ptr<ExprAST>> indices;
    std::unique_ptr<ExprAST> value;
    
public:
    ArrayStoreExprAST(std::unique_ptr<ExprAST> arrayExpr,
                     std::vector<std::unique_ptr<ExprAST>> idxExprs,
                     std::unique_ptr<ExprAST> val)
        : array(std::move(arrayExpr)), indices(std::move(idxExprs)), value(std::move(val)) {}
        
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

#endif // AST_HPP