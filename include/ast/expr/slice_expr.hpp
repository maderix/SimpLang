#ifndef AST_EXPR_SLICE_EXPR_HPP
#define AST_EXPR_SLICE_EXPR_HPP

#include <string>
#include <memory>
#include "../base/ast_base.hpp"
#include "../../slice_type.hpp"

class SliceTypeAST : public ExprAST {
    SliceType type;
public:
    SliceTypeAST(SliceType t) : type(t) {}
    virtual ~SliceTypeAST() {}

    virtual llvm::Value* codeGen(CodeGenContext& context) override;
    virtual ASTKind getKind() const override {
        return type == SliceType::SSE_SLICE ? ASTKind::SSESliceExpr : ASTKind::AVXSliceExpr;
    }
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
    virtual ASTKind getKind() const override {
        return type == SliceType::SSE_SLICE ? ASTKind::SSESliceExpr : ASTKind::AVXSliceExpr;
    }

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
    virtual ASTKind getKind() const override { return ASTKind::SliceGetExpr; }

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
    virtual ASTKind getKind() const override { return ASTKind::SliceSetExpr; }
};

#endif // AST_EXPR_SLICE_EXPR_HPP