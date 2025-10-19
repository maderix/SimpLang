#ifndef AST_EXPR_SIMD_EXPR_HPP
#define AST_EXPR_SIMD_EXPR_HPP

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include "../base/ast_base.hpp"

class SIMDTypeExprAST : public ExprAST {
    std::vector<ExprAST*> elements;
    bool isAVX;
public:
    SIMDTypeExprAST(const std::vector<ExprAST*>& elems, bool avx = false)
        : elements(elems), isAVX(avx) {}
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
    virtual ASTKind getKind() const override {
        return isAVX ? ASTKind::AVXExpr : ASTKind::SSEExpr;
    }
};

class SIMDIntrinsicExprAST : public ExprAST {
    std::string intrinsic;
    std::vector<ExprAST*> args;
public:
    SIMDIntrinsicExprAST(const std::string& name, std::vector<ExprAST*>& arguments)
        : intrinsic(name), args(arguments) {}
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
    virtual ASTKind getKind() const override {
        // Using SSEExpr as a generic SIMD intrinsic kind
        return ASTKind::SSEExpr;
    }
};

class VectorCreationExprAST : public ExprAST {
    std::vector<std::unique_ptr<ExprAST>> elements_;
    bool isAVX_;

public:
    VectorCreationExprAST(std::vector<std::unique_ptr<ExprAST>> elements, bool isAVX)
        : elements_(std::move(elements)), isAVX_(isAVX) {
        // Validate vector size at construction
        // When USE_MLIR is defined, MLIR code is compiled with -fno-exceptions
        // so skip the validation here (will be done at codegen time)
#ifndef USE_MLIR
        size_t expected = isAVX ? 8 : 2;
        if (elements_.size() != expected) {
            std::string msg = "Vector size mismatch. Got " +
                            std::to_string(elements_.size()) +
                            " elements but expected " +
                            std::to_string(expected);
            throw std::runtime_error(msg);
        }
#endif
    }

    virtual llvm::Value* codeGen(CodeGenContext& context) override;
    virtual ASTKind getKind() const override {
        return isAVX_ ? ASTKind::AVXExpr : ASTKind::SSEExpr;
    }
};

#endif // AST_EXPR_SIMD_EXPR_HPP