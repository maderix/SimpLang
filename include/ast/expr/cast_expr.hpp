#ifndef AST_EXPR_CAST_EXPR_HPP
#define AST_EXPR_CAST_EXPR_HPP

#include <memory>
#include "../base/ast_base.hpp"
#include "../type/type_info.hpp"

// Type cast expression: expr as type
class CastExprAST : public ExprAST {
    std::unique_ptr<ExprAST> expr_;
    std::unique_ptr<TypeInfo> targetType_;

public:
    CastExprAST(std::unique_ptr<ExprAST> expr, std::unique_ptr<TypeInfo> targetType)
        : expr_(std::move(expr)), targetType_(std::move(targetType)) {}

    // Accessors
    ExprAST* getExpr() const { return expr_.get(); }
    TypeInfo* getTargetType() const { return targetType_.get(); }

    virtual llvm::Value* codeGen(CodeGenContext& context) override;
    virtual ASTKind getKind() const override { return ASTKind::CastExpr; }
};

#endif // AST_EXPR_CAST_EXPR_HPP
