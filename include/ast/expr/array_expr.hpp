#ifndef AST_EXPR_ARRAY_EXPR_HPP
#define AST_EXPR_ARRAY_EXPR_HPP

#include <vector>
#include <memory>
#include "../base/ast_base.hpp"
#include "../type/type_info.hpp"

// Array creation: array<f32>([10, 20, 30]) or array<f32>([1.0, 2.0, 3.0])
// Supports both dimension specification (integers) and initializer lists (values)
class ArrayCreateExprAST : public ExprAST {
    std::unique_ptr<TypeInfo> elementType;
    std::vector<std::unique_ptr<ExprAST>> dimensionExprs; // Runtime dimensions or initializer values
    bool isInitializerList;  // true if this is an initializer list, false if dimensions

public:
    ArrayCreateExprAST(std::unique_ptr<TypeInfo> elemType,
                      std::vector<std::unique_ptr<ExprAST>> dimensions,
                      bool initList = false)
        : elementType(std::move(elemType)), dimensionExprs(std::move(dimensions)),
          isInitializerList(initList) {}

    virtual llvm::Value* codeGen(CodeGenContext& context) override;
    virtual ASTKind getKind() const override { return ASTKind::ArrayCreateExpr; }

    TypeInfo* getElementType() const { return elementType.get(); }
    size_t getDimensionCount() const { return dimensionExprs.size(); }
    const std::vector<std::unique_ptr<ExprAST>>& getDimensions() const { return dimensionExprs; }
    bool isInitializer() const { return isInitializerList; }
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
    virtual ASTKind getKind() const override { return ASTKind::ArrayCreateExpr; }

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
    virtual ASTKind getKind() const override { return ASTKind::ArrayAccessExpr; }

    bool hasVectorSlice() const;
    ExprAST* getArray() const { return array.get(); }
    const std::vector<std::unique_ptr<ExprAST>>& getIndices() const { return indices; }
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

    // Accessors for MLIR lowering
    ExprAST* getArray() const { return array.get(); }
    const std::vector<std::unique_ptr<ExprAST>>& getIndices() const { return indices; }
    ExprAST* getValue() const { return value.get(); }

    virtual llvm::Value* codeGen(CodeGenContext& context) override;
    virtual ASTKind getKind() const override { return ASTKind::ArrayStoreExpr; }
};

#endif // AST_EXPR_ARRAY_EXPR_HPP