#ifndef AST_STMT_FUNCTION_STMT_HPP
#define AST_STMT_FUNCTION_STMT_HPP

#include <string>
#include <vector>
#include <memory>
#include "../base/ast_base.hpp"
#include "../type/type_info.hpp"

class VariableDeclarationAST;
class BlockAST;

class FunctionAST : public StmtAST {
    std::string name;
    std::vector<VariableDeclarationAST*> arguments;
    BlockAST* body;
    std::unique_ptr<TypeInfo> returnType;  // Optional static return type
public:
    FunctionAST(const std::string& name,
                std::vector<VariableDeclarationAST*>* arguments,
                BlockAST* body,
                std::unique_ptr<TypeInfo> retType = nullptr)
        : name(name), arguments(*arguments), body(body), returnType(std::move(retType)) {}

    bool hasStaticReturnType() const { return returnType && returnType->isStaticallyTyped(); }
    TypeKind getReturnTypeKind() const { return returnType ? returnType->kind : TypeKind::Dynamic; }
    const TypeInfo* getReturnType() const { return returnType.get(); }

    // Accessors for MLIR lowering
    const std::string& getName() const { return name; }
    const std::vector<VariableDeclarationAST*>& getArguments() const { return arguments; }
    BlockAST* getBody() const { return body; }

    virtual llvm::Value* codeGen(CodeGenContext& context) override;
    virtual ASTKind getKind() const override { return ASTKind::FunctionDecl; }
};

#endif // AST_STMT_FUNCTION_STMT_HPP