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

    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

#endif // AST_STMT_FUNCTION_STMT_HPP