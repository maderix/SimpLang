#ifndef AST_STMT_DECLARATION_STMT_HPP
#define AST_STMT_DECLARATION_STMT_HPP

#include <string>
#include <memory>
#include "../base/ast_base.hpp"
#include "../type/type_info.hpp"
#include "../expr/slice_expr.hpp"

class VariableDeclarationAST : public StmtAST {
    std::string name;
    ExprAST* assignmentExpr;
    SliceTypeAST* sliceType;
    std::unique_ptr<TypeInfo> staticType;  // Optional static type
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
          staticType(std::move(type)), isGlobal(global),
          typeName(staticType ? staticType->toString() : "double") {
        setLocation(line);  // Use base class location tracking
    }

    // Legacy constructor for backward compatibility
    VariableDeclarationAST(const std::string& name,
                          ExprAST* expr,
                          SliceTypeAST* slice,
                          unsigned line,
                          bool global,
                          const std::string& type)
        : name(name), assignmentExpr(expr), sliceType(slice),
          staticType(nullptr), isGlobal(global), typeName(type) {
        setLocation(line);  // Use base class location tracking
    }

    virtual ~VariableDeclarationAST() {
        delete assignmentExpr;
        delete sliceType;
    }

    const std::string& getName() const { return name; }
    bool isSlice() const { return sliceType != nullptr; }
    SliceType getSliceType() const {
        return sliceType ? sliceType->getType() : SliceType::SSE_SLICE;
    }
    // getLine() is now inherited from AST base class
    bool isGlobalVariable() const { return isGlobal; }
    const std::string& getTypeName() const { return typeName; }
    ExprAST* getAssignmentExpr() const { return assignmentExpr; }

    // Methods for static typing
    bool isStaticallyTyped() const { return staticType && staticType->isStaticallyTyped(); }
    TypeKind getTypeKind() const { return staticType ? staticType->kind : TypeKind::Dynamic; }
    const TypeInfo* getStaticType() const { return staticType.get(); }

    virtual llvm::Value* codeGen(CodeGenContext& context) override;
    virtual ASTKind getKind() const override { return ASTKind::VariableDecl; }
};

#endif // AST_STMT_DECLARATION_STMT_HPP