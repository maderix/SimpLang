//===- type_inference.hpp - Type Inference Pass ------------------*- C++ -*-===//
//
// Type inference pass that propagates types through the AST
// Handles C++ style type promotion and updates variable types
//
//===----------------------------------------------------------------------===//

#ifndef TYPE_INFERENCE_HPP
#define TYPE_INFERENCE_HPP

#include "ast/ast.hpp"
#include <map>
#include <string>
#include <memory>

/// Type inference context that tracks variable types through program flow
class TypeInferenceContext {
public:
    TypeInferenceContext() = default;

    /// Run type inference on the entire program
    bool inferTypes(BlockAST* program);

    /// Get the inferred type for a variable
    TypeInfo* getVariableType(const std::string& name);

private:
    /// Infer types for a block of statements
    void inferBlock(BlockAST* block);

    /// Infer types for a statement
    void inferStatement(StmtAST* stmt);

    /// Infer and return the type of an expression
    TypeInfo* inferExpression(ExprAST* expr);

    /// Apply C++ usual arithmetic conversions to determine result type
    TypeInfo* promoteTypes(TypeInfo* lhs, TypeInfo* rhs);

    /// Get the wider of two numeric types
    TypeInfo* getWiderType(TypeInfo* t1, TypeInfo* t2);

    /// Type precedence for promotion (higher = wider)
    int getTypePrecedence(TypeInfo* type);

    /// Declare or update a variable's type
    void setVariableType(const std::string& name, TypeInfo* type);

    /// Symbol table: variable name → inferred type
    std::map<std::string, TypeInfo*> variableTypes;

    /// Track whether we're in a loop (affects type widening)
    bool inLoop = false;
};

#endif // TYPE_INFERENCE_HPP
