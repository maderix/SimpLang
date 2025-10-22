#ifndef AST_BASE_AST_VISITOR_HPP
#define AST_BASE_AST_VISITOR_HPP

// Forward declarations for all AST node types
class NumberExprAST;
class VariableExprAST;
class UnaryExprAST;
class BinaryExprAST;
class AssignmentExprAST;
class CallExprAST;
class SliceTypeAST;
class SliceExprAST;
class SliceAccessExprAST;
class SliceStoreExprAST;
class SIMDTypeExprAST;
class SIMDIntrinsicExprAST;
class VectorCreationExprAST;
class ArrayCreateExprAST;
class SIMDArrayCreateExprAST;
class ArrayAccessExprAST;
class ArrayStoreExprAST;
class VectorSliceExprAST;
class ExpressionStmtAST;
class VariableDeclarationAST;
class BlockAST;
class FunctionAST;
class IfAST;
class WhileAST;
class ReturnAST;
class IncludeStmtAST;

// Placeholder for future visitor pattern implementation
// Currently, the codebase uses virtual codeGen() method for traversal
// This file is created for future enhancements when a visitor pattern is needed

/*
// Example visitor interface (not currently used)
class ASTVisitor {
public:
    virtual ~ASTVisitor() = default;

    // Expression visitors
    virtual void visit(NumberExprAST& node) = 0;
    virtual void visit(VariableExprAST& node) = 0;
    virtual void visit(UnaryExprAST& node) = 0;
    virtual void visit(BinaryExprAST& node) = 0;
    // ... etc for all node types
};
*/

#endif // AST_BASE_AST_VISITOR_HPP