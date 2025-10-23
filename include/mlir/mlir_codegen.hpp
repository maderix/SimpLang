//===- mlir_codegen.hpp - MLIR Code Generation from AST ---------*- C++ -*-===//
//
// Part of the SimpLang Project
//
// This file defines the MLIRCodeGenContext class for lowering SimpLang AST
// to MLIR Simp dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CODEGEN_HPP
#define MLIR_CODEGEN_HPP

#include "mlir/simp_dialect.hpp"
#include "mlir/simp_types.hpp"
#include "mlir/simp_ops.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // For FuncOp in MLIR 14
#include <map>
#include <set>
#include <string>
#include <vector>
#include <memory>

// Forward declarations - avoid including full AST to prevent exception issues
class BlockAST;
class StmtAST;
class ExprAST;
class NumberExprAST;
class VariableExprAST;
class AssignmentExprAST;
class BinaryExprAST;
class UnaryExprAST;
class ArrayCreateExprAST;
class ArrayAccessExprAST;
class ArrayStoreExprAST;
class MatMulExprAST;
class CallExprAST;
class VariableDeclarationAST;
class ReturnAST;
class IfAST;
class WhileAST;
class ExpressionStmtAST;
class FunctionAST;
class TypeInfo;

namespace mlir {
namespace simp {

//===----------------------------------------------------------------------===//
// MLIRCodeGenContext
//===----------------------------------------------------------------------===//

/// Context for lowering SimpLang AST to MLIR Simp dialect.
/// Manages MLIR builder infrastructure, symbol tables, and scope management.
class MLIRCodeGenContext {
public:
  explicit MLIRCodeGenContext(const std::string& moduleName);
  ~MLIRCodeGenContext() = default;

  // Disable copy/move
  MLIRCodeGenContext(const MLIRCodeGenContext&) = delete;
  MLIRCodeGenContext& operator=(const MLIRCodeGenContext&) = delete;

  //===--------------------------------------------------------------------===//
  // Core Infrastructure
  //===--------------------------------------------------------------------===//

  /// Get the MLIR context
  mlir::MLIRContext* getMLIRContext() { return &mlirContext; }

  /// Get the OpBuilder
  mlir::OpBuilder& getBuilder() { return builder; }

  /// Get the module
  mlir::ModuleOp getModule() { return module; }

  //===--------------------------------------------------------------------===//
  // Symbol Table Management
  //===--------------------------------------------------------------------===//

  /// Declare a new variable in the current scope
  void declareVariable(const std::string& name, mlir::Value value);

  /// Lookup a variable in the symbol table (searches all scopes)
  mlir::Value lookupVariable(const std::string& name);

  /// Check if a variable exists in the current scope
  bool variableExists(const std::string& name);

  /// Push a new scope (for entering blocks, functions, etc.)
  void pushScope();

  /// Pop the current scope
  void popScope();

  //===--------------------------------------------------------------------===//
  // Function Management
  //===--------------------------------------------------------------------===//

  /// Set the current function being lowered
  void setCurrentFunction(mlir::FuncOp func) { currentFunction = func; }

  /// Get the current function
  mlir::FuncOp getCurrentFunction() { return currentFunction; }

  //===--------------------------------------------------------------------===//
  // AST Lowering Entry Points
  //===--------------------------------------------------------------------===//

  /// Lower a complete AST to MLIR module
  mlir::ModuleOp lowerAST(BlockAST* programBlock);

  /// Lower a function definition
  mlir::FuncOp lowerFunction(FunctionAST* funcAst);

  /// Lower a statement
  mlir::LogicalResult lowerStatement(StmtAST* stmt);

  /// Lower an expression
  mlir::Value lowerExpression(ExprAST* expr);

  //===--------------------------------------------------------------------===//
  // Expression Lowering
  //===--------------------------------------------------------------------===//

  /// Lower a literal expression to simp.constant
  mlir::Value lowerLiteral(NumberExprAST* literal);

  /// Lower a variable reference
  mlir::Value lowerVariable(VariableExprAST* varExpr);

  /// Lower an assignment expression (updates symbol table)
  mlir::Value lowerAssignment(AssignmentExprAST* assignment);

  /// Lower a binary operation (add, sub, mul, div)
  mlir::Value lowerBinaryOp(BinaryExprAST* binOp);

  /// Lower a unary operation (neg)
  mlir::Value lowerUnaryOp(UnaryExprAST* unaryOp);

  /// Lower an array creation expression
  mlir::Value lowerArrayCreate(ArrayCreateExprAST* arrayCreate);

  /// Lower an array access expression (get or set)
  mlir::Value lowerArrayAccess(ArrayAccessExprAST* arrayAccess,
                                mlir::Value newValue = nullptr);

  /// Lower an array store expression (arr[idx] = val)
  mlir::Value lowerArrayStore(ArrayStoreExprAST* arrayStore);

  /// Lower a matrix multiplication expression
  mlir::Value lowerMatMul(MatMulExprAST* matmul);

  /// Lower a function call
  mlir::Value lowerCall(CallExprAST* call);

  //===--------------------------------------------------------------------===//
  // Statement Lowering
  //===--------------------------------------------------------------------===//

  /// Lower a variable declaration
  mlir::LogicalResult lowerDeclaration(VariableDeclarationAST* decl);

  /// Lower a return statement
  mlir::LogicalResult lowerReturn(ReturnAST* ret);

  /// Lower an if statement
  mlir::LogicalResult lowerIf(IfAST* ifStmt);

  /// Lower a while loop
  mlir::LogicalResult lowerWhile(WhileAST* whileLoop);

  /// Lower an expression statement
  mlir::LogicalResult lowerExpressionStmt(ExpressionStmtAST* exprStmt);

  //===--------------------------------------------------------------------===//
  // Control Flow Helpers
  //===--------------------------------------------------------------------===//

  /// Track variables modified in a scope (for if/while value passing)
  /// Only tracks modifications to variables that already exist in the symbol table
  std::set<std::string> trackModifiedVariables(BlockAST* block,
                                                 const std::set<std::string>& existingVars);

  /// Get all currently declared variable names
  std::set<std::string> getCurrentVariableNames() const;

  /// Collect current values of variables (for yielding from if/while)
  std::vector<mlir::Value> collectVariableValues(const std::set<std::string>& varNames);

  /// Update symbol table with yielded values from control flow
  void updateSymbolTableWithResults(const std::set<std::string>& varNames,
                                     mlir::ValueRange results);

  //===--------------------------------------------------------------------===//
  // Type Conversion Helpers
  //===--------------------------------------------------------------------===//

  /// Convert a SimpLang type to an MLIR type
  mlir::Type convertType(const std::string& simpType);

  /// Get MLIR type from AST type info
  mlir::Type getMLIRType(TypeInfo* typeInfo);

  /// Get the element type from an array/tensor type string
  mlir::Type getElementType(const std::string& typeStr);

  //===--------------------------------------------------------------------===//
  // Location Management
  //===--------------------------------------------------------------------===//

  /// Get a location for debugging (file:line:col)
  mlir::Location getLocation(int line, int col);

  /// Get an unknown location (fallback)
  mlir::Location getUnknownLocation() { return builder.getUnknownLoc(); }

private:
  //===--------------------------------------------------------------------===//
  // Private Members
  //===--------------------------------------------------------------------===//

  /// MLIR context (owns all MLIR objects)
  mlir::MLIRContext mlirContext;

  /// OpBuilder for creating MLIR operations
  mlir::OpBuilder builder;

  /// The MLIR module being constructed
  mlir::ModuleOp module;

  /// Symbol table: maps variable names to MLIR SSA values
  /// Uses vector of maps for nested scopes
  std::vector<std::map<std::string, mlir::Value>> symbolTable;

  /// Dimension tracking: maps variable names to their runtime dimension values
  /// Used for computing flattened indices in multi-dimensional array access
  /// Using variable names instead of SSA values since values change through assignments
  std::map<std::string, llvm::SmallVector<mlir::Value, 4>> arrayDimensions;

  /// Current function being lowered
  mlir::FuncOp currentFunction;

  /// Source file name for debug locations
  std::string sourceFileName;

  //===--------------------------------------------------------------------===//
  // Private Helpers
  //===--------------------------------------------------------------------===//

  /// Initialize the MLIR context and load required dialects
  void initializeMLIRContext();

  /// Create the MLIR module
  void createModule(const std::string& moduleName);

  /// Get or create a function declaration
  mlir::FuncOp getOrCreateFunction(const std::string& name,
                                   mlir::FunctionType funcType);
};

} // namespace simp
} // namespace mlir

#endif // MLIR_CODEGEN_HPP
