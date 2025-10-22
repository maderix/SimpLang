//===- mlir_codegen.cpp - MLIR Code Generation Implementation ------------===//
//
// Part of the SimpLang Project
//
// This file implements the MLIRCodeGenContext class for lowering SimpLang AST
// to MLIR Simp dialect operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/mlir_codegen.hpp"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>

// Include full AST definitions here in the .cpp file
// This way the header doesn't pull in exception-throwing code
#include "ast/ast.hpp"

using namespace mlir;
using namespace mlir::simp;

//===----------------------------------------------------------------------===//
// Constructor & Initialization
//===----------------------------------------------------------------------===//

MLIRCodeGenContext::MLIRCodeGenContext(const std::string& moduleName)
    : builder(&mlirContext), sourceFileName(moduleName) {
  initializeMLIRContext();
  createModule(moduleName);

  // Push global scope
  pushScope();
}

void MLIRCodeGenContext::initializeMLIRContext() {
  // Load required dialects
  mlirContext.getOrLoadDialect<mlir::simp::SimpDialect>();
  mlirContext.getOrLoadDialect<mlir::StandardOpsDialect>();
  mlirContext.getOrLoadDialect<mlir::scf::SCFDialect>();
  mlirContext.getOrLoadDialect<mlir::arith::ArithmeticDialect>();
}

void MLIRCodeGenContext::createModule(const std::string& moduleName) {
  // Create the MLIR module
  // ModuleOp::create expects Optional<StringRef>
  module = mlir::ModuleOp::create(builder.getUnknownLoc(), llvm::StringRef(moduleName));
}

//===----------------------------------------------------------------------===//
// Symbol Table Management
//===----------------------------------------------------------------------===//

void MLIRCodeGenContext::declareVariable(const std::string& name, mlir::Value value) {
  if (symbolTable.empty()) {
    llvm::errs() << "Error: No scope to declare variable '" << name << "'\n";
    return;
  }

  // Add to current scope (top of stack)
  symbolTable.back()[name] = value;
}

mlir::Value MLIRCodeGenContext::lookupVariable(const std::string& name) {
  // Search from innermost to outermost scope
  for (auto it = symbolTable.rbegin(); it != symbolTable.rend(); ++it) {
    auto found = it->find(name);
    if (found != it->end()) {
      return found->second;
    }
  }

  // Not found
  llvm::errs() << "Error: Undefined variable '" << name << "'\n";
  return nullptr;
}

bool MLIRCodeGenContext::variableExists(const std::string& name) {
  if (symbolTable.empty()) {
    return false;
  }

  // Check current scope only
  return symbolTable.back().find(name) != symbolTable.back().end();
}

void MLIRCodeGenContext::pushScope() {
  symbolTable.push_back(std::map<std::string, mlir::Value>());
}

void MLIRCodeGenContext::popScope() {
  if (!symbolTable.empty()) {
    symbolTable.pop_back();
  }
}

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

mlir::Type MLIRCodeGenContext::convertType(const std::string& simpType) {
  // Handle basic types
  if (simpType == "double" || simpType == "f64") {
    return builder.getF64Type();
  } else if (simpType == "float" || simpType == "f32") {
    return builder.getF32Type();
  } else if (simpType == "int" || simpType == "i64") {
    return builder.getI64Type();
  } else if (simpType == "i32") {
    return builder.getI32Type();
  } else if (simpType == "bool" || simpType == "i1") {
    return builder.getI1Type();
  }

  // Handle array types: "array<T>"
  if (simpType.find("array<") == 0) {
    // Extract element type from "array<f64>"
    size_t start = simpType.find('<') + 1;
    size_t end = simpType.find('>');
    std::string elemTypeStr = simpType.substr(start, end - start);
    mlir::Type elemType = convertType(elemTypeStr);

    return mlir::simp::ArrayType::get(&mlirContext, elemType);
  }

  // Handle tensor types: "tensor<10x20xf64>" or "tensor<?x?xf32>"
  if (simpType.find("tensor<") == 0) {
    // Parse shape and element type
    // For now, return a placeholder - full parsing in future
    return mlir::simp::SimpTensorType::get(&mlirContext, {-1}, builder.getF32Type());
  }

  // Default to f32 (matches existing SimpLang compiler)
  llvm::errs() << "Warning: Unknown type '" << simpType << "', defaulting to f32\n";
  return builder.getF32Type();
}

mlir::Type MLIRCodeGenContext::getMLIRType(TypeInfo* typeInfo) {
  if (!typeInfo) {
    return builder.getF32Type(); // Default (matches existing compiler)
  }

  // Use the toString() method from TypeInfo
  return convertType(typeInfo->toString());
}

mlir::Type MLIRCodeGenContext::getElementType(const std::string& typeStr) {
  // Extract element type from array/tensor string
  if (typeStr.find('<') != std::string::npos) {
    size_t start = typeStr.rfind('<') + 1;
    size_t end = typeStr.rfind('>');
    if (end != std::string::npos && end > start) {
      std::string elemStr = typeStr.substr(start, end - start);

      // Remove any shape information (e.g., "10x20xf64" -> "f64")
      size_t lastX = elemStr.rfind('x');
      if (lastX != std::string::npos) {
        elemStr = elemStr.substr(lastX + 1);
      }

      return convertType(elemStr);
    }
  }

  return builder.getF32Type();
}

//===----------------------------------------------------------------------===//
// Location Management
//===----------------------------------------------------------------------===//

mlir::Location MLIRCodeGenContext::getLocation(int line, int col) {
  return mlir::FileLineColLoc::get(&mlirContext, sourceFileName, line, col);
}

//===----------------------------------------------------------------------===//
// AST Lowering - Main Entry Point
//===----------------------------------------------------------------------===//

mlir::ModuleOp MLIRCodeGenContext::lowerAST(BlockAST* programBlock) {
  if (!programBlock) {
    llvm::errs() << "Error: Null program block\n";
    return nullptr;
  }

  // Set insertion point to module body
  builder.setInsertionPointToEnd(module.getBody());

  // Lower all statements in the program block
  // Note: BlockAST::statements is a vector of raw pointers
  for (auto* stmt : programBlock->statements) {
    // Use getKind() to identify statement type
    if (stmt->getKind() == ASTKind::FunctionDecl) {
      FunctionAST* func = static_cast<FunctionAST*>(stmt);
      if (!lowerFunction(func)) {
        llvm::errs() << "Error: Failed to lower function\n";
        return nullptr;
      }
    } else {
      // Handle other top-level statements
      if (failed(lowerStatement(stmt))) {
        llvm::errs() << "Error: Failed to lower statement\n";
        return nullptr;
      }
    }
  }

  // Verify the module
  if (failed(mlir::verify(module))) {
    llvm::errs() << "Error: Module verification failed\n";
    module.dump();
    return nullptr;
  }

  return module;
}

//===----------------------------------------------------------------------===//
// Expression Lowering
//===----------------------------------------------------------------------===//

mlir::Value MLIRCodeGenContext::lowerExpression(ExprAST* expr) {
  if (!expr) {
    llvm::errs() << "Error: Null expression\n";
    return nullptr;
  }

  // Use getKind() for type identification without RTTI
  switch (expr->getKind()) {
    case ASTKind::NumberExpr:
      return lowerLiteral(static_cast<NumberExprAST*>(expr));

    case ASTKind::VariableExpr:
      return lowerVariable(static_cast<VariableExprAST*>(expr));

    case ASTKind::AssignmentExpr:
      return lowerAssignment(static_cast<AssignmentExprAST*>(expr));

    case ASTKind::BinaryExpr:
      return lowerBinaryOp(static_cast<BinaryExprAST*>(expr));

    case ASTKind::UnaryExpr:
      return lowerUnaryOp(static_cast<UnaryExprAST*>(expr));

    case ASTKind::CallExpr:
      return lowerCall(static_cast<CallExprAST*>(expr));

    case ASTKind::ArrayCreateExpr:
      return lowerArrayCreate(static_cast<ArrayCreateExprAST*>(expr));

    case ASTKind::ArrayAccessExpr:
      return lowerArrayAccess(static_cast<ArrayAccessExprAST*>(expr));

    case ASTKind::ArrayStoreExpr:
      return lowerArrayStore(static_cast<ArrayStoreExprAST*>(expr));

    case ASTKind::MatMulExpr:
      return lowerMatMul(static_cast<MatMulExprAST*>(expr));

    default:
      llvm::errs() << "Error: Unsupported expression kind " << static_cast<int>(expr->getKind())
                   << " in MLIR lowering\n";
      return nullptr;
  }
}

mlir::Value MLIRCodeGenContext::lowerLiteral(NumberExprAST* literal) {
  auto loc = getUnknownLocation();

  // Determine type based on the literal value using accessor
  double val = literal->getValue();
  mlir::Type type = builder.getF32Type(); // Default float type (matches existing compiler)
  mlir::Attribute value;

  // Check if it's an integer or floating-point
  if (val == std::floor(val) && literal->isIntegerLiteral()) {
    // Integer value
    type = builder.getI64Type();
    value = builder.getI64IntegerAttr(static_cast<int64_t>(val));
  } else {
    // Floating-point value - use f32 to match existing compiler
    type = builder.getF32Type();
    value = builder.getF32FloatAttr(static_cast<float>(val));
  }

  // Create simp.constant operation
  return builder.create<mlir::simp::ConstantOp>(loc, type, value);
}

mlir::Value MLIRCodeGenContext::lowerVariable(VariableExprAST* varExpr) {
  return lookupVariable(varExpr->getName());
}

mlir::Value MLIRCodeGenContext::lowerAssignment(AssignmentExprAST* assignment) {
  // In SSA form, assignment just creates a new SSA value
  // and updates the symbol table

  // Lower the RHS expression
  mlir::Value value = lowerExpression(assignment->getRHS());
  if (!value) {
    llvm::errs() << "Error: Failed to lower assignment RHS\n";
    return nullptr;
  }

  // Get the variable name from LHS
  const std::string& varName = assignment->getLHS()->getName();

  // Update the symbol table with the new value
  // In SSA, this creates a new binding for the variable name
  declareVariable(varName, value);

  // Return the assigned value
  return value;
}

mlir::Value MLIRCodeGenContext::lowerBinaryOp(BinaryExprAST* binOp) {
  auto loc = getUnknownLocation();

  // Lower left and right operands using accessors
  mlir::Value lhs = lowerExpression(binOp->getLeft());
  mlir::Value rhs = lowerExpression(binOp->getRight());

  if (!lhs || !rhs) {
    llvm::errs() << "Error: Failed to lower binary operation operands\n";
    return nullptr;
  }

  // Result type is the same as operand type (assuming both operands have same type)
  mlir::Type resultType = lhs.getType();

  // Create the appropriate operation based on the operator
  // Note: MLIR operations need result type as first argument
  switch (binOp->getOp()) {
    case OpAdd:
      return builder.create<mlir::simp::AddOp>(loc, resultType, lhs, rhs);
    case OpSub:
      return builder.create<mlir::simp::SubOp>(loc, resultType, lhs, rhs);
    case OpMul:
      return builder.create<mlir::simp::MulOp>(loc, resultType, lhs, rhs);
    case OpDiv:
      return builder.create<mlir::simp::DivOp>(loc, resultType, lhs, rhs);

    // Comparison operations - use arith dialect
    // Result type is always i1 (boolean) for comparisons
    // Need to check operand type: use CmpIOp for integers, CmpFOp for floats
    case OpLT:
      if (lhs.getType().isa<mlir::IntegerType>()) {
        return builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::slt, lhs, rhs);
      } else {
        return builder.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OLT, lhs, rhs);
      }
    case OpGT:
      if (lhs.getType().isa<mlir::IntegerType>()) {
        return builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::sgt, lhs, rhs);
      } else {
        return builder.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OGT, lhs, rhs);
      }
    case OpLE:
      if (lhs.getType().isa<mlir::IntegerType>()) {
        return builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::sle, lhs, rhs);
      } else {
        return builder.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OLE, lhs, rhs);
      }
    case OpGE:
      if (lhs.getType().isa<mlir::IntegerType>()) {
        return builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::sge, lhs, rhs);
      } else {
        return builder.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OGE, lhs, rhs);
      }
    case OpEQ:
      if (lhs.getType().isa<mlir::IntegerType>()) {
        return builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::eq, lhs, rhs);
      } else {
        return builder.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OEQ, lhs, rhs);
      }
    case OpNE:
      if (lhs.getType().isa<mlir::IntegerType>()) {
        return builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::ne, lhs, rhs);
      } else {
        return builder.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::ONE, lhs, rhs);
      }

    default:
      llvm::errs() << "Error: Unsupported binary operator: " << binOp->getOp() << "\n";
      return nullptr;
  }
}

mlir::Value MLIRCodeGenContext::lowerUnaryOp(UnaryExprAST* unaryOp) {
  auto loc = getUnknownLocation();

  // Lower the operand
  mlir::Value operand = lowerExpression(unaryOp->getOperand());
  if (!operand) {
    llvm::errs() << "Error: Failed to lower unary operand\n";
    return nullptr;
  }

  // Get result type (same as operand type for negation)
  mlir::Type resultType = operand.getType();

  // Handle the unary operation
  switch (unaryOp->getOp()) {
    case OpNeg:
      return builder.create<mlir::simp::NegOp>(loc, resultType, operand);

    default:
      llvm::errs() << "Error: Unsupported unary operator: " << unaryOp->getOp() << "\n";
      return nullptr;
  }
}

mlir::Value MLIRCodeGenContext::lowerArrayCreate(ArrayCreateExprAST* arrayCreate) {
  auto loc = getUnknownLocation();

  // Get dimensions
  const auto& dimensions = arrayCreate->getDimensions();
  if (dimensions.empty()) {
    llvm::errs() << "Error: Array must have at least one dimension\n";
    return nullptr;
  }

  // Lower the first dimension as size
  mlir::Value size = lowerExpression(dimensions[0].get());
  if (!size) {
    llvm::errs() << "Error: Failed to lower array size\n";
    return nullptr;
  }

  // Get element type
  mlir::Type elemType = getMLIRType(arrayCreate->getElementType());

  // Create the array type
  mlir::Type arrayType = mlir::simp::ArrayType::get(&mlirContext, elemType);

  // Create simp.array_create operation
  return builder.create<mlir::simp::ArrayCreateOp>(loc, arrayType, size);
}

mlir::Value MLIRCodeGenContext::lowerArrayAccess(ArrayAccessExprAST* arrayAccess,
                                                  mlir::Value newValue) {
  auto loc = getUnknownLocation();

  // Lower the array expression
  mlir::Value array = lowerExpression(arrayAccess->getArray());
  if (!array) {
    llvm::errs() << "Error: Failed to lower array expression\n";
    return nullptr;
  }

  // Lower the index expression (first index for now)
  const auto& indices = arrayAccess->getIndices();
  if (indices.empty()) {
    llvm::errs() << "Error: Array access requires at least one index\n";
    return nullptr;
  }

  mlir::Value index = lowerExpression(indices[0].get());
  if (!index) {
    llvm::errs() << "Error: Failed to lower array index\n";
    return nullptr;
  }

  // If newValue is provided, this is an array_set
  if (newValue) {
    return builder.create<mlir::simp::ArraySetOp>(loc, array.getType(),
                                                   array, index, newValue);
  } else {
    // Otherwise, it's an array_get
    // Get element type from array type
    auto arrayType = array.getType().dyn_cast<mlir::simp::ArrayType>();
    if (!arrayType) {
      llvm::errs() << "Error: Array access on non-array type\n";
      return nullptr;
    }

    mlir::Type elemType = arrayType.getElementType();
    return builder.create<mlir::simp::ArrayGetOp>(loc, elemType, array, index);
  }
}

mlir::Value MLIRCodeGenContext::lowerArrayStore(ArrayStoreExprAST* arrayStore) {
  auto loc = getUnknownLocation();

  // Check if we're storing to a variable (need to update symbol table)
  std::string varName;
  if (arrayStore->getArray()->getKind() == ASTKind::VariableExpr) {
    auto* varExpr = static_cast<VariableExprAST*>(arrayStore->getArray());
    varName = varExpr->getName();
  }

  // Lower the array expression
  mlir::Value array = lowerExpression(arrayStore->getArray());
  if (!array) {
    llvm::errs() << "Error: Failed to lower array expression in store\n";
    return nullptr;
  }

  // Lower the index expression (first index for now)
  const auto& indices = arrayStore->getIndices();
  if (indices.empty()) {
    llvm::errs() << "Error: Array store requires at least one index\n";
    return nullptr;
  }

  mlir::Value index = lowerExpression(indices[0].get());
  if (!index) {
    llvm::errs() << "Error: Failed to lower array index in store\n";
    return nullptr;
  }

  // Lower the value to store
  mlir::Value value = lowerExpression(arrayStore->getValue());
  if (!value) {
    llvm::errs() << "Error: Failed to lower value in array store\n";
    return nullptr;
  }

  // Create simp.array_set operation (returns new array in SSA form)
  mlir::Value newArray = builder.create<mlir::simp::ArraySetOp>(
      loc, array.getType(), array, index, value);

  // Update the symbol table if storing to a variable
  if (!varName.empty()) {
    declareVariable(varName, newArray);
  }

  return newArray;
}

mlir::Value MLIRCodeGenContext::lowerMatMul(MatMulExprAST* matmul) {
  auto loc = getUnknownLocation();

  // Lower the left-hand side matrix (A: MxK)
  mlir::Value lhs = lowerExpression(matmul->getLHS());
  if (!lhs) {
    llvm::errs() << "Error: Failed to lower matmul LHS\n";
    return nullptr;
  }

  // Lower the right-hand side matrix (B: KxN)
  mlir::Value rhs = lowerExpression(matmul->getRHS());
  if (!rhs) {
    llvm::errs() << "Error: Failed to lower matmul RHS\n";
    return nullptr;
  }

  // Lower the dimension arguments: m, k, n
  mlir::Value m = lowerExpression(matmul->getM());
  if (!m) {
    llvm::errs() << "Error: Failed to lower matmul dimension m\n";
    return nullptr;
  }

  mlir::Value k = lowerExpression(matmul->getK());
  if (!k) {
    llvm::errs() << "Error: Failed to lower matmul dimension k\n";
    return nullptr;
  }

  mlir::Value n = lowerExpression(matmul->getN());
  if (!n) {
    llvm::errs() << "Error: Failed to lower matmul dimension n\n";
    return nullptr;
  }

  // Lower the output buffer (pre-allocated by caller)
  mlir::Value output = lowerExpression(matmul->getOutput());
  if (!output) {
    llvm::errs() << "Error: Failed to lower matmul output buffer\n";
    return nullptr;
  }

  // Get the array type
  auto arrayType = output.getType().dyn_cast<mlir::simp::ArrayType>();
  if (!arrayType) {
    llvm::errs() << "Error: MatMul output is not an array type\n";
    return nullptr;
  }

  // Create the simp.matmul operation
  // The output buffer is pre-allocated - matmul writes in-place
  return builder.create<mlir::simp::MatMulOp>(
      loc, arrayType, lhs, rhs, output, m, k, n);
}

mlir::Value MLIRCodeGenContext::lowerCall(CallExprAST* call) {
  auto loc = getUnknownLocation();

  // Get the callee name
  const std::string& calleeName = call->getCallee();

  // Lower all argument expressions
  llvm::SmallVector<mlir::Value, 4> args;
  for (auto* argExpr : call->getArguments()) {
    mlir::Value arg = lowerExpression(argExpr);
    if (!arg) {
      llvm::errs() << "Error: Failed to lower call argument\n";
      return nullptr;
    }
    args.push_back(arg);
  }

  // Look up the function in the module
  mlir::FuncOp callee = module.lookupSymbol<mlir::FuncOp>(calleeName);
  if (!callee) {
    llvm::errs() << "Error: Undefined function '" << calleeName << "'\n";
    return nullptr;
  }

  // Create the call operation (Standard dialect in MLIR 14)
  // CallOp takes: location, callee symbol, arguments
  auto callOp = builder.create<mlir::CallOp>(
      loc,
      callee,
      args
  );

  // Return the first result (functions return a single value)
  if (callOp.getNumResults() == 0) {
    llvm::errs() << "Error: Function '" << calleeName << "' has no return value\n";
    return nullptr;
  }

  return callOp.getResult(0);
}

//===----------------------------------------------------------------------===//
// Statement Lowering
//===----------------------------------------------------------------------===//

mlir::LogicalResult MLIRCodeGenContext::lowerStatement(StmtAST* stmt) {
  if (!stmt) {
    return mlir::failure();
  }

  // Use getKind() for type identification without RTTI
  switch (stmt->getKind()) {
    case ASTKind::VariableDecl:
      return lowerDeclaration(static_cast<VariableDeclarationAST*>(stmt));

    case ASTKind::ReturnStmt:
      return lowerReturn(static_cast<ReturnAST*>(stmt));

    case ASTKind::ExpressionStmt:
      return lowerExpressionStmt(static_cast<ExpressionStmtAST*>(stmt));

    case ASTKind::IfStmt:
      return lowerIf(static_cast<IfAST*>(stmt));

    case ASTKind::WhileStmt:
      return lowerWhile(static_cast<WhileAST*>(stmt));

    case ASTKind::FunctionDecl:
      // Functions are handled at top level, not as statements
      llvm::errs() << "Warning: Function declarations should be lowered at module level\n";
      return mlir::success();

    case ASTKind::BlockStmt:
    case ASTKind::IncludeStmt:
      // Not yet implemented
      llvm::errs() << "Warning: Statement kind not yet implemented in MLIR lowering\n";
      return mlir::success();

    default:
      llvm::errs() << "Error: Unknown statement kind in MLIR lowering\n";
      return mlir::failure();
  }
}

mlir::LogicalResult MLIRCodeGenContext::lowerDeclaration(VariableDeclarationAST* decl) {
  // Lower the initialization expression
  mlir::Value initValue = lowerExpression(decl->getAssignmentExpr());
  if (!initValue) {
    return mlir::failure();
  }

  // Declare the variable in the symbol table
  declareVariable(decl->getName(), initValue);

  return mlir::success();
}

mlir::LogicalResult MLIRCodeGenContext::lowerReturn(ReturnAST* ret) {
  auto loc = getUnknownLocation();

  // Lower the return value expression
  mlir::Value returnValue = lowerExpression(ret->getExpression());
  if (!returnValue) {
    return mlir::failure();
  }

  // Create return operation (Standard dialect in MLIR 14)
  builder.create<mlir::ReturnOp>(loc, returnValue);

  return mlir::success();
}

mlir::LogicalResult MLIRCodeGenContext::lowerExpressionStmt(ExpressionStmtAST* exprStmt) {
  // Just lower the expression (side effects will be captured)
  mlir::Value result = lowerExpression(exprStmt->getExpression());
  return result ? mlir::success() : mlir::failure();
}

mlir::LogicalResult MLIRCodeGenContext::lowerIf(IfAST* ifStmt) {
  auto loc = getUnknownLocation();

  // Lower the condition expression
  mlir::Value condition = lowerExpression(ifStmt->getCondition());
  if (!condition) {
    llvm::errs() << "Error: Failed to lower if condition\n";
    return mlir::failure();
  }

  // The condition must be i1 type for scf.if
  if (!condition.getType().isInteger(1)) {
    // For now, we'll just warn about this limitation
    llvm::errs() << "Warning: Non-boolean condition in if statement\n";
  }

  // Get the set of variables that exist before the if statement
  auto existingVars = getCurrentVariableNames();

  // Track which variables are modified in the if/else blocks
  // Only track modifications to pre-existing variables (not new locals)
  auto thenModified = trackModifiedVariables(ifStmt->getThenBlock(), existingVars);
  auto elseModified = trackModifiedVariables(ifStmt->getElseBlock(), existingVars);

  // Find the union of variables modified in either branch
  std::set<std::string> allModified;
  allModified.insert(thenModified.begin(), thenModified.end());
  allModified.insert(elseModified.begin(), elseModified.end());

  // Collect initial values of modified variables (for else branch defaults)
  std::vector<mlir::Value> initialValues = collectVariableValues(allModified);

  // Determine result types for the scf.if operation
  llvm::SmallVector<mlir::Type, 4> resultTypes;
  for (const auto& value : initialValues) {
    resultTypes.push_back(value.getType());
  }

  // Create the scf.if operation with result types
  auto ifOp = builder.create<mlir::scf::IfOp>(
      loc, resultTypes, condition,
      /*withElseRegion=*/true);  // Always create else region when we have results

  // Lower the then block
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

    pushScope();
    BlockAST* thenBlock = ifStmt->getThenBlock();
    if (thenBlock) {
      for (auto* stmt : thenBlock->statements) {
        if (failed(lowerStatement(stmt))) {
          popScope();
          return mlir::failure();
        }
      }
    }

    // Collect final values of modified variables for yielding
    std::vector<mlir::Value> thenValues;
    for (const auto& varName : allModified) {
      mlir::Value value = lookupVariable(varName);
      if (!value) {
        // Variable wasn't modified in then branch, use initial value
        auto it = std::find(allModified.begin(), allModified.end(), varName);
        size_t idx = std::distance(allModified.begin(), it);
        value = initialValues[idx];
      }
      thenValues.push_back(value);
    }

    popScope();

    // Terminate with scf.yield, passing the modified values
    builder.create<mlir::scf::YieldOp>(loc, thenValues);
  }

  // Lower the else block
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());

    if (ifStmt->getElseBlock()) {
      pushScope();
      BlockAST* elseBlock = ifStmt->getElseBlock();
      for (auto* stmt : elseBlock->statements) {
        if (failed(lowerStatement(stmt))) {
          popScope();
          return mlir::failure();
        }
      }

      // Collect final values of modified variables for yielding
      std::vector<mlir::Value> elseValues;
      for (const auto& varName : allModified) {
        mlir::Value value = lookupVariable(varName);
        if (!value) {
          // Variable wasn't modified in else branch, use initial value
          auto it = std::find(allModified.begin(), allModified.end(), varName);
          size_t idx = std::distance(allModified.begin(), it);
          value = initialValues[idx];
        }
        elseValues.push_back(value);
      }

      popScope();

      // Terminate with scf.yield, passing the modified values
      builder.create<mlir::scf::YieldOp>(loc, elseValues);
    } else {
      // No else block - create implicit else that yields initial values
      std::vector<mlir::Value> elseValues(initialValues);
      builder.create<mlir::scf::YieldOp>(loc, elseValues);
    }
  }

  // Update symbol table with the results from scf.if
  if (!allModified.empty()) {
    // Extract modified variable values
    std::vector<mlir::Value> modifiedResults;
    for (size_t i = 0; i < allModified.size(); ++i) {
      modifiedResults.push_back(ifOp.getResult(i));
    }
    updateSymbolTableWithResults(allModified, modifiedResults);
  }

  return mlir::success();
}

mlir::LogicalResult MLIRCodeGenContext::lowerWhile(WhileAST* whileLoop) {
  auto loc = getUnknownLocation();

  // Get the set of variables that exist before the while loop
  auto existingVars = getCurrentVariableNames();

  // Track which variables are modified in the loop body
  // Only track modifications to pre-existing variables (not new loop locals)
  auto modifiedVars = trackModifiedVariables(whileLoop->getBody(), existingVars);

  // Collect initial values of loop-carried variables
  std::vector<mlir::Value> initialValues = collectVariableValues(modifiedVars);

  // Determine types for iter_args
  llvm::SmallVector<mlir::Type, 4> iterTypes;
  for (const auto& value : initialValues) {
    iterTypes.push_back(value.getType());
  }

  // Create the scf.while operation with iter_args
  // scf.while has two regions: "before" (condition) and "after" (body)
  auto whileOp = builder.create<mlir::scf::WhileOp>(loc, iterTypes, initialValues);

  // Build the "before" region (condition check)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Block* beforeBlock = builder.createBlock(&whileOp.getBefore());

    // Add block arguments for loop-carried values
    for (auto type : iterTypes) {
      beforeBlock->addArgument(type, loc);
    }

    builder.setInsertionPointToStart(beforeBlock);

    // Update symbol table with block arguments
    pushScope();
    auto varIt = modifiedVars.begin();
    for (size_t i = 0; i < beforeBlock->getNumArguments(); ++i, ++varIt) {
      declareVariable(*varIt, beforeBlock->getArgument(i));
    }

    // Lower the condition
    mlir::Value condition = lowerExpression(whileLoop->getCondition());
    if (!condition) {
      llvm::errs() << "Error: Failed to lower while condition\n";
      popScope();
      return mlir::failure();
    }

    // The condition must be i1 type
    if (!condition.getType().isInteger(1)) {
      llvm::errs() << "Warning: Non-boolean condition in while loop\n";
    }

    // Collect current values for passing to loop body
    std::vector<mlir::Value> conditionValues;
    for (const auto& varName : modifiedVars) {
      mlir::Value value = lookupVariable(varName);
      if (!value) {
        // Use the block argument if not found
        auto it = std::find(modifiedVars.begin(), modifiedVars.end(), varName);
        size_t idx = std::distance(modifiedVars.begin(), it);
        value = beforeBlock->getArgument(idx);
      }
      conditionValues.push_back(value);
    }

    popScope();

    // Terminate with scf.condition, passing values to the body
    builder.create<mlir::scf::ConditionOp>(loc, condition, conditionValues);
  }

  // Build the "after" region (loop body)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Block* afterBlock = builder.createBlock(&whileOp.getAfter());

    // Add block arguments for loop-carried values
    for (auto type : iterTypes) {
      afterBlock->addArgument(type, loc);
    }

    builder.setInsertionPointToStart(afterBlock);

    pushScope();

    // Update symbol table with block arguments
    auto varIt = modifiedVars.begin();
    for (size_t i = 0; i < afterBlock->getNumArguments(); ++i, ++varIt) {
      declareVariable(*varIt, afterBlock->getArgument(i));
    }

    // Lower the loop body
    BlockAST* body = whileLoop->getBody();
    if (body) {
      for (auto* stmt : body->statements) {
        if (failed(lowerStatement(stmt))) {
          popScope();
          return mlir::failure();
        }
      }
    }

    // Collect updated values for the next iteration
    std::vector<mlir::Value> nextIterValues;
    for (const auto& varName : modifiedVars) {
      mlir::Value value = lookupVariable(varName);
      if (!value) {
        // This shouldn't happen if tracking is correct
        auto it = std::find(modifiedVars.begin(), modifiedVars.end(), varName);
        size_t idx = std::distance(modifiedVars.begin(), it);
        value = afterBlock->getArgument(idx);
      }
      nextIterValues.push_back(value);
    }

    popScope();

    // Terminate with scf.yield, passing updated values back to condition
    builder.create<mlir::scf::YieldOp>(loc, nextIterValues);
  }

  // Update symbol table with the final results from the while loop
  if (!modifiedVars.empty()) {
    updateSymbolTableWithResults(modifiedVars, whileOp.getResults());
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Control Flow Helpers
//===----------------------------------------------------------------------===//

std::set<std::string> MLIRCodeGenContext::getCurrentVariableNames() const {
  std::set<std::string> varNames;

  // Collect all variable names from all scopes
  for (const auto& scope : symbolTable) {
    for (const auto& entry : scope) {
      varNames.insert(entry.first);
    }
  }

  return varNames;
}

std::set<std::string> MLIRCodeGenContext::trackModifiedVariables(
    BlockAST* block, const std::set<std::string>& existingVars) {
  std::set<std::string> modifiedVars;

  if (!block) return modifiedVars;

  // Scan through all statements to find assignments to existing variables
  for (auto* stmt : block->statements) {
    if (!stmt) continue;

    switch (stmt->getKind()) {
      case ASTKind::VariableDecl: {
        // Variable declarations: check if re-declaring an existing variable
        auto* decl = static_cast<VariableDeclarationAST*>(stmt);
        const std::string& varName = decl->getName();

        // Only track if this variable existed before this block
        if (existingVars.find(varName) != existingVars.end()) {
          modifiedVars.insert(varName);
        }
        // Otherwise it's a new local variable, not loop-carried
        break;
      }

      case ASTKind::ExpressionStmt: {
        auto* exprStmt = static_cast<ExpressionStmtAST*>(stmt);
        if (!exprStmt->getExpression()) break;

        // Check for regular assignment: x = value
        if (exprStmt->getExpression()->getKind() == ASTKind::AssignmentExpr) {
          auto* assign = static_cast<AssignmentExprAST*>(exprStmt->getExpression());
          // Extract variable name from LHS (assuming it's a VariableExprAST)
          if (assign->getLHS() &&
              assign->getLHS()->getKind() == ASTKind::VariableExpr) {
            auto* varExpr = static_cast<VariableExprAST*>(assign->getLHS());
            const std::string& varName = varExpr->getName();

            // Only track if this variable existed before this block
            if (existingVars.find(varName) != existingVars.end()) {
              modifiedVars.insert(varName);
            }
          }
        }
        // Check for array element assignment: A[i] = value
        // In Simp dialect, this is functional: A = array_set(A, i, value)
        else if (exprStmt->getExpression()->getKind() == ASTKind::ArrayStoreExpr) {
          auto* arrayStore = static_cast<ArrayStoreExprAST*>(exprStmt->getExpression());
          // Extract variable name from the array being modified
          if (arrayStore->getArray() &&
              arrayStore->getArray()->getKind() == ASTKind::VariableExpr) {
            auto* varExpr = static_cast<VariableExprAST*>(arrayStore->getArray());
            const std::string& varName = varExpr->getName();

            // Only track if this variable existed before this block
            if (existingVars.find(varName) != existingVars.end()) {
              modifiedVars.insert(varName);
            }
          }
        }
        break;
      }

      case ASTKind::IfStmt: {
        // Recursively track modified vars in nested if
        auto* ifStmt = static_cast<IfAST*>(stmt);
        auto thenVars = trackModifiedVariables(ifStmt->getThenBlock(), existingVars);
        auto elseVars = trackModifiedVariables(ifStmt->getElseBlock(), existingVars);
        modifiedVars.insert(thenVars.begin(), thenVars.end());
        modifiedVars.insert(elseVars.begin(), elseVars.end());
        break;
      }

      case ASTKind::WhileStmt: {
        // Recursively track modified vars in nested while
        auto* whileStmt = static_cast<WhileAST*>(stmt);
        auto bodyVars = trackModifiedVariables(whileStmt->getBody(), existingVars);
        modifiedVars.insert(bodyVars.begin(), bodyVars.end());
        break;
      }

      default:
        // Other statement types don't modify variables
        break;
    }
  }

  return modifiedVars;
}

std::vector<mlir::Value> MLIRCodeGenContext::collectVariableValues(
    const std::set<std::string>& varNames) {
  std::vector<mlir::Value> values;

  for (const auto& name : varNames) {
    mlir::Value value = lookupVariable(name);
    if (value) {
      values.push_back(value);
    } else {
      // This shouldn't happen if tracking is correct
      llvm::errs() << "Warning: Variable '" << name
                   << "' not found when collecting values\n";
    }
  }

  return values;
}

void MLIRCodeGenContext::updateSymbolTableWithResults(
    const std::set<std::string>& varNames,
    mlir::ValueRange results) {

  // Ensure we have the right number of results
  if (varNames.size() != results.size()) {
    llvm::errs() << "Error: Mismatch between variable count ("
                 << varNames.size() << ") and result count ("
                 << results.size() << ")\n";
    return;
  }

  // Update symbol table with new SSA values
  auto varIt = varNames.begin();
  for (size_t i = 0; i < results.size(); ++i, ++varIt) {
    declareVariable(*varIt, results[i]);
  }
}

//===----------------------------------------------------------------------===//
// Function Lowering
//===----------------------------------------------------------------------===//

mlir::FuncOp MLIRCodeGenContext::lowerFunction(FunctionAST* funcAst) {
  auto loc = getUnknownLocation();

  // Save the current insertion point (should be at module level)
  auto savedInsertionPoint = builder.saveInsertionPoint();

  // Build function type
  llvm::SmallVector<mlir::Type, 4> argTypes;
  for (auto* arg : funcAst->getArguments()) {
    // Get type from VariableDeclarationAST
    mlir::Type argType = builder.getF32Type(); // Default (matches existing compiler)
    if (arg->isStaticallyTyped()) {
      argType = getMLIRType(const_cast<TypeInfo*>(arg->getStaticType()));
    }
    argTypes.push_back(argType);
  }

  // Get return type
  mlir::Type returnType = builder.getF32Type(); // Default (matches existing compiler)
  if (funcAst->hasStaticReturnType()) {
    returnType = getMLIRType(const_cast<TypeInfo*>(funcAst->getReturnType()));
  }

  auto funcType = builder.getFunctionType(argTypes, returnType);

  // Create the function (Standard dialect in MLIR 14)
  // Note: builder.create() automatically inserts at the current insertion point
  auto func = builder.create<mlir::FuncOp>(loc, funcAst->getName(), funcType);

  // Create entry block
  auto* entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Set as current function
  setCurrentFunction(func);

  // Push new scope for function body
  pushScope();

  // Declare function arguments
  const auto& args = funcAst->getArguments();
  for (size_t i = 0; i < args.size(); ++i) {
    declareVariable(args[i]->getName(), entryBlock->getArgument(i));
  }

  // Lower function body
  BlockAST* body = funcAst->getBody();
  if (body) {
    for (auto* stmt : body->statements) {
      if (failed(lowerStatement(stmt))) {
        func.erase();
        popScope();
        builder.restoreInsertionPoint(savedInsertionPoint);
        return nullptr;
      }
    }
  }

  // Pop function scope
  popScope();

  // Restore insertion point to module level for next function
  builder.restoreInsertionPoint(savedInsertionPoint);

  return func;
}

mlir::FuncOp MLIRCodeGenContext::getOrCreateFunction(const std::string& name,
                                                     mlir::FunctionType funcType) {
  // Look for existing function in module
  if (auto func = module.lookupSymbol<mlir::FuncOp>(name)) {
    return func;
  }

  // Create new function declaration (Standard dialect in MLIR 14)
  // Note: builder.create() automatically inserts at the current insertion point
  auto loc = getUnknownLocation();
  auto func = builder.create<mlir::FuncOp>(loc, name, funcType);

  return func;
}
