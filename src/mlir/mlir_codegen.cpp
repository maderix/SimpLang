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
    return mlir::simp::SimpTensorType::get(&mlirContext, {-1}, builder.getF64Type());
  }

  // Default to f64
  llvm::errs() << "Warning: Unknown type '" << simpType << "', defaulting to f64\n";
  return builder.getF64Type();
}

mlir::Type MLIRCodeGenContext::getMLIRType(TypeInfo* typeInfo) {
  if (!typeInfo) {
    return builder.getF64Type(); // Default
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

  return builder.getF64Type();
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

    case ASTKind::CallExpr:
      return lowerCall(static_cast<CallExprAST*>(expr));

    case ASTKind::ArrayCreateExpr:
      return lowerArrayCreate(static_cast<ArrayCreateExprAST*>(expr));

    case ASTKind::ArrayAccessExpr:
      return lowerArrayAccess(static_cast<ArrayAccessExprAST*>(expr));

    default:
      llvm::errs() << "Error: Unsupported expression kind in MLIR lowering\n";
      return nullptr;
  }
}

mlir::Value MLIRCodeGenContext::lowerLiteral(NumberExprAST* literal) {
  auto loc = getUnknownLocation();

  // Determine type based on the literal value using accessor
  double val = literal->getValue();
  mlir::Type type = builder.getF64Type();
  mlir::Attribute value;

  // Check if it's an integer or floating-point
  if (val == std::floor(val) && literal->isIntegerLiteral()) {
    // Integer value
    type = builder.getI64Type();
    value = builder.getI64IntegerAttr(static_cast<int64_t>(val));
  } else {
    // Floating-point value
    value = builder.getF64FloatAttr(val);
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
    case OpLT:
      return builder.create<mlir::arith::CmpFOp>(
          loc, mlir::arith::CmpFPredicate::OLT, lhs, rhs);
    case OpGT:
      return builder.create<mlir::arith::CmpFOp>(
          loc, mlir::arith::CmpFPredicate::OGT, lhs, rhs);
    case OpLE:
      return builder.create<mlir::arith::CmpFOp>(
          loc, mlir::arith::CmpFPredicate::OLE, lhs, rhs);
    case OpGE:
      return builder.create<mlir::arith::CmpFOp>(
          loc, mlir::arith::CmpFPredicate::OGE, lhs, rhs);
    case OpEQ:
      return builder.create<mlir::arith::CmpFOp>(
          loc, mlir::arith::CmpFPredicate::OEQ, lhs, rhs);
    case OpNE:
      return builder.create<mlir::arith::CmpFOp>(
          loc, mlir::arith::CmpFPredicate::ONE, lhs, rhs);

    default:
      llvm::errs() << "Error: Unsupported binary operator: " << binOp->getOp() << "\n";
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
  // If it's not, we need to convert it (e.g., compare with zero)
  if (!condition.getType().isInteger(1)) {
    // Convert to boolean by comparing with 0.0
    mlir::Value zero = builder.create<mlir::simp::ConstantOp>(
        loc, condition.getType(),
        builder.getZeroAttr(condition.getType()));

    // Create a comparison (not equal to zero means true)
    // For now, assume f64 - we'd use arith.cmpf in a full implementation
    // Since we don't have comparison ops in simp dialect yet,
    // we'll just document this limitation
    llvm::errs() << "Warning: Boolean conversion not fully implemented, assuming i1 type\n";
  }

  // Create the scf.if operation
  auto ifOp = builder.create<mlir::scf::IfOp>(loc, condition, /*withElseRegion=*/ifStmt->getElseBlock() != nullptr);

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
    popScope();

    // Add yield to terminate the region
    builder.create<mlir::scf::YieldOp>(loc);
  }

  // Lower the else block if it exists
  if (ifStmt->getElseBlock()) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());

    pushScope();
    BlockAST* elseBlock = ifStmt->getElseBlock();
    for (auto* stmt : elseBlock->statements) {
      if (failed(lowerStatement(stmt))) {
        popScope();
        return mlir::failure();
      }
    }
    popScope();

    // Add yield to terminate the region
    builder.create<mlir::scf::YieldOp>(loc);
  }

  return mlir::success();
}

mlir::LogicalResult MLIRCodeGenContext::lowerWhile(WhileAST* whileLoop) {
  auto loc = getUnknownLocation();

  // Create the scf.while operation
  // scf.while has two regions: "before" (condition) and "after" (body)
  auto whileOp = builder.create<mlir::scf::WhileOp>(loc, llvm::None, llvm::None);

  // Build the "before" region (condition check)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Block* beforeBlock = builder.createBlock(&whileOp.getBefore());
    builder.setInsertionPointToStart(beforeBlock);

    // Lower the condition
    mlir::Value condition = lowerExpression(whileLoop->getCondition());
    if (!condition) {
      llvm::errs() << "Error: Failed to lower while condition\n";
      return mlir::failure();
    }

    // The condition must be i1 type
    if (!condition.getType().isInteger(1)) {
      llvm::errs() << "Warning: Boolean conversion not fully implemented for while, assuming i1 type\n";
    }

    // Terminate with scf.condition
    builder.create<mlir::scf::ConditionOp>(loc, condition, llvm::None);
  }

  // Build the "after" region (loop body)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Block* afterBlock = builder.createBlock(&whileOp.getAfter());
    builder.setInsertionPointToStart(afterBlock);

    pushScope();
    BlockAST* body = whileLoop->getBody();
    if (body) {
      for (auto* stmt : body->statements) {
        if (failed(lowerStatement(stmt))) {
          popScope();
          return mlir::failure();
        }
      }
    }
    popScope();

    // Terminate with scf.yield
    builder.create<mlir::scf::YieldOp>(loc);
  }

  return mlir::success();
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
    mlir::Type argType = builder.getF64Type(); // Default for now
    if (arg->isStaticallyTyped()) {
      argType = getMLIRType(const_cast<TypeInfo*>(arg->getStaticType()));
    }
    argTypes.push_back(argType);
  }

  // Get return type
  mlir::Type returnType = builder.getF64Type(); // Default
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
