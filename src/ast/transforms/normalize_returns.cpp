//===- normalize_returns.cpp - Return Normalization Implementation -------===//
//
// Part of the SimpLang Project
//
// This file implements the return normalization transformation pass.
//
//===----------------------------------------------------------------------===//

#include "ast/transforms/normalize_returns.hpp"
#include "ast/ast.hpp"
#include <iostream>

namespace ast {
namespace transforms {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Count the number of return statements in a block (recursive)
int ReturnNormalizer::countReturns(BlockAST* block) {
  if (!block) return 0;

  int count = 0;
  for (auto* stmt : block->statements) {
    if (!stmt) continue;

    // Direct return statement
    if (stmt->getKind() == ASTKind::ReturnStmt) {
      count++;
      continue;
    }

    // Check nested blocks
    if (stmt->getKind() == ASTKind::IfStmt) {
      auto* ifStmt = static_cast<IfAST*>(stmt);
      count += countReturns(ifStmt->getThenBlock());
      count += countReturns(ifStmt->getElseBlock());
    } else if (stmt->getKind() == ASTKind::WhileStmt) {
      auto* whileStmt = static_cast<WhileAST*>(stmt);
      count += countReturns(whileStmt->getBody());
    }
  }

  return count;
}

/// Replace return statements with assignments to the result variable
bool ReturnNormalizer::replaceReturnsWithAssignments(BlockAST* block,
                                                      const std::string& resultVar) {
  if (!block) return false;

  bool modified = false;

  for (size_t i = 0; i < block->statements.size(); ++i) {
    auto* stmt = block->statements[i];
    if (!stmt) continue;

    // Replace return statement with assignment
    if (stmt->getKind() == ASTKind::ReturnStmt) {
      auto* retStmt = static_cast<ReturnAST*>(stmt);
      auto* returnExpr = retStmt->getExpression();

      if (!returnExpr) {
        // Skip returns with no expression (shouldn't happen in well-formed code)
        continue;
      }

      // Create assignment: __mlir_result = returnExpr
      // Note: We need to transfer ownership of returnExpr from retStmt to assignExpr
      auto* varExpr = new VariableExprAST(resultVar, true);  // true = write access
      auto* assignExpr = new AssignmentExprAST(varExpr, std::unique_ptr<ExprAST>(returnExpr));

      // Wrap in expression statement
      auto* assignStmt = new ExpressionStmtAST(assignExpr);

      // Replace return with assignment
      // Note: This will delete retStmt, but we've already transferred ownership of returnExpr
      delete retStmt;
      block->statements[i] = assignStmt;
      modified = true;
      continue;
    }

    // Recursively process nested blocks
    if (stmt->getKind() == ASTKind::IfStmt) {
      auto* ifStmt = static_cast<IfAST*>(stmt);
      bool thenMod = replaceReturnsWithAssignments(ifStmt->getThenBlock(), resultVar);
      bool elseMod = replaceReturnsWithAssignments(ifStmt->getElseBlock(), resultVar);
      modified = modified || thenMod || elseMod;
    } else if (stmt->getKind() == ASTKind::WhileStmt) {
      auto* whileStmt = static_cast<WhileAST*>(stmt);
      bool bodyMod = replaceReturnsWithAssignments(whileStmt->getBody(), resultVar);
      modified = modified || bodyMod;
    }
  }

  return modified;
}

//===----------------------------------------------------------------------===//
// Function Transformation
//===----------------------------------------------------------------------===//

/// Transform a single function to single-exit form
bool ReturnNormalizer::transformFunction(FunctionAST* func) {
  if (!func) return false;

  auto* body = func->getBody();
  if (!body) return false;

  // Count return statements
  int returnCount = countReturns(body);

  // If 0 or 1 return (and it's at the end), no transformation needed
  if (returnCount <= 1) {
    // Check if the single return is already at the end
    if (returnCount == 1 && !body->statements.empty()) {
      auto* lastStmt = body->statements.back();
      if (lastStmt && lastStmt->getKind() == ASTKind::ReturnStmt) {
        // Already in single-exit form
        return false;
      }
    }
    if (returnCount == 0) {
      // No returns at all (void function or missing return)
      return false;
    }
  }

  std::cerr << "[AST Transform] Normalizing function '" << func->getName()
            << "' (" << returnCount << " returns)" << std::endl;

  // Create result variable name (use unique name to avoid conflicts)
  const std::string resultVarName = "__mlir_result";

  // Get return type from function (default to Dynamic/float if not specified)
  auto* returnType = func->getReturnType();
  TypeKind typeKind = returnType ? returnType->kind : TypeKind::Dynamic;

  // Create default initializer (0.0 for now, will be overwritten)
  auto* zeroInit = new NumberExprAST(0.0);

  // Create variable declaration for result: var __mlir_result = 0.0;
  auto* resultDecl = new VariableDeclarationAST(
      resultVarName,
      zeroInit,  // initialize with zero
      std::make_unique<TypeInfo>(typeKind),
      nullptr,  // no slice type
      0,        // line number
      false     // not global
  );

  // Insert result variable at the beginning of the function body
  body->statements.insert(body->statements.begin(), resultDecl);

  // Replace all return statements with assignments
  replaceReturnsWithAssignments(body, resultVarName);

  // Add final return statement: return __mlir_result;
  auto* resultVarExpr = new VariableExprAST(resultVarName);
  auto* finalReturn = new ReturnAST(resultVarExpr);
  body->statements.push_back(finalReturn);

  return true;
}

//===----------------------------------------------------------------------===//
// Main Entry Point
//===----------------------------------------------------------------------===//

/// Normalize all functions in the program block
bool ReturnNormalizer::normalizeReturns(BlockAST* programBlock) {
  if (!programBlock) return false;

  bool anyModified = false;

  // Iterate through top-level statements looking for functions
  for (auto* stmt : programBlock->statements) {
    if (!stmt) continue;

    if (stmt->getKind() == ASTKind::FunctionDecl) {
      auto* func = static_cast<FunctionAST*>(stmt);
      bool modified = transformFunction(func);
      anyModified = anyModified || modified;
    }
  }

  return anyModified;
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

bool normalizeAllReturns(BlockAST* programBlock) {
  ReturnNormalizer normalizer;
  return normalizer.normalizeReturns(programBlock);
}

} // namespace transforms
} // namespace ast
