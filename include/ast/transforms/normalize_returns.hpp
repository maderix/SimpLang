//===- normalize_returns.hpp - Return Normalization Transform ---*- C++ -*-===//
//
// Part of the SimpLang Project
//
// This file defines the return normalization transformation pass that converts
// functions with multiple return statements into single-exit form. This is
// required for MLIR lowering which expects structured control flow.
//
//===----------------------------------------------------------------------===//

#ifndef NORMALIZE_RETURNS_HPP
#define NORMALIZE_RETURNS_HPP

#include <memory>

// Forward declarations
class BlockAST;
class FunctionAST;

namespace ast {
namespace transforms {

//===----------------------------------------------------------------------===//
// ReturnNormalizer
//===----------------------------------------------------------------------===//

/// Transforms functions with multiple return statements into single-exit form.
///
/// This pass is essential for MLIR lowering because MLIR's structured control
/// flow (scf.if) cannot contain return operations. The transformation adds a
/// result variable and converts all returns into assignments followed by a
/// single return at the end.
///
/// Example:
///   fn foo(x) {                    fn foo(x) {
///     if (x < 0) {                   var __mlir_result;
///       return 0;        --->        if (x < 0) {
///     }                                __mlir_result = 0;
///     return x * 2;                  } else {
///   }                                  __mlir_result = x * 2;
///                                    }
///                                    return __mlir_result;
///                                  }
class ReturnNormalizer {
public:
  ReturnNormalizer() = default;

  /// Normalize all functions in the program block to single-exit form
  /// Returns true if any transformations were applied
  bool normalizeReturns(BlockAST* programBlock);

private:
  /// Count the number of return statements in a block (recursive)
  int countReturns(BlockAST* block);

  /// Transform a single function to single-exit form
  /// Returns true if transformation was applied
  bool transformFunction(FunctionAST* func);

  /// Replace return statements with assignments to result variable
  /// Returns true if any returns were replaced
  bool replaceReturnsWithAssignments(BlockAST* block, const std::string& resultVar);
};

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

/// Normalize all functions in the program to single-exit form
/// This is the main entry point for the transformation pass
bool normalizeAllReturns(BlockAST* programBlock);

} // namespace transforms
} // namespace ast

#endif // NORMALIZE_RETURNS_HPP
