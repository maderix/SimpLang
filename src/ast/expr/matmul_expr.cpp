#include "ast/expr/matmul_expr.hpp"
#include "codegen.hpp"

// Stub implementation for LLVM codegen
// MatMul is primarily for MLIR backend; LLVM backend not implemented
llvm::Value* MatMulExprAST::codeGen(CodeGenContext& context) {
    // TODO: Implement naive matmul for LLVM backend if needed
    // For now, this is MLIR-only
    llvm::errs() << "Error: matmul() is only supported with --emit-mlir flag\n";
    return nullptr;
}
