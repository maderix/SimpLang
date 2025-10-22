#ifndef AST_UTILS_HPP
#define AST_UTILS_HPP

#include <string>
#include <llvm/IR/Value.h>
#include <llvm/IR/Type.h>

class CodeGenContext;

// Generic type conversion utility
llvm::Value* convertType(llvm::Value* value, llvm::Type* targetType,
                         CodeGenContext& context, const std::string& name = "conv");

#endif // AST_UTILS_HPP