#include "ast/expr/variable_expr.hpp"
#include "ast/stmt/function_stmt.hpp"
#include "logger.hpp"
#include "codegen.hpp"
#include "../ast_utils.hpp"
#include <llvm/IR/Value.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/GlobalVariable.h>

// VariableExprAST implementation
llvm::Value* VariableExprAST::codeGen(CodeGenContext& context) {
    LOG_DEBUG("Creating variable reference: ", name);

    llvm::Value* value = context.getSymbolValue(name);
    if (!value) {
        LOG_ERROR("Unknown variable name: ", name);
        return nullptr;
    }

    // Don't load from slice pointers (sse_slice_t* or avx_slice_t*)
    if (value->getType()->isPointerTy()) {
        // For opaque pointers in LLVM 14+, we need to check the symbol table for type info
        // or use explicit type information from our type system

        // Try to get type info from symbol table
        std::string varName = name;
        llvm::Value* symbolValue = context.getSymbolValue(varName);
        if (symbolValue && symbolValue == value) {
            // This is a function parameter or variable - check if it should be loaded
            // For function parameters, we generally don't load unless it's a local variable
            llvm::BasicBlock* currentBlock = context.getBuilder().GetInsertBlock();
            if (currentBlock) {
                llvm::Function* currentFunc = currentBlock->getParent();
                for (auto& arg : currentFunc->args()) {
                    if (&arg == value) {
                        // This is a function argument - return directly (no load)
                        return value;
                    }
                }
            }
            // If we're in global context, skip the function argument check
        }

        // For local variables allocated with alloca, we need to load
        if (llvm::isa<llvm::AllocaInst>(value)) {
            // Get the allocated type from the alloca instruction
            llvm::AllocaInst* allocaInst = llvm::cast<llvm::AllocaInst>(value);
            llvm::Type* allocatedType = allocaInst->getAllocatedType();
            return context.getBuilder().CreateLoad(
                allocatedType,
                value,
                name.c_str()
            );
        }

        // For global variables, we need to load them to get their value
        if (llvm::isa<llvm::GlobalVariable>(value)) {
            llvm::GlobalVariable* globalVar = llvm::cast<llvm::GlobalVariable>(value);
            llvm::Type* globalType = globalVar->getValueType();

            // Check if we have a valid insert block (function context)
            llvm::BasicBlock* currentBlock = context.getBuilder().GetInsertBlock();
            if (currentBlock) {
                // We're in a function - can use load instruction
                return context.getBuilder().CreateLoad(
                    globalType,
                    value,
                    name.c_str()
                );
            } else {
                // We're in global context - try to get the constant initializer
                if (globalVar->hasInitializer()) {
                    llvm::Constant* initializer = globalVar->getInitializer();
                    return initializer;
                } else {
                    LOG_ERROR("Global variable ", name, " accessed in global context without initializer");
                    return nullptr;
                }
            }
        }

        // For other cases, return the pointer directly (function parameters, etc.)
        return value;
    }

    return value;
}

// AssignmentExprAST implementation
llvm::Value* AssignmentExprAST::codeGen(CodeGenContext& context) {
    LOG_DEBUG("Generating assignment for ", lhs_->getName());

    llvm::Value* rhsValue = rhs_->codeGen(context);
    if (!rhsValue) {
        LOG_ERROR("Invalid right-hand side in assignment");
        return nullptr;
    }

    // Get the variable from the symbol table
    llvm::Value* variable = context.getSymbolValue(lhs_->getName());
    if (!variable) {
        LOG_ERROR("Undefined variable ", lhs_->getName());
        return nullptr;
    }

    // Check types and convert if needed
    llvm::Type* varType = nullptr;
    if (llvm::AllocaInst* allocaInst = llvm::dyn_cast<llvm::AllocaInst>(variable)) {
        varType = allocaInst->getAllocatedType();
    } else {
        // For other cases, try to infer from RHS type
        varType = rhsValue->getType();
    }
    rhsValue = convertType(rhsValue, varType, context, "assignconv");

    // Create the store instruction
    context.getBuilder().CreateStore(rhsValue, variable);

    // Return the assigned value
    return rhsValue;
}
