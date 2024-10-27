#include "ast.hpp"
#include "codegen.hpp"
#include <llvm/IR/Value.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <iostream>

using namespace std;

// Implementation of the Expression Statement code generation
llvm::Value* ExpressionStmtAST::codeGen(CodeGenContext& context) {
    return expression->codeGen(context);
}

// Implementation of the Block Statement code generation
llvm::Value* BlockAST::codeGen(CodeGenContext& context) {
    llvm::Value *last = nullptr;
    for (StmtAST* statement : statements) {
        last = statement->codeGen(context);
    }
    return last;
}

// Definition for AssignmentExprAST::codeGen
llvm::Value* AssignmentExprAST::codeGen(CodeGenContext& context) {
    if (!lhs) {
        std::cerr << "Error: Left-hand side of assignment is not a variable." << std::endl;
        return nullptr;
    }

    llvm::Value* rhsValue = rhs->codeGen(context);
    if (!rhsValue) {
        std::cerr << "Error: RHS of assignment could not be generated." << std::endl;
        return nullptr;
    }

    llvm::Value* variable = context.getNamedValue(lhs->getName());
    if (!variable) {
        std::cerr << "Error: Undefined variable '" << lhs->getName() << "'." << std::endl;
        return nullptr;
    }

    // Create a store instruction to assign the RHS value to the variable
    context.getBuilder().CreateStore(rhsValue, variable);
    return rhsValue;
}

// Definition for CallExprAST::codeGen
llvm::Value* CallExprAST::codeGen(CodeGenContext& context) {
    // Retrieve the function from the module
    llvm::Function* calleeF = context.getModule()->getFunction(callee);
    if (!calleeF) {
        std::cerr << "Error: Unknown function referenced: " << callee << std::endl;
        return nullptr;
    }

    // Check if the number of arguments matches
    if (calleeF->arg_size() != arguments.size()) {
        std::cerr << "Error: Incorrect number of arguments passed to function '" << callee << "'." << std::endl;
        return nullptr;
    }

    std::vector<llvm::Value*> argsV;
    for (unsigned i = 0, e = arguments.size(); i != e; ++i) {
        argsV.push_back(arguments[i]->codeGen(context));
        if (!argsV.back()) {
            std::cerr << "Error: Argument " << i << " could not be generated." << std::endl;
            return nullptr;
        }
    }

    // Create a call instruction to invoke the function
    return context.getBuilder().CreateCall(calleeF, argsV, "calltmp");
}


// Implementation of the Variable Declaration code generation
llvm::Value* VariableDeclarationAST::codeGen(CodeGenContext& context) {
    llvm::Type* type = context.getDoubleType();
    llvm::Function* function = context.getBuilder().GetInsertBlock()->getParent();
    
    // Create an alloca instruction in the entry block of the function
    llvm::IRBuilder<> TmpB(&function->getEntryBlock(), 
                          function->getEntryBlock().begin());
    llvm::AllocaInst* alloca = TmpB.CreateAlloca(type, nullptr, name);
    
    // Store the variable in our symbol table
    context.setNamedValue(name, alloca);
    
    // Generate the initialization value
    if (assignmentExpr != nullptr) {
        llvm::Value* initVal = assignmentExpr->codeGen(context);
        if (initVal == nullptr) return nullptr;
        context.getBuilder().CreateStore(initVal, alloca);
    }
    
    return alloca;
}

llvm::Value* FunctionAST::codeGen(CodeGenContext& context) {
    std::vector<llvm::Type*> argTypes(arguments.size(), context.getDoubleType());
    llvm::FunctionType* functionType = llvm::FunctionType::get(
        context.getDoubleType(), argTypes, false);
    
    llvm::Function* function = llvm::Function::Create(
        functionType, llvm::Function::ExternalLinkage, name, context.getModule());
    
    if (!function) return nullptr;
    
    // Create a new basic block to start insertion into
    llvm::BasicBlock* BB = llvm::BasicBlock::Create(context.getContext(), 
                                                   "entry", function);
    context.getBuilder().SetInsertPoint(BB);
    
    // Record the function arguments in the NamedValues map
    size_t idx = 0;
    for (auto &Arg : function->args()) {
        // Create an alloca for this variable
        llvm::IRBuilder<> TmpB(&function->getEntryBlock(), 
                              function->getEntryBlock().begin());
        llvm::AllocaInst* alloca = TmpB.CreateAlloca(context.getDoubleType(), 
                                                    nullptr, 
                                                    arguments[idx]->getName());
        
        // Store the initial value into the alloca
        context.getBuilder().CreateStore(&Arg, alloca);
        
        // Add to symbol table
        context.setNamedValue(arguments[idx]->getName(), alloca);
        idx++;
    }
    
    if (!body->codeGen(context)) {
        // Error generating body, remove function
        function->eraseFromParent();
        return nullptr;
    }
    
    // Check if the current block is not terminated
    if (!context.getBuilder().GetInsertBlock()->getTerminator()) {
        if (function->getReturnType()->isVoidTy()) {
            context.getBuilder().CreateRetVoid();
        } else {
            context.getBuilder().CreateRet(llvm::ConstantFP::get(context.getContext(), llvm::APFloat(0.0)));
        }
    }
    
    return function;
}


// Implementation of the Return Statement code generation
llvm::Value* ReturnAST::codeGen(CodeGenContext& context) {
    llvm::Value* returnValue = expression->codeGen(context);
    if (!returnValue) return nullptr;
    
    return context.getBuilder().CreateRet(returnValue);
}

// Implementation of Number Expression code generation
llvm::Value* NumberExprAST::codeGen(CodeGenContext& context) {
    return llvm::ConstantFP::get(context.getContext(), llvm::APFloat(value));
}

// Implementation of Variable Expression code generation
llvm::Value* VariableExprAST::codeGen(CodeGenContext& context) {
    llvm::Value* value = context.getNamedValue(name);
    if (!value) {
        std::cerr << "Unknown variable name: " << name << std::endl;
        return nullptr;
    }
    
    return context.getBuilder().CreateLoad(value->getType()->getPointerElementType(), 
                                         value, name.c_str());
}

// Implementation of Binary Expression code generation
llvm::Value* BinaryExprAST::codeGen(CodeGenContext& context) {
    llvm::Value* L = lhs->codeGen(context);
    llvm::Value* R = rhs->codeGen(context);
    if (!L || !R) return nullptr;

    switch (op) {
        case OpAdd:
            return context.getBuilder().CreateFAdd(L, R, "addtmp");
        case OpSub:
            return context.getBuilder().CreateFSub(L, R, "subtmp");
        case OpMul:
            return context.getBuilder().CreateFMul(L, R, "multmp");
        case OpDiv:
            return context.getBuilder().CreateFDiv(L, R, "divtmp");
        case OpLT:
            L = context.getBuilder().CreateFCmpULT(L, R, "lttmp");
            return context.getBuilder().CreateUIToFP(L, context.getDoubleType(), "booltmp");
        case OpGT:
            L = context.getBuilder().CreateFCmpUGT(L, R, "gttmp");
            return context.getBuilder().CreateUIToFP(L, context.getDoubleType(), "booltmp");
        case OpLE:
            L = context.getBuilder().CreateFCmpULE(L, R, "letmp");
            return context.getBuilder().CreateUIToFP(L, context.getDoubleType(), "booltmp");
        case OpGE:
            L = context.getBuilder().CreateFCmpUGE(L, R, "getmp");
            return context.getBuilder().CreateUIToFP(L, context.getDoubleType(), "booltmp");
        case OpEQ:
            L = context.getBuilder().CreateFCmpUEQ(L, R, "eqtmp");
            return context.getBuilder().CreateUIToFP(L, context.getDoubleType(), "booltmp");
        case OpNE:
            L = context.getBuilder().CreateFCmpUNE(L, R, "netmp");
            return context.getBuilder().CreateUIToFP(L, context.getDoubleType(), "booltmp");
        default:
            std::cerr << "Invalid binary operator" << std::endl;
            return nullptr;
    }
}


// Implementation of If Statement code generation
llvm::Value* IfAST::codeGen(CodeGenContext& context) {
    llvm::Value* conditionValue = condition->codeGen(context);
    if (!conditionValue) return nullptr;
    
    // Convert condition to a bool by comparing non-equal to 0.0
    conditionValue = context.getBuilder().CreateFCmpONE(
        conditionValue, 
        llvm::ConstantFP::get(context.getContext(), llvm::APFloat(0.0)), 
        "ifcond");
    
    llvm::Function* function = context.getBuilder().GetInsertBlock()->getParent();
    
    // Create blocks for the then and else cases. Insert the 'then' block at the end of the function
    llvm::BasicBlock* thenBB = llvm::BasicBlock::Create(context.getContext(), "then", function);
    llvm::BasicBlock* elseBB = llvm::BasicBlock::Create(context.getContext(), "else");
    llvm::BasicBlock* mergeBB = llvm::BasicBlock::Create(context.getContext(), "ifcont");
    
    context.getBuilder().CreateCondBr(conditionValue, thenBB, elseBB);
    
    // Emit then value
    context.getBuilder().SetInsertPoint(thenBB);
    llvm::Value* thenValue = thenBlock->codeGen(context);
    if (!thenValue) return nullptr;
    
    context.getBuilder().CreateBr(mergeBB);
    
    // Emit else block
    function->getBasicBlockList().push_back(elseBB);
    context.getBuilder().SetInsertPoint(elseBB);
    
    llvm::Value* elseValue = nullptr;
    if (elseBlock) {
        elseValue = elseBlock->codeGen(context);
        if (!elseValue) return nullptr;
    }
    
    context.getBuilder().CreateBr(mergeBB);
    
    // Emit merge block
    function->getBasicBlockList().push_back(mergeBB);
    context.getBuilder().SetInsertPoint(mergeBB);
    
    return llvm::ConstantFP::get(context.getContext(), llvm::APFloat(0.0));
}

// Implementation of While Statement code generation
llvm::Value* WhileAST::codeGen(CodeGenContext& context) {
    llvm::Function* function = context.getBuilder().GetInsertBlock()->getParent();
    
    // Create basic blocks for the loop
    llvm::BasicBlock* condBB = llvm::BasicBlock::Create(context.getContext(), "loopcond", function);
    llvm::BasicBlock* loopBB = llvm::BasicBlock::Create(context.getContext(), "loop");
    llvm::BasicBlock* afterBB = llvm::BasicBlock::Create(context.getContext(), "afterloop");
    
    // Branch to condition block
    context.getBuilder().CreateBr(condBB);
    context.getBuilder().SetInsertPoint(condBB);
    
    // Generate condition value
    llvm::Value* conditionValue = condition->codeGen(context);
    if (!conditionValue) return nullptr;
    
    // Convert condition to a bool by comparing non-equal to 0.0
    conditionValue = context.getBuilder().CreateFCmpONE(
        conditionValue, 
        llvm::ConstantFP::get(context.getContext(), llvm::APFloat(0.0)), 
        "loopcond");
    
    context.getBuilder().CreateCondBr(conditionValue, loopBB, afterBB);
    
    // Emit loop block
    function->getBasicBlockList().push_back(loopBB);
    context.getBuilder().SetInsertPoint(loopBB);
    
    if (!body->codeGen(context)) return nullptr;
    
    context.getBuilder().CreateBr(condBB);
    
    // Emit after loop block
    function->getBasicBlockList().push_back(afterBB);
    context.getBuilder().SetInsertPoint(afterBB);
    
    return llvm::Constant::getNullValue(context.getDoubleType());
}
