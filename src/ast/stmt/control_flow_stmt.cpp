#include "ast/stmt/control_flow_stmt.hpp"
#include "logger.hpp"
#include "codegen.hpp"
#include <llvm/IR/Value.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>

// IfAST implementation
llvm::Value* IfAST::codeGen(CodeGenContext& context) {
    llvm::Value* condV = condition->codeGen(context);
    if (!condV) return nullptr;

    // Convert condition to bool if it's a double
    if (condV->getType()->isDoubleTy()) {
        condV = context.getBuilder().CreateFCmpONE(
            condV,
            llvm::ConstantFP::get(context.getContext(), llvm::APFloat(0.0)),
            "ifcond"
        );
    }

    llvm::Function* theFunction = context.getBuilder().GetInsertBlock()->getParent();
    llvm::BasicBlock* entryBlock = context.getBuilder().GetInsertBlock();

    // Create basic blocks
    llvm::BasicBlock* thenBB = llvm::BasicBlock::Create(context.getContext(), "then", theFunction);
    llvm::BasicBlock* elseBB = elseBlock ?
        llvm::BasicBlock::Create(context.getContext(), "else") : nullptr;
    llvm::BasicBlock* mergeBB = llvm::BasicBlock::Create(context.getContext(), "merge");

    // Create conditional branch
    context.getBuilder().CreateCondBr(condV, thenBB, elseBB ? elseBB : mergeBB);

    // Generate then block
    context.getBuilder().SetInsertPoint(thenBB);
    llvm::Value* thenV = thenBlock->codeGen(context);
    if (!thenV) return nullptr;

    // If there's no terminator, create branch to merge
    if (!context.getBuilder().GetInsertBlock()->getTerminator()) {
        context.getBuilder().CreateBr(mergeBB);
    }

    // Record the block where 'then' part ended
    llvm::BasicBlock* thenEndBB = context.getBuilder().GetInsertBlock();

    // Generate else block
    llvm::BasicBlock* elseEndBB = nullptr;
    llvm::Value* elseV = nullptr;
    if (elseBlock) {
        theFunction->getBasicBlockList().push_back(elseBB);
        context.getBuilder().SetInsertPoint(elseBB);
        elseV = elseBlock->codeGen(context);
        if (!elseV) return nullptr;
        if (!context.getBuilder().GetInsertBlock()->getTerminator()) {
            context.getBuilder().CreateBr(mergeBB);
        }
        elseEndBB = context.getBuilder().GetInsertBlock();
    }

    // Add merge block
    theFunction->getBasicBlockList().push_back(mergeBB);
    context.getBuilder().SetInsertPoint(mergeBB);

    // If both then and else terminate (e.g., with return), no PHI needed
    if (thenEndBB->getTerminator() &&
        (!elseBlock || elseEndBB->getTerminator())) {
        return llvm::Constant::getNullValue(llvm::Type::getDoubleTy(context.getContext()));
    }

    // If values are generated and the blocks don't terminate, create PHI
    if (thenV->getType() != llvm::Type::getVoidTy(context.getContext())) {
        llvm::PHINode* PN = context.getBuilder().CreatePHI(
            thenV->getType(), 2, "iftmp");

        // Only add incoming value from 'then' if it doesn't terminate
        if (!thenEndBB->getTerminator()) {
            PN->addIncoming(thenV, thenEndBB);
        }

        // Handle the else/default path
        llvm::Value* elseVal = elseV ? elseV :
            llvm::Constant::getNullValue(thenV->getType());

        llvm::BasicBlock* incomingBlock = elseBlock ? elseEndBB : entryBlock;
        if (incomingBlock && !incomingBlock->getTerminator()) {
            PN->addIncoming(elseVal, incomingBlock);
        }

        return PN;
    }

    return nullptr;
}

// WhileAST implementation
llvm::Value* WhileAST::codeGen(CodeGenContext& context) {
    llvm::Function* function = context.getBuilder().GetInsertBlock()->getParent();

    // Create blocks for the loop
    llvm::BasicBlock* condBB = llvm::BasicBlock::Create(context.getContext(), "cond", function);
    llvm::BasicBlock* loopBB = llvm::BasicBlock::Create(context.getContext(), "loop", function);
    llvm::BasicBlock* afterBB = llvm::BasicBlock::Create(context.getContext(), "afterloop", function);

    // Branch to condition block
    context.getBuilder().CreateBr(condBB);

    // Emit condition block
    context.getBuilder().SetInsertPoint(condBB);
    llvm::Value* condValue = condition->codeGen(context);
    if (!condValue)
        return nullptr;

    // Convert condition to bool if it's a double
    if (condValue->getType()->isDoubleTy()) {
        condValue = context.getBuilder().CreateFCmpOLE(
            condValue,
            llvm::ConstantFP::get(context.getContext(), llvm::APFloat(0.0)),
            "whilecond"
        );
    }

    // Create conditional branch
    context.getBuilder().CreateCondBr(condValue, loopBB, afterBB);

    // Emit loop block
    context.getBuilder().SetInsertPoint(loopBB);
    context.pushBlock();

    llvm::Value* bodyVal = body->codeGen(context);
    if (!bodyVal)
        return nullptr;

    context.popBlock();

    // Create back edge if no terminator exists
    if (!context.getBuilder().GetInsertBlock()->getTerminator()) {
        context.getBuilder().CreateBr(condBB);
    }

    // Move to after block
    context.getBuilder().SetInsertPoint(afterBB);

    return llvm::ConstantFP::get(context.getContext(), llvm::APFloat(0.0));
}
