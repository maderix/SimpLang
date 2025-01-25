#include "ProfilingPass.hpp"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Casting.h"

char ProfilingPass::ID = 0;

// Change run to runOnFunction for legacy pass manager
bool ProfilingPass::runOnFunction(llvm::Function &F) {
    if (F.isDeclaration()) {
        return false;
    }

    // Safety check - make sure function has basic blocks
    if (F.empty()) {
        return false;
    }

    auto &M = *F.getParent();
    
    try {
        insertFunctionInstrumentation(F, M);

        for (auto &BB : F) {
            insertBlockInstrumentation(BB, M);
        }
    } catch (const std::exception& e) {
        llvm::errs() << "Error in ProfilingPass: " << e.what() << "\n";
        return false;
    }

    return true;
}

void ProfilingPass::insertFunctionInstrumentation(llvm::Function &F, llvm::Module &M) {
    // Safety check - make sure function has entry block
    if (F.empty()) {
        return;
    }

    auto &Ctx = M.getContext();
    llvm::IRBuilder<> Builder(Ctx);

    // Get function enter/exit hooks
    auto *VoidPtrTy = llvm::Type::getInt8PtrTy(Ctx);
    auto *ProfileFuncTy = llvm::FunctionType::get(
        llvm::Type::getVoidTy(Ctx),
        {VoidPtrTy, VoidPtrTy},
        false
    );

    auto EnterFunc = getOrInsertFunction(M, "__cyg_profile_func_enter", ProfileFuncTy);
    auto ExitFunc = getOrInsertFunction(M, "__cyg_profile_func_exit", ProfileFuncTy);

    // Insert enter probe at start
    auto &EntryBB = F.getEntryBlock();
    if (!EntryBB.empty()) {
        Builder.SetInsertPoint(&EntryBB, EntryBB.getFirstInsertionPt());
        auto *FuncPtr = Builder.CreateBitCast(&F, VoidPtrTy);
        Builder.CreateCall(EnterFunc, {FuncPtr, llvm::ConstantPointerNull::get(VoidPtrTy)});
    }

    // Insert exit probe before each return
    for (auto &BB : F) {
        auto *Term = BB.getTerminator();
        if (!Term) continue;
        
        if (auto *RI = llvm::dyn_cast<llvm::ReturnInst>(Term)) {
            Builder.SetInsertPoint(RI);
            auto *FuncPtr = Builder.CreateBitCast(&F, VoidPtrTy);
            Builder.CreateCall(ExitFunc, {FuncPtr, llvm::ConstantPointerNull::get(VoidPtrTy)});
        }
    }
}

void ProfilingPass::insertBlockInstrumentation(llvm::BasicBlock &BB, llvm::Module &M) {
    // Safety check - make sure block has terminator
    if (BB.empty() || !BB.getTerminator()) {
        return;
    }

    auto &Ctx = M.getContext();
    llvm::IRBuilder<> Builder(Ctx);

    // Get block enter/exit hooks
    auto *Int8PtrTy = llvm::Type::getInt8PtrTy(Ctx);
    auto *TraceBlockTy = llvm::FunctionType::get(
        llvm::Type::getVoidTy(Ctx),
        {Int8PtrTy},
        false
    );

    auto EnterFunc = getOrInsertFunction(M, "__trace_block_enter", TraceBlockTy);
    auto ExitFunc = getOrInsertFunction(M, "__trace_block_exit", TraceBlockTy);

    // Create block name string
    std::string BlockName = BB.getParent()->getName().str() + "." + BB.getName().str();
    auto *BlockNameGV = Builder.CreateGlobalStringPtr(BlockName);

    // Insert enter probe at start of block
    Builder.SetInsertPoint(&BB, BB.getFirstInsertionPt());
    Builder.CreateCall(EnterFunc, {BlockNameGV});

    // Insert exit probe before terminator
    Builder.SetInsertPoint(BB.getTerminator());
    Builder.CreateCall(ExitFunc, {BlockNameGV});
}

llvm::FunctionCallee ProfilingPass::getOrInsertFunction(llvm::Module &M,
                                                       const std::string &Name,
                                                       llvm::FunctionType *FTy) {
    if (auto *ExistingFunc = M.getFunction(Name)) {
        return ExistingFunc;
    }
    return M.getOrInsertFunction(Name, FTy);
} 