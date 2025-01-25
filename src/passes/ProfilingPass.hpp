#ifndef PROFILING_PASS_HPP
#define PROFILING_PASS_HPP

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"

class ProfilingPass : public llvm::FunctionPass {
public:
    static char ID;
    ProfilingPass() : FunctionPass(ID) {}

    bool runOnFunction(llvm::Function &F) override;
    
private:
    void insertFunctionInstrumentation(llvm::Function &F, llvm::Module &M);
    void insertBlockInstrumentation(llvm::BasicBlock &BB, llvm::Module &M);
    
    llvm::FunctionCallee getOrInsertFunction(llvm::Module &M, const std::string &Name, 
                                           llvm::FunctionType *FTy);
};

#endif