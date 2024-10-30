#ifndef CODEGEN_HPP
#define CODEGEN_HPP

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/Host.h>
#include "ast.hpp"
#include <map>
#include <string>
#include <vector>
#include <memory>

class CodeGenBlock {
public:
    std::map<std::string, llvm::Value*> locals;
};

class CodeGenContext {
    std::vector<CodeGenBlock*> blocks;
    llvm::LLVMContext context;
    std::unique_ptr<llvm::Module> module;
    llvm::IRBuilder<> builder;
    std::unique_ptr<llvm::legacy::FunctionPassManager> fpm;
    std::unique_ptr<llvm::TargetMachine> targetMachine;
    std::string targetTriple;

public:
    CodeGenContext();

    llvm::LLVMContext& getContext();
    llvm::Module* getModule();
    llvm::IRBuilder<>& getBuilder();
    llvm::Type* getDoubleType();
    llvm::Function* currentFunction();
    llvm::TargetMachine* getTargetMachine();

    void generateCode(BlockAST& root);
    void pushBlock();
    void popBlock();
    void setSymbolValue(const std::string& name, llvm::Value* value);
    llvm::Value* getSymbolValue(const std::string& name);
    llvm::Type* getVectorType(unsigned width);
    llvm::Type* getCurrentFunctionType(const std::string& name);
};

#endif // CODEGEN_HPP