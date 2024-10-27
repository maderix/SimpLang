#ifndef CODEGEN_HPP
#define CODEGEN_HPP

#include <stack>
#include <map>
#include <memory>
#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/BasicBlock.h>

class BlockAST;
class AST;

class CodeGenContext {
    std::unique_ptr<llvm::LLVMContext> context;
    std::unique_ptr<llvm::IRBuilder<>> builder;
    std::unique_ptr<llvm::Module> module;
    std::map<std::string, llvm::Value*> namedValues;
    std::stack<llvm::BasicBlock*> blocks;
    llvm::Function* mainFunc;
    
public:
    CodeGenContext();
    
    llvm::Module* getModule() { return module.get(); }
    llvm::LLVMContext& getContext() { return *context; }
    llvm::IRBuilder<>& getBuilder() { return *builder; }
    llvm::Function* getCurrentFunction() { return mainFunc; }
    
    void setCurrentBlock(llvm::BasicBlock *block) { blocks.push(block); }
    void popBlock() { blocks.pop(); }
    llvm::BasicBlock* currentBlock() { return blocks.empty() ? nullptr : blocks.top(); }
    
    void setNamedValue(const std::string& name, llvm::Value* value) {
        namedValues[name] = value;
    }
    
    llvm::Value* getNamedValue(const std::string& name) {
        auto it = namedValues.find(name);
        if (it != namedValues.end()) {
            return it->second;
        }
        return nullptr;
    }
    
    llvm::Type* getDoubleType() { return llvm::Type::getDoubleTy(*context); }
    llvm::Type* getInt32Type() { return llvm::Type::getInt32Ty(*context); }
    llvm::Type* getVoidType() { return llvm::Type::getVoidTy(*context); }
    
    void generateCode(BlockAST& root);
};

#endif
