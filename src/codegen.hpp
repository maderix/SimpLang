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

// Forward declarations
class BlockAST;

class CodeGenBlock {
public:
    std::map<std::string, llvm::Value*> locals;
};

class CodeGenContext {
private:
    std::vector<CodeGenBlock*> blocks;
    llvm::LLVMContext context;
    std::unique_ptr<llvm::Module> module;
    llvm::IRBuilder<> builder;
    std::unique_ptr<llvm::legacy::FunctionPassManager> fpm;
    std::unique_ptr<llvm::TargetMachine> targetMachine;
    std::string targetTriple;

    // Cache for commonly used types and functions
    llvm::StructType* sseSliceType;
    llvm::StructType* avxSliceType;
    llvm::Function* mallocFunc;
    llvm::Function* freeFunc;
    llvm::Function* errorFunc;

    // Initialize runtime functions
    void initializeRuntimeFunctions();
    void initializeSliceTypes();

public:
    CodeGenContext();
    ~CodeGenContext();

    // Explicitly delete copy operations
    CodeGenContext(const CodeGenContext&) = delete;
    CodeGenContext& operator=(const CodeGenContext&) = delete;

    // Basic accessors
    llvm::LLVMContext& getContext() { return context; }
    llvm::Module* getModule() { return module.get(); }
    llvm::IRBuilder<>& getBuilder() { return builder; }
    llvm::Type* getDoubleType() { return llvm::Type::getDoubleTy(context); }
    llvm::Function* currentFunction() { return builder.GetInsertBlock()->getParent(); }
    llvm::TargetMachine* getTargetMachine() { return targetMachine.get(); }

    // SIMD support
    llvm::Type* getVectorType(unsigned width);

    // Runtime function access
    llvm::Function* getMallocFunc() { return mallocFunc; }
    llvm::Function* getFreeFunc() { return freeFunc; }
    void emitError(const std::string& message);

    // Slice support
    llvm::Type* getSliceType(SliceType type);
    llvm::Value* createSlice(SliceType type, llvm::Value* len, llvm::Value* cap);
    llvm::Value* createSlice(SliceType type, llvm::Value* len) {
        return createSlice(type, len, len); // Default: capacity = length
    }
    
    llvm::Value* createSliceWithCap(SliceType type, llvm::Value* len, llvm::Value* cap);
    
    // Slice field access
    llvm::Value* getSliceData(llvm::Value* slice);
    llvm::Value* getSliceLen(llvm::Value* slice);
    llvm::Value* getSliceCap(llvm::Value* slice);
    void setSliceData(llvm::Value* slice, llvm::Value* data);
    void setSliceLen(llvm::Value* slice, llvm::Value* len);
    void setSliceCap(llvm::Value* slice, llvm::Value* cap);

    // Code generation
    void generateCode(BlockAST& root);
    void pushBlock();
    void popBlock();
    void setSymbolValue(const std::string& name, llvm::Value* value);
    llvm::Value* getSymbolValue(const std::string& name);
    
    // Function type handling
    llvm::Type* getCurrentFunctionType(const std::string& name) {
        if (name.find("_sse") != std::string::npos) {
            return getVectorType(4);
        } else if (name.find("_avx") != std::string::npos) {
            return getVectorType(8);
        }
        return getDoubleType();
    }

    // Type checks and conversions
    bool isSliceType(llvm::Type* type) const;
    bool isVectorType(llvm::Type* type) const;
    unsigned getVectorWidth(llvm::Type* type) const;
    //debug function:
    void declareRuntimeFunctions();
};

// Slice runtime structure (matches LLVM struct type)
struct SliceRuntime {
    void* data;      // Pointer to vector data
    size_t len;      // Current length
    size_t cap;      // Capacity
};

#endif // CODEGEN_HPP