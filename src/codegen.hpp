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
#include <iostream>
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

private:
    bool integerContextFlag = false;

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
    void dumpBlocks() const;
    void dumpSymbols() const;

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

    void setIntegerContext(bool flag) { integerContextFlag = flag; }
    bool isIntegerContext() const { return integerContextFlag; }

    void createVoidCall(llvm::Function* func, std::vector<llvm::Value*> args) {
        builder.CreateCall(func, args);
    }

    // Helper method for non-void calls
    llvm::Value* createCall(llvm::Function* func, std::vector<llvm::Value*> args, const std::string& name = "") {
        return builder.CreateCall(func, args, name);
    }

    // Helper for slice operations
    void createSliceSet(llvm::Value* slice, llvm::Value* idx, llvm::Value* val, bool isSSE) {
        // First, ensure we have a proper slice pointer, not a pointer to a pointer
        if (slice->getType()->isPointerTy() && 
            slice->getType()->getPointerElementType()->isPointerTy()) {
            slice = builder.CreateLoad(slice->getType()->getPointerElementType(), slice);
        }

        llvm::Function* setFunc = module->getFunction(isSSE ? "slice_set_sse" : "slice_set_avx");
        if (!setFunc) {
            std::cerr << "Set function not found" << std::endl;
            return;
        }
        createVoidCall(setFunc, {slice, idx, val});
    }

    // Helper for slice creation
    llvm::Value* createMakeSlice(llvm::Value* len, bool isSSE) {
        // Ensure len is i64
        if (len->getType()->isDoubleTy()) {
            len = builder.CreateFPToSI(len, builder.getInt64Ty());
        }
        llvm::Function* makeFunc = module->getFunction(isSSE ? "make_sse_slice" : "make_avx_slice");
        if (!makeFunc) {
            std::cerr << "Make function not found" << std::endl;
            return nullptr;
        }
        return createCall(makeFunc, {len}, "slice.create");
    }
};

// Slice runtime structure (matches LLVM struct type)
struct SliceRuntime {
    void* data;      // Pointer to vector data
    size_t len;      // Current length
    size_t cap;      // Capacity
};

#endif // CODEGEN_HPP