#ifndef CODEGEN_HPP
#define CODEGEN_HPP

#include "slice_type.hpp"
#include "simd_interface.hpp"
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/Host.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DebugLoc.h>
#include <llvm/IR/DebugInfo.h>
#include "ast.hpp"
#include "memory_tracker.hpp"
#include <map>
#include <iostream>
#include <string>
#include <vector>
#include <memory>

// Forward declarations
class BlockAST;
class MemoryTracker;

class CodeGenBlock {
public:
    std::map<std::string, llvm::Value*> locals;
    std::map<std::string, llvm::DILocalVariable*> debugLocals;
    llvm::DIScope* debugScope;
    
    CodeGenBlock(llvm::DIScope* scope = nullptr) : debugScope(scope) {}
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

    // SIMD support
    bool simd_enabled = false;
    void initializeMallocFree();
    void initializeSIMDFunctions();
    void initializeSimpBlasFunctions();
    std::unique_ptr<SIMDInterface> simdInterface;

    // Cache for commonly used types
    llvm::StructType* sseSliceType;
    llvm::StructType* avxSliceType;
    llvm::Function* mallocFunc;
    llvm::Function* freeFunc;
    llvm::Function* errorFunc;
    
    // SimpBLAS function declarations
    llvm::Function* simpblasEwAddF32Func;
    llvm::Function* simpblasEwMulF32Func;
    llvm::Function* simpblasEwReluF32Func;
    llvm::Function* simpblasGemmF32Func;

    // Private initialization methods
    void initializeModuleAndPassManager();
    void initializeTargetMachine();
    void initializeRuntimeFunctions();
    void initializeSliceTypes();
    void declareRuntimeFunctions();

    // Debug info support
    void createDebugInfoForModule(const std::string& filename);
    std::unique_ptr<llvm::DIBuilder> debugBuilder;
    llvm::DIFile* debugFile;
    llvm::DICompileUnit* debugCompileUnit;
    llvm::DIScope* currentDebugScope;
    unsigned currentDebugLine;
    std::string currentDebugFile;
    
    // Variable tracking
    std::shared_ptr<MemoryTracker> memoryTracker;
    std::map<std::string, llvm::DILocalVariable*> debugVariables;
    std::map<std::string, llvm::Value*> globalDebugValues;

    bool integerContextFlag = false;

    llvm::Value* createSliceWithCap(SliceType type, llvm::Value* len, llvm::Value* cap);

public:
    CodeGenContext();
    ~CodeGenContext();

    // SIMD control methods
    void enableSIMD() { simd_enabled = true; }
    bool isSIMDEnabled() const { return simd_enabled; }

    // Debug info methods
    void initializeDebugInfo(const std::string& filename);
    void setCurrentDebugLocation(unsigned line, const std::string& filename = "");
    llvm::DILocalVariable* createDebugVariable(const std::string& name, 
                                             llvm::Type* type,
                                             llvm::Value* storage,
                                             unsigned line);
    void finalizeDebugInfo();
    llvm::DIScope* getCurrentDebugScope() const { return currentDebugScope; }
    unsigned getCurrentDebugLine() const { return currentDebugLine; }

    // Variable tracking
    void trackVariable(const std::string& name, llvm::Value* value, llvm::Type* type);
    void updateVariableValue(const std::string& name, llvm::Value* value);
    llvm::Value* getVariableValue(const std::string& name) const;
    void registerVariableWrite(llvm::Value* ptr, const std::string& name);
    void registerVariableRead(llvm::Value* ptr, const std::string& name);

    // Memory tracking
    void trackMemoryAccess(llvm::Value* ptr, size_t size, bool isWrite);
    void addMemoryAccess(llvm::Value* ptr, bool isWrite);
    void addVariableAccess(const std::string& name, bool isWrite);

    // Basic accessors
    llvm::LLVMContext& getContext() { return context; }
    llvm::Module* getModule() { return module.get(); }
    llvm::IRBuilder<>& getBuilder() { return builder; }
    llvm::DIBuilder* getDebugBuilder() { return debugBuilder.get(); }
    MemoryTracker* getMemoryTracker() { return memoryTracker.get(); }
    
    void setMemoryTracker(std::shared_ptr<MemoryTracker> tracker) {
        memoryTracker = tracker;
    }

    // Type helpers and checks
    llvm::Type* getDoubleType() { return llvm::Type::getDoubleTy(context); }
    llvm::Type* getVectorType(unsigned width);
    llvm::Function* currentFunction() { return builder.GetInsertBlock()->getParent(); }
    llvm::TargetMachine* getTargetMachine();
    bool isSliceType(llvm::Type* type) const;
    bool isVectorType(llvm::Type* type) const;
    unsigned getVectorWidth(llvm::Type* type) const;

    // SIMD support
    llvm::Type* getSliceType(SliceType type);
    llvm::Value* createSlice(SliceType type, llvm::Value* len);
    llvm::Value* getSliceData(llvm::Value* slice);
    llvm::Value* getSliceLen(llvm::Value* slice);
    llvm::Value* getSliceCap(llvm::Value* slice);
    void setSliceData(llvm::Value* slice, llvm::Value* data);
    void setSliceLen(llvm::Value* slice, llvm::Value* len);
    void setSliceCap(llvm::Value* slice, llvm::Value* cap);

    // Runtime function access
    llvm::Function* getMallocFunc() { return mallocFunc; }
    llvm::Function* getFreeFunc() { return freeFunc; }
    void emitError(const std::string& message);

    // Code generation and blocks
    void generateCode(BlockAST& root);
    void pushBlock(llvm::DIScope* debugScope = nullptr);
    void popBlock();
    
    // Symbol table management
    void setSymbolValue(const std::string& name, llvm::Value* value);
    llvm::Value* getSymbolValue(const std::string& name);
    void dumpSymbols() const;
    void dumpBlocks() const;

    // Scope and debug handling
    void enterFunction(llvm::Function* func, llvm::DISubprogram* debugInfo);
    void exitFunction();
    void enterScope(llvm::DIScope* scope = nullptr);
    void exitScope();
    void declareVariable(const std::string& name, llvm::Value* value, 
                        llvm::DILocalVariable* debugVar = nullptr);

    // Variable scope management
    void enterVariableScope(const std::string& name);
    void exitVariableScope(const std::string& name);

    // Integer context flag for handling integer literals
    void setIntegerContext(bool flag) { integerContextFlag = flag; }
    bool isIntegerContext() const { return integerContextFlag; }
    bool isTrackingEnabled() const { return memoryTracker != nullptr; }

    // SIMD operations
    SIMDInterface* getSIMDInterface() { return simdInterface.get(); }

    llvm::Function* createFunction(const std::string& name,
                                 const std::vector<std::pair<std::string, llvm::Type*>>& args);

    llvm::Value* createAlignedVector(llvm::Type* vectorType, llvm::Value* dataPtr);

    llvm::Value* createAVXVector(const std::vector<double>& values);

    void emitSliceSet(llvm::Value* slice, llvm::Value* index, llvm::Value* value);
};

#endif // CODEGEN_HPP