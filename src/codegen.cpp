#include "codegen.hpp"
#include "ast.hpp"
#include <llvm/IR/Verifier.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/Host.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm-c/Target.h>
#include <iostream>

CodeGenContext::CodeGenContext() : builder(context) {
    module = std::make_unique<llvm::Module>("simple-lang", context);

    // Set target triple and data layout
    targetTriple = llvm::sys::getDefaultTargetTriple();
    module->setTargetTriple(targetTriple);

    std::string error;
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
    if (!target) {
        llvm::errs() << error << "\n";
        exit(1);
    }

    llvm::TargetOptions opt;
    auto RM = llvm::Optional<llvm::Reloc::Model>();

    targetMachine = std::unique_ptr<llvm::TargetMachine>(
        target->createTargetMachine(targetTriple, "generic", "", opt, RM));

    module->setDataLayout(targetMachine->createDataLayout());

    // Initialize optimization passes
    fpm = std::make_unique<llvm::legacy::FunctionPassManager>(module.get());
    fpm->doInitialization();

    // Initialize SIMD interface based on target architecture
    std::string arch = targetMachine->getTargetCPU().str();
    if (arch.find("avx") != std::string::npos) {
        simdInterface = std::unique_ptr<SIMDInterface>(createSIMDInterface("avx"));
    } else {
        simdInterface = std::unique_ptr<SIMDInterface>(createSIMDInterface("sse"));
    }

    // Don't initialize slice types until needed
    sseSliceType = nullptr;
    avxSliceType = nullptr;
}

CodeGenContext::~CodeGenContext() {
    for (auto block : blocks) {
        delete block;
    }
    blocks.clear();
}


void CodeGenContext::initializeRuntimeFunctions() {
    // malloc function
    std::vector<llvm::Type*> mallocArgs = {llvm::Type::getInt64Ty(context)};
    llvm::FunctionType* mallocType = llvm::FunctionType::get(
        llvm::Type::getInt8PtrTy(context), mallocArgs, false);
    mallocFunc = llvm::Function::Create(mallocType, llvm::Function::ExternalLinkage,
                                      "malloc", module.get());

    // free function
    std::vector<llvm::Type*> freeArgs = {llvm::Type::getInt8PtrTy(context)};
    llvm::FunctionType* freeType = llvm::FunctionType::get(
        llvm::Type::getVoidTy(context), freeArgs, false);
    freeFunc = llvm::Function::Create(freeType, llvm::Function::ExternalLinkage,
                                    "free", module.get());

    // error function (for bounds checking)
    std::vector<llvm::Type*> errorArgs = {llvm::Type::getInt8PtrTy(context)};
    llvm::FunctionType* errorType = llvm::FunctionType::get(
        llvm::Type::getVoidTy(context), errorArgs, false);
    errorFunc = llvm::Function::Create(errorType, llvm::Function::ExternalLinkage,
                                     "error", module.get());
}

void CodeGenContext::initializeSliceTypes() {
    // SSE Slice type (4 doubles)
    std::vector<llvm::Type*> sseFields = {
        llvm::Type::getInt8PtrTy(context),  // data pointer
        llvm::Type::getInt64Ty(context),    // length
        llvm::Type::getInt64Ty(context)     // capacity
    };
    sseSliceType = llvm::StructType::create(context, sseFields, "SSESlice");

    // AVX Slice type (8 doubles)
    std::vector<llvm::Type*> avxFields = {
        llvm::Type::getInt8PtrTy(context),  // data pointer
        llvm::Type::getInt64Ty(context),    // length
        llvm::Type::getInt64Ty(context)     // capacity
    };
    avxSliceType = llvm::StructType::create(context, avxFields, "AVXSlice");
}

llvm::Type* CodeGenContext::getSliceType(SliceType type) {
    return type == SliceType::SSE_SLICE ? sseSliceType : avxSliceType;
}

llvm::Value* CodeGenContext::createSlice(SliceType type, llvm::Value* len, llvm::Value* cap) {
    // Convert length to i64 if it's a double
    if (len->getType()->isDoubleTy()) {
        len = builder.CreateFPToSI(len, builder.getInt64Ty(), "len.conv");
    }
    
    // Convert capacity to i64 if it's a double
    if (cap && cap->getType()->isDoubleTy()) {
        cap = builder.CreateFPToSI(cap, builder.getInt64Ty(), "cap.conv");
    }

    // Get the appropriate make function
    llvm::Function* makeFunc = type == SliceType::SSE_SLICE ? 
        module->getFunction("make_sse_slice") : 
        module->getFunction("make_avx_slice");

    if (!makeFunc) {
        std::cerr << "Make function not found for slice type" << std::endl;
        return nullptr;
    }

    // Create the slice
    return builder.CreateCall(makeFunc, {len}, "slice.create");
}

llvm::Value* CodeGenContext::createSliceWithCap(SliceType type, llvm::Value* len, llvm::Value* cap) {
    auto& builder = getBuilder();
    llvm::Type* sliceTy = getSliceType(type);
    
    // Allocate slice struct
    llvm::Value* slice = builder.CreateAlloca(sliceTy, nullptr, "slice");
    
    // Get data pointer type based on slice type
    llvm::Type* vecTy = getVectorType(type == SliceType::SSE_SLICE ? 4 : 8);
    llvm::Type* ptrTy = vecTy->getPointerTo();
    
    // Calculate allocation size (capacity * vector_size)
    llvm::Value* vecSize = llvm::ConstantInt::get(cap->getType(), 
        module->getDataLayout().getTypeAllocSize(vecTy));
    llvm::Value* allocSize = builder.CreateMul(cap, vecSize);
    
    // Allocate data buffer
    auto allocFunc = module->getOrInsertFunction("aligned_alloc",
        llvm::FunctionType::get(
            builder.getInt8PtrTy(),
            {builder.getInt64Ty(), builder.getInt64Ty()},
            false
        ));
    
    llvm::Value* alignment = llvm::ConstantInt::get(builder.getInt64Ty(), 
        type == SliceType::SSE_SLICE ? 32 : 64);
    
    llvm::Value* data = builder.CreateCall(allocFunc, {alignment, allocSize});
    data = builder.CreateBitCast(data, ptrTy);
    
    // Store fields
    builder.CreateStore(data, 
        builder.CreateStructGEP(sliceTy, slice, 0));
    builder.CreateStore(len,
        builder.CreateStructGEP(sliceTy, slice, 1));
    builder.CreateStore(cap,
        builder.CreateStructGEP(sliceTy, slice, 2));
        
    return slice;
}

void CodeGenContext::declareVariable(const std::string& name, llvm::Value* value, 
                                   llvm::DILocalVariable* debugVar) {
    std::cout << "Declaring variable: " << name << std::endl;
    if (!blocks.empty()) {
        blocks.back()->locals[name] = value;
        if (debugVar) {
            blocks.back()->debugLocals[name] = debugVar;
        }

        // Track the variable in memory tracker if available
        if (memoryTracker && value) {
            llvm::Type* varType = value->getType()->getPointerElementType();
            trackVariable(name, value, varType);
        }
    } else {
        std::cerr << "Error: No active block for variable declaration" << std::endl;
    }
}

void CodeGenContext::createDebugInfoForModule(const std::string& filename) {
    if (!debugBuilder) {
        return;
    }

    // Create file descriptor
    debugFile = debugBuilder->createFile(
        llvm::StringRef(filename),
        llvm::StringRef(".")  // Current directory
    );

    // Create compile unit
    debugCompileUnit = debugBuilder->createCompileUnit(
        llvm::dwarf::DW_LANG_C,          // Source language
        debugFile,                        // File descriptor
        "SimpleLang Compiler",           // Producer
        false,                           // isOptimized
        "",                             // Compiler flags
        0,                              // Runtime version
        "",                             // Split name
        llvm::DICompileUnit::DebugEmissionKind::FullDebug,  // Debug emission
        0,                              // DWOId
        true,                           // Split Debug Inlining
        false,                          // Debug Info for profiling
        llvm::DICompileUnit::DebugNameTableKind::Default,  // Use default name table
        false,                          // Range lists
        "",                             // Sysroot
        ""                              // SDK
    );

    // Set compile unit as current debug scope
    currentDebugScope = debugCompileUnit;
    
    // Record current file
    currentDebugFile = filename;
}

void CodeGenContext::trackVariable(const std::string& name, llvm::Value* value, llvm::Type* type) {
    if (!memoryTracker) return;

    MemoryTracker::VarType varType;
    if (type->isDoubleTy()) {
        varType = MemoryTracker::VarType::Double;
    } else if (type->isIntegerTy()) {
        varType = MemoryTracker::VarType::Int;
    } else if (isVectorType(type)) {
        varType = (getVectorWidth(type) == 4) ? MemoryTracker::VarType::SSE_Vector : 
                                               MemoryTracker::VarType::AVX_Vector;
    } else if (isSliceType(type)) {
        varType = (type == sseSliceType) ? MemoryTracker::VarType::SSE_Slice : 
                                          MemoryTracker::VarType::AVX_Slice;
    } else {
        return;
    }

    memoryTracker->trackVariable(name, value, varType);
}

void CodeGenContext::updateVariableValue(const std::string& name, llvm::Value* value) {
    if (!value || !memoryTracker) return;
    
    auto* type = value->getType();
    if (auto* loadInst = llvm::dyn_cast<llvm::LoadInst>(value)) {
        value = loadInst->getPointerOperand();
        type = value->getType()->getPointerElementType();
    }
    
    size_t size = module->getDataLayout().getTypeAllocSize(type);
    memoryTracker->trackAccess(value, size, true);
}

void CodeGenContext::registerVariableWrite(llvm::Value* ptr, const std::string& name) {
    if (!ptr || !memoryTracker) return;
    trackMemoryAccess(ptr, module->getDataLayout().getTypeAllocSize(
        ptr->getType()->getPointerElementType()), true);
}

void CodeGenContext::registerVariableRead(llvm::Value* ptr, const std::string& name) {
    if (!ptr || !memoryTracker) return;
    trackMemoryAccess(ptr, module->getDataLayout().getTypeAllocSize(
        ptr->getType()->getPointerElementType()), false);
}

void CodeGenContext::trackMemoryAccess(llvm::Value* ptr, size_t size, bool isWrite) {
    if (memoryTracker) {
        memoryTracker->trackAccess(ptr, size, isWrite);
    }
}

void CodeGenContext::addMemoryAccess(llvm::Value* ptr, bool isWrite) {
    if (!ptr || !memoryTracker) return;
    trackMemoryAccess(ptr, module->getDataLayout().getTypeAllocSize(
        ptr->getType()->getPointerElementType()), isWrite);
}

// Debug Information
void CodeGenContext::initializeDebugInfo(const std::string& filename) {
    debugBuilder = std::make_unique<llvm::DIBuilder>(*module);
    createDebugInfoForModule(filename);
}

void CodeGenContext::setCurrentDebugLocation(unsigned line, const std::string& filename) {
    currentDebugLine = line;
    if (!filename.empty()) {
        currentDebugFile = filename;
    }
    
    if (debugBuilder) {
        auto scope = getCurrentDebugScope() ? getCurrentDebugScope() : debugCompileUnit;
        auto loc = llvm::DILocation::get(context, line, 0, scope);
        builder.SetCurrentDebugLocation(loc);
    }
}

llvm::DILocalVariable* CodeGenContext::createDebugVariable(
    const std::string& name, llvm::Type* type, llvm::Value* storage, unsigned line) {
    if (!debugBuilder || !currentDebugScope) return nullptr;

    auto diType = debugBuilder->createBasicType(
        type->isDoubleTy() ? "double" : "int",
        type->isDoubleTy() ? 64 : 32,
        type->isDoubleTy() ? llvm::dwarf::DW_ATE_float : llvm::dwarf::DW_ATE_signed);

    auto var = debugBuilder->createAutoVariable(
        currentDebugScope, name, debugFile, line, diType);
    
    debugBuilder->insertDeclare(
        storage, var, debugBuilder->createExpression(),
        llvm::DILocation::get(context, line, 0, currentDebugScope),
        builder.GetInsertBlock());

    return var;
}

// Scope Management
void CodeGenContext::enterScope(llvm::DIScope* scope) {
    if (scope) currentDebugScope = scope;
    pushBlock(scope);
}

void CodeGenContext::exitScope() {
    popBlock();
}

void CodeGenContext::enterFunction(llvm::Function* func, llvm::DISubprogram* debugInfo) {
    if (debugInfo) currentDebugScope = debugInfo;
    pushBlock(debugInfo);
}

void CodeGenContext::exitFunction() {
    popBlock();
    if (!blocks.empty() && blocks.back()->debugScope) {
        currentDebugScope = blocks.back()->debugScope;
    }
}

void CodeGenContext::emitError(const std::string& message) {
    llvm::Value* msgGlobal = builder.CreateGlobalStringPtr(message);
    builder.CreateCall(errorFunc, {msgGlobal});
}

llvm::Value* CodeGenContext::getSliceData(llvm::Value* slice) {
    llvm::Value* dataPtr = builder.CreateStructGEP(
        slice->getType()->getPointerElementType(), 
        slice, 
        0, 
        "data.ptr");
    return builder.CreateLoad(
        dataPtr->getType()->getPointerElementType(),
        dataPtr,
        "slice.data");
}

llvm::Value* CodeGenContext::getSliceLen(llvm::Value* slice) {
    llvm::Value* lenPtr = builder.CreateStructGEP(
        slice->getType()->getPointerElementType(), 
        slice, 
        1, 
        "len.ptr");
    return builder.CreateLoad(
        lenPtr->getType()->getPointerElementType(),
        lenPtr,
        "slice.len");
}

llvm::Value* CodeGenContext::getSliceCap(llvm::Value* slice) {
    llvm::Value* capPtr = builder.CreateStructGEP(
        slice->getType()->getPointerElementType(), 
        slice, 
        2, 
        "cap.ptr");
    return builder.CreateLoad(
        capPtr->getType()->getPointerElementType(),
        capPtr,
        "slice.cap");
}

void CodeGenContext::setSliceData(llvm::Value* slice, llvm::Value* data) {
    llvm::Value* dataPtr = builder.CreateStructGEP(
        slice->getType()->getPointerElementType(), 
        slice, 
        0);
    builder.CreateStore(data, dataPtr);
}

void CodeGenContext::setSliceLen(llvm::Value* slice, llvm::Value* len) {
    llvm::Value* lenPtr = builder.CreateStructGEP(
        slice->getType()->getPointerElementType(), 
        slice, 
        1);
    builder.CreateStore(len, lenPtr);
}

void CodeGenContext::setSliceCap(llvm::Value* slice, llvm::Value* cap) {
    llvm::Value* capPtr = builder.CreateStructGEP(
        slice->getType()->getPointerElementType(), 
        slice, 
        2);
    builder.CreateStore(cap, capPtr);
}

void CodeGenContext::generateCode(BlockAST& root) {
    if (!module) {
        module = std::make_unique<llvm::Module>("simple-lang", context);
    }

    // Set target triple if not already set
    if (module->getTargetTriple().empty()) {
        module->setTargetTriple(llvm::sys::getDefaultTargetTriple());
    }

    // Generate code for each top-level statement
    for (StmtAST* stmt : root.statements) {
        if (!stmt) continue;
        llvm::Value* val = stmt->codeGen(*this);
        if (!val) {
            std::cerr << "Failed to generate code for statement" << std::endl;
        }
    }

    // Verify the complete module
    std::string error;
    llvm::raw_string_ostream errorStream(error);
    
    if (llvm::verifyModule(*module, &errorStream)) {
        std::cerr << "Error verifying module: " << error << std::endl;
        return;
    }

    // Initialize native target
    LLVMInitializeNativeTarget();
    LLVMInitializeNativeAsmParser();
    LLVMInitializeNativeAsmPrinter();
}

void CodeGenContext::pushBlock(llvm::DIScope* debugScope) {
    std::cout << "Pushing new block" << std::endl;
    blocks.push_back(new CodeGenBlock(debugScope));
    if (debugScope) {
        currentDebugScope = debugScope;
    }
}

void CodeGenContext::popBlock() {
    std::cout << "Popping block" << std::endl;
    if (!blocks.empty()) {
        CodeGenBlock* top = blocks.back();
        blocks.pop_back();
        
        // Update debug scope
        if (!blocks.empty() && blocks.back()->debugScope) {
            currentDebugScope = blocks.back()->debugScope;
        }
        
        // Clean up the block
        delete top;
        
        // If we're popping to an empty state, make sure we don't have any dangling insert points
        if (blocks.empty()) {
            builder.ClearInsertionPoint();
        }
    }
}

void CodeGenContext::setSymbolValue(const std::string& name, llvm::Value* value) {
        std::cout << "Setting symbol: " << name << std::endl;
        blocks.back()->locals[name] = value;
}

llvm::Value* CodeGenContext::getSymbolValue(const std::string& name) {
        for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) {
            auto value = (*it)->locals.find(name);
            if (value != (*it)->locals.end()) {
                std::cout << "Found symbol: " << name << std::endl;
                return value->second;
            }
        }
        std::cerr << "Symbol not found: " << name << std::endl;
        return nullptr;
}

void CodeGenContext::dumpBlocks() const {
    std::cout << "Current block stack (size=" << blocks.size() << "):" << std::endl;
    int i = 0;
    for (auto block : blocks) {
        std::cout << "Block " << i++ << " has " << block->locals.size() << " symbols" << std::endl;
    }
}

void CodeGenContext::dumpSymbols() const {
    if (blocks.empty()) {
        std::cout << "No active blocks" << std::endl;
        return;
    }
    
    std::cout << "Current scope symbols:" << std::endl;
    for (const auto& pair : blocks.back()->locals) {
        std::cout << "  " << pair.first << std::endl;
    }
}

llvm::Type* CodeGenContext::getVectorType(unsigned width) {
    if (simdInterface) {
        return simdInterface->getVectorType(context);
    }
    // Fallback to scalar type if no SIMD interface
    return llvm::Type::getDoubleTy(context);
}

bool CodeGenContext::isSliceType(llvm::Type* type) const {
    if (auto structTy = llvm::dyn_cast<llvm::StructType>(type)) {
        return structTy == sseSliceType || structTy == avxSliceType;
    }
    return false;
}

bool CodeGenContext::isVectorType(llvm::Type* type) const {
    return type->isVectorTy() && type->getScalarType()->isDoubleTy();
}

unsigned CodeGenContext::getVectorWidth(llvm::Type* type) const {
    if (auto vecTy = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
        return vecTy->getNumElements();
    }
    return 0;
}

void CodeGenContext::declareRuntimeFunctions() {
    // Empty by default - runtime functions will be declared on demand
}

llvm::Function* createFunction(CodeGenContext& context, 
                             const std::string& name,
                             const std::vector<std::string>& args) {
    std::vector<llvm::Type*> argTypes(args.size(), 
        llvm::Type::getDoubleTy(context.getContext()));
    
    llvm::FunctionType* funcType = llvm::FunctionType::get(
        llvm::Type::getDoubleTy(context.getContext()),
        argTypes,
        false
    );

    // Check if function already exists
    if (llvm::Function* existingFunc = context.getModule()->getFunction(name)) {
        // If it exists but has wrong linkage, recreate it
        if (name == "kernel_main" && existingFunc->getLinkage() != llvm::Function::ExternalLinkage) {
            existingFunc->eraseFromParent();
        } else {
            return existingFunc;
        }
    }

    // Use external linkage for kernel_main, internal for others
    llvm::Function::LinkageTypes linkage = 
        (name == "kernel_main") ? llvm::Function::ExternalLinkage 
                               : llvm::Function::InternalLinkage;

    llvm::Function* function = llvm::Function::Create(
        funcType,
        linkage,
        name,
        context.getModule()
    );

    // Create entry block
    llvm::BasicBlock* bb = llvm::BasicBlock::Create(
        context.getContext(),
        "entry",
        function
    );
    context.getBuilder().SetInsertPoint(bb);

    return function;
}