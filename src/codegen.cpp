#include "logger.hpp"
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
    
    // Enable SIMD by default
    simd_enabled = true;
    
    // Initialize target machine first
    targetTriple = llvm::sys::getDefaultTargetTriple();
    module->setTargetTriple(targetTriple);
    
    std::string error;
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
    if (!target) {
        std::cerr << "Target lookup failed: " << error << std::endl;
        return;
    }

    llvm::TargetOptions opt;
    auto RM = llvm::Optional<llvm::Reloc::Model>();
    targetMachine = std::unique_ptr<llvm::TargetMachine>(
        target->createTargetMachine(
            targetTriple,
            "generic",  // CPU
            "+avx512f,+avx512dq",  // Enable AVX-512 features
            opt,
            RM
        )
    );

    if (!targetMachine) {
        std::cerr << "Could not create target machine" << std::endl;
        return;
    }

    // Set data layout before initializing types
    module->setDataLayout(targetMachine->createDataLayout());
    
    // Initialize SIMD interface using the factory function
    simdInterface.reset(createSIMDInterface("avx512"));
    
    // Now initialize runtime functions
    initializeRuntimeFunctions();
    
    // Initialize optimization passes
    fpm = std::make_unique<llvm::legacy::FunctionPassManager>(module.get());
    fpm->doInitialization();
}

CodeGenContext::~CodeGenContext() {
    for (auto block : blocks) {
        delete block;
    }
    blocks.clear();
}


void CodeGenContext::initializeRuntimeFunctions() {
    // Always initialize basic runtime functions
    initializeMallocFree();
    
    // Always declare error function
    std::vector<llvm::Type*> errorArgs = {llvm::Type::getInt8PtrTy(context)};
    llvm::FunctionType* errorType = llvm::FunctionType::get(
        llvm::Type::getVoidTy(context),
        errorArgs,
        false
    );
    errorFunc = llvm::Function::Create(
        errorType,
        llvm::Function::ExternalLinkage,
        "error",
        module.get()
    );
    
    // Initialize SIMD functions only if enabled
    if (simd_enabled) {
        initializeSliceTypes();
        initializeSIMDFunctions();
    }
}

void CodeGenContext::initializeSliceTypes() {
    // Get vector types for SSE (2 doubles) and AVX (8 doubles)
    auto sseVecTy = llvm::VectorType::get(llvm::Type::getDoubleTy(context), 2, false);
    auto avxVecTy = llvm::VectorType::get(llvm::Type::getDoubleTy(context), 8, false);

    // Create slice types with vector pointers
    std::vector<llvm::Type*> sseFields = {
        llvm::PointerType::get(sseVecTy, 0),  // sse_vector_t*
        llvm::Type::getInt64Ty(context),      // size_t size
        llvm::Type::getInt64Ty(context)       // size_t capacity
    };
    sseSliceType = llvm::StructType::create(context, sseFields, "SSESlice");

    std::vector<llvm::Type*> avxFields = {
        llvm::PointerType::get(avxVecTy, 0),  // avx_vector_t*
        llvm::Type::getInt64Ty(context),      // size_t size
        llvm::Type::getInt64Ty(context)       // size_t capacity
    };
    avxSliceType = llvm::StructType::create(context, avxFields, "AVXSlice");
}

llvm::Type* CodeGenContext::getSliceType(SliceType type) {
    switch (type) {
        case SliceType::SSE_SLICE: 
            if (!sseSliceType) {
                std::cerr << "SSESlice type not initialized" << std::endl;
                return nullptr;
            }
            return sseSliceType;
        case SliceType::AVX_SLICE:
            if (!avxSliceType) {
                std::cerr << "AVXSlice type not initialized" << std::endl;
                return nullptr;
            }
            return avxSliceType;
        default:
            std::cerr << "Unknown slice type" << std::endl;
            return nullptr;
    }
}

llvm::Value* CodeGenContext::createSlice(SliceType type, llvm::Value* len) {
    std::cout << "Creating slice of type " << (type == SliceType::SSE_SLICE ? "SSE" : "AVX") << std::endl;
    
    // Convert length to i64 if needed
    if (len->getType()->isDoubleTy()) {
        len = builder.CreateFPToSI(len, builder.getInt64Ty(), "len.conv");
    }

    // Get the appropriate make function
    const char* funcName = (type == SliceType::SSE_SLICE) ? "make_sse_slice" : "make_avx_slice";
    llvm::Function* makeFunc = module->getFunction(funcName);
    
    if (!makeFunc) {
        std::cerr << "Error: Make function " << funcName << " not found" << std::endl;
        return nullptr;
    }

    // Create the call
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
    if (!errorFunc) {
        std::cerr << "Error function not initialized" << std::endl;
        return;
    }

    // Create a global string constant for the error message
    llvm::Constant* strConstant = llvm::ConstantDataArray::getString(
        context, 
        message
    );
    
    // Create a global variable to hold the string
    auto global = new llvm::GlobalVariable(
        *module,
        strConstant->getType(),
        true,
        llvm::GlobalValue::PrivateLinkage,
        strConstant,
        ".str"
    );
    
    // Get pointer to the start of the string
    std::vector<llvm::Value*> indices = {
        llvm::ConstantInt::get(context, llvm::APInt(64, 0)),
        llvm::ConstantInt::get(context, llvm::APInt(64, 0))
    };
    llvm::Value* strPtr = builder.CreateInBoundsGEP(
        global->getValueType(),
        global,
        indices,
        "str"
    );
    
    // Call error function
    builder.CreateCall(errorFunc, {strPtr});
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

    // Initialize target machine if not already done
    if (!targetMachine) {
        std::string error;
        const llvm::Target* target = llvm::TargetRegistry::lookupTarget(
            module->getTargetTriple(), error);
        
        if (!target) {
            std::cerr << "Target lookup failed: " << error << std::endl;
            return;
        }

        llvm::TargetOptions opt;
        auto RM = llvm::Optional<llvm::Reloc::Model>();
        targetMachine = std::unique_ptr<llvm::TargetMachine>(
            target->createTargetMachine(
                module->getTargetTriple(),
                "generic",  // CPU
                "",        // Features
                opt,
                RM
            )
        );

        if (!targetMachine) {
            std::cerr << "Could not create target machine" << std::endl;
            return;
        }

        // Set data layout
        module->setDataLayout(targetMachine->createDataLayout());
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
    LOG_TRACE("Pushing new block");
    blocks.push_back(new CodeGenBlock(debugScope));
    if (debugScope) {
        currentDebugScope = debugScope;
    }
}

void CodeGenContext::popBlock() {
    LOG_TRACE("Popping block");
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
        LOG_TRACE("Setting symbol: ", name);
        blocks.back()->locals[name] = value;
}

llvm::Value* CodeGenContext::getSymbolValue(const std::string& name) {
        for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) {
            auto value = (*it)->locals.find(name);
            if (value != (*it)->locals.end()) {
                LOG_TRACE("Found symbol: ", name);
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

llvm::Function* CodeGenContext::createFunction(const std::string& name, 
                                             const std::vector<std::pair<std::string, llvm::Type*>>& args) {
    std::vector<llvm::Type*> argTypes;
    
    // Special case for kernel_main
    if (name == "kernel_main") {
        // Use slice types directly for kernel_main
        argTypes = {
            sseSliceType->getPointerTo(),  // out_sse: SSESlice*
            avxSliceType->getPointerTo()   // out_avx: AVXSlice*
        };
    } else {
        // For other functions, use the provided types
        for (const auto& arg : args) {
            argTypes.push_back(arg.second);
        }
    }

    llvm::FunctionType* funcType = llvm::FunctionType::get(
        llvm::Type::getDoubleTy(context),  // Return type is always double
        argTypes,
        false  // Not variadic
    );

    llvm::Function* function = llvm::Function::Create(
        funcType,
        llvm::Function::ExternalLinkage,
        name,
        module.get()
    );

    // Set argument names
    unsigned idx = 0;
    for (auto& arg : function->args()) {
        if (idx < args.size()) {
            arg.setName(args[idx].first);
        }
        idx++;
    }

    return function;
}

void CodeGenContext::initializeMallocFree() {
    // Malloc function
    std::vector<llvm::Type*> mallocArgs = {llvm::Type::getInt64Ty(context)};
    llvm::FunctionType* mallocType = llvm::FunctionType::get(
        llvm::Type::getInt8PtrTy(context),
        mallocArgs,
        false
    );
    mallocFunc = llvm::Function::Create(
        mallocType,
        llvm::Function::ExternalLinkage,
        "malloc",
        module.get()
    );

    // Free function
    std::vector<llvm::Type*> freeArgs = {llvm::Type::getInt8PtrTy(context)};
    llvm::FunctionType* freeType = llvm::FunctionType::get(
        llvm::Type::getVoidTy(context),
        freeArgs,
        false
    );
    freeFunc = llvm::Function::Create(
        freeType,
        llvm::Function::ExternalLinkage,
        "free",
        module.get()
    );
}

void CodeGenContext::initializeSIMDFunctions() {
    // Get vector types (already defined in initializeSliceTypes)
    auto sseVecTy = llvm::VectorType::get(llvm::Type::getDoubleTy(context), 2, false);
    auto avxVecTy = llvm::VectorType::get(llvm::Type::getDoubleTy(context), 8, false);

    // Declare make functions
    auto makeSseFuncTy = llvm::FunctionType::get(
        llvm::PointerType::get(sseSliceType, 0),  // Returns SSESlice*
        {llvm::Type::getInt64Ty(context)},        // Takes size_t argument
        false
    );

    auto makeAvxFuncTy = llvm::FunctionType::get(
        llvm::PointerType::get(avxSliceType, 0),  // Returns AVXSlice*
        {llvm::Type::getInt64Ty(context)},        // Takes size_t argument
        false
    );

    // Create function declarations
    llvm::Function::Create(makeSseFuncTy, llvm::Function::ExternalLinkage,
                          "make_sse_slice", module.get());
    llvm::Function::Create(makeAvxFuncTy, llvm::Function::ExternalLinkage,
                          "make_avx_slice", module.get());

    // Declare slice set functions
    auto setSSEFuncTy = llvm::FunctionType::get(
        llvm::Type::getVoidTy(context),
        {
            llvm::PointerType::get(sseSliceType, 0),  // SSESlice*
            llvm::Type::getInt64Ty(context),          // size_t index
            sseVecTy                                  // sse_vector_t value
        },
        false
    );

    auto setAVXFuncTy = llvm::FunctionType::get(
        llvm::Type::getVoidTy(context),
        {
            llvm::PointerType::get(avxSliceType, 0),  // AVXSlice*
            llvm::Type::getInt64Ty(context),          // size_t index
            avxVecTy                                  // avx_vector_t value
        },
        false
    );

    llvm::Function::Create(setSSEFuncTy, llvm::Function::ExternalLinkage,
                          "slice_set_sse", module.get());
    llvm::Function::Create(setAVXFuncTy, llvm::Function::ExternalLinkage,
                          "slice_set_avx", module.get());

    // Add declarations for slice_get functions
    auto getSSEFuncTy = llvm::FunctionType::get(
        sseVecTy,  // Returns sse_vector_t
        {
            llvm::PointerType::get(sseSliceType, 0),  // SSESlice*
            llvm::Type::getInt64Ty(context)           // size_t index
        },
        false
    );

    auto getAVXFuncTy = llvm::FunctionType::get(
        avxVecTy,  // Returns avx_vector_t
        {
            llvm::PointerType::get(avxSliceType, 0),  // AVXSlice*
            llvm::Type::getInt64Ty(context)           // size_t index
        },
        false
    );

    // Create function declarations for get operations
    llvm::Function::Create(getSSEFuncTy, llvm::Function::ExternalLinkage,
                          "slice_get_sse", module.get());
    llvm::Function::Create(getAVXFuncTy, llvm::Function::ExternalLinkage,
                          "slice_get_avx", module.get());
}

llvm::TargetMachine* CodeGenContext::getTargetMachine() {
    return targetMachine.get();
}

llvm::Value* CodeGenContext::createAVXVector(const std::vector<double>& values) {
    if (values.size() != 8) {
        emitError("AVX vector must have exactly 8 elements");
        return nullptr;
    }

    // Create vector constant from doubles
    std::vector<llvm::Constant*> constants;
    for (double val : values) {
        constants.push_back(llvm::ConstantFP::get(getDoubleType(), val));
    }
    
    // Create vector type for AVX (8 doubles)
    auto vectorType = llvm::VectorType::get(getDoubleType(), 8, false);
    return llvm::ConstantVector::get(constants);
}

void CodeGenContext::emitSliceSet(llvm::Value* slice, llvm::Value* index, llvm::Value* value) {
    if (!isVectorType(value->getType())) {
        emitError("Expected vector type for slice set operation");
        return;
    }

    unsigned width = getVectorWidth(value->getType());
    
    // Debug output
    printf("Emitting slice set for vector width %u\n", width);
    
    if (width == 2) {
        // SSE vector (2 doubles)
        auto func = module->getFunction("slice_set_sse");
        if (!func) {
            emitError("slice_set_sse function not found");
            return;
        }
        builder.CreateCall(func, {slice, index, value});
    } 
    else if (width == 8) {
        // AVX vector (8 doubles)
        auto func = module->getFunction("slice_set_avx");
        if (!func) {
            emitError("slice_set_avx function not found");
            return;
        }
        builder.CreateCall(func, {slice, index, value});
    }
    else {
        emitError("Unsupported vector width for slice set operation");
    }
}