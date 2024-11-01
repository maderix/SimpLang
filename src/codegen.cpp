#include "codegen.hpp"
#include "ast.hpp"
#include <llvm/IR/Verifier.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/Host.h>
#include <llvm/IR/LegacyPassManager.h>
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

    // Function Pass Manager for optimizations
    fpm = std::make_unique<llvm::legacy::FunctionPassManager>(module.get());
    fpm->doInitialization();

    // Initialize slice types
    sseSliceType = nullptr;
    avxSliceType = nullptr;
    declareRuntimeFunctions();
}

CodeGenContext::~CodeGenContext() {
    // Clean up blocks
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
    llvm::Type* sliceType = getSliceType(type);
    llvm::Value* slice = builder.CreateAlloca(sliceType, nullptr, "slice");
    
    // Initialize to zero
    builder.CreateMemSet(slice, 
        llvm::ConstantInt::get(llvm::Type::getInt8Ty(context), 0),
        module->getDataLayout().getTypeAllocSize(sliceType),
        llvm::MaybeAlign(),
        false);
    
    return slice;
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
    std::cout << "Generating code..." << std::endl;
    root.codeGen(*this);
    llvm::verifyModule(*module, &llvm::errs());
    fpm->doFinalization();
    std::cout << "Code generation complete." << std::endl;
}

void CodeGenContext::pushBlock() {
    blocks.push_back(new CodeGenBlock());
}

void CodeGenContext::popBlock() {
    CodeGenBlock* top = blocks.back();
    blocks.pop_back();
    delete top;
}

void CodeGenContext::setSymbolValue(const std::string& name, llvm::Value* value) {
    blocks.back()->locals[name] = value;
}

llvm::Value* CodeGenContext::getSymbolValue(const std::string& name) {
    for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) {
        auto value = (*it)->locals.find(name);
        if (value != (*it)->locals.end()) {
            return value->second;
        }
    }
    return nullptr;
}

llvm::Type* CodeGenContext::getVectorType(unsigned width) {
    return llvm::VectorType::get(getDoubleType(), width, false);
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
    auto& context = getContext();
    auto module = getModule();
    
    // Get base types
    auto i8PtrTy = llvm::Type::getInt8PtrTy(context);
    auto i64Ty = llvm::Type::getInt64Ty(context);
    auto voidTy = llvm::Type::getVoidTy(context);
    
    // Declare slice types if not already declared
    if (!sseSliceType) {
        sseSliceType = llvm::StructType::create(context, "sse_slice_t");
        sseSliceType->setBody({
            getVectorType(4)->getPointerTo(),
            i64Ty,  // len
            i64Ty   // cap
        });
    }
    
    if (!avxSliceType) {
        avxSliceType = llvm::StructType::create(context, "avx_slice_t");
        avxSliceType->setBody({
            getVectorType(8)->getPointerTo(),
            i64Ty,  // len
            i64Ty   // cap
        });
    }
    
    // Declare slice functions
    llvm::FunctionType* make_slice_ty = llvm::FunctionType::get(
        sseSliceType->getPointerTo(),
        {i64Ty},
        false
    );
    
    module->getOrInsertFunction("make_sse_slice", make_slice_ty);
    
    make_slice_ty = llvm::FunctionType::get(
        avxSliceType->getPointerTo(),
        {i64Ty},
        false
    );
    
    module->getOrInsertFunction("make_avx_slice", make_slice_ty);
    
    // Declare slice access functions
    llvm::FunctionType* get_sse_ty = llvm::FunctionType::get(
        getVectorType(4),
        {sseSliceType->getPointerTo(), i64Ty},
        false
    );
    
    module->getOrInsertFunction("slice_get_sse", get_sse_ty);
    
    llvm::FunctionType* get_avx_ty = llvm::FunctionType::get(
        getVectorType(8),
        {avxSliceType->getPointerTo(), i64Ty},
        false
    );
    
    module->getOrInsertFunction("slice_get_avx", get_avx_ty);

    llvm::FunctionType* set_sse_ty = llvm::FunctionType::get(
    voidTy,
    {sseSliceType->getPointerTo(), i64Ty, getVectorType(4)},
    false
    );

    module->getOrInsertFunction("slice_set_sse", set_sse_ty);

    llvm::FunctionType* set_avx_ty = llvm::FunctionType::get(
        voidTy,
        {avxSliceType->getPointerTo(), i64Ty, getVectorType(8)},
        false
    );

    module->getOrInsertFunction("slice_set_avx", set_avx_ty);
}