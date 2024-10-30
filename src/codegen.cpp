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
}

llvm::LLVMContext& CodeGenContext::getContext() {
    return context;
}

llvm::Module* CodeGenContext::getModule() {
    return module.get();
}

llvm::IRBuilder<>& CodeGenContext::getBuilder() {
    return builder;
}

llvm::Type* CodeGenContext::getDoubleType() {
    return llvm::Type::getDoubleTy(context);
}

llvm::Function* CodeGenContext::currentFunction() {
    return builder.GetInsertBlock()->getParent();
}

llvm::TargetMachine* CodeGenContext::getTargetMachine() {
    return targetMachine.get();
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

llvm::Type* CodeGenContext::getCurrentFunctionType(const std::string& name) {
    if (name.find("sse") != std::string::npos) {
        return getVectorType(4);
    } else if (name.find("avx") != std::string::npos) {
        return getVectorType(8);
    }
    return getDoubleType();
}