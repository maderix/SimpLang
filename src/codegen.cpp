// src/codegen.cpp

#include "codegen.hpp"
#include "ast.hpp"
#include <iostream>
#include <llvm/IR/Verifier.h>

CodeGenContext::CodeGenContext() {
    // Initialize the LLVM context, module, and builder
    context = std::make_unique<llvm::LLVMContext>();
    module = std::make_unique<llvm::Module>("simple-lang", *context);
    builder = std::make_unique<llvm::IRBuilder<>>(*context);
}

void CodeGenContext::generateCode(BlockAST& root) {
    std::cout << "Generating code...\n";
    
    // Generate code for the root block, which includes function declarations
    if (!root.codeGen(*this)) {
        std::cerr << "Code generation failed." << std::endl;
        return;
    }
    
    // Verify the entire module for correctness
    std::string error;
    llvm::raw_string_ostream errorStream(error);
    if (llvm::verifyModule(*module, &errorStream)) {
        std::cerr << "Error verifying module: " << error << std::endl;
        return;
    }
    
    // Optionally, print the LLVM IR to stdout
    module->print(llvm::outs(), nullptr);
}
