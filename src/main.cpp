// main.cpp

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstring>

#include "ast.hpp"
#include "codegen.hpp"
#include "parser.hpp"

#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Host.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Target/TargetMachine.h>

extern BlockAST* programBlock;
extern int yyparse();
extern FILE* yyin;

int main(int argc, char** argv) {
    bool debug = false;
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " source.sl [-d]" << std::endl;
        return 1;
    }

    // Check for debug flag
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--debug") == 0) {
            debug = true;
            yydebug = 1;
        }
    }

    // Initialize LLVM
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    // Set input file
    std::cout << "Opening " << argv[1] << std::endl;
    yyin = fopen(argv[1], "r");
    if (!yyin) {
        std::cerr << "Error: Failed to open " << argv[1] << std::endl;
        return 1;
    }

    std::cout << "Parsing..." << std::endl;
    if (yyparse()) {
        std::cerr << "Error parsing!" << std::endl;
        return 1;
    }

    // Generate code
    CodeGenContext context;
    context.generateCode(*programBlock);

    // Print the generated LLVM IR
    std::cout << "\nGenerated LLVM IR:" << std::endl;
    std::cout << "==================" << std::endl;
    context.getModule()->print(llvm::outs(), nullptr);
    std::cout << "==================" << std::endl;

    // Write LLVM IR to file
    std::error_code EC;
    llvm::raw_fd_ostream dest("output.ll", EC);
    if (EC) {
        llvm::errs() << "Could not open file: " << EC.message() << "\n";
        return 1;
    }
    context.getModule()->print(dest, nullptr);
    dest.flush();

    // Generate object code
    llvm::legacy::PassManager pass;
    llvm::raw_fd_ostream destObj("output.o", EC);
    if (EC) {
        llvm::errs() << "Could not open file: " << EC.message() << "\n";
        return 1;
    }

    llvm::TargetMachine* targetMachine = context.getTargetMachine();
    auto FileType = llvm::CGFT_ObjectFile;

    if (targetMachine->addPassesToEmitFile(pass, destObj, nullptr, FileType)) {
        llvm::errs() << "TargetMachine can't emit a file of this type\n";
        return 1;
    }

    pass.run(*context.getModule());
    destObj.flush();

    std::cout << "Object code generated in 'output.o'" << std::endl;

    return 0;
}
