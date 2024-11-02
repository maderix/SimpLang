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
    std::string outputPath;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " source.sl [-o output] [-d]" << std::endl;
        return 1;
    }

    // Parse command line arguments
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--debug") == 0) {
            debug = true;
            yydebug = 1;
        }
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            outputPath = argv[++i];
        }
    }

    // Initialize LLVM
    std::cout << "Initializing LLVM..." << std::endl;
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

    if (!programBlock) {
        std::cerr << "Error: No program block generated!" << std::endl;
        return 1;
    }

    std::cout << "Creating CodeGen context..." << std::endl;
    CodeGenContext context;

    // Generate code
    std::cout << "Generating code..." << std::endl;
    try {
        context.generateCode(*programBlock);
    } catch (const std::exception& e) {
        std::cerr << "Exception during code generation: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception during code generation!" << std::endl;
        return 1;
    }

    // Print the generated LLVM IR
    std::cout << "\nGenerated LLVM IR:" << std::endl;
    std::cout << "==================" << std::endl;
    context.getModule()->print(llvm::outs(), nullptr);
    std::cout << "==================" << std::endl;

    // Determine output paths
    std::string objFile, llFile;
    if (!outputPath.empty()) {
        // Use provided output path
        size_t lastDot = outputPath.find_last_of('.');
        std::string basePath = outputPath.substr(0, lastDot);
        objFile = outputPath;
        llFile = basePath + ".ll";
    } else {
        // Use input path as base
        std::string inputPath = argv[1];
        size_t lastDot = inputPath.find_last_of('.');
        std::string basePath = inputPath.substr(0, lastDot);
        objFile = basePath + ".o";
        llFile = basePath + ".ll";
    }

    // Write LLVM IR to file
    std::error_code EC;
    llvm::raw_fd_ostream dest(llFile, EC, llvm::sys::fs::OF_None);
    if (EC) {
        llvm::errs() << "Could not open file: " << EC.message() << "\n";
        return 1;
    }
    context.getModule()->print(dest, nullptr);
    dest.flush();
    std::cout << "LLVM IR written to: " << llFile << std::endl;

    // Generate object code
    llvm::legacy::PassManager pass;
    llvm::raw_fd_ostream destObj(objFile, EC, llvm::sys::fs::OF_None);
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
    std::cout << "Object code written to: " << objFile << std::endl;

    return 0;
}