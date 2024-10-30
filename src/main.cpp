#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <system_error>

#include "ast.hpp"
#include "codegen.hpp"
#include "parser.hpp"

#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/Program.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Target/TargetMachine.h>

extern BlockAST* programBlock;
extern int yyparse();
extern FILE* yyin;

bool CompileAndLink(const std::string& objFile, const std::string& outFile) {
    // Find clang executable
    std::string clangPath = llvm::sys::findProgramByName("clang").get();
    if (clangPath.empty()) {
        clangPath = llvm::sys::findProgramByName("clang-14").get();
    }
    if (clangPath.empty()) {
        std::cerr << "Error: Could not find clang or clang-14 in PATH" << std::endl;
        return false;
    }

    std::vector<llvm::StringRef> args;
    args.push_back(clangPath);
    args.push_back(objFile);
    args.push_back("-o");
    args.push_back(outFile);

    // Add required libraries for SIMD
    args.push_back("-mavx");
    args.push_back("-msse4.2");

    std::string ErrMsg;
    int result = llvm::sys::ExecuteAndWait(clangPath, args, llvm::None, {}, 0, 0, &ErrMsg);
    
    if (result != 0) {
        std::cerr << "Error linking: " << ErrMsg << std::endl;
        return false;
    }
    
    std::cout << "Linked using: " << clangPath << std::endl;
    return true;
}

int main(int argc, char** argv) {
    bool debug = false;
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " source.sl [-d]" << std::endl;
        return 1;
    }

    // Parse input file name without extension
    std::string inputFile(argv[1]);
    size_t lastDot = inputFile.find_last_of(".");
    std::string baseName = inputFile.substr(0, lastDot);

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

    if (debug) {
        std::cout << "\nGenerated LLVM IR:" << std::endl;
        std::cout << "==================" << std::endl;
        context.getModule()->print(llvm::outs(), nullptr);
        std::cout << "==================" << std::endl;
    }

    // Write LLVM IR to file
    std::string irFile = baseName + ".ll";
    std::error_code EC;
    llvm::raw_fd_ostream dest(irFile, EC);
    if (EC) {
        llvm::errs() << "Could not open file: " << EC.message() << "\n";
        return 1;
    }
    context.getModule()->print(dest, nullptr);
    dest.flush();
    std::cout << "LLVM IR written to '" << irFile << "'" << std::endl;

    // Generate object code
    std::string objFile = baseName + ".o";
    llvm::legacy::PassManager pass;
    llvm::raw_fd_ostream destObj(objFile, EC);
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
    std::cout << "Object code written to '" << objFile << "'" << std::endl;

    // Link executable
    std::string exeFile = baseName + ".exe";
    if (!CompileAndLink(objFile, exeFile)) {
        return 1;
    }
    std::cout << "Executable generated: " << exeFile << std::endl;

    return 0;
}