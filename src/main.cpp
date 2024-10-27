#include <iostream>
#include <fstream>
#include "codegen.hpp"
#include "ast.hpp"
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <cstring>

extern int yyparse();
extern BlockAST* programBlock;
extern FILE* yyin;
extern int yydebug;

int main(int argc, char **argv) {
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
    llvm::InitializeNativeTargetAsmParser();
    llvm::InitializeNativeTargetAsmPrinter();

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

    return 0;
}