/*
 * Lum DSL Compiler
 * Main entry point for the lum command-line tool
 *
 * Usage:
 *   lum schedule.lum --emit-transform -o output.mlir
 *   lum schedule.lum input.mlir -o output.mlir  (future)
 */

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>

#include "ast/base.hpp"
#include "ast/pattern.hpp"
#include "ast/schedule.hpp"
#include "ast/transform.hpp"
#include "codegen/transform_emitter.hpp"

// Flex/Bison declarations
extern FILE* yyin;
extern int yyparse();
extern lum::Program* programRoot;
extern int yylineno;

void printUsage(const char* progName) {
    std::cerr << "Lum DSL Compiler - Transform Dialect Generator\n\n";
    std::cerr << "Usage:\n";
    std::cerr << "  " << progName << " <schedule.lum> --emit-transform [-o output.mlir]\n";
    std::cerr << "  " << progName << " <schedule.lum> --dump-ast\n";
    std::cerr << "\n";
    std::cerr << "Options:\n";
    std::cerr << "  --emit-transform  Generate MLIR Transform Dialect output\n";
    std::cerr << "  --dump-ast        Print the parsed AST (for debugging)\n";
    std::cerr << "  -o <file>         Output file (default: stdout)\n";
    std::cerr << "  --help, -h        Show this help message\n";
    std::cerr << "\n";
    std::cerr << "Example:\n";
    std::cerr << "  " << progName << " matmul.lum --emit-transform -o matmul.mlir\n";
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string inputFile;
    std::string outputFile;
    bool emitTransform = false;
    bool dumpAST = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--emit-transform") {
            emitTransform = true;
        } else if (arg == "--dump-ast") {
            dumpAST = true;
        } else if (arg == "-o" && i + 1 < argc) {
            outputFile = argv[++i];
        } else if (arg[0] != '-') {
            if (inputFile.empty()) {
                inputFile = arg;
            } else {
                std::cerr << "Error: Multiple input files not supported\n";
                return 1;
            }
        } else {
            std::cerr << "Error: Unknown option '" << arg << "'\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    // Validate arguments
    if (inputFile.empty()) {
        std::cerr << "Error: No input file specified\n";
        printUsage(argv[0]);
        return 1;
    }

    if (!emitTransform && !dumpAST) {
        std::cerr << "Error: Specify --emit-transform or --dump-ast\n";
        printUsage(argv[0]);
        return 1;
    }

    // Open input file
    FILE* file = fopen(inputFile.c_str(), "r");
    if (!file) {
        std::cerr << "Error: Cannot open input file '" << inputFile << "'\n";
        return 1;
    }

    // Set input for lexer
    yyin = file;

    // Parse
    std::cerr << "Parsing " << inputFile << "...\n";
    int parseResult = yyparse();
    fclose(file);

    if (parseResult != 0) {
        std::cerr << "Error: Parsing failed\n";
        return 1;
    }

    if (!programRoot) {
        std::cerr << "Error: No program produced\n";
        return 1;
    }

    std::cerr << "Parsed successfully!\n";

    // Handle --dump-ast
    if (dumpAST) {
        programRoot->dump(std::cout);
        delete programRoot;
        return 0;
    }

    // Handle --emit-transform
    if (emitTransform) {
        lum::TransformEmitter emitter;
        std::string output = emitter.emit(*programRoot);

        // Check for errors
        if (emitter.hasErrors()) {
            std::cerr << "Errors during code generation:\n";
            for (const auto& err : emitter.getErrors()) {
                std::cerr << "  - " << err << "\n";
            }
            delete programRoot;
            return 1;
        }

        // Write output
        if (outputFile.empty()) {
            std::cout << output;
        } else {
            std::ofstream outStream(outputFile);
            if (!outStream) {
                std::cerr << "Error: Cannot open output file '" << outputFile << "'\n";
                delete programRoot;
                return 1;
            }
            outStream << output;
            std::cerr << "Wrote " << outputFile << "\n";
        }
    }

    delete programRoot;
    return 0;
}
