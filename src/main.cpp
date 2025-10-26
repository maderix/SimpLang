#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstring>

#include "ast/ast.hpp"
#include "codegen.hpp"
#include "parser.hpp"
#include "logger.hpp"

#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Host.h>
#include <llvm/MC/SubtargetFeature.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Transforms/Vectorize.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/MC/TargetRegistry.h>

#ifdef USE_MLIR
#include "mlir/mlir_codegen.hpp"
#include "mlir/mlir_pipeline.hpp"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/PromoteLargeAllocaToHeap.h"
#include "ast/transforms/normalize_returns.hpp"
#endif

extern BlockAST* programBlock;
extern int yyparse();
extern FILE* yyin;

int main(int argc, char** argv) {
    bool debug = false;
    bool printIR = false;
    bool emitMLIR = false;
    bool enableTiling = true;  // MLIR tiling optimization (enabled by default)
    int tileSize = 8;  // Default optimal tile size
    bool dumpMLIRPasses = false;  // Dump MLIR at each pipeline stage
    std::string outputPath;
    std::string logLevel = "INFO";  // Default log level

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " source.sl [-o output] [-d] [--log-level LEVEL] [--print-ir] [--emit-mlir] [--enable-tiling] [--dump-mlir-passes]" << std::endl;
        std::cerr << "  --log-level LEVEL: Set logging level (ERROR, WARNING, INFO, DEBUG, TRACE)" << std::endl;
        std::cerr << "  -q, --quiet:       Equivalent to --log-level ERROR" << std::endl;
        std::cerr << "  -v, --verbose:     Equivalent to --log-level DEBUG" << std::endl;
        std::cerr << "  --print-ir:        Print LLVM IR to console" << std::endl;
#ifdef USE_MLIR
        std::cerr << "  --emit-mlir:       Emit MLIR instead of LLVM IR" << std::endl;
        std::cerr << "  --enable-tiling:   Enable loop tiling optimization for matmul (default: on)" << std::endl;
        std::cerr << "  --no-tiling:       Disable loop tiling optimization" << std::endl;
        std::cerr << "  --tile-size N:     Set tile size for matmul (default: 8)" << std::endl;
        std::cerr << "  --dump-mlir-passes: Dump MLIR IR at each pipeline stage" << std::endl;
#endif
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
        else if (strcmp(argv[i], "--log-level") == 0 && i + 1 < argc) {
            logLevel = argv[++i];
        }
        else if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--quiet") == 0) {
            logLevel = "ERROR";
        }
        else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            logLevel = "DEBUG";
        }
        else if (strcmp(argv[i], "--no-color") == 0) {
            Logger::setColorEnabled(false);
        }
        else if (strcmp(argv[i], "--print-ir") == 0) {
            printIR = true;
        }
#ifdef USE_MLIR
        else if (strcmp(argv[i], "--emit-mlir") == 0) {
            emitMLIR = true;
        }
        else if (strcmp(argv[i], "--enable-tiling") == 0) {
            enableTiling = true;
        }
        else if (strcmp(argv[i], "--no-tiling") == 0) {
            enableTiling = false;
        }
        else if (strcmp(argv[i], "--tile-size") == 0 && i + 1 < argc) {
            tileSize = std::atoi(argv[++i]);
            if (tileSize <= 0) {
                std::cerr << "Error: Invalid tile size " << tileSize << std::endl;
                return 1;
            }
        }
        else if (strcmp(argv[i], "--dump-mlir-passes") == 0) {
            dumpMLIRPasses = true;
        }
#endif
    }
    
    // Initialize logger
    Logger::setLevelFromString(logLevel);

    // Initialize LLVM
    LOG_INFO("Initializing LLVM...");
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

#ifdef USE_MLIR
    // Initialize LLVM optimization passes (required for makeOptimizingTransformer)
    mlir::initializeLLVMPasses();
#endif

    // Set input file
    LOG_INFO("Opening ", argv[1]);
    yyin = fopen(argv[1], "r");
    if (!yyin) {
        std::cerr << "Error: Failed to open " << argv[1] << std::endl;
        return 1;
    }

    LOG_INFO("Parsing...");
    if (yyparse()) {
        std::cerr << "Error parsing!" << std::endl;
        return 1;
    }

    if (!programBlock) {
        std::cerr << "Error: No program block generated!" << std::endl;
        return 1;
    }

#ifdef USE_MLIR
    if (emitMLIR) {
        LOG_INFO("MLIR mode enabled - generating MLIR...");

        // Apply return normalization pass (required for MLIR structured control flow)
        LOG_INFO("Applying return normalization pass...");
        ast::transforms::normalizeAllReturns(programBlock);

        // Create MLIR code generation context
        std::string moduleName = argv[1];
        mlir::simp::MLIRCodeGenContext mlirContext(moduleName);

        // Lower AST to MLIR Simp dialect
        mlir::ModuleOp mlirModule = mlirContext.lowerAST(programBlock);

        if (!mlirModule) {
            std::cerr << "Error: Failed to lower AST to MLIR!" << std::endl;
            return 1;
        }

        // Determine output paths
        std::string basePath;
        if (!outputPath.empty()) {
            size_t lastDot = outputPath.find_last_of('.');
            basePath = outputPath.substr(0, lastDot);
        } else {
            std::string inputPath = argv[1];
            size_t lastDot = inputPath.find_last_of('.');
            basePath = inputPath.substr(0, lastDot);
        }

        std::string mlirFile = basePath + ".mlir";
        std::string llFile = basePath + ".ll";
        std::string objFile = outputPath.empty() ? basePath + ".o" : outputPath;

        // Write initial MLIR (Simp dialect) to file for debugging
        {
            std::error_code EC;
            llvm::raw_fd_ostream dest(mlirFile, EC, llvm::sys::fs::OF_None);
            if (EC) {
                llvm::errs() << "Could not open file: " << EC.message() << "\n";
                return 1;
            }
            mlirModule.print(dest);
            dest.flush();
            std::cout << "Initial MLIR (Simp dialect) written to: " << mlirFile << std::endl;
        }

        // Print initial MLIR if requested
        if (printIR) {
            std::cout << "\n=== Initial MLIR (Simp Dialect) ===" << std::endl;
            mlirModule.print(llvm::outs());
            std::cout << "====================================\n" << std::endl;
        }

        // Create MLIR compilation pipeline
        LOG_INFO("Running MLIR lowering passes...");
        mlir::simp::MLIRCompilationPipeline pipeline(mlirModule);

        // Configure pipeline options
        pipeline.setEnableTiling(enableTiling);
        pipeline.setTileSize(tileSize);
        if (enableTiling) {
            LOG_INFO("Loop tiling optimization enabled (" + std::to_string(tileSize) + "x" + std::to_string(tileSize) + "x" + std::to_string(tileSize) + ")");
        }

        pipeline.setDumpIntermediateIR(dumpMLIRPasses);
        if (dumpMLIRPasses) {
            LOG_INFO("MLIR intermediate IR dumping enabled");
        }

        // Run progressive lowering passes (Simp → LLVM dialect)
        if (!pipeline.runPasses()) {
            std::cerr << "Error: MLIR lowering passes failed!" << std::endl;
            return 1;
        }

        // Print lowered MLIR (LLVM dialect) if requested
        if (printIR) {
            std::cout << "\n=== Lowered MLIR (LLVM Dialect) ===" << std::endl;
            pipeline.getModule().print(llvm::outs());
            std::cout << "====================================\n" << std::endl;
        }

        // Translate MLIR LLVM dialect to LLVM IR
        LOG_INFO("Translating MLIR to LLVM IR...");
        llvm::LLVMContext llvmContext;
        auto llvmModule = pipeline.translateToLLVMIR(llvmContext);

        if (!llvmModule) {
            std::cerr << "Error: Failed to translate MLIR to LLVM IR!" << std::endl;
            return 1;
        }

        // Write LLVM IR to file
        {
            std::error_code EC;
            llvm::raw_fd_ostream dest(llFile, EC, llvm::sys::fs::OF_None);
            if (EC) {
                llvm::errs() << "Could not open file: " << EC.message() << "\n";
                return 1;
            }
            llvmModule->print(dest, nullptr);
            dest.flush();
            std::cout << "LLVM IR written to: " << llFile << std::endl;
        }

        // Print unoptimized LLVM IR if requested
        if (printIR) {
            std::cout << "\n=== Generated LLVM IR (Before Optimization) ===" << std::endl;
            llvmModule->print(llvm::outs(), nullptr);
            std::cout << "==================================================\n" << std::endl;
        }

        // Generate object code using existing infrastructure
        // We need to create a target machine for the LLVM module
        LOG_INFO("Generating object code...");

        std::string targetTriple = llvm::sys::getDefaultTargetTriple();
        llvmModule->setTargetTriple(targetTriple);

        std::string error;
        auto target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
        if (!target) {
            llvm::errs() << error;
            return 1;
        }

        // Use native CPU features for maximum performance
        std::string cpu = llvm::sys::getHostCPUName().str();
        llvm::StringMap<bool> featureMap;
        llvm::sys::getHostCPUFeatures(featureMap);
        llvm::SubtargetFeatures subtargetFeatures;
        for (auto &feature : featureMap) {
            subtargetFeatures.AddFeature(feature.first(), feature.second);
        }
        std::string features = subtargetFeatures.getString();

        llvm::TargetOptions opt;
        auto rm = llvm::Optional<llvm::Reloc::Model>();
        auto targetMachine = target->createTargetMachine(targetTriple, cpu, features, opt, rm);

        llvmModule->setDataLayout(targetMachine->createDataLayout());

        // RUN LLVM OPTIMIZATION PASSES (Critical for performance!)
        // Use MLIR's optimization transformer (same as Toy tutorial Ch6)
        LOG_INFO("Running LLVM optimization passes (O3)...");
        auto optPipeline = mlir::makeOptimizingTransformer(
            /*optLevel=*/3,        // O3 optimization (enables loop vectorization!)
            /*sizeLevel=*/0,       // Optimize for speed, not size
            /*targetMachine=*/targetMachine);

        if (auto err = optPipeline(llvmModule.get())) {
            llvm::errs() << "Failed to optimize LLVM IR: " << err << "\n";
            return 1;
        }

        // AFTER O3: Promote large stack allocations to heap
        // This must run AFTER optimization because O3 can recreate large allocas
        LOG_INFO("Promoting large stack allocations to heap...");
        {
            llvm::legacy::PassManager heapPromotionPM;
            heapPromotionPM.add(llvm::createPromoteLargeAllocaToHeapPass());
            heapPromotionPM.run(*llvmModule);
        }

        // Print optimized IR if requested
        if (printIR) {
            std::cout << "\n=== Optimized LLVM IR (After O3) ===" << std::endl;
            llvmModule->print(llvm::outs(), nullptr);
            std::cout << "=========================================\n" << std::endl;
        }

        // Generate object file
        std::error_code EC;
        llvm::raw_fd_ostream destObj(objFile, EC, llvm::sys::fs::OF_None);
        if (EC) {
            llvm::errs() << "Could not open file: " << EC.message() << "\n";
            return 1;
        }

        llvm::legacy::PassManager pass;

        // Run heap promotion AGAIN before codegen
        // Some late optimization passes recreate large allocas, so we promote them again
        LOG_INFO("Final heap promotion pass before codegen...");
        pass.add(llvm::createPromoteLargeAllocaToHeapPass());

        auto fileType = llvm::CGFT_ObjectFile;

        if (targetMachine->addPassesToEmitFile(pass, destObj, nullptr, fileType)) {
            llvm::errs() << "TargetMachine can't emit a file of this type\n";
            return 1;
        }

        pass.run(*llvmModule);
        destObj.flush();
        std::cout << "Object code written to: " << objFile << std::endl;

        return 0;
    }
#endif

    LOG_INFO("Creating CodeGen context...");
    CodeGenContext context;

    // Generate code
    LOG_INFO("Generating code...");
    try {
        context.generateCode(*programBlock);
    } catch (const std::exception& e) {
        std::cerr << "Exception during code generation: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception during code generation!" << std::endl;
        return 1;
    }

    // Print the generated LLVM IR if requested
    if (printIR) {
        std::cout << "\nGenerated LLVM IR:" << std::endl;
        std::cout << "==================" << std::endl;
        context.getModule()->print(llvm::outs(), nullptr);
        std::cout << "==================" << std::endl;
    }

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

    // Run module-level optimization passes including vectorization
    llvm::legacy::PassManager modulePM;

    // FIRST: Promote large stack allocations to heap
    // This must run BEFORE any optimization to fix MLIR's stack allocation bug
    modulePM.add(llvm::createPromoteLargeAllocaToHeapPass());

    // Add target transform info for vectorizer
    modulePM.add(llvm::createTargetTransformInfoWrapperPass(context.getTargetMachine()->getTargetIRAnalysis()));

    // Add vectorization passes directly
    modulePM.add(llvm::createLoopVectorizePass());        // Loop vectorizer
    modulePM.add(llvm::createSLPVectorizerPass());        // SLP vectorizer

    // Run the optimization passes
    modulePM.run(*context.getModule());

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