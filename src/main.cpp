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
#include <llvm/Support/CommandLine.h>
#include <llvm/MC/SubtargetFeature.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Transforms/Vectorize.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Analysis/CGSCCPassManager.h>

#ifdef USE_MLIR
#include "mlir/mlir_codegen.hpp"
#include "mlir/mlir_pipeline.hpp"
#include "mlir/Passes.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/PromoteLargeAllocaToHeap.h"
#include "mlir/passes/VNNIPass.h"
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
    int tileSize = 16;  // Optimal tile size: 39% of NumPy @512x512, 20% @1024x1024
    bool enableHierarchicalTiling = false;  // Multi-level cache-aware tiling (experimental)
    bool enableOpenMP = false;  // OpenMP parallelization (disabled by default)
    bool enableO3 = true;  // LLVM O3 optimization (enabled by default)
    bool enablePrefetch = true;  // Prefetch insertion for memory latency hiding (enabled by default)
    bool llvmVectorize = false;  // Skip MLIR vectorization, let LLVM handle it (better for INT8/INT4)
    bool dumpMLIRPasses = false;  // Dump MLIR at each pipeline stage
    bool emitGPU = false;  // Enable GPU code generation (requires USE_CUDA)
    std::string cudaArch = "sm_80";  // CUDA compute capability (default: A100)
    std::string gpuMatMulStrategy = "cublas";  // GPU matmul strategy
    std::string outputPath;
    std::string logLevel = "INFO";  // Default log level
    std::string targetArch = "";  // Target architecture (empty = native)

    // Helper function to print help
    auto printHelp = [&]() {
        std::cout << "SimpLang Compiler - DSL for SIMD Hardware Optimization\n" << std::endl;
        std::cout << "Usage: " << argv[0] << " <source.sl> [OPTIONS]\n" << std::endl;

        std::cout << "Basic Options:" << std::endl;
        std::cout << "  -h, --help         Show this help message" << std::endl;
        std::cout << "  -o <file>          Output file path (default: <input>.o)" << std::endl;
        std::cout << "  -d, --debug        Enable parser debug mode" << std::endl;
        std::cout << "  --print-ir         Print LLVM IR to console" << std::endl;
        std::cout << "  --target <arch>    Target architecture (x86_64, aarch64, armv7)" << std::endl;
        std::cout << std::endl;

        std::cout << "Logging Options:" << std::endl;
        std::cout << "  --log-level LEVEL  Set logging level (ERROR, WARNING, INFO, DEBUG, TRACE)" << std::endl;
        std::cout << "  -q, --quiet        Quiet mode (--log-level ERROR)" << std::endl;
        std::cout << "  -v, --verbose      Verbose mode (--log-level DEBUG)" << std::endl;
        std::cout << "  --no-color         Disable colored output" << std::endl;
        std::cout << std::endl;

#ifdef USE_MLIR
        std::cout << "MLIR Backend Options:" << std::endl;
        std::cout << "  --emit-mlir        Use MLIR backend (required for tensor ops, transformers)" << std::endl;
        std::cout << "  --dump-mlir-passes Dump MLIR IR at each pipeline stage" << std::endl;
        std::cout << std::endl;

        std::cout << "Optimization Options (MLIR):" << std::endl;
        std::cout << "  --enable-tiling    Enable loop tiling for matmul (default: ON)" << std::endl;
        std::cout << "  --no-tiling        Disable loop tiling optimization" << std::endl;
        std::cout << "  --tile-size N      Set tile size for matmul (default: 16)" << std::endl;
        std::cout << "  --hierarchical-tiling  Multi-level cache-aware tiling (L1/L2/L3)" << std::endl;
        std::cout << "  --enable-openmp    Enable OpenMP parallelization (multi-threading)" << std::endl;
        std::cout << "  --no-prefetch      Disable prefetch insertion (memory latency hiding)" << std::endl;
        std::cout << "  --llvm-vectorize   Use LLVM vectorization instead of MLIR (better for INT8/INT4)" << std::endl;
        std::cout << "  --no-opt           Disable LLVM O3 optimization (faster compilation)" << std::endl;
        std::cout << std::endl;

#ifdef USE_CUDA
        std::cout << "GPU Backend Options (requires CUDA):" << std::endl;
        std::cout << "  --emit-gpu         Enable GPU code generation (CUDA)" << std::endl;
        std::cout << "  --cuda-arch ARCH   Target CUDA architecture (default: sm_80 for A100)" << std::endl;
        std::cout << "                     sm_70 = V100, sm_80 = A100, sm_90 = H100" << std::endl;
        std::cout << "  --gpu-matmul STRAT GPU matmul strategy: cublas (default), mlir, auto" << std::endl;
        std::cout << std::endl;
#endif
#endif

        std::cout << "Examples:" << std::endl;
        std::cout << "  # Basic compilation" << std::endl;
        std::cout << "  " << argv[0] << " kernel.sl -o kernel.o" << std::endl;
        std::cout << std::endl;
        std::cout << "  # Compile with MLIR backend (for tensor operations)" << std::endl;
        std::cout << "  " << argv[0] << " transformer.sl --emit-mlir -o model.o" << std::endl;
        std::cout << std::endl;
        std::cout << "  # Optimize for specific tile size" << std::endl;
        std::cout << "  " << argv[0] << " matmul.sl --emit-mlir --tile-size 64 -o matmul.o" << std::endl;
        std::cout << std::endl;
        std::cout << "  # Enable parallelization with OpenMP" << std::endl;
        std::cout << "  " << argv[0] << " compute.sl --emit-mlir --enable-openmp -o compute.o" << std::endl;
        std::cout << "  # Link: gcc -shared -o compute.so compute.o -lm -fopenmp" << std::endl;
        std::cout << std::endl;
    };

    if (argc < 2) {
        printHelp();
        return 1;
    }

    // Check for help flag first (can be anywhere in args)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printHelp();
            return 0;
        }
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
        else if (strcmp(argv[i], "--target") == 0 && i + 1 < argc) {
            targetArch = argv[++i];
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
        else if (strcmp(argv[i], "--hierarchical-tiling") == 0) {
            enableHierarchicalTiling = true;
        }
        else if (strcmp(argv[i], "--enable-openmp") == 0) {
            enableOpenMP = true;
        }
        else if (strcmp(argv[i], "--no-prefetch") == 0) {
            enablePrefetch = false;
        }
        else if (strcmp(argv[i], "--llvm-vectorize") == 0) {
            llvmVectorize = true;
        }
        else if (strcmp(argv[i], "--no-opt") == 0) {
            enableO3 = false;
        }
        else if (strcmp(argv[i], "--dump-mlir-passes") == 0) {
            dumpMLIRPasses = true;
        }
#ifdef USE_CUDA
        else if (strcmp(argv[i], "--emit-gpu") == 0) {
            emitGPU = true;
            emitMLIR = true;  // GPU backend requires MLIR
        }
        else if (strcmp(argv[i], "--cuda-arch") == 0 && i + 1 < argc) {
            cudaArch = argv[++i];
            // Validate architecture format (sm_XX)
            if (cudaArch.substr(0, 3) != "sm_" || cudaArch.length() < 5) {
                std::cerr << "Error: Invalid CUDA architecture '" << cudaArch
                          << "'. Expected format: sm_XX (e.g., sm_80)" << std::endl;
                return 1;
            }
        }
        else if (strcmp(argv[i], "--gpu-matmul") == 0 && i + 1 < argc) {
            gpuMatMulStrategy = argv[++i];
            if (gpuMatMulStrategy != "cublas" && gpuMatMulStrategy != "mlir" && gpuMatMulStrategy != "auto") {
                std::cerr << "Error: Invalid GPU matmul strategy '" << gpuMatMulStrategy
                          << "'. Valid options: cublas, mlir, auto" << std::endl;
                return 1;
            }
        }
#endif
#endif
    }
    
    // Initialize logger
    Logger::setLevelFromString(logLevel);

    // Initialize LLVM - initialize ALL targets for cross-compilation support
    LOG_INFO("Initializing LLVM...");
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();

#ifdef USE_MLIR
    // Initialize LLVM optimization passes (required for makeOptimizingTransformer)
    // Note: For CUDA builds with shared libMLIR.so, this symbol isn't exported,
    // so we skip it. GPU codegen doesn't use JIT-style optimization anyway.
#ifndef USE_CUDA
    mlir::initializeLLVMPasses();
#endif
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

        // Register all SimpLang MLIR passes and pipelines for CLI usage
        mlir::simp::registerSimpPasses();
        mlir::simp::registerSimpPipelines();

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
        pipeline.setEnableHierarchicalTiling(enableHierarchicalTiling);
        pipeline.setEnableOpenMP(enableOpenMP);
        pipeline.setEnablePrefetch(enablePrefetch);
        pipeline.setSkipMLIRVectorization(llvmVectorize);
        if (llvmVectorize) {
            LOG_INFO("Using LLVM vectorization (MLIR vectorization disabled)");
        }
        if (enableTiling) {
            if (enableHierarchicalTiling) {
                LOG_INFO("Hierarchical tiling enabled (L3: 128x128x128, L2: 32x32x32, L1: 8x8x8)");
            } else {
                LOG_INFO("Loop tiling optimization enabled (" + std::to_string(tileSize) + "x" + std::to_string(tileSize) + "x" + std::to_string(tileSize) + ")");
            }
        }
        if (enableOpenMP) {
            LOG_INFO("OpenMP parallelization enabled (multi-threading)");
        }

        pipeline.setDumpIntermediateIR(dumpMLIRPasses);
        pipeline.setOutputPath(outputPath);
        if (dumpMLIRPasses) {
            LOG_INFO("MLIR intermediate IR dumping enabled");
        }

#ifdef USE_CUDA
        // Configure GPU backend options
        if (emitGPU) {
            pipeline.setEnableGPU(true);
            pipeline.setCudaArch(cudaArch);
            pipeline.setGPUMatMulStrategy(gpuMatMulStrategy);
            LOG_INFO("GPU code generation enabled");
            LOG_INFO("  Target CUDA architecture: " + cudaArch);
            LOG_INFO("  GPU MatMul strategy: " + gpuMatMulStrategy);
        }
#endif

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

        // DEBUG: Dump MLIR before LLVM translation to see what vector ops remain
        {
            std::string preTranslationPath = std::string(outputPath) + "_pre_llvm.mlir";
            std::error_code EC;
            llvm::raw_fd_ostream preTranslationOut(preTranslationPath, EC);
            if (!EC) {
                pipeline.getModule().print(preTranslationOut);
                std::cout << "Pre-translation MLIR written to: " << preTranslationPath << std::endl;
            }
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

        // Determine target triple based on --target flag
        std::string targetTriple;
        if (targetArch.empty()) {
            // No --target specified, use native target
            targetTriple = llvm::sys::getDefaultTargetTriple();
            LOG_INFO("Target: native (" + targetTriple + ")");
        } else {
            // User specified target for cross-compilation
            if (targetArch == "aarch64" || targetArch == "arm64") {
                targetTriple = "aarch64-unknown-linux-gnu";
            } else if (targetArch == "armv7" || targetArch == "arm") {
                targetTriple = "armv7-unknown-linux-gnueabihf";
            } else if (targetArch == "x86_64" || targetArch == "x86-64") {
                targetTriple = "x86_64-unknown-linux-gnu";
            } else {
                // Assume user provided full triple
                targetTriple = targetArch;
            }
            LOG_INFO("Target: " + targetTriple + " (cross-compilation)");
        }
        llvmModule->setTargetTriple(targetTriple);

        std::string error;
        auto target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
        if (!target) {
            llvm::errs() << error;
            return 1;
        }

        // CPU and features configuration
        std::string cpu;
        std::string features;

        if (targetArch.empty()) {
            // Native compilation: use host CPU features for maximum performance
            cpu = llvm::sys::getHostCPUName().str();
            llvm::StringMap<bool> featureMap;
            llvm::sys::getHostCPUFeatures(featureMap);
            llvm::SubtargetFeatures subtargetFeatures;
            for (auto &feature : featureMap) {
                subtargetFeatures.AddFeature(feature.first(), feature.second);
            }
            features = subtargetFeatures.getString();
            LOG_INFO("CPU: " + cpu + " (native)");
        } else {
            // Cross-compilation: use generic CPU for target architecture
            if (targetArch == "aarch64" || targetArch == "arm64") {
                cpu = "generic";
                features = "+neon";  // Enable NEON SIMD
            } else if (targetArch == "armv7" || targetArch == "arm") {
                cpu = "generic";
                features = "+neon,+vfp4";  // Enable NEON and VFPv4
            } else {
                cpu = "generic";
                features = "";
            }
            LOG_INFO("CPU: " + cpu + " (generic for cross-compilation)");
        }

        llvm::TargetOptions opt;
        auto rm = llvm::Optional<llvm::Reloc::Model>(llvm::Reloc::PIC_);
        auto targetMachine = target->createTargetMachine(targetTriple, cpu, features, opt, rm);

        llvmModule->setDataLayout(targetMachine->createDataLayout());

        // Run VNNI optimization pass BEFORE O3 to identify and transform i8 dot product loops
        LOG_INFO("Running VNNI optimization pass...");
        {
            llvm::legacy::PassManager vnniPM;
            // First simplify loops to create proper preheaders
            vnniPM.add(llvm::createLoopSimplifyPass());
            vnniPM.add(llvm::createLCSSAPass());
            // Now run VNNI pass
            vnniPM.add(llvm::createVNNIPass());
            vnniPM.run(*llvmModule);
        }

        // Dump IR after VNNI pass for debugging
        {
            std::error_code EC;
            llvm::raw_fd_ostream vnniIR("/tmp/after_vnni.ll", EC, llvm::sys::fs::OF_None);
            llvmModule->print(vnniIR, nullptr);
        }

        // RUN LLVM OPTIMIZATION PASSES (Critical for performance!)
        // Use custom pass pipeline to respect loop unroll metadata from VNNI pass
        if (enableO3) {
            LOG_INFO("Running LLVM optimization passes (O3)...");

            // Use new pass manager for better control
            llvm::LoopAnalysisManager LAM;
            llvm::FunctionAnalysisManager FAM;
            llvm::CGSCCAnalysisManager CGAM;
            llvm::ModuleAnalysisManager MAM;

            llvm::PassBuilder PB(targetMachine);

            // Configure PassBuilder to skip loop unrolling (preserves VNNI loops)
            llvm::PipelineTuningOptions PTO;
            PTO.LoopUnrolling = false;  // CRITICAL: Disable loop unrolling to preserve VNNI structure
            PTO.LoopVectorization = true;  // Keep SIMD vectorization

            llvm::PassBuilder PB2(targetMachine, PTO);

            // Register analysis passes
            PB2.registerModuleAnalyses(MAM);
            PB2.registerCGSCCAnalyses(CGAM);
            PB2.registerFunctionAnalyses(FAM);
            PB2.registerLoopAnalyses(LAM);
            PB2.crossRegisterProxies(LAM, FAM, CGAM, MAM);

            // Build O3 pipeline - now with unrolling disabled
            llvm::ModulePassManager MPM = PB2.buildPerModuleDefaultPipeline(
                llvm::OptimizationLevel::O3);

            MPM.run(*llvmModule, MAM);

            // Dump IR after O3 for debugging
            {
                std::error_code EC;
                llvm::raw_fd_ostream afterO3("/tmp/after_o3.ll", EC, llvm::sys::fs::OF_None);
                llvmModule->print(afterO3, nullptr);
            }
        } else {
            LOG_INFO("Skipping LLVM O3 optimization (--no-opt)");
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

#ifdef USE_MLIR
    // FIRST: Promote large stack allocations to heap
    // This must run BEFORE any optimization to fix MLIR's stack allocation bug
    modulePM.add(llvm::createPromoteLargeAllocaToHeapPass());
#endif

    // Add target transform info for vectorizer
    modulePM.add(llvm::createTargetTransformInfoWrapperPass(context.getTargetMachine()->getTargetIRAnalysis()));

    // Configure and add loop data prefetch pass
    // LoopDataPrefetch requires canonical loops (LoopSimplify + LCSSA)
    llvm::legacy::FunctionPassManager fpm(context.getModule());
    fpm.add(llvm::createLoopSimplifyPass());
    fpm.doInitialization();
    for (auto &F : *context.getModule()) {
        if (!F.isDeclaration())
            fpm.run(F);
    }
    fpm.doFinalization();

    // Set prefetch distance (default is 0 which disables prefetching)
    std::cout << "[Prefetch] Configuring loop data prefetch (distance=256, stride=1)" << std::endl;
    const char* prefetch_args[] = {"simplang", "-prefetch-distance=256", "-min-prefetch-stride=1"};
    llvm::cl::ParseCommandLineOptions(3, prefetch_args, "", &llvm::errs());
    modulePM.add(llvm::createLoopDataPrefetchPass());

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