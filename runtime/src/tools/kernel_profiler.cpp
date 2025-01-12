#include "profiler/profiler.hpp"
#include <iostream>
#include <cstring>

void printUsage(const char* program) {
    std::cout << "Usage: " << program << " [options] <kernel.so>\n"
              << "Options:\n"
              << "  --trace             Enable execution tracing\n"
              << "  --trace-file <path> Specify trace output file (default: kernel_trace.json)\n"
              << "  --warmup <n>        Number of warmup iterations (default: 100)\n"
              << "  --iterations <n>    Number of measurement iterations (default: 1000)\n"
              << "  --track-memory      Enable memory tracking\n"
              << "  --verbose           Enable verbose output\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    KernelProfiler::Config config;
    std::string kernel_path;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--trace") == 0) {
            config.enable_tracing = true;
        }
        else if (strcmp(argv[i], "--trace-file") == 0 && i + 1 < argc) {
            config.trace_path = argv[++i];
            config.enable_tracing = true;
        }
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            config.warmup_iterations = std::stoul(argv[++i]);
        }
        else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            config.total_iterations = std::stoul(argv[++i]);
        }
        else if (strcmp(argv[i], "--track-memory") == 0) {
            config.track_memory = true;
        }
        else if (strcmp(argv[i], "--verbose") == 0) {
            config.verbose = true;
        }
        else if (argv[i][0] != '-') {
            kernel_path = argv[i];
        }
        else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    if (kernel_path.empty()) {
        std::cerr << "No kernel specified\n";
        printUsage(argv[0]);
        return 1;
    }

    try {
        KernelProfiler profiler(config);
        profiler.profileKernel(kernel_path);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
