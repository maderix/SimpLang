#include "profiler/profiler.hpp"
#include <iostream>
#include <string>
#include <map>

void printUsage(const char* program) {
    std::cout << "Usage: " << program << " [options] -k <kernel.so>\n"
              << "Options:\n"
              << "  -k, --kernel <file>      Kernel .so file to profile\n"
              << "  -w, --warmup <n>         Warmup iterations (default: 10)\n"
              << "  -i, --iterations <n>     Measurement iterations (default: 100)\n"
              << "  -m, --memory             Enable memory tracking (default: true)\n"
              << "  -o, --output <dir>       Output directory (default: .)\n"
              << "  -v, --verbose            Verbose output\n"
              << "  -h, --help               Print this help message\n";
}

int main(int argc, char* argv[]) {
    std::string kernel_path;
    KernelProfiler::ProfileConfig config;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "-k" || arg == "--kernel") {
            if (i + 1 < argc) kernel_path = argv[++i];
        }
        else if (arg == "-w" || arg == "--warmup") {
            if (i + 1 < argc) config.warmup_iterations = std::stoul(argv[++i]);
        }
        else if (arg == "-i" || arg == "--iterations") {
            if (i + 1 < argc) config.total_iterations = std::stoul(argv[++i]);
        }
        else if (arg == "-m" || arg == "--memory") {
            config.track_memory = true;
        }
        else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) config.output_dir = argv[++i];
        }
        else if (arg == "-v" || arg == "--verbose") {
            config.verbose = true;
        }
        else {
            std::cerr << "Unknown option: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (kernel_path.empty()) {
        std::cerr << "Error: No kernel file specified\n";
        printUsage(argv[0]);
        return 1;
    }

    try {
        KernelProfiler profiler(config);
        profiler.profileKernel(kernel_path);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
