#include "kernel_runner.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <numeric>
#include <cmath>
#include <fstream>
#include <algorithm>

class ExecutionMetrics {
public:
    struct Measurement {
        double duration_us;  // microseconds
        double result;
    };

    ExecutionMetrics(size_t warmup = 10, size_t total = 100) 
        : warmup_iterations(warmup)
        , total_iterations(total) {}

    void addMeasurement(double duration_us, double result) {
        measurements.push_back({duration_us, result});
    }

    // Make these accessible to ProfiledKernelRunner
    size_t getWarmupIterations() const { return warmup_iterations; }
    size_t getTotalIterations() const { return total_iterations; }

    void printStatistics() const {
        if (measurements.empty()) {
            std::cout << "No measurements recorded.\n";
            return;
        }

        // Skip warmup iterations for statistics
        auto start = measurements.begin() + warmup_iterations;
        auto end = measurements.end();
        
        std::vector<double> durations;
        durations.reserve(end - start);
        
        for (auto it = start; it != end; ++it) {
            durations.push_back(it->duration_us);
        }

        // Calculate statistics
        double avg = std::accumulate(durations.begin(), durations.end(), 0.0) / durations.size();
        
        std::vector<double> sorted_durations = durations;
        std::sort(sorted_durations.begin(), sorted_durations.end());
        
        double median = sorted_durations[sorted_durations.size() / 2];
        double min = sorted_durations.front();
        double max = sorted_durations.back();
        
        double sq_sum = std::inner_product(durations.begin(), durations.end(), 
                                         durations.begin(), 0.0);
        double std_dev = std::sqrt(sq_sum / durations.size() - avg * avg);

        // Print results
        std::cout << "\n=== Execution Metrics ===\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Sample size: " << durations.size() << " iterations";
        std::cout << " (+" << warmup_iterations << " warmup iterations)\n";
        std::cout << "Average time: " << avg << " μs\n";
        std::cout << "Median time: " << median << " μs\n";
        std::cout << "Min time: " << min << " μs\n";
        std::cout << "Max time: " << max << " μs\n";
        std::cout << "Std deviation: " << std_dev << " μs\n";
        
        std::cout << "\nFirst measured result: " << measurements[warmup_iterations].result << "\n";
    }

    void exportCSV(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open " << filename << " for writing\n";
            return;
        }

        file << "Iteration,Duration_us,Result\n";
        for (size_t i = 0; i < measurements.size(); ++i) {
            file << i << "," 
                 << measurements[i].duration_us << "," 
                 << measurements[i].result << "\n";
        }
    }

private:
    std::vector<Measurement> measurements;
    size_t warmup_iterations;
    size_t total_iterations;
};

class ProfiledKernelRunner {
private:
    KernelRunner runner;
    ExecutionMetrics metrics;
    bool verbose;

public:
    ProfiledKernelRunner(size_t warmup = 10, size_t total = 100, bool verbose = false) 
        : metrics(warmup, total)
        , verbose(verbose) {}

    void runBenchmark(const char* kernel_path) {
        try {
            runner.loadLibrary(kernel_path);
            
            if (verbose) {
                std::cout << "Running kernel: " << kernel_path << "\n";
                std::cout << "Warming up...\n";
            }

            // Warmup runs
            for (size_t i = 0; i < metrics.getWarmupIterations(); ++i) {
                auto start = std::chrono::high_resolution_clock::now();
                double result = runner.runKernel();
                auto end = std::chrono::high_resolution_clock::now();
                
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                metrics.addMeasurement(duration.count(), result);
            }

            if (verbose) {
                std::cout << "Running benchmark...\n";
            }

            // Actual measurements
            for (size_t i = 0; i < metrics.getTotalIterations(); ++i) {
                auto start = std::chrono::high_resolution_clock::now();
                double result = runner.runKernel();
                auto end = std::chrono::high_resolution_clock::now();
                
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                metrics.addMeasurement(duration.count(), result);
            }

            metrics.printStatistics();
            metrics.exportCSV("kernel_profile.csv");

        } catch (const std::exception& e) {
            std::cerr << "Error running kernel: " << e.what() << "\n";
            throw;
        }
    }
};

void printUsage(const char* program) {
    std::cout << "Usage: " << program << " <kernel.so> [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --warmup N     Number of warmup iterations (default: 10)\n";
    std::cout << "  --iterations N Number of measured iterations (default: 100)\n";
    std::cout << "  --verbose      Enable verbose output\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    const char* kernel_path = argv[1];
    size_t warmup = 10;
    size_t iterations = 100;
    bool verbose = false;

    // Parse command line arguments
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::stoul(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoul(argv[++i]);
        } else if (arg == "--verbose") {
            verbose = true;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    try {
        ProfiledKernelRunner profiler(warmup, iterations, verbose);
        profiler.runBenchmark(kernel_path);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}