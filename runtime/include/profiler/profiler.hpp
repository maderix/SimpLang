#pragma once
#include "kernel_runner.hpp"
#include "profiler/execution_metrics.hpp"
#include "profiler/memory_metrics.hpp"
#include <string>

class KernelProfiler {
public:
    struct ProfileConfig {
        bool verbose = false;
        bool track_memory = false;
        std::string output_dir = ".";
        size_t warmup_iterations = 10;
        size_t total_iterations = 100;
    };

    explicit KernelProfiler(const ProfileConfig& cfg);
    
    void profileKernel(const std::string& kernel_path);
    void compareWithBaseline(const std::string& kernel_path, double (*baseline_func)());
    
private:
    void runWarmup();
    void runMeasurements();
    void runWarmupScalar();  // Add scalar version
    void runMeasurementsScalar();  // Add scalar version
    void generateReport();

    ProfileConfig config;
    ExecutionMetrics exec_metrics;
    MemoryMetrics mem_metrics;
    KernelRunner runner;
};
