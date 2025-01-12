#pragma once
#include "kernel_runner.hpp"
#include "execution_metrics.hpp"
#include "memory_metrics.hpp"
#include "trace_events.hpp"
#include <string>
#include <memory>
#include <filesystem>

class KernelProfiler {
public:
    struct Config {
        size_t warmup_iterations;
        size_t total_iterations;
        bool track_memory;
        bool verbose;
        bool enable_tracing;
        std::string trace_path;
        std::string output_dir;

        Config() 
            : warmup_iterations(100)
            , total_iterations(1000)
            , track_memory(false)
            , verbose(false)
            , enable_tracing(false)
            , trace_path("kernel_trace.json")
            , output_dir("profile_output") {}
    };

    explicit KernelProfiler(const Config& config = Config{});

    void profileKernel(const std::string& kernel_path);
    void compareWithBaseline(const std::string& kernel_path, double (*baseline_func)());

private:
    void runWarmup();
    void runMeasurements();
    void runWarmupScalar();
    void runMeasurementsScalar();
    void generateReport();

    Config config;
    KernelRunner runner;
    ExecutionMetrics exec_metrics;
    MemoryMetrics mem_metrics;
    std::unique_ptr<TraceEvents> trace_events;
};
