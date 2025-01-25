#pragma once
#include "kernel_runner.hpp"
#include "execution_metrics.hpp"
#include "memory_metrics.hpp"
#include "trace_events.hpp"
#include <string>
#include <memory>
#include <filesystem>
#include <chrono>

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
        std::string ir_file;

        Config() 
            : warmup_iterations(100)
            , total_iterations(1000)
            , track_memory(false)
            , verbose(false)
            , enable_tracing(false)
            , trace_path("kernel_trace.json")
            , output_dir("profile_output")
            , ir_file("") {}
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
    std::string findIRFile(const std::string& kernel_path);
    
    // Helper for instrumentation
    uint64_t getTimestamp() {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(
            now - start_time
        ).count();
    }

    Config config;
    KernelRunner runner;
    ExecutionMetrics exec_metrics;
    MemoryMetrics mem_metrics;
    std::unique_ptr<TraceEvents> trace_events;
    std::chrono::steady_clock::time_point start_time;
};
