#include "profiler/execution_metrics.hpp"
#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <fstream>

ExecutionMetrics::ExecutionMetrics(size_t warmup, size_t total) 
    : warmup_iterations(warmup)
    , total_iterations(total) {
    measurements.reserve(warmup + total);
}

void ExecutionMetrics::addMeasurement(double duration_us, double result, size_t memory_used) {
    measurements.push_back({duration_us, result, memory_used});
}

void ExecutionMetrics::printStatistics() const {
    if (measurements.empty()) {
        std::cout << "No measurements recorded.\n";
        return;
    }

    // Skip warmup iterations for statistics
    auto start = measurements.begin() + warmup_iterations;
    auto end = measurements.end();
    
    std::vector<double> durations;
    std::vector<size_t> memory_usage;
    durations.reserve(end - start);
    memory_usage.reserve(end - start);
    
    for (auto it = start; it != end; ++it) {
        durations.push_back(it->duration_us);
        memory_usage.push_back(it->memory_used);
    }

    // Calculate timing statistics
    double avg_duration = std::accumulate(durations.begin(), durations.end(), 0.0) / durations.size();
    std::vector<double> sorted_durations = durations;
    std::sort(sorted_durations.begin(), sorted_durations.end());
    double median_duration = sorted_durations[sorted_durations.size() / 2];
    double min_duration = sorted_durations.front();
    double max_duration = sorted_durations.back();
    
    double sq_sum = std::inner_product(durations.begin(), durations.end(), 
                                     durations.begin(), 0.0);
    double std_dev = std::sqrt(sq_sum / durations.size() - avg_duration * avg_duration);

    // Calculate memory statistics
    size_t avg_memory = std::accumulate(memory_usage.begin(), memory_usage.end(), 0ULL) / memory_usage.size();
    size_t peak_memory = *std::max_element(memory_usage.begin(), memory_usage.end());

    // Print results
    std::cout << "\n=== Execution Metrics ===\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Sample size: " << durations.size() << " iterations";
    std::cout << " (+" << warmup_iterations << " warmup iterations)\n";
    
    std::cout << "\nTiming Metrics:\n";
    std::cout << "  Average time: " << avg_duration << " μs\n";
    std::cout << "  Median time:  " << median_duration << " μs\n";
    std::cout << "  Min time:     " << min_duration << " μs\n";
    std::cout << "  Max time:     " << max_duration << " μs\n";
    std::cout << "  Std dev:      " << std_dev << " μs\n";
    
    // Calculate operations per second
    double ops_per_sec = 1e6 / avg_duration;  // Convert μs to ops/sec
    std::cout << "  Ops/sec:      " << std::fixed << std::setprecision(2) 
              << (ops_per_sec < 1e6 ? ops_per_sec : ops_per_sec / 1e6)
              << (ops_per_sec < 1e6 ? " ops/sec" : " Mops/sec") << "\n";
    
    std::cout << "\nMemory Metrics:\n";
    std::cout << "  Average usage: " << (avg_memory / 1024.0) << " KB\n";
    std::cout << "  Peak usage:    " << (peak_memory / 1024.0) << " KB\n";
    
    std::cout << "\nFirst measured result: " << measurements[warmup_iterations].result << "\n";
}

void ExecutionMetrics::exportCSV(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing\n";
        return;
    }

    file << "Iteration,Duration_us,Result,Memory_Bytes\n";
    for (size_t i = 0; i < measurements.size(); ++i) {
        file << i << "," 
             << measurements[i].duration_us << "," 
             << measurements[i].result << ","
             << measurements[i].memory_used << "\n";
    }
}
