#pragma once
#include <vector>
#include <string>
#include <chrono>

class ExecutionMetrics {
public:
    struct Measurement {
        double duration_us;  // Keep using microseconds consistently
        double result;
        size_t memory_used;
    };

    ExecutionMetrics(size_t warmup = 10, size_t total = 100);
    
    void addMeasurement(double duration_us, double result, size_t memory_used);
    void printStatistics() const;
    void exportCSV(const std::string& filename) const;
    
    size_t getWarmupIterations() const { return warmup_iterations; }
    size_t getTotalIterations() const { return total_iterations; }

private:
    std::vector<Measurement> measurements;
    size_t warmup_iterations;
    size_t total_iterations;
};