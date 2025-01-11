#pragma once
#include <kernel_debugger/memory_tracker.hpp>
#include <string>
#include <map>

// Wrapper around MemoryTracker to provide profiling-specific functionality
class MemoryMetrics {
public:
    MemoryMetrics();
    void trackAllocation(void* ptr, size_t size, const std::string& type, bool aligned = false);
    void trackDeallocation(void* ptr);
    size_t getCurrentUsage() const;
    size_t getPeakUsage() const;
    void printReport() const;

private:
    MemoryTracker tracker;
    std::map<void*, size_t> allocations;
    size_t current_usage;
    size_t peak_usage;
};
