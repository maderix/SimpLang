#include "profiler/memory_metrics.hpp"
#include <iostream>
#include <algorithm>

MemoryMetrics::MemoryMetrics() : current_usage(0), peak_usage(0) {
    tracker.enableTracking(true);
}

void MemoryMetrics::trackAllocation(void* ptr, size_t size, const std::string& type, bool aligned) {
    tracker.trackAllocation(ptr, size, type, "profiler");
    
    // Update our own tracking
    current_usage += size;
    peak_usage = std::max(peak_usage, current_usage);
    allocations[ptr] = size;
}

void MemoryMetrics::trackDeallocation(void* ptr) {
    auto it = allocations.find(ptr);
    if (it != allocations.end()) {
        current_usage -= it->second;
        allocations.erase(it);
    }
    tracker.trackDeallocation(ptr);
}

size_t MemoryMetrics::getCurrentUsage() const {
    return current_usage;
}

size_t MemoryMetrics::getPeakUsage() const {
    return peak_usage;
}

void MemoryMetrics::printReport() const {
    std::cout << "\n=== Memory Profiling Report ===\n";
    std::cout << "Current memory usage: " << (current_usage / 1024.0) << " KB\n";
    std::cout << "Peak memory usage: " << (peak_usage / 1024.0) << " KB\n";
    std::cout << "Active allocations: " << allocations.size() << "\n\n";
    
    // Also print the detailed tracker report
    tracker.generateReport();
}
