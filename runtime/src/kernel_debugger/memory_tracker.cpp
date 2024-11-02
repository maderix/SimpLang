#include "kernel_debugger/memory_tracker.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <numeric>
#include <ctime>

// Helper function to format memory sizes
static std::string formatMemorySize(size_t size) {
    constexpr size_t KB = 1024;
    constexpr size_t MB = KB * 1024;
    constexpr size_t GB = MB * 1024;

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);

    if (size >= GB) {
        ss << static_cast<double>(size) / GB << " GB";
    } else if (size >= MB) {
        ss << static_cast<double>(size) / MB << " MB";
    } else if (size >= KB) {
        ss << static_cast<double>(size) / KB << " KB";
    } else {
        ss << size << " bytes";
    }
    return ss.str();
}

void MemoryTracker::trackOperation(const std::string& op, void* addr, size_t size,
                                 const std::string& description) {
    std::lock_guard<std::mutex> lock(mutex);
    auto now = std::chrono::system_clock::now();

    if (op == "allocate") {
        // Track allocation
        auto [it, inserted] = allocatedBlocks.emplace(addr, 
            MemoryBlock(addr, size, description));
        if (inserted) {
            it->second.stackTrace = getCurrentStackTrace();
            
            // Update statistics
            totalAllocated += size;
            currentMemory += size;
            peakMemory = std::max(peakMemory, currentMemory);
            allocationPatterns[size]++;

            // Log operation
            operations.emplace_back(
                MemoryOperation::Type::ALLOCATE,
                addr,
                size,
                description
            );

            checkAllocationThresholds();
        }
    } else if (op == "free") {
        auto it = allocatedBlocks.find(addr);
        if (it != allocatedBlocks.end()) {
            // Calculate lifetime
            auto lifetime = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - it->second.allocationTime).count();
            blockLifetimes.push_back(lifetime);

            // Update statistics
            currentMemory -= it->second.size;
            freedMemory += it->second.size;

            // Log operation
            operations.emplace_back(
                MemoryOperation::Type::FREE,
                addr,
                it->second.size,
                description
            );

            allocatedBlocks.erase(it);
        } else {
            logError("Attempt to free untracked memory", addressToString(addr));
        }
    } else if (op == "access") {
        auto it = allocatedBlocks.find(addr);
        if (it != allocatedBlocks.end()) {
            it->second.lastAccessTime = now;
            it->second.accessCount++;

            operations.emplace_back(
                MemoryOperation::Type::ACCESS,
                addr,
                size,
                description
            );
        } else {
            logError("Access to untracked memory", addressToString(addr));
        }
    }

    // Maintain operation history size
    if (operations.size() > maxOperationHistory) {
        operations.erase(operations.begin(), 
                        operations.begin() + operations.size() - maxOperationHistory);
    }
}

void MemoryTracker::printSummary() const {
    std::lock_guard<std::mutex> lock(mutex);

    std::cout << "\n=== Memory Usage Summary ===\n";
    std::cout << "Total allocated: " << formatMemorySize(totalAllocated) << "\n";
    std::cout << "Current usage:   " << formatMemorySize(currentMemory) << "\n";
    std::cout << "Peak usage:      " << formatMemorySize(peakMemory) << "\n";
    std::cout << "Freed memory:    " << formatMemorySize(freedMemory) << "\n";
    std::cout << "Active blocks:   " << allocatedBlocks.size() << "\n";

    if (!allocationPatterns.empty()) {
        std::cout << "\nAllocation Size Patterns:\n";
        for (const auto& [size, count] : allocationPatterns) {
            std::cout << "  " << std::setw(10) << formatMemorySize(size) 
                      << ": " << count << " times\n";
        }
    }

    if (!blockLifetimes.empty()) {
        auto [minIt, maxIt] = std::minmax_element(blockLifetimes.begin(), 
                                                 blockLifetimes.end());
        double avgLifetime = std::accumulate(blockLifetimes.begin(), 
                                           blockLifetimes.end(), 0.0) / 
                                           blockLifetimes.size();

        std::cout << "\nBlock Lifetime Statistics:\n";
        std::cout << "  Minimum: " << *minIt << " ms\n";
        std::cout << "  Maximum: " << *maxIt << " ms\n";
        std::cout << "  Average: " << std::fixed << std::setprecision(2) 
                  << avgLifetime << " ms\n";
    }
}

void MemoryTracker::printCurrentState() const {
    std::lock_guard<std::mutex> lock(mutex);

    std::cout << "\nCurrent Memory State:\n";
    std::cout << "Using " << formatMemorySize(currentMemory) 
              << " across " << allocatedBlocks.size() << " blocks\n";

    if (!operations.empty()) {
        std::cout << "\nLast 5 memory operations:\n";
        auto start = operations.size() <= 5 ? operations.begin() 
                                          : operations.end() - 5;
        for (auto it = start; it != operations.end(); ++it) {
            printOperation(*it);
        }
    }
}

std::vector<MemoryTracker::MemoryBlock> MemoryTracker::detectLeaks() const {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<MemoryBlock> leaks;
    auto now = std::chrono::system_clock::now();

    for (const auto& [addr, block] : allocatedBlocks) {
        auto lifetime = std::chrono::duration_cast<std::chrono::seconds>(
            now - block.allocationTime).count();

        if (lifetime > leakThresholdSeconds) {
            leaks.push_back(block);
        }
    }

    return leaks;
}

void MemoryTracker::analyzeFragmentation() const {
    std::lock_guard<std::mutex> lock(mutex);

    std::cout << "\nMemory Fragmentation Analysis:\n";

    std::vector<std::pair<void*, MemoryBlock>> sortedBlocks(
        allocatedBlocks.begin(), allocatedBlocks.end());
    std::sort(sortedBlocks.begin(), sortedBlocks.end(),
              [](const auto& a, const auto& b) {
                  return a.first < b.first;
              });

    size_t totalGaps = 0;
    size_t maxGap = 0;
    int gapCount = 0;

    for (size_t i = 1; i < sortedBlocks.size(); i++) {
        uintptr_t prevEnd = reinterpret_cast<uintptr_t>(sortedBlocks[i-1].first) + 
                           sortedBlocks[i-1].second.size;
        uintptr_t currStart = reinterpret_cast<uintptr_t>(sortedBlocks[i].first);

        if (currStart > prevEnd) {
            size_t gap = currStart - prevEnd;
            totalGaps += gap;
            maxGap = std::max(maxGap, gap);
            gapCount++;
        }
    }

    std::cout << "Number of gaps: " << gapCount << "\n";
    std::cout << "Total gap size: " << formatMemorySize(totalGaps) << "\n";
    std::cout << "Largest gap: " << formatMemorySize(maxGap) << "\n";
    
    if (gapCount > 0) {
        double avgGap = static_cast<double>(totalGaps) / gapCount;
        std::cout << "Average gap size: " << formatMemorySize(static_cast<size_t>(avgGap)) << "\n";
    }

    double fragmentationRatio = static_cast<double>(totalGaps) / 
                              (currentMemory + totalGaps);
    std::cout << "Fragmentation ratio: " << std::fixed << std::setprecision(2)
              << (fragmentationRatio * 100) << "%\n";
}

void MemoryTracker::clear() {
    std::lock_guard<std::mutex> lock(mutex);
    allocatedBlocks.clear();
    operations.clear();
    allocationPatterns.clear();
    blockLifetimes.clear();
    totalAllocated = 0;
    currentMemory = 0;
    peakMemory = 0;
    freedMemory = 0;
}

void MemoryTracker::checkAllocationThresholds() {
    if (currentMemory > memoryWarningThreshold) {
        logWarning("Memory usage exceeds warning threshold",
                  "Current: " + formatMemorySize(currentMemory));
    }

    if (allocatedBlocks.size() > blockCountWarningThreshold) {
        logWarning("Block count exceeds warning threshold",
                  "Count: " + std::to_string(allocatedBlocks.size()));
    }
}

void MemoryTracker::logError(const std::string& message, 
                           const std::string& details) {
    std::cerr << "Memory Error: " << message << "\n";
    if (!details.empty()) {
        std::cerr << "Details: " << details << "\n";
    }
    errors.push_back({message, details, std::chrono::system_clock::now()});
}

void MemoryTracker::logWarning(const std::string& message,
                             const std::string& details) {
    std::cout << "Memory Warning: " << message << "\n";
    if (!details.empty()) {
        std::cout << "Details: " << details << "\n";
    }
    warnings.push_back({message, details, std::chrono::system_clock::now()});
}

std::string MemoryTracker::addressToString(void* addr) const {
    std::stringstream ss;
    ss << "0x" << std::hex << std::setfill('0') << std::setw(12)
       << reinterpret_cast<uintptr_t>(addr);
    return ss.str();
}

void MemoryTracker::printOperation(const MemoryOperation& op) const {
    std::stringstream ss;
    auto timePoint = std::chrono::system_clock::to_time_t(op.timestamp);
    ss << std::put_time(std::localtime(&timePoint), "%H:%M:%S") << " - ";

    switch (op.type) {
        case MemoryOperation::Type::ALLOCATE:
            ss << "Allocated ";
            break;
        case MemoryOperation::Type::FREE:
            ss << "Freed ";
            break;
        case MemoryOperation::Type::ACCESS:
            ss << "Accessed ";
            break;
    }

    ss << formatMemorySize(op.size) << " at " << addressToString(op.address);
    if (!op.description.empty()) {
        ss << " (" << op.description << ")";
    }

    std::cout << ss.str() << "\n";
}

std::vector<std::string> MemoryTracker::getCurrentStackTrace() const {
    // Platform-specific implementation would go here
    return std::vector<std::string>();
}