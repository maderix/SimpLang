#ifndef KERNEL_DEBUGGER_MEMORY_TRACKER_HPP
#define KERNEL_DEBUGGER_MEMORY_TRACKER_HPP

#include <string>
#include <map>
#include <vector>
#include <cstddef>
#include <sstream>
#include <chrono>
#include <mutex>

class MemoryTracker {
public:
    struct MemoryBlock {
        void* address{nullptr};
        size_t size{0};
        std::string description;
        std::chrono::system_clock::time_point allocationTime;
        size_t accessCount{0};
        std::chrono::system_clock::time_point lastAccessTime;
        std::vector<std::string> stackTrace;
        
        // Add default constructor
        MemoryBlock() 
            : allocationTime(std::chrono::system_clock::now())
            , lastAccessTime(allocationTime)
        {}
        
        // Keep existing constructor
        MemoryBlock(void* addr, size_t sz, const std::string& desc)
            : address(addr)
            , size(sz)
            , description(desc)
            , allocationTime(std::chrono::system_clock::now())
            , lastAccessTime(allocationTime)
        {}
    };

    struct MemoryOperation {
        enum class Type {
            ALLOCATE,
            FREE,
            ACCESS
        };

        Type type;
        void* address;
        size_t size;
        std::string description;
        std::chrono::system_clock::time_point timestamp;

        MemoryOperation(Type t, void* addr, size_t sz, const std::string& desc)
            : type(t)
            , address(addr)
            , size(sz)
            , description(desc)
            , timestamp(std::chrono::system_clock::now())
        {}
    };

private:
    std::map<void*, MemoryBlock> allocatedBlocks;
    std::vector<MemoryOperation> operations;
    std::map<size_t, size_t> allocationPatterns;
    std::vector<int64_t> blockLifetimes;
    
    size_t totalAllocated{0};
    size_t peakMemory{0};
    size_t currentMemory{0};
    size_t freedMemory{0};
    
    const size_t maxOperationHistory{1000};
    const size_t memoryWarningThreshold{1024 * 1024 * 1024}; // 1GB
    const size_t blockCountWarningThreshold{1000};
    const int64_t leakThresholdSeconds{3600}; // 1 hour
    
    mutable std::mutex mutex;
    
    struct LogEntry {
        std::string message;
        std::string details;
        std::chrono::system_clock::time_point timestamp;
    };
    
    std::vector<LogEntry> errors;
    std::vector<LogEntry> warnings;

    // Private helper methods
    void checkAllocationThresholds();
    void logError(const std::string& message, const std::string& details = "");
    void logWarning(const std::string& message, const std::string& details = "");
    std::string addressToString(void* addr) const;
    void printOperation(const MemoryOperation& op) const;
    std::vector<std::string> getCurrentStackTrace() const;

public:
    MemoryTracker() = default;

    void trackOperation(const std::string& op, void* addr, size_t size,
                       const std::string& description);
    void clear();
    void printSummary() const;
    void printCurrentState() const;
    std::vector<MemoryBlock> detectLeaks() const;
    void analyzeFragmentation() const;

    // Getters
    size_t getTotalAllocated() const { return totalAllocated; }
    size_t getPeakMemory() const { return peakMemory; }
    size_t getCurrentMemory() const { return currentMemory; }
    size_t getActiveAllocations() const { return allocatedBlocks.size(); }
    const std::vector<MemoryOperation>& getOperations() const { return operations; }
};

#endif // KERNEL_DEBUGGER_MEMORY_TRACKER_HPP