#ifndef MEMORY_TRACKER_HPP
#define MEMORY_TRACKER_HPP

#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <chrono>
#include <memory>
#include <immintrin.h>
#include <iostream>

class MemoryTracker {
public:
    enum class VarType {
        Int,
        Double,
        SSE_Vector,
        AVX_Vector,
        SSE_Slice,
        AVX_Slice
    };

    struct VariableState {
        std::string name;
        void* address;
        VarType type;
        size_t size;
        bool isActive;
        std::vector<std::pair<std::chrono::system_clock::time_point, std::string>> valueHistory;
        
        VariableState(const std::string& n = "", void* addr = nullptr, 
                     VarType t = VarType::Double, size_t sz = 0)
            : name(n), address(addr), type(t), size(sz), isActive(true) {}
    };

    struct MemoryStats {
        size_t totalAllocated{0};
        size_t currentlyAllocated{0};
        size_t peakAllocated{0};
        size_t totalOperations{0};
        std::map<VarType, size_t> simdOperations;
        std::map<std::string, size_t> operationTypes;
        std::map<size_t, size_t> alignmentStats;
    };

    struct Allocation {
        void* address;
        size_t size;
        std::string name;
        bool isActive;
        bool isAligned;
        size_t alignment;
        VarType type;
        std::chrono::system_clock::time_point timestamp;
        
        Allocation(void* addr = nullptr, size_t sz = 0, 
                  const std::string& n = "", VarType t = VarType::Double,
                  bool aligned = false, size_t align = 0)
            : address(addr), size(sz), name(n), isActive(true)
            , isAligned(aligned), alignment(align), type(t)
            , timestamp(std::chrono::system_clock::now()) {}
    };

    struct SimdOperation {
        VarType type;
        std::string operation;
        void* result;
        void* operand1;
        void* operand2;
        std::chrono::system_clock::time_point timestamp;
        std::string location;
        
        SimdOperation(VarType t, const std::string& op, void* res, void* op1, void* op2,
                     const std::string& loc)
            : type(t), operation(op), result(res), operand1(op1), operand2(op2)
            , timestamp(std::chrono::system_clock::now()), location(loc) {}
    };

    // Constructor and basic interface
    MemoryTracker();
    void enableTracking(bool enable) { isEnabled = enable; }
    bool isTrackingEnabled() const { return isEnabled; }

    // Memory tracking
    void trackAllocation(void* ptr, size_t size, const std::string& type, const std::string& location);
    void trackSimdAllocation(void* ptr, size_t size, VarType type, size_t alignment,
                           const std::string& location);
    void trackDeallocation(void* ptr);
    void trackAccess(void* ptr, size_t size, bool isWrite);

    // SIMD operations
    void trackSimdOperation(VarType type, const std::string& operation,
                          void* result, void* op1, void* op2,
                          const std::string& location);
    const SimdOperation* getLastOperation(void* ptr) const;
    std::vector<SimdOperation> getOperationHistory(void* ptr) const;
    bool validateSimdAccess(void* ptr, size_t size, VarType type) const;
    bool isSimdAligned(void* ptr) const;

    // Variable tracking
    void trackVariable(const std::string& name, void* address, VarType type);
    void updateVariableValue(void* address);
    const std::vector<VariableState> getActiveVariables() const;
    std::vector<Allocation> getActiveAllocations() const;
    const Allocation* getAllocationInfo(void* ptr) const;

    // Memory validation
    bool isValidPointer(void* ptr) const;
    size_t getAllocationSize(void* ptr) const;

    // Value tracking
    void updateVariableValue(const std::string& name, const std::string& value) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = variableStates.find(name);
        if (it != variableStates.end()) {
            it->second.valueHistory.emplace_back(
                std::chrono::system_clock::now(),
                value
            );
        }
    }

    std::string getLatestValue(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = variableStates.find(name);
        if (it != variableStates.end() && !it->second.valueHistory.empty()) {
            return it->second.valueHistory.back().second;
        }
        return "<undefined>";
    }

    std::map<std::string, std::string> getCurrentScopeVariables() const {
        std::lock_guard<std::mutex> lock(mutex);
        std::map<std::string, std::string> result;
        for (const auto& [name, var] : variableStates) {
            if (var.isActive && !var.valueHistory.empty()) {
                result[name] = var.valueHistory.back().second;
            }
        }
        return result;
    }

    // Reporting
    void generateSimdReport(std::ostream& out = std::cout) const;
    void dumpSimdState(std::ostream& out = std::cout) const;
    void generateReport(std::ostream& out = std::cout) const;
    void reset();

private:
    mutable std::mutex mutex;
    bool isEnabled;
    std::map<void*, Allocation> allocations;         // For memory allocations
    std::map<std::string, VariableState> variableStates;  // For variable tracking
    std::vector<SimdOperation> operationHistory;
    std::map<void*, std::string> addressToName;
    MemoryStats stats;
    static constexpr size_t MAX_HISTORY_SIZE = 1000;

    // Helper methods
    void updateStats(const Allocation& alloc, bool isAllocation);
    void pruneOperationHistory();
    std::string formatSize(size_t size) const;
    std::string formatTimestamp(const std::chrono::system_clock::time_point& time) const;
    void checkMemoryLeaks() const;
    bool checkAlignment(void* ptr, size_t required) const;
};

#endif // MEMORY_TRACKER_HPP