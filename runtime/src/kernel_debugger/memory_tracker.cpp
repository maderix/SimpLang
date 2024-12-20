#include "kernel_debugger/memory_tracker.hpp"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cstring>

MemoryTracker::MemoryTracker() : isEnabled(true) {}

void MemoryTracker::trackAllocation(void* ptr, size_t size, const std::string& type, 
                                  const std::string& location) {
    if (!isEnabled || !ptr) return;
    
    std::lock_guard<std::mutex> lock(mutex);
    
    Allocation alloc(ptr, size, type);
    allocations[ptr] = alloc;
    updateStats(allocations[ptr], true);

    std::cout << "DEBUG: Tracked allocation of " << type << " at " << ptr 
              << " (size: " << size << ")\n";
}

void MemoryTracker::trackSimdAllocation(void* ptr, size_t size, VarType type,
                                      size_t alignment, const std::string& location) {
    if (!isEnabled || !ptr) return;
    
    std::lock_guard<std::mutex> lock(mutex);
    
    Allocation alloc(ptr, size, "", type, true, alignment);
    allocations[ptr] = alloc;
    updateStats(alloc, true);
    stats.alignmentStats[alignment]++;
}

void MemoryTracker::trackVariable(const std::string& name, void* address, VarType type) {
    if (!isEnabled || !address) return;
    
    std::lock_guard<std::mutex> lock(mutex);
    
    size_t size;
    switch (type) {
        case VarType::Double: size = sizeof(double); break;
        case VarType::Int: size = sizeof(int); break;
        case VarType::SSE_Vector: size = sizeof(__m256d); break;
        case VarType::AVX_Vector: size = sizeof(__m512d); break;
        case VarType::SSE_Slice:
        case VarType::AVX_Slice:
            size = sizeof(void*) + 2 * sizeof(size_t);
            break;
    }
    
    variableStates[name] = VariableState(name, address, type, size);
    addressToName[address] = name;
}

void MemoryTracker::updateVariableValue(void* address) {
    if (!isEnabled || !address) return;
    
    std::lock_guard<std::mutex> lock(mutex);
    
    auto it = addressToName.find(address);
    if (it == addressToName.end()) return;
    
    auto& var = variableStates[it->second];
    std::string value;
    
    switch (var.type) {
        case VarType::Double: {
            double val = *static_cast<double*>(address);
            std::stringstream ss;
            ss << val;
            value = ss.str();
            break;
        }
        case VarType::Int: {
            int val = *static_cast<int*>(address);
            std::stringstream ss;
            ss << val;
            value = ss.str();
            break;
        }
        case VarType::SSE_Vector: {
            __m256d vec = *static_cast<__m256d*>(address);
            alignas(32) double values[4];
            _mm256_store_pd(values, vec);
            std::stringstream ss;
            ss << "[" << values[0];
            for (int i = 1; i < 4; i++) ss << ", " << values[i];
            ss << "]";
            value = ss.str();
            break;
        }
        case VarType::AVX_Vector: {
            __m512d vec = *static_cast<__m512d*>(address);
            alignas(64) double values[8];
            _mm512_store_pd(values, vec);
            std::stringstream ss;
            ss << "[" << values[0];
            for (int i = 1; i < 8; i++) ss << ", " << values[i];
            ss << "]";
            value = ss.str();
            break;
        }
        default:
            value = "<unknown type>";
    }
    
    var.valueHistory.emplace_back(std::chrono::system_clock::now(), value);
}




void MemoryTracker::trackDeallocation(void* ptr) {
    if (!isEnabled || !ptr) return;
    
    std::lock_guard<std::mutex> lock(mutex);
    
    auto it = allocations.find(ptr);
    if (it != allocations.end()) {
        if (it->second.isAligned) {
            stats.alignmentStats[it->second.alignment]--;
        }
        updateStats(it->second, false);
        allocations.erase(it);
    }
}

void MemoryTracker::trackAccess(void* ptr, size_t size, bool isWrite) {
    if (!isEnabled || !ptr) return;
    
    std::lock_guard<std::mutex> lock(mutex);
    
    auto it = allocations.find(ptr);
    if (it != allocations.end() && it->second.isActive) {
        size_t accessEnd = reinterpret_cast<size_t>(ptr) + size;
        size_t allocEnd = reinterpret_cast<size_t>(it->second.address) + it->second.size;
        
        if (accessEnd > allocEnd) {
            throw std::runtime_error("Memory access out of bounds");
        }
        
        stats.totalOperations++;
    } else {
        throw std::runtime_error("Invalid memory access");
    }
}


bool MemoryTracker::isValidPointer(void* ptr) const {
    if (!isEnabled || !ptr) return false;
    
    std::lock_guard<std::mutex> lock(mutex);
    
    auto it = allocations.find(ptr);
    return it != allocations.end() && it->second.isActive;
}


void MemoryTracker::generateReport(std::ostream& out) const {
    std::lock_guard<std::mutex> lock(mutex);
    
    out << "\nVariable Tracking Report\n"
        << "=====================\n\n"
        << "Active Variables:\n";
    
    for (const auto& [name, var] : variableStates) {
        if (var.isActive) {
            out << name << ": ";
            if (!var.valueHistory.empty()) {
                out << var.valueHistory.back().second;
            } else {
                out << "<no value>";
            }
            out << "\n";
        }
    }
    
    out << "\nMemory Statistics:\n"
        << "Total Allocated: " << formatSize(stats.totalAllocated) << "\n"
        << "Currently Allocated: " << formatSize(stats.currentlyAllocated) << "\n"
        << "Peak Allocated: " << formatSize(stats.peakAllocated) << "\n"
        << "Total Operations: " << stats.totalOperations << "\n";
}


const std::vector<MemoryTracker::VariableState> MemoryTracker::getActiveVariables() const {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<VariableState> result;
    
    for (const auto& [name, var] : variableStates) {
        if (var.isActive) {
            result.push_back(var);
        }
    }
    
    return result;
}

std::vector<MemoryTracker::Allocation> MemoryTracker::getActiveAllocations() const {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<Allocation> result;
    
    for (const auto& [_, alloc] : allocations) {
        if (alloc.isActive) {
            result.push_back(alloc);
        }
    }
    
    return result;
}

size_t MemoryTracker::getAllocationSize(void* ptr) const {
    std::lock_guard<std::mutex> lock(mutex);
    
    auto it = allocations.find(ptr);
    return (it != allocations.end()) ? it->second.size : 0;
}

const MemoryTracker::Allocation* MemoryTracker::getAllocationInfo(void* ptr) const {
    std::lock_guard<std::mutex> lock(mutex);
    
    auto it = allocations.find(ptr);
    if (it != allocations.end()) {
        return &(it->second);
    }
    return nullptr;
}

void MemoryTracker::reset() {
    std::lock_guard<std::mutex> lock(mutex);
    
    allocations.clear();
    variableStates.clear();
    addressToName.clear();
    operationHistory.clear();
    stats = MemoryStats();
}

// Private helper methods
void MemoryTracker::updateStats(const Allocation& alloc, bool isAllocation) {
    if (isAllocation) {
        stats.totalAllocated += alloc.size;
        stats.currentlyAllocated += alloc.size;
        stats.peakAllocated = std::max(stats.peakAllocated, stats.currentlyAllocated);
    } else {
        stats.currentlyAllocated -= alloc.size;
    }
}

void MemoryTracker::pruneOperationHistory() {
    if (operationHistory.size() > MAX_HISTORY_SIZE) {
        operationHistory.erase(
            operationHistory.begin(),
            operationHistory.begin() + (operationHistory.size() - MAX_HISTORY_SIZE)
        );
    }
}

void MemoryTracker::checkMemoryLeaks() const {
    size_t leakCount = 0;
    size_t leakSize = 0;
    
    for (const auto& [ptr, alloc] : allocations) {
        if (alloc.isActive) {
            leakCount++;
            leakSize += alloc.size;
        }
    }
    
    if (leakCount > 0) {
        std::cerr << "WARNING: " << leakCount << " memory leaks detected, "
                  << "total size: " << formatSize(leakSize) << std::endl;
    }
}

std::string MemoryTracker::formatSize(size_t size) const {
    const char* units[] = {"B", "KB", "MB", "GB"};
    int unit = 0;
    double dsize = static_cast<double>(size);
    
    while (dsize >= 1024.0 && unit < 3) {
        dsize /= 1024.0;
        unit++;
    }
    
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << dsize << " " << units[unit];
    return ss.str();
}

std::string MemoryTracker::formatTimestamp(
    const std::chrono::system_clock::time_point& time) const {
    auto timer = std::chrono::system_clock::to_time_t(time);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&timer), "%H:%M:%S");
    return ss.str();
}

bool MemoryTracker::checkAlignment(void* ptr, size_t required) const {
    return reinterpret_cast<uintptr_t>(ptr) % required == 0;
}