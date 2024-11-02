#ifndef KERNEL_DEBUGGER_HPP
#define KERNEL_DEBUGGER_HPP

#include "kernel.h"
#include "breakpoint.hpp"
#include "event_logger.hpp"
#include "memory_tracker.hpp"
#include <immintrin.h>
#include <string>
#include <map>
#include <mutex>
#include <memory>

class KernelDebugger {
public:
    // Execution modes
    enum class Mode {
        RUN,            // Run normally
        STEP,           // Step by step execution
        BREAKPOINT      // Run until breakpoint
    };

    // SIMD operation types for tracking
    enum class SIMDOp {
        ADD_SSE,
        MUL_SSE,
        ADD_AVX,
        MUL_AVX
    };

private:
    // Singleton pattern
    static KernelDebugger* instance;
    static std::mutex mutex;

    // Component managers
    std::unique_ptr<BreakpointManager> breakpointMgr;
    std::unique_ptr<EventLogger> eventLogger;
    std::unique_ptr<MemoryTracker> memoryTracker;

    // Debugger state
    Mode currentMode;
    bool isRunning;
    size_t stepCount;
    std::map<std::string, size_t> opCounts;

    // Private constructor for singleton
    KernelDebugger();

public:
    // Prevent copying
    KernelDebugger(const KernelDebugger&) = delete;
    KernelDebugger& operator=(const KernelDebugger&) = delete;

    // Singleton access
    static KernelDebugger* getInstance();

    // Debugger control
    void start();
    void stop();
    void setMode(Mode mode);

    // SIMD operation tracking
    void onSIMDOperation(SIMDOp op, const void* a, const void* b, const void* result);
    void onSliceOperation(const std::string& op, const void* slice, size_t index);
    void onMemoryOperation(const std::string& op, void* addr, size_t size, 
                          const std::string& description);

    // Breakpoint management
    int addBreakpoint(const std::string& location, 
                     std::function<bool()> condition = nullptr);
    void removeBreakpoint(int id);

    // Debug information
    void printSummary() const;
    void printCurrentState() const;

private:
    // Helper methods
    void handleBreakpoint(const std::string& location);
    void handleStep(const std::string& location);
    void waitForCommand();
    std::string getModeString(Mode mode) const;
    bool isSSEVector(const void* vec) const;
    std::string formatSIMDVector(const void* vec, SIMDOp op) const;
    std::string formatSliceElement(const void* slice, size_t index) const;
};

#endif // KERNEL_DEBUGGER_HPP