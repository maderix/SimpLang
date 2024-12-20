#ifndef KERNEL_DEBUGGER_HPP
#define KERNEL_DEBUGGER_HPP

#include "breakpoint.hpp"
#include "event_logger.hpp"
#include "memory_tracker.hpp"
#include "source_manager.hpp"
#include "call_stack.hpp"
#include "command_processor.hpp"
#include "ui_helper.hpp"
#include "config.hpp"
#include <immintrin.h>
#include <string>
#include <map>
#include <mutex>
#include <memory>
#include <vector>
#include <thread>
#include <atomic>

class KernelDebugger {
public:
    enum class Mode {
        RUN,        // Normal execution
        STEP,       // Single step
        NEXT,       // Step over
        FINISH      // Run until current function returns
    };

    enum class SIMDOp {
        ADD_SSE,
        MUL_SSE,
        ADD_AVX,
        MUL_AVX
    };

    struct VectorRegister {
        union {
            __m256d sse;
            __m512d avx;
        } value;
        bool is_avx;
        
        VectorRegister() : is_avx(false) {
            value.sse = _mm256_setzero_pd();
        }
    };

    // Singleton access
    static KernelDebugger& getInstance();
    
    // Initialization and cleanup
    void initialize();
    void setEventLogger(std::shared_ptr<EventLogger> logger);
    void setMemoryTracker(std::shared_ptr<MemoryTracker> tracker);
    
    // Mode control
    Mode getCurrentMode() const { return currentMode; }
    void setMode(Mode mode) {
        std::lock_guard<std::mutex> lock(mutex);
        currentMode = mode;
    }
    std::string getCurrentFile() const { 
        return currentFile.empty() ? sourceManager.getCurrentFile() : currentFile;
    }
    void setCurrentFile(const std::string& file) {
        currentFile = file;
    }
    void setCurrentFunction(const std::string& function);
    std::string getCurrentFunction() const { return currentFunction; }
    
    // Debugging commands
    bool loadKernel(const std::string& filename);
    void start();
    void stop();
    void continueExecution();
    void stepIn();
    void stepOver();
    void stepOut();
    void selectFrame(int frameNum);
    
    // Breakpoint management
    int addBreakpoint(const std::string& file, int line, const std::string& condition = "");
    bool removeBreakpoint(int id);
    void enableBreakpoint(int id, bool enable = true);
    void clearBreakpoints();
    void listBreakpoints() const;
    
    // Information display
    void listSourceFiles() const;
    void showSource(const std::string& file = "", int startLine = 0, int count = 10) const;
    void printLocation() const;
    void printBacktrace() const;
    void printLocals() const;
    void printVectorState() const;
    void displayMemory(void* addr, size_t size) const;
    void printExpression(const std::string& expr) const;
    
    // Event handlers
    void onDebugEvent(const std::string& event, 
                     const std::string& file, 
                     int line, 
                     const std::string& details);
    void onSIMDOperation(SIMDOp op, void* result, void* op1, void* op2);
    void onMemoryAccess(void* addr, size_t size, bool isWrite);

    UIHelper* getUIHelper() const { return uiHelper.get(); }
    
    // Configuration access
    const DebuggerConfig& getConfig() const { return *config; }
    void updateConfig(const DebuggerConfig::DisplayConfig& display,
                     const DebuggerConfig::DebugConfig& debug,
                     const DebuggerConfig::UIConfig& ui) {
        if (config) {
            config->display() = display;
            config->debug() = debug;
            config->ui() = ui;
        }
    }
    
    // State access
    MemoryTracker& getMemoryTracker() { return *memoryTracker; }
    CallStack& getCallStack() { return callStack; }
    bool isPaused() const { return pauseFlag; }
    bool shouldBreak() const;
    
    // UI related
    void printError(const std::string& message) const {
        if (uiHelper) {
            uiHelper->printError(message);
        }
    }
    void updateLocation(const std::string& file, int line);

    void printInfo(const std::string& message) const {
        if (uiHelper) {
            uiHelper->printInfo(message);
        }
    }

private:
    KernelDebugger();
    ~KernelDebugger();
    KernelDebugger(const KernelDebugger&) = delete;
    KernelDebugger& operator=(const KernelDebugger&) = delete;
    void initializeComponents();
    void performInitialization();

    // Component managers
    std::shared_ptr<EventLogger> eventLogger;
    std::shared_ptr<MemoryTracker> memoryTracker;
    std::unique_ptr<CommandProcessor> cmdProcessor;
    std::unique_ptr<UIHelper> uiHelper;
    std::shared_ptr<DebuggerConfig> config;
    SourceManager sourceManager;
    BreakpointManager breakpointMgr;
    CallStack callStack;

    // Debugger state
    bool isInitialized = false;
    std::string currentFile;    // Current source file being debugged
    std::string currentFunction;  // Current function being debugged
    Mode currentMode{Mode::RUN};
    bool isRunning{false};
    std::atomic<bool> pauseFlag{false};
    size_t stepCount{0};
    int stepStopDepth{-1};
    std::map<std::string, VectorRegister> vectorRegs;
    std::map<SIMDOp, size_t> opStats;
    mutable std::mutex mutex;

    // Event handling
    std::unique_ptr<class DebugEventQueue> eventQueue;
    std::unique_ptr<class IDebugEventHandler> eventHandler;
    std::thread eventThread;
    void initializeEventHandler();
    void cleanupEventHandler();
    void eventLoop();
    void postEvent(class DebugEvent event);

    // Execution control
    void pauseExecution();
    void resumeExecution();
    bool shouldBreakAtCurrentLocation() const;
    void executeNextInstruction();
    bool isExecutionComplete() const;

    // Helper methods
    void logDebugEvent(const std::string& event, const std::string& details = "");
    void updateVectorRegister(const std::string& name, const void* value, bool is_avx);
    std::string formatVectorRegister(const VectorRegister& reg) const;
    void validateMemoryAccess(void* addr, size_t size) const;
    bool validateBreakpoint(const std::string& file, int line) const;
    std::string getSIMDOpName(SIMDOp op) const;
    size_t getSimdOperandSize(SIMDOp op) const;
    bool checkAlignment(void* ptr, size_t required) const;

};

#endif // KERNEL_DEBUGGER_HPP