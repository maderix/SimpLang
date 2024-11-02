#include "kernel_debugger/debugger.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>

// Initialize static members
KernelDebugger* KernelDebugger::instance = nullptr;
std::mutex KernelDebugger::mutex;

// Constructor
KernelDebugger::KernelDebugger()
    : breakpointMgr(std::make_unique<BreakpointManager>())
    , eventLogger(std::make_unique<EventLogger>())
    , memoryTracker(std::make_unique<MemoryTracker>())
    , currentMode(Mode::RUN)
    , isRunning(false)
    , stepCount(0)
{}

KernelDebugger* KernelDebugger::getInstance() {
    std::lock_guard<std::mutex> lock(mutex);
    if (instance == nullptr) {
        instance = new KernelDebugger();
    }
    return instance;
}

void KernelDebugger::start() {
    std::lock_guard<std::mutex> lock(mutex);
    
    if (isRunning) {
        std::cout << "Debugger is already running" << std::endl;
        return;
    }

    isRunning = true;
    stepCount = 0;
    opCounts.clear();
    
    eventLogger->logEvent(
        EventLogger::EventType::BREAKPOINT,
        "Debugger started",
        "",
        nullptr,
        0,
        "Mode: " + getModeString(currentMode)
    );

    std::cout << "\n=== Kernel Debugger Started ===\n"
              << "Mode: " << getModeString(currentMode) << "\n"
              << "===============================" << std::endl;
}

void KernelDebugger::stop() {
    std::lock_guard<std::mutex> lock(mutex);
    
    if (!isRunning) {
        std::cout << "Debugger is not running" << std::endl;
        return;
    }

    eventLogger->logEvent(
        EventLogger::EventType::BREAKPOINT,
        "Debugger stopped",
        "",
        nullptr,
        0,
        "Total steps: " + std::to_string(stepCount)
    );

    isRunning = false;
    printSummary();
}

void KernelDebugger::onSIMDOperation(SIMDOp op, const void* a, const void* b, const void* result) {
    if (!isRunning) return;
    std::lock_guard<std::mutex> lock(mutex);

    stepCount++;
    std::string opName = getModeString(currentMode);
    opCounts[opName]++;

    std::stringstream desc;
    desc << opName;
    if (a) desc << "\n  Input A: " << formatSIMDVector(a, op);
    if (b) desc << "\n  Input B: " << formatSIMDVector(b, op);
    if (result) desc << "\n  Result: " << formatSIMDVector(result, op);

    eventLogger->logEvent(
        EventLogger::EventType::SIMD_OP,
        desc.str(),
        std::to_string(stepCount)
    );

    if (currentMode == Mode::BREAKPOINT && 
        breakpointMgr->checkBreakpoint(opName)) {
        handleBreakpoint(opName);
    }

    if (currentMode == Mode::STEP) {
        handleStep(opName);
    }
}

void KernelDebugger::onMemoryOperation(const std::string& op, void* addr, 
                                     size_t size, const std::string& description) {
    if (!isRunning) return;
    std::lock_guard<std::mutex> lock(mutex);

    memoryTracker->trackOperation(op, addr, size, description);

    eventLogger->logEvent(
        EventLogger::EventType::MEMORY_OP,
        op + " " + std::to_string(size) + " bytes",
        std::to_string(stepCount),
        addr,
        size,
        description
    );
}

void KernelDebugger::onSliceOperation(const std::string& op, const void* slice, 
                                    size_t index) {
    if (!isRunning) return;
    std::lock_guard<std::mutex> lock(mutex);

    stepCount++;
    std::string sliceType = isSSEVector(slice) ? "SSE" : "AVX";
    std::string opName = sliceType + "_slice_" + op;
    opCounts[opName]++;

    std::stringstream desc;
    desc << sliceType << " Slice " << op << " at index " << index;
    if (op == "get") {
        desc << "\n  Content: " << formatSliceElement(slice, index);
    }

    eventLogger->logEvent(
        EventLogger::EventType::SLICE_OP,
        desc.str(),
        std::to_string(stepCount)
    );

    if (currentMode == Mode::BREAKPOINT && 
        breakpointMgr->checkBreakpoint(opName)) {
        handleBreakpoint(opName);
    }

    if (currentMode == Mode::STEP) {
        handleStep(opName);
    }
}

int KernelDebugger::addBreakpoint(const std::string& location, 
                                 std::function<bool()> condition) {
    return breakpointMgr->addBreakpoint(
        BreakpointManager::Type::INSTRUCTION,
        location,
        condition
    );
}

void KernelDebugger::removeBreakpoint(int id) {
    breakpointMgr->removeBreakpoint(id);
}

void KernelDebugger::setMode(Mode mode) {
    std::lock_guard<std::mutex> lock(mutex);
    
    currentMode = mode;
    std::string modeStr = getModeString(mode);
    
    std::cout << "Debugger mode changed to: " << modeStr << std::endl;
    
    if (mode == Mode::STEP) {
        std::cout << "Step-by-step execution enabled.\n";
    }

    eventLogger->logEvent(
        EventLogger::EventType::BREAKPOINT,
        "Mode changed to " + modeStr,
        std::to_string(stepCount)
    );
}

void KernelDebugger::handleBreakpoint(const std::string& location) {
    std::cout << "\nBreakpoint hit at " << location << " (step " << stepCount << ")\n";
    printCurrentState();
    waitForCommand();
}

void KernelDebugger::handleStep(const std::string& location) {
    std::cout << "\nStep " << stepCount << ": " << location << "\n";
    printCurrentState();
    waitForCommand();
}

void KernelDebugger::printCurrentState() const {
    eventLogger->printEventHistory(1);  // Print the most recent event
    memoryTracker->printCurrentState();
}

void KernelDebugger::printSummary() const {
    std::cout << "\n=== Debug Session Summary ===\n";
    std::cout << "Total steps executed: " << stepCount << "\n\n";

    // Print operation statistics
    std::cout << "Operation Counts:\n";
    for (const auto& [op, count] : opCounts) {
        std::cout << std::setw(25) << std::left << op << ": " 
                  << std::setw(6) << count << "\n";
    }

    // Print memory statistics and recent events
    memoryTracker->printSummary();
    eventLogger->printEventHistory(1);  // Print the most recent event
}

void KernelDebugger::waitForCommand() {
    std::cout << "\nDebugger Commands:\n"
              << "  [Enter] - Continue\n"
              << "  p      - Print full state\n"
              << "  m      - Print memory details\n"
              << "  b      - List breakpoints\n"
              << "  q      - Quit debugging\n"
              << "Command: ";

    std::string cmd;
    std::getline(std::cin, cmd);

    switch (cmd[0]) {
        case 'p':
            eventLogger->printEventHistory(10);
            break;
        case 'm':
            memoryTracker->printSummary();
            break;
        case 'b':
            breakpointMgr->listBreakpoints();
            break;
        case 'q':
            stop();
            exit(0);
            break;
    }
}

std::string KernelDebugger::getModeString(Mode mode) const {
    switch (mode) {
        case Mode::RUN: return "Run";
        case Mode::STEP: return "Step";
        case Mode::BREAKPOINT: return "Breakpoint";
        default: return "Unknown";
    }
}

bool KernelDebugger::isSSEVector(const void* vec) const {
    return (reinterpret_cast<uintptr_t>(vec) % 64) != 0;
}

std::string KernelDebugger::formatSIMDVector(const void* vec, SIMDOp op) const {
    std::stringstream ss;
    
    if (op == SIMDOp::ADD_AVX || op == SIMDOp::MUL_AVX) {
        auto* v = static_cast<const __m512d*>(vec);
        alignas(64) double values[8];
        _mm512_store_pd(values, *v);
        ss << "[";
        for (int i = 0; i < 8; i++) {
            if (i > 0) ss << ", ";
            ss << values[i];
        }
        ss << "]";
    } else {
        auto* v = static_cast<const __m256d*>(vec);
        alignas(32) double values[4];
        _mm256_store_pd(values, *v);
        ss << "[";
        for (int i = 0; i < 4; i++) {
            if (i > 0) ss << ", ";
            ss << values[i];
        }
        ss << "]";
    }
    
    return ss.str();
}

std::string KernelDebugger::formatSliceElement(const void* slice, size_t index) const {
    if (isSSEVector(slice)) {
        auto s = static_cast<const sse_slice_t*>(slice);
        return formatSIMDVector(&s->data[index], SIMDOp::ADD_SSE);
    } else {
        auto s = static_cast<const avx_slice_t*>(slice);
        return formatSIMDVector(&s->data[index], SIMDOp::ADD_AVX);
    }
}