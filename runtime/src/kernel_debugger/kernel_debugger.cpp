#include "kernel_debugger/debugger.hpp"
#include "kernel_debugger/debug_events.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>

KernelDebugger& KernelDebugger::getInstance() {
    static KernelDebugger instance;
    return instance;
}

KernelDebugger::KernelDebugger() 
    : sourceManager(this)
    , currentFile("")      // Initialize currentFile to empty string
    , currentFunction("") // Initialize currentFunction to empty string
    , isRunning(false)
    , pauseFlag(false)
    , stepCount(0)
    , stepStopDepth(-1)
{
    initializeComponents();
}
void KernelDebugger::initializeComponents() {
        // Initialize configuration
        config = DebuggerConfig::getInstance();

        // Initialize UI Helper
        uiHelper = std::make_unique<UIHelper>(UIHelper::PromptOptions(
            config->ui().prompt,
            config->ui().enableHistory,
            config->ui().enableCompletion,
            config->ui().maxHistorySize
        )); 

        // Initialize Memory Tracker
        memoryTracker = std::make_shared<MemoryTracker>();
        memoryTracker->enableTracking(config->debug().enableMemoryTracking);

        // Initialize Event Logger
        eventLogger = std::make_shared<EventLogger>(config->debug().maxEventLogSize);
        eventLogger->setEnabled(config->debug().logDebugEvents);

        // Initialize Command Processor - moved to after other components
        cmdProcessor = std::make_unique<CommandProcessor>(*this);

        // Initialize Event Handler
        initializeEventHandler();
        
        isInitialized = true;
    }

KernelDebugger::~KernelDebugger() {
    stop();
    cleanupEventHandler();
}

void KernelDebugger::initialize() {
    std::cout << "DEBUG: KernelDebugger::initialize() start\n";
    std::lock_guard<std::mutex> lock(mutex);
    
    isRunning = false;
    stepCount = 0;
    currentMode = Mode::RUN;
    stepStopDepth = -1;
    vectorRegs.clear();
    opStats.clear();
    pauseFlag = false;
    currentFile = "";
    currentFunction = "";
    
    if (eventLogger) {
        eventLogger->clear();
    }
    if (memoryTracker) {
        memoryTracker->reset();
    }
    
    sourceManager.reset();
    breakpointMgr.clearAllBreakpoints();
    callStack.clear();  // Clear call stack
    std::cout << "DEBUG: KernelDebugger::initialize() complete\n";
}

void KernelDebugger::performInitialization() {
        isRunning = false;
        stepCount = 0;
        currentMode = Mode::RUN;
        stepStopDepth = -1;
        vectorRegs.clear();
        opStats.clear();
        pauseFlag = false;
        currentFile = "";
        currentFunction = "";
        
        if (eventLogger) {
            eventLogger->clear();
        }
        if (memoryTracker) {
            memoryTracker->reset();
        }
        
        sourceManager.reset();
        breakpointMgr.clearAllBreakpoints();
        callStack.clear();
}

bool KernelDebugger::loadKernel(const std::string& filename) {
    std::lock_guard<std::mutex> lock(mutex);
    
    if (sourceManager.setKernel(filename)) {
        logDebugEvent("loadKernel", "Loaded " + filename);
        return true;
    }
    return false;
}

void KernelDebugger::start() {
    std::cout << "DEBUG: KernelDebugger::start() begin\n";
    
    {
        std::lock_guard<std::mutex> lock(mutex);
        isRunning = true;
        std::cout << "DEBUG: Set isRunning to true\n";
    }
    
    logDebugEvent("start", "Debugger started");
    std::cout << "DEBUG: Logged start event\n";
    
    auto loc = sourceManager.getCurrentLocation();
    std::cout << "DEBUG: Got current location: " << loc.file << ":" << loc.line << "\n";
    
    if (loc.line > 0) {
        std::cout << "DEBUG: About to print location\n";
        printLocation();
        std::cout << "DEBUG: Location printed\n";
    } else {
        if (uiHelper) {
            std::cout << "DEBUG: About to print ready message\n";
            uiHelper->printInfo("Ready to load source file");
            std::cout << "DEBUG: Ready message printed\n";
        }
    }
    std::cout << "DEBUG: KernelDebugger::start() complete\n";
}

void KernelDebugger::stop() {
    std::lock_guard<std::mutex> lock(mutex);
    
    isRunning = false;
    logDebugEvent("stop", "Debugger stopped");
}

void KernelDebugger::executeNextInstruction() {
    std::cout << "DEBUG: KernelDebugger::executeNextInstruction - start\n";
    if (!isRunning) {
        std::cout << "DEBUG: Not running, returning\n";
        return;
    }

    auto loc = sourceManager.getCurrentLocation();
    std::cout << "DEBUG: Executing line " << loc.line << " in " << loc.file << "\n";
    
    // Execute the line
    sourceManager.executeCurrentLine();
    std::cout << "DEBUG: Line executed\n";

    // Get current values from memory tracker
    if (memoryTracker) {
        std::cout << "Current variable values:\n";
        auto activeVars = memoryTracker->getActiveVariables();
        for (const auto& var : activeVars) {
            if (var.isActive) {
                if (var.valueHistory.empty()) continue;
                
                // Get most recent value
                const auto& lastValue = var.valueHistory.back().second;
                std::cout << "  " << var.name << " = " << lastValue << "\n";
            }
        }
    }

    sourceManager.advanceLine();
    std::cout << "DEBUG: Advanced to next line\n";
    
    stepCount++;
    std::cout << "DEBUG: Step count: " << stepCount << "\n";
}

void KernelDebugger::continueExecution() {
    std::cout << "DEBUG: KernelDebugger::continueExecution - start\n";
    
    // Check if we're running
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!isRunning) {
            std::cout << "DEBUG: Debugger is not running\n";
            uiHelper->printError("Debugger is not running");
            return;
        }
    }

    std::cout << "DEBUG: Setting mode to RUN\n";
    currentMode = Mode::RUN;
    stepStopDepth = -1;
    
    std::cout << "DEBUG: About to resume execution\n";
    pauseFlag = false;  // Clear pause flag
    std::cout << "DEBUG: Resumed execution\n";
    
    while (!isExecutionComplete() && !shouldBreak()) {
        std::cout << "DEBUG: Executing next instruction\n";
        executeNextInstruction();
    }
    
    std::cout << "DEBUG: Execution loop complete\n";
    if (!isExecutionComplete()) {
        printLocation();
        std::cout << "DEBUG: Location printed\n";
    } else {
        uiHelper->printInfo("Program finished");
        std::cout << "DEBUG: Program finished message printed\n";
    }
    std::cout << "DEBUG: KernelDebugger::continueExecution - end\n";
}

void KernelDebugger::stepIn() {
    std::lock_guard<std::mutex> lock(mutex);
    
    if (!isRunning) return;
    
    currentMode = Mode::STEP;
    executeNextInstruction();
    printLocation();
}

void KernelDebugger::stepOver() {
    std::lock_guard<std::mutex> lock(mutex);
    
    if (!isRunning) return;
    
    currentMode = Mode::NEXT;
    stepStopDepth = callStack.getDepth();
    
    do {
        executeNextInstruction();
    } while (!isExecutionComplete() && !shouldBreak());
    
    printLocation();
}

void KernelDebugger::stepOut() {
    std::lock_guard<std::mutex> lock(mutex);
    
    if (!isRunning || callStack.getDepth() == 0) return;
    
    currentMode = Mode::FINISH;
    stepStopDepth = callStack.getDepth() - 1;
    
    while (!isExecutionComplete() && !shouldBreak()) {
        executeNextInstruction();
    }
    
    printLocation();
}

int KernelDebugger::addBreakpoint(const std::string& file, int line, const std::string& condition) {
    std::lock_guard<std::mutex> lock(mutex);
    
    if (!validateBreakpoint(file, line)) {
        return -1;
    }
    
    int id = breakpointMgr.addBreakpoint(file, line, condition);
    if (id >= 0) {
        logDebugEvent("breakpoint", "Added breakpoint " + std::to_string(id) + 
                     " at " + file + ":" + std::to_string(line));
    }
    return id;
}

bool KernelDebugger::removeBreakpoint(int id) {
    std::lock_guard<std::mutex> lock(mutex);
    if (breakpointMgr.removeBreakpoint(id)) {
        logDebugEvent("breakpoint", "Removed breakpoint " + std::to_string(id));
        return true;
    }
    return false;
}

void KernelDebugger::enableBreakpoint(int id, bool enable) {
    std::lock_guard<std::mutex> lock(mutex);
    breakpointMgr.enableBreakpoint(id, enable);
}

void KernelDebugger::onSIMDOperation(SIMDOp op, void* result, void* op1, void* op2) {
    std::lock_guard<std::mutex> lock(mutex);
    
    opStats[op]++;
    
    std::string opStr;
    switch (op) {
        case SIMDOp::ADD_SSE: opStr = "SSE Addition"; break;
        case SIMDOp::MUL_SSE: opStr = "SSE Multiplication"; break;
        case SIMDOp::ADD_AVX: opStr = "AVX Addition"; break;
        case SIMDOp::MUL_AVX: opStr = "AVX Multiplication"; break;
    }

    bool isAVX = (op == SIMDOp::ADD_AVX || op == SIMDOp::MUL_AVX);
    
    if (eventLogger) {
        eventLogger->logEvent(
            EventLogger::EventType::SIMD_OP,
            opStr,
            sourceManager.getCurrentLocation().file + ":" + 
            std::to_string(sourceManager.getCurrentLocation().line),
            result,
            isAVX ? sizeof(__m512d) : sizeof(__m256d)
        );
    }

    // Post SIMD operation event
    postEvent(DebugEvent(DebugEvent::Type::SIMD_OPERATION, opStr, 
        sourceManager.getCurrentLocation().file,
        sourceManager.getCurrentLocation().line,
        std::make_tuple(result, op1, op2)));
}

void KernelDebugger::onDebugEvent(const std::string& event, 
                                 const std::string& file, 
                                 int line, 
                                 const std::string& details) {
    if (event == "line") {
        postEvent(DebugEvent(DebugEvent::Type::SOURCE_LINE, details, file, line));
    }
    else if (event == "enterFunction") {
        postEvent(DebugEvent(DebugEvent::Type::FUNCTION_ENTRY, details, file, line));
    }
    else if (event == "exitFunction") {
        postEvent(DebugEvent(DebugEvent::Type::FUNCTION_EXIT, details, file, line));
    }
}

void KernelDebugger::printLocation() const {
    std::cout << "DEBUG: KernelDebugger::printLocation - start\n";
    
    // Get location without lock
    auto loc = sourceManager.getCurrentLocation();
    std::cout << "DEBUG: Got location: " << loc.file << ":" << loc.line << "\n";
    
    if (uiHelper) {
        std::cout << "DEBUG: About to print location info\n";
        uiHelper->printInfo(loc.file + ":" + std::to_string(loc.line) + 
                          (loc.function.empty() ? "" : " in " + loc.function));
        std::cout << "DEBUG: Location info printed\n";
    }

    if (config->display().showSourceOnBreak) {
        std::cout << "DEBUG: About to show source\n";
        int contextLines = config->display().contextLines;
        showSource("", loc.line - contextLines/2, contextLines);
        std::cout << "DEBUG: Source shown\n";
    }
    std::cout << "DEBUG: KernelDebugger::printLocation - end\n";
}

void KernelDebugger::printVectorState() const {
    std::lock_guard<std::mutex> lock(mutex);
    
    uiHelper->printInfo("\nVector Registers:");
    for (const auto& [name, reg] : vectorRegs) {
        std::cout << std::setw(10) << name << ": " << formatVectorRegister(reg) << "\n";
    }
    
    uiHelper->printInfo("\nSIMD Operation Stats:");
    for (const auto& [op, count] : opStats) {
        std::string opStr;
        switch (op) {
            case SIMDOp::ADD_SSE: opStr = "ADD_SSE"; break;
            case SIMDOp::MUL_SSE: opStr = "MUL_SSE"; break;
            case SIMDOp::ADD_AVX: opStr = "ADD_AVX"; break;
            case SIMDOp::MUL_AVX: opStr = "MUL_AVX"; break;
        }
        std::cout << std::left << std::setw(10) << opStr << ": " << count << "\n";
    }
}

bool KernelDebugger::shouldBreak() const {
    std::cout << "DEBUG: Checking should break\n";
    return shouldBreakAtCurrentLocation() || pauseFlag;
}

bool KernelDebugger::shouldBreakAtCurrentLocation() const {
    auto loc = sourceManager.getCurrentLocation();
    
    if (breakpointMgr.shouldBreak(loc.file, loc.line)) {
        return true;
    }
    
    switch (currentMode) {
        case Mode::STEP:
            return true;
        case Mode::NEXT:
            return callStack.getDepth() <= static_cast<size_t>(stepStopDepth);
        case Mode::FINISH:
            return callStack.getDepth() == static_cast<size_t>(stepStopDepth);
        default:
            return false;
    }
}

bool KernelDebugger::isExecutionComplete() const {
    std::cout << "DEBUG: Checking execution completion\n";
    return !isRunning || sourceManager.isAtEnd();
}


std::string KernelDebugger::formatVectorRegister(const VectorRegister& reg) const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);
    
    if (reg.is_avx) {
        alignas(64) double values[8];
        _mm512_store_pd(values, reg.value.avx);
        ss << "[" << values[0];
        for (int i = 1; i < 8; i++) {
            ss << ", " << values[i];
        }
    } else {
        alignas(32) double values[4];
        _mm256_store_pd(values, reg.value.sse);
        ss << "[" << values[0];
        for (int i = 1; i < 4; i++) {
            ss << ", " << values[i];
        }
    }
    ss << "]";
    return ss.str();
}

void KernelDebugger::validateMemoryAccess(void* addr, size_t size) const {
    if (!memoryTracker->isValidPointer(addr)) {
        throw std::runtime_error("Invalid memory access");
    }
    
    size_t allocSize = memoryTracker->getAllocationSize(addr);
    if (size > allocSize) {
        throw std::runtime_error("Memory access out of bounds");
    }
}

bool KernelDebugger::validateBreakpoint(const std::string& file, int line) const {
    if (!sourceManager.hasSource(file)) {
        uiHelper->printError("No such file: " + file);
        return false;
    }
    
    if (line <= 0 || line > sourceManager.getLineCount(file)) {
        uiHelper->printError("Invalid line number: " + std::to_string(line));
        return false;
    }
    
    return true;
}

void KernelDebugger::logDebugEvent(const std::string& event, const std::string& details) {
    if (eventLogger) {
        auto loc = sourceManager.getCurrentLocation();
        eventLogger->logEvent(
            EventLogger::EventType::STEP,
            event,
            loc.file + ":" + std::to_string(loc.line),
            nullptr, 0, details
        );
    }
}
std::string KernelDebugger::getSIMDOpName(SIMDOp op) const {
    switch (op) {
        case SIMDOp::ADD_SSE: return "SSE Add";
        case SIMDOp::MUL_SSE: return "SSE Multiply";
        case SIMDOp::ADD_AVX: return "AVX Add";
        case SIMDOp::MUL_AVX: return "AVX Multiply";
        default: return "Unknown SIMD Operation";
    }
}
size_t KernelDebugger::getSimdOperandSize(SIMDOp op) const {
    switch (op) {
        case SIMDOp::ADD_SSE:
        case SIMDOp::MUL_SSE:
            return sizeof(__m256d);  // SSE operations use 256-bit vectors
        case SIMDOp::ADD_AVX:
        case SIMDOp::MUL_AVX:
            return sizeof(__m512d);  // AVX operations use 512-bit vectors
        default:
            return 0;
    }
}

bool KernelDebugger::checkAlignment(void* ptr, size_t required) const {
    return reinterpret_cast<uintptr_t>(ptr) % required == 0;
}

// Add missing function implementations
void KernelDebugger::showSource(const std::string& file, int line, int count) const {
    std::cout << "DEBUG: KernelDebugger::showSource - start\n";
    std::cout << "DEBUG: file=" << file << ", line=" << line << ", count=" << count << "\n";
    
    std::string sourceFile = file.empty() ? sourceManager.getCurrentFile() : file;
    std::cout << "DEBUG: Using source file: " << sourceFile << "\n";
    
    // If line is negative, center around current line
    if (line < 0) {
        auto loc = sourceManager.getCurrentLocation();
        line = loc.line + line;  // line is negative, so this centers around current line
        std::cout << "DEBUG: Adjusted line to: " << line << "\n";
    }
    
    // Ensure line is positive
    line = std::max(1, line);
    std::cout << "DEBUG: Final line number: " << line << "\n";
    
    try {
        std::cout << "DEBUG: About to print lines\n";
        sourceManager.printLines(sourceFile, line, count, std::cout);
        std::cout << "DEBUG: Lines printed\n";
    } catch (const std::exception& e) {
        std::cerr << "Error showing source: " << e.what() << std::endl;
    }
    std::cout << "DEBUG: KernelDebugger::showSource - end\n";
}

void KernelDebugger::onMemoryAccess(void* addr, size_t size, bool isWrite) {
    if (memoryTracker) {
        memoryTracker->trackAccess(addr, size, isWrite);
    }
}

void KernelDebugger::resumeExecution() {
    pauseFlag = false;
}

void KernelDebugger::printBacktrace() const {
    std::lock_guard<std::mutex> lock(mutex);
    callStack.printBacktrace();
}

void KernelDebugger::initializeEventHandler() {
    eventQueue = std::make_unique<DebugEventQueue>();
    eventThread = std::thread([this]() { eventLoop(); });
}

void KernelDebugger::postEvent(DebugEvent event) {
    if (eventQueue) {
        eventQueue->push(std::move(event));
    }
}

void KernelDebugger::printLocals() const {
    std::lock_guard<std::mutex> lock(mutex);
    callStack.printLocals();
}

void KernelDebugger::printExpression(const std::string& expr) const {
    // For now, just handle simple variable names
    const auto* var = callStack.getLocal(expr);
    if (var) {
        std::cout << expr << " = ";
        callStack.printLocals();
    } else {
        uiHelper->printError("Unknown variable: " + expr);
    }
}

void KernelDebugger::displayMemory(void* addr, size_t size) const {
    std::lock_guard<std::mutex> lock(mutex);
    
    try {
        validateMemoryAccess(addr, size);
        
        // Display memory in hex format
        const uint8_t* bytes = static_cast<const uint8_t*>(addr);
        for (size_t i = 0; i < size; i += 16) {
            // Print address
            std::cout << std::hex << std::setw(8) << std::setfill('0') 
                      << reinterpret_cast<uintptr_t>(addr) + i << ": ";
            
            // Print hex values
            for (size_t j = 0; j < 16 && (i + j) < size; j++) {
                std::cout << std::hex << std::setw(2) << std::setfill('0') 
                          << static_cast<int>(bytes[i + j]) << " ";
            }
            
            // Print ASCII representation
            std::cout << "  ";
            for (size_t j = 0; j < 16 && (i + j) < size; j++) {
                char c = bytes[i + j];
                std::cout << (std::isprint(c) ? c : '.');
            }
            std::cout << "\n";
        }
    } catch (const std::exception& e) {
        uiHelper->printError(e.what());
    }
}

void KernelDebugger::cleanupEventHandler() {
    if (eventQueue) {
        eventQueue->stop();
    }
    if (eventThread.joinable()) {
        eventThread.join();
    }
}

void KernelDebugger::listBreakpoints() const {
    std::lock_guard<std::mutex> lock(mutex);
    
    auto bps = breakpointMgr.getAllBreakpoints();
    if (bps.empty()) {
        uiHelper->printInfo("No breakpoints set");
        return;
    }
    
    std::cout << "Num    Type           Disp    Location\n";
    for (const auto& bp : bps) {
        std::cout << std::left << std::setw(7) << bp.id 
                  << std::setw(14) << "breakpoint"
                  << std::setw(8) << (bp.enabled ? "enable" : "disable")
                  << bp.file << ":" << bp.line;
        if (!bp.condition.empty()) {
            std::cout << " if " << bp.condition;
        }
        std::cout << "\n";
    }
}

// Add event loop function
void KernelDebugger::eventLoop() {
    while (isRunning) {
        DebugEvent event;
        if (eventQueue && eventQueue->pop(event)) {
            // Process the event based on its type
            switch (event.type) {
                case DebugEvent::Type::BREAKPOINT_HIT:
                    pauseFlag = true;
                    printLocation();
                    break;
                case DebugEvent::Type::SOURCE_LINE:
                    updateLocation(event.file, event.line);
                    break;
                case DebugEvent::Type::SIMD_OPERATION:
                case DebugEvent::Type::MEMORY_ACCESS:
                case DebugEvent::Type::ERROR:
                    // Log these events if event logger is enabled
                    if (eventLogger) {
                        eventLogger->logEvent(
                            EventLogger::EventType::SIMD_OP,
                            event.description,
                            event.file + ":" + std::to_string(event.line)
                        );
                    }
                    break;
                default:
                    break;
            }
        }
    }
}
void KernelDebugger::updateLocation(const std::string& file, int line) {
    std::lock_guard<std::mutex> lock(mutex);
    
    // Update source manager location
    sourceManager.setLocation(file, line, currentFunction);
    
    // Update call stack if needed
    if (!callStack.isEmpty()) {
        callStack.updateLocation(file, line);
    }
    
    // Log the location update if event logger is enabled
    if (eventLogger) {
        eventLogger->logEvent(
            EventLogger::EventType::STEP,
            "Location update",
            file + ":" + std::to_string(line)
        );
    }
    
    // Post location update event
    postEvent(DebugEvent(
        DebugEvent::Type::SOURCE_LINE,
        "Location update",
        file,
        line
    ));
}

void KernelDebugger::setCurrentFunction(const std::string& function) {
    std::lock_guard<std::mutex> lock(mutex);
    currentFunction = function;
    
    // Log function change if event logger is enabled
    if (eventLogger) {
        eventLogger->logEvent(
            EventLogger::EventType::STEP,
            "Function change",
            getCurrentFile() + ":" + std::to_string(sourceManager.getCurrentLocation().line),
            nullptr, 0,
            "Entered function: " + function
        );
    }
}