#include "kernel_debugger/debugger.hpp"
#include <chrono>
#include <thread>

class DebugLoop {
private:
    KernelDebugger& debugger;
    std::atomic<bool> shouldStop{false};
    std::atomic<bool> isPaused{false};
    
    enum class StepResult {
        CONTINUE,
        BREAK,
        ERROR,
        COMPLETE
    };

public:
    explicit DebugLoop(KernelDebugger& dbg) : debugger(dbg) {}

    void run() {
        shouldStop = false;
        isPaused = false;

        while (!shouldStop) {
            if (!isPaused) {
                auto result = executeStep();
                switch (result) {
                    case StepResult::BREAK:
                        handleBreakpoint();
                        break;
                    case StepResult::ERROR:
                        handleError();
                        return;
                    case StepResult::COMPLETE:
                        handleCompletion();
                        return;
                    case StepResult::CONTINUE:
                        // Continue execution
                        break;
                }
            } else {
                // Small sleep to prevent busy waiting when paused
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }

    void stop() {
        shouldStop = true;
    }

    void pause() {
        isPaused = true;
    }

    void resume() {
        isPaused = false;
    }

private:
    StepResult executeStep() {
        try {
            // Get current location
            auto loc = debugger.sourceManager.getCurrentLocation();
            
            // Check for breakpoints
            if (debugger.breakpointMgr.shouldBreak(loc.file, loc.line)) {
                return StepResult::BREAK;
            }

            // Handle stepping modes
            switch (debugger.currentMode) {
                case KernelDebugger::Mode::STEP:
                    debugger.stepIn();
                    return StepResult::BREAK;

                case KernelDebugger::Mode::NEXT:
                    if (debugger.callStack.getDepth() <= debugger.stepStopDepth) {
                        return StepResult::BREAK;
                    }
                    debugger.stepOver();
                    break;

                case KernelDebugger::Mode::FINISH:
                    if (debugger.callStack.getDepth() == debugger.stepStopDepth) {
                        return StepResult::BREAK;
                    }
                    debugger.stepOut();
                    break;

                default:
                    debugger.executeNextInstruction();
                    break;
            }

            // Check if execution is complete
            if (debugger.isExecutionComplete()) {
                return StepResult::COMPLETE;
            }

            return StepResult::CONTINUE;
        }
        catch (const std::exception& e) {
            debugger.logDebugEvent("error", std::string("Execution error: ") + e.what());
            return StepResult::ERROR;
        }
    }

    void handleBreakpoint() {
        auto loc = debugger.sourceManager.getCurrentLocation();
        debugger.logDebugEvent("break", "Hit breakpoint at " + loc.file + ":" + std::to_string(loc.line));
        
        // Display current location
        debugger.printLocation();
        
        // If configured, show additional information
        auto& config = DebuggerConfig::getInstance();
        if (config.display().showRegistersOnBreak) {
            debugger.printVectorState();
        }
        
        pause();  // Pause execution for user input
    }

    void handleError() {
        debugger.logDebugEvent("error", "Execution stopped due to error");
        debugger.printLocation();
        pause();
    }

    void handleCompletion() {
        debugger.logDebugEvent("complete", "Program execution completed");
        pause();
    }
};

// Add these methods to KernelDebugger class
void KernelDebugger::runDebugLoop() {
    DebugLoop loop(*this);
    debugLoop = &loop;  // Store reference to debug loop
    
    // Start the debugging loop
    loop.run();
}

void KernelDebugger::pauseExecution() {
    if (debugLoop) {
        debugLoop->pause();
    }
}

void KernelDebugger::resumeExecution() {
    if (debugLoop) {
        debugLoop->resume();
    }
}

void KernelDebugger::stopExecution() {
    if (debugLoop) {
        debugLoop->stop();
    }
}

// Add this helper class for managing the debug loop lifecycle
class DebugSession {
private:
    KernelDebugger& debugger;
    std::unique_ptr<DebugLoop> loop;
    std::thread debugThread;

public:
    explicit DebugSession(KernelDebugger& dbg) : debugger(dbg) {}

    void start() {
        // Initialize the debug loop
        loop = std::make_unique<DebugLoop>(debugger);
        
        // Start the debug loop in a separate thread
        debugThread = std::thread([this]() {
            loop->run();
        });
    }

    void stop() {
        if (loop) {
            loop->stop();
            if (debugThread.joinable()) {
                debugThread.join();
            }
        }
    }

    ~DebugSession() {
        stop();
    }
};