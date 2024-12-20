// runtime/src/kernel_debugger/debug_events.cpp

#include "kernel_debugger/debug_events.hpp"
#include "kernel_debugger/debugger.hpp"
#include "kernel_debugger/memory_tracker.hpp"
#include "kernel_debugger/ui_helper.hpp"
#include <any>
#include <tuple>

void DebugEventLoop::processEvents() {
    while (running) {
        DebugEvent event;
        if (eventQueue.pop(event)) {
            handleEvent(event);
        }
    }
}

void DebugEventLoop::handleEvent(const DebugEvent& event) {
    try {
        switch (event.type) {
            case DebugEvent::Type::BREAKPOINT_HIT:
                handleBreakpoint(event);
                break;
            
            case DebugEvent::Type::STEP_COMPLETE:
                handleStepComplete(event);
                break;
            
            case DebugEvent::Type::SIMD_OPERATION:
                handleSimdOperation(event);
                break;
            
            case DebugEvent::Type::MEMORY_ACCESS:
                handleMemoryAccess(event);
                break;
            
            case DebugEvent::Type::ERROR:
                handleError(event);
                break;

            case DebugEvent::Type::SOURCE_LINE:
                debugger.printLocation();
                break;

            case DebugEvent::Type::FUNCTION_ENTRY:
                debugger.printInfo("Entering function: " + event.description);
                break;

            case DebugEvent::Type::FUNCTION_EXIT:
                debugger.printInfo("Exiting function: " + event.description);
                break;
        }
    } catch (const std::exception& e) {
        debugger.printError(std::string("Error handling event: ") + e.what());
    }
}

void DebugEventLoop::handleBreakpoint(const DebugEvent& event) {
    // Stop execution and print breakpoint info
    debugger.printInfo("Breakpoint hit: " + event.description);
    debugger.printLocation();
    debugger.setMode(KernelDebugger::Mode::STEP); // Force step mode on breakpoint
}

void DebugEventLoop::handleStepComplete(const DebugEvent& /* event */) {
    if (debugger.getCurrentMode() != KernelDebugger::Mode::RUN) {
        debugger.printLocation();
    }
}

void DebugEventLoop::handleSimdOperation(const DebugEvent& event) {
    try {
        auto [result, op1, op2] = std::any_cast<std::tuple<void*, void*, void*>>(event.data);
        debugger.onSIMDOperation(
            KernelDebugger::SIMDOp::ADD_SSE,  // You might want to make this dynamic
            result, 
            op1, 
            op2
        );
    } catch (const std::bad_any_cast& e) {
        debugger.printError("Invalid SIMD operation data");
    }
}

void DebugEventLoop::handleMemoryAccess(const DebugEvent& event) {
    try {
        auto [addr, size] = std::any_cast<std::pair<void*, size_t>>(event.data);
        debugger.onMemoryAccess(addr, size, event.description == "write");
    } catch (const std::bad_any_cast& e) {
        debugger.printError("Invalid memory access data");
    }
}

void DebugEventLoop::handleError(const DebugEvent& event) {
    debugger.printError(event.description);
}