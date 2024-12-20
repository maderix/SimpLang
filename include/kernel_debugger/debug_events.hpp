// include/kernel_debugger/debug_events.hpp

#ifndef KERNEL_DEBUGGER_DEBUG_EVENTS_HPP
#define KERNEL_DEBUGGER_DEBUG_EVENTS_HPP

#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <any>
#include <memory>
#include <thread>

class KernelDebugger;

// Debug event structure
struct DebugEvent {
    enum class Type {
        BREAKPOINT_HIT,
        STEP_COMPLETE,
        SIMD_OPERATION,
        MEMORY_ACCESS,
        ERROR,
        SOURCE_LINE,
        FUNCTION_ENTRY,
        FUNCTION_EXIT
    };

    Type type;
    std::string description;
    std::string file;
    int line;
    std::any data;  // Additional context-specific data

    DebugEvent() = default;  // Default constructor needed for queue operations
    
    DebugEvent(Type t, std::string desc = "", std::string f = "", int l = 0, std::any d = std::any())
        : type(t), description(std::move(desc)), file(std::move(f)), line(l), data(std::move(d)) {}
};

// Thread-safe event queue
class DebugEventQueue {
public:
    void push(DebugEvent event) {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push(std::move(event));
        cv.notify_one();
    }

    bool pop(DebugEvent& event) {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this] { return !queue.empty() || stopped; });
        
        if (stopped && queue.empty()) {
            return false;
        }

        event = std::move(queue.front());
        queue.pop();
        return true;
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex);
        stopped = true;
        cv.notify_all();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex);
        std::queue<DebugEvent> empty;
        std::swap(queue, empty);
    }

private:
    std::queue<DebugEvent> queue;
    std::mutex mutex;
    std::condition_variable cv;
    bool stopped{false};
};

// Event handler interface
class IDebugEventHandler {
public:
    virtual ~IDebugEventHandler() = default;
    virtual void handleBreakpoint(const DebugEvent& event) = 0;
    virtual void handleStepComplete(const DebugEvent& event) = 0;
    virtual void handleSimdOperation(const DebugEvent& event) = 0;
    virtual void handleMemoryAccess(const DebugEvent& event) = 0;
    virtual void handleError(const DebugEvent& event) = 0;
};

// Debug event loop
class DebugEventLoop {
public:
    explicit DebugEventLoop(KernelDebugger& dbg) : debugger(dbg) {}

    void start() {
        running = true;
        eventThread = std::thread([this] { processEvents(); });
    }

    void stop() {
        running = false;
        eventQueue.stop();
        if (eventThread.joinable()) {
            eventThread.join();
        }
    }

    void postEvent(DebugEvent event) {
        eventQueue.push(std::move(event));
    }

private:
    friend class KernelDebugger;  // Allow KernelDebugger to access private members
    
    KernelDebugger& debugger;
    std::atomic<bool> running{false};
    std::thread eventThread;
    DebugEventQueue eventQueue;

    void processEvents();
    void handleEvent(const DebugEvent& event);
    void handleBreakpoint(const DebugEvent& event);
    void handleStepComplete(const DebugEvent& event);
    void handleSimdOperation(const DebugEvent& event);
    void handleMemoryAccess(const DebugEvent& event);
    void handleError(const DebugEvent& event);
};

#endif // KERNEL_DEBUGGER_DEBUG_EVENTS_HPP