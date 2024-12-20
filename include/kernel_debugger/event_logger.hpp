#ifndef KERNEL_DEBUGGER_EVENT_LOGGER_HPP
#define KERNEL_DEBUGGER_EVENT_LOGGER_HPP

#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include <mutex>
#include <map>
#include <functional>

class EventLogger {
public:
    enum class EventType {
        SIMD_OP,      // SIMD operation execution
        MEMORY_OP,    // Memory allocation/access
        BREAKPOINT,   // Breakpoint hit
        STEP,         // Single step execution
        ERROR         // Error condition
    };

    struct Event {
        EventType type;
        std::string description;
        std::string location;  // file:line
        std::chrono::system_clock::time_point timestamp;
        
        // Optional fields for specific event types
        void* address{nullptr};
        size_t size{0};
        std::string extraInfo;
        
        Event(EventType t, 
              const std::string& desc, 
              const std::string& loc = "",
              void* addr = nullptr, 
              size_t sz = 0, 
              const std::string& extra = "");

        std::string getFormattedTime() const;
        std::string toString() const;
    };

    explicit EventLogger(size_t maxEvents = 1000);

    // Core logging functionality
    void logEvent(EventType type, 
                 const std::string& description,
                 const std::string& location = "",
                 void* address = nullptr,
                 size_t size = 0,
                 const std::string& extraInfo = "");

    // Event retrieval and display
    void printEvent(const Event& event) const;
    void printLastEvents(size_t count = 10) const;
    void generateSummary() const;
    
    // Event filtering
    std::vector<Event> getEventsByType(EventType type) const;
    std::vector<Event> getEventsInTimeRange(
        std::chrono::system_clock::time_point start,
        std::chrono::system_clock::time_point end) const;

    // Configuration
    void setRealTimeLogging(bool enable) { realTimeLogging = enable; }
    void clear();
    void setEnabled(bool enable) { enabled = enable; }
    
    // Event notification
    void addListener(std::function<void(const Event&)> listener);
    void clearListeners();

private:
    std::vector<Event> events;
    std::vector<std::function<void(const Event&)>> listeners;
    const size_t maxEvents;
    bool enabled{true};
    bool realTimeLogging{false};
    mutable std::mutex mutex;

    // Helper methods
    void notifyListeners(const Event& event);
    static std::string getEventTypeName(EventType type);
    void pruneOldEvents();
};

#endif // KERNEL_DEBUGGER_EVENT_LOGGER_HPP