#ifndef KERNEL_DEBUGGER_EVENT_LOGGER_HPP
#define KERNEL_DEBUGGER_EVENT_LOGGER_HPP

#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include <sstream>
#include <mutex>
#include <functional>
#include <fstream>
#include <numeric>

class EventLogger {
public:
    enum class EventType {
        SIMD_OP,
        MEMORY_OP,
        SLICE_OP,
        BREAKPOINT,
        ERROR
    };

    struct Event {
        EventType type;
        std::string description;
        std::string location;
        std::chrono::system_clock::time_point timestamp;
        void* address;
        size_t size;
        std::string extraInfo;
        
        Event(EventType t, const std::string& desc, const std::string& loc = "",
              void* addr = nullptr, size_t sz = 0, const std::string& extra = "")
            : type(t)
            , description(desc)
            , location(loc)
            , timestamp(std::chrono::system_clock::now())
            , address(addr)
            , size(sz)
            , extraInfo(extra) {}

        std::string getFormattedTime() const;
        std::string toString() const;
    };

    // Constructor
    EventLogger(size_t max_size = 1000)
        : logging_enabled(true)
        , max_events(max_size)
        , real_time_logging(false) {}

    // Event logging
    void logEvent(EventType type, 
                 const std::string& description,
                 const std::string& location = "",
                 void* address = nullptr,
                 size_t size = 0,
                 const std::string& extraInfo = "");

    // Event display
    void printEvent(const Event& event) const;
    void printEventHistory(size_t last_n = 0, const std::string& filter = "") const;
    void generateSummary() const;

    // Event filtering
    std::vector<Event> getEventsByType(EventType type) const;
    std::vector<Event> getEventsInRange(
        std::chrono::system_clock::time_point start,
        std::chrono::system_clock::time_point end) const;

    // Listener management
    void addListener(std::function<void(const Event&)> listener);
    void clearListeners();

    // Configuration
    void setRealTimeLogging(bool enable);
    void clear() { events.clear(); }
    void setLoggingEnabled(bool enable) { logging_enabled = enable; }

    // Export and analysis
    void exportEvents(const std::string& filename) const;
    void analyzePerformance() const;

private:
    std::vector<Event> events;
    std::vector<std::function<void(const Event&)>> listeners;
    bool logging_enabled;
    bool real_time_logging;
    size_t max_events;
    mutable std::mutex mutex;

    // Private helper methods
    void notifyListeners(const Event& event) const;
    std::string getEventTypeName(EventType type) const;
};

#endif // KERNEL_DEBUGGER_EVENT_LOGGER_HPP