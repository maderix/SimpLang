#pragma once
#include <string>
#include <vector>
#include <chrono>
#include <fstream>

class TraceEvents {
public:
    struct Event {
        std::string name;
        std::string cat;  // category
        std::string ph;   // phase (B=begin, E=end, X=complete)
        uint64_t ts;      // timestamp in microseconds
        uint64_t dur;     // duration in microseconds (for complete events)
        uint32_t tid;     // thread id
        uint32_t pid;     // process id
    };

    void beginEvent(const std::string& name, const std::string& category = "kernel");
    void endEvent(const std::string& name);
    void completeEvent(const std::string& name, uint64_t duration_us, 
                      const std::string& category = "kernel");
    void writeToFile(const std::string& path);

private:
    std::vector<Event> events;
    std::chrono::steady_clock::time_point start_time;
    uint64_t getTimestamp();
}; 