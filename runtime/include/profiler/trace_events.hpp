#ifndef TRACE_EVENTS_HPP
#define TRACE_EVENTS_HPP

#include <string>
#include <vector>
#include <cstdint>
#include <chrono>

class TraceEvents {
public:
    struct Event {
        std::string name;
        std::string cat;  // category
        std::string ph;   // phase
        uint64_t ts;      // timestamp
        uint32_t tid;     // thread id
        uint32_t pid;     // process id
    };

    static TraceEvents& getInstance();
    
    void addFunctionEntry(const std::string& name, uint64_t timestamp);
    void addFunctionExit(const std::string& name, uint64_t timestamp);
    void addBlockEntry(const std::string& name, uint64_t timestamp);
    void addBlockExit(const std::string& name, uint64_t timestamp);
    void writeToFile(const std::string& filename);

    void handleFunctionEnter(void* func_addr, void* caller_addr);
    void handleFunctionExit(void* func_addr, void* caller_addr);
    void handleBlockEnter(const char* name);
    void handleBlockExit(const char* name);

private:
    std::vector<Event> events;
    uint64_t getTimestamp();
    std::chrono::steady_clock::time_point start_time;
};

// C-style exports for LLVM instrumentation
extern "C" {
    void __cyg_profile_func_enter(void* func_addr, void* caller_addr);
    void __cyg_profile_func_exit(void* func_addr, void* caller_addr);
    void __trace_block_enter(const char* name);
    void __trace_block_exit(const char* name);
}

#endif