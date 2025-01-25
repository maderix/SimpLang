#include "profiler/trace_events.hpp"
#include <fstream>
#include <iostream>
#include <chrono>
#include <dlfcn.h>

TraceEvents& TraceEvents::getInstance() {
    static TraceEvents instance;
    return instance;
}

uint64_t TraceEvents::getTimestamp() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(
        now - start_time
    ).count();
}

void TraceEvents::addFunctionEntry(const std::string& name, uint64_t timestamp) {
    Event event{};
    event.name = name;
    event.cat = "function";
    event.ph = "B";  // Begin event
    event.ts = timestamp;
    event.tid = 1;   // Single thread for now
    event.pid = 1;   // Single process
    events.push_back(std::move(event));
}

void TraceEvents::addFunctionExit(const std::string& name, uint64_t timestamp) {
    Event event{};
    event.name = name;
    event.cat = "function";
    event.ph = "E";  // End event
    event.ts = timestamp;
    event.tid = 1;
    event.pid = 1;
    events.push_back(std::move(event));
}

void TraceEvents::addBlockEntry(const std::string& name, uint64_t timestamp) {
    Event event{};
    event.name = name;
    event.cat = "block";
    event.ph = "B";
    event.ts = timestamp;
    event.tid = 1;
    event.pid = 1;
    events.push_back(std::move(event));
}

void TraceEvents::addBlockExit(const std::string& name, uint64_t timestamp) {
    Event event{};
    event.name = name;
    event.cat = "block";
    event.ph = "E";
    event.ts = timestamp;
    event.tid = 1;
    event.pid = 1;
    events.push_back(std::move(event));
}

void TraceEvents::writeToFile(const std::string& filename) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Failed to open trace file: " << filename << std::endl;
        return;
    }

    file << "{\n\"traceEvents\": [\n";
    
    bool first = true;
    for (const auto& event : events) {
        if (!first) {
            file << ",\n";
        }
        first = false;
        
        file << "  {\n";
        file << "    \"name\": \"" << event.name << "\",\n";
        file << "    \"cat\": \"" << event.cat << "\",\n";
        file << "    \"ph\": \"" << event.ph << "\",\n";
        file << "    \"ts\": " << event.ts << ",\n";
        file << "    \"tid\": " << event.tid << ",\n";
        file << "    \"pid\": " << event.pid << "\n";
        file << "  }";
    }
    
    file << "\n]\n}\n";
}

// New profiling hook implementations
void TraceEvents::handleFunctionEnter(void* func_addr, void* caller_addr) {
    Dl_info info;
    if (dladdr(func_addr, &info)) {
        addFunctionEntry(
            info.dli_sname ? info.dli_sname : "unknown",
            getTimestamp()
        );
    }
}

void TraceEvents::handleFunctionExit(void* func_addr, void* caller_addr) {
    Dl_info info;
    if (dladdr(func_addr, &info)) {
        addFunctionExit(
            info.dli_sname ? info.dli_sname : "unknown",
            getTimestamp()
        );
    }
}

void TraceEvents::handleBlockEnter(const char* name) {
    addBlockEntry(name, getTimestamp());
}

void TraceEvents::handleBlockExit(const char* name) {
    addBlockExit(name, getTimestamp());
}

// C-style exports implementation
extern "C" {
    void __cyg_profile_func_enter(void* func_addr, void* caller_addr) {
        TraceEvents::getInstance().handleFunctionEnter(func_addr, caller_addr);
    }

    void __cyg_profile_func_exit(void* func_addr, void* caller_addr) {
        TraceEvents::getInstance().handleFunctionExit(func_addr, caller_addr);
    }

    void __trace_block_enter(const char* name) {
        TraceEvents::getInstance().handleBlockEnter(name);
    }

    void __trace_block_exit(const char* name) {
        TraceEvents::getInstance().handleBlockExit(name);
    }
} 