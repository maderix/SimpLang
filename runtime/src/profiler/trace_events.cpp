#include "profiler/trace_events.hpp"
#include <thread>
#include <sstream>
#include <iomanip>
#include <unistd.h>

void TraceEvents::beginEvent(const std::string& name, const std::string& category) {
    Event event{
        .name = name,
        .cat = category,
        .ph = "B",
        .ts = getTimestamp(),
        .dur = 0,
        .tid = static_cast<uint32_t>(std::hash<std::thread::id>{}(std::this_thread::get_id())),
        .pid = static_cast<uint32_t>(getpid())
    };
    events.push_back(event);
}

void TraceEvents::endEvent(const std::string& name) {
    Event event{
        .name = name,
        .cat = events.back().cat,
        .ph = "E",
        .ts = getTimestamp(),
        .dur = 0,
        .tid = events.back().tid,
        .pid = events.back().pid
    };
    events.push_back(event);
}

void TraceEvents::completeEvent(const std::string& name, uint64_t duration_us, 
                              const std::string& category) {
    Event event{
        .name = name,
        .cat = category,
        .ph = "X",
        .ts = getTimestamp(),
        .dur = duration_us,
        .tid = static_cast<uint32_t>(std::hash<std::thread::id>{}(std::this_thread::get_id())),
        .pid = static_cast<uint32_t>(getpid())
    };
    events.push_back(event);
}

void TraceEvents::writeToFile(const std::string& path) {
    std::ofstream file(path);
    file << "{\n  \"traceEvents\": [\n";
    
    for (size_t i = 0; i < events.size(); ++i) {
        const auto& event = events[i];
        file << "    {\n";
        file << "      \"name\": \"" << event.name << "\",\n";
        file << "      \"cat\": \"" << event.cat << "\",\n";
        file << "      \"ph\": \"" << event.ph << "\",\n";
        file << "      \"ts\": " << event.ts << ",\n";
        if (event.ph == "X") {
            file << "      \"dur\": " << event.dur << ",\n";
        }
        file << "      \"tid\": " << event.tid << ",\n";
        file << "      \"pid\": " << event.pid << "\n";
        file << "    }" << (i < events.size() - 1 ? "," : "") << "\n";
    }
    
    file << "  ]\n}\n";
}

uint64_t TraceEvents::getTimestamp() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();
} 