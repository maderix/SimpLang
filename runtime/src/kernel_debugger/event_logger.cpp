#include "kernel_debugger/event_logger.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <map>
#include <chrono>
#include <ctime>
#include <cmath>

// Event struct member functions
std::string EventLogger::Event::getFormattedTime() const {
    auto timer = std::chrono::system_clock::to_time_t(timestamp);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&timer), "%Y-%m-%d %H:%M:%S");
    
    // Add milliseconds
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        timestamp.time_since_epoch()) % 1000;
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    
    return ss.str();
}

std::string EventLogger::Event::toString() const {
    std::stringstream ss;
    ss << "[" << getFormattedTime() << "] ";
    
    switch (type) {
        case EventType::SIMD_OP:
            ss << "[SIMD] ";
            break;
        case EventType::MEMORY_OP:
            ss << "[MEM] ";
            break;
        case EventType::SLICE_OP:
            ss << "[SLICE] ";
            break;
        case EventType::BREAKPOINT:
            ss << "[BRK] ";
            break;
        case EventType::ERROR:
            ss << "[ERROR] ";
            break;
    }

    ss << description;

    if (!location.empty()) {
        ss << " at " << location;
    }

    if (address) {
        ss << " (address: " << address;
        if (size > 0) {
            ss << ", size: " << size << " bytes";
        }
        ss << ")";
    }

    if (!extraInfo.empty()) {
        ss << "\n    " << extraInfo;
    }

    return ss.str();
}

// Implementation of EventLogger member functions
void EventLogger::logEvent(EventType type, 
                        const std::string& description,
                        const std::string& location,
                        void* address,
                        size_t size,
                        const std::string& extraInfo) {
    if (!logging_enabled) return;

    std::lock_guard<std::mutex> lock(mutex);
    
    events.emplace_back(type, description, location, address, size, extraInfo);

    // Keep event log size under control
    if (events.size() > max_events) {
        events.erase(events.begin(), events.begin() + events.size() - max_events);
    }

    // Print real-time event if enabled
    if (real_time_logging) {
        printEvent(events.back());
    }

    // Notify event listeners if any
    notifyListeners(events.back());
}

void EventLogger::printEvent(const Event& event) const {
    std::lock_guard<std::mutex> lock(mutex);
    std::cout << event.toString() << std::endl;
}

void EventLogger::printEventHistory(size_t last_n, const std::string& filter) const {
    std::lock_guard<std::mutex> lock(mutex);

    if (events.empty()) {
        std::cout << "No events recorded." << std::endl;
        return;
    }

    std::cout << "\nEvent History:";
    if (last_n > 0) {
        std::cout << " (Last " << std::min(last_n, events.size()) << " events)";
    }
    if (!filter.empty()) {
        std::cout << " [Filter: " << filter << "]";
    }
    std::cout << "\n" << std::string(80, '-') << std::endl;

    auto start = events.begin();
    if (last_n > 0 && last_n < events.size()) {
        start = events.end() - last_n;
    }

    for (auto it = start; it != events.end(); ++it) {
        if (filter.empty() || 
            it->description.find(filter) != std::string::npos ||
            it->location.find(filter) != std::string::npos) {
            printEvent(*it);
        }
    }
}

void EventLogger::generateSummary() const {
    std::lock_guard<std::mutex> lock(mutex);

    std::cout << "\nEvent Summary:\n" << std::string(80, '=') << std::endl;

    // Count events by type
    std::map<EventType, size_t> typeCounts;
    for (const auto& event : events) {
        typeCounts[event.type]++;
    }

    // Print event type statistics
    std::cout << "Event Counts by Type:\n";
    for (const auto& [type, count] : typeCounts) {
        std::cout << std::setw(15) << getEventTypeName(type) << ": " 
                  << count << std::endl;
    }

    // Calculate time statistics
    if (!events.empty()) {
        auto firstTime = events.front().timestamp;
        auto lastTime = events.back().timestamp;
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            lastTime - firstTime);

        std::cout << "\nTime Statistics:\n";
        std::cout << "Start time: " << events.front().getFormattedTime() << std::endl;
        std::cout << "End time:   " << events.back().getFormattedTime() << std::endl;
        std::cout << "Duration:   " << duration.count() << "ms" << std::endl;
        std::cout << "Events/sec: " << 
            (events.size() * 1000.0 / duration.count()) << std::endl;
    }

    // Print error statistics if any
    auto errors = getEventsByType(EventType::ERROR);
    if (!errors.empty()) {
        std::cout << "\nErrors (" << errors.size() << "):\n";
        for (const auto& error : errors) {
            std::cout << "- " << error.description << 
                        " at " << error.getFormattedTime() << std::endl;
        }
    }
}

std::vector<EventLogger::Event> EventLogger::getEventsByType(EventType type) const {
    std::lock_guard<std::mutex> lock(mutex);
    
    std::vector<Event> filtered;
    std::copy_if(events.begin(), events.end(), std::back_inserter(filtered),
                 [type](const Event& event) { return event.type == type; });
    return filtered;
}

std::vector<EventLogger::Event> EventLogger::getEventsInRange(
    std::chrono::system_clock::time_point start,
    std::chrono::system_clock::time_point end) const {
    std::lock_guard<std::mutex> lock(mutex);
    
    std::vector<Event> filtered;
    std::copy_if(events.begin(), events.end(), std::back_inserter(filtered),
                 [start, end](const Event& event) {
                     return event.timestamp >= start && event.timestamp <= end;
                 });
    return filtered;
}

void EventLogger::addListener(std::function<void(const Event&)> listener) {
    std::lock_guard<std::mutex> lock(mutex);
    listeners.push_back(listener);
}

void EventLogger::clearListeners() {
    std::lock_guard<std::mutex> lock(mutex);
    listeners.clear();
}

void EventLogger::setRealTimeLogging(bool enable) {
    real_time_logging = enable;
}

void EventLogger::notifyListeners(const Event& event) const {
    for (const auto& listener : listeners) {
        try {
            listener(event);
        } catch (const std::exception& e) {
            std::cerr << "Error in event listener: " << e.what() << std::endl;
        }
    }
}

std::string EventLogger::getEventTypeName(EventType type) const {
    switch (type) {
        case EventType::SIMD_OP: return "SIMD Operation";
        case EventType::MEMORY_OP: return "Memory Operation";
        case EventType::SLICE_OP: return "Slice Operation";
        case EventType::BREAKPOINT: return "Breakpoint";
        case EventType::ERROR: return "Error";
        default: return "Unknown";
    }
}

void EventLogger::exportEvents(const std::string& filename) const {
    std::lock_guard<std::mutex> lock(mutex);
    
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Failed to open file for export: " << filename << std::endl;
        return;
    }

    // Write CSV header
    file << "Timestamp,Type,Description,Location,Address,Size,ExtraInfo\n";

    // Write events
    for (const auto& event : events) {
        file << event.getFormattedTime() << ","
             << getEventTypeName(event.type) << ","
             << "\"" << event.description << "\","
             << "\"" << event.location << "\","
             << event.address << ","
             << event.size << ","
             << "\"" << event.extraInfo << "\"\n";
    }

    std::cout << "Events exported to: " << filename << std::endl;
}

void EventLogger::analyzePerformance() const {
    std::lock_guard<std::mutex> lock(mutex);

    if (events.empty()) {
        std::cout << "No events to analyze." << std::endl;
        return;
    }

    std::cout << "\nPerformance Analysis:\n" << std::string(80, '=') << std::endl;

    // Analyze operation timing
    std::map<std::string, std::vector<double>> operationTiming;
    std::map<std::string, size_t> operationCounts;

    for (size_t i = 1; i < events.size(); i++) {
        const auto& prev = events[i-1];
        const auto& curr = events[i];

        std::string opType = getEventTypeName(curr.type);
        double duration = std::chrono::duration_cast<std::chrono::microseconds>(
            curr.timestamp - prev.timestamp).count();

        operationTiming[opType].push_back(duration);
        operationCounts[opType]++;
    }

    // Print timing statistics for each operation type
    for (const auto& [opType, timings] : operationTiming) {
        if (timings.empty()) continue;

        // Calculate statistics
        double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
        double mean = sum / timings.size();
        
        auto minmax = std::minmax_element(timings.begin(), timings.end());
        double min = *minmax.first;
        double max = *minmax.second;

        // Calculate standard deviation
        double sq_sum = std::inner_product(timings.begin(), timings.end(), 
                                         timings.begin(), 0.0);
        double stdev = std::sqrt(sq_sum / timings.size() - mean * mean);

        std::cout << "\nOperation Type: " << opType << "\n";
        std::cout << "  Count:     " << operationCounts[opType] << "\n";
        std::cout << "  Avg Time:  " << std::fixed << std::setprecision(2) 
                  << mean << "µs\n";
        std::cout << "  Min Time:  " << min << "µs\n";
        std::cout << "  Max Time:  " << max << "µs\n";
        std::cout << "  Std Dev:   " << stdev << "µs\n";
    }
}