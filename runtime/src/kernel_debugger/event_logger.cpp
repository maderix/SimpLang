#include "kernel_debugger/event_logger.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>

EventLogger::Event::Event(EventType t, 
                         const std::string& desc, 
                         const std::string& loc,
                         void* addr, 
                         size_t sz, 
                         const std::string& extra)
    : type(t)
    , description(desc)
    , location(loc)
    , timestamp(std::chrono::system_clock::now())
    , address(addr)
    , size(sz)
    , extraInfo(extra)
{}

std::string EventLogger::Event::getFormattedTime() const {
    auto timer = std::chrono::system_clock::to_time_t(timestamp);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&timer), "%H:%M:%S");
    
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
        case EventType::SIMD_OP:    ss << "[SIMD] "; break;
        case EventType::MEMORY_OP:   ss << "[MEM]  "; break;
        case EventType::BREAKPOINT:  ss << "[BRK]  "; break;
        case EventType::STEP:        ss << "[STEP] "; break;
        case EventType::ERROR:       ss << "[ERR]  "; break;
    }

    ss << description;
    if (!location.empty()) {
        ss << " at " << location;
    }
    if (address) {
        ss << " (addr: 0x" << std::hex << std::setw(12) << std::setfill('0')
           << reinterpret_cast<uintptr_t>(address);
        if (size > 0) {
            ss << ", size: " << std::dec << size << " bytes";
        }
        ss << ")";
    }
    if (!extraInfo.empty()) {
        ss << "\n    " << extraInfo;
    }
    return ss.str();
}

EventLogger::EventLogger(size_t maxEvents)
    : maxEvents(maxEvents)
{}

void EventLogger::logEvent(EventType type,
                         const std::string& description,
                         const std::string& location,
                         void* address,
                         size_t size,
                         const std::string& extraInfo) {
    if (!enabled) return;
    
    std::lock_guard<std::mutex> lock(mutex);
    
    Event event(type, description, location, address, size, extraInfo);
    events.push_back(event);
    
    if (realTimeLogging) {
        printEvent(event);
    }
    
    notifyListeners(event);
    pruneOldEvents();
}

void EventLogger::printEvent(const Event& event) const {
    std::cout << event.toString() << std::endl;
}

void EventLogger::printLastEvents(size_t count) const {
    std::lock_guard<std::mutex> lock(mutex);
    
    size_t start = events.size() > count ? events.size() - count : 0;
    for (size_t i = start; i < events.size(); ++i) {
        printEvent(events[i]);
    }
}

void EventLogger::generateSummary() const {
    std::lock_guard<std::mutex> lock(mutex);
    
    std::map<EventType, size_t> typeCounts;
    for (const auto& event : events) {
        typeCounts[event.type]++;
    }
    
    std::cout << "\nEvent Summary:\n"
              << "=============\n";
    for (const auto& [type, count] : typeCounts) {
        std::cout << std::left << std::setw(12) 
                 << getEventTypeName(type) << ": " 
                 << count << "\n";
    }
    std::cout << "Total Events: " << events.size() << "\n";
}

std::vector<EventLogger::Event> EventLogger::getEventsByType(EventType type) const {
    std::lock_guard<std::mutex> lock(mutex);
    
    std::vector<Event> filtered;
    std::copy_if(events.begin(), events.end(), std::back_inserter(filtered),
                 [type](const Event& e) { return e.type == type; });
    return filtered;
}

std::vector<EventLogger::Event> EventLogger::getEventsInTimeRange(
    std::chrono::system_clock::time_point start,
    std::chrono::system_clock::time_point end) const {
    std::lock_guard<std::mutex> lock(mutex);
    
    std::vector<Event> filtered;
    std::copy_if(events.begin(), events.end(), std::back_inserter(filtered),
                 [start, end](const Event& e) {
                     return e.timestamp >= start && e.timestamp <= end;
                 });
    return filtered;
}

void EventLogger::clear() {
    std::lock_guard<std::mutex> lock(mutex);
    events.clear();
}

void EventLogger::addListener(std::function<void(const Event&)> listener) {
    std::lock_guard<std::mutex> lock(mutex);
    listeners.push_back(std::move(listener));
}

void EventLogger::clearListeners() {
    std::lock_guard<std::mutex> lock(mutex);
    listeners.clear();
}

void EventLogger::notifyListeners(const Event& event) {
    for (const auto& listener : listeners) {
        listener(event);
    }
}

std::string EventLogger::getEventTypeName(EventType type) {
    switch (type) {
        case EventType::SIMD_OP:    return "SIMD";
        case EventType::MEMORY_OP:   return "Memory";
        case EventType::BREAKPOINT:  return "Breakpoint";
        case EventType::STEP:        return "Step";
        case EventType::ERROR:       return "Error";
        default:                     return "Unknown";
    }
}

void EventLogger::pruneOldEvents() {
    if (events.size() > maxEvents) {
        events.erase(events.begin(), 
                    events.begin() + (events.size() - maxEvents));
    }
}