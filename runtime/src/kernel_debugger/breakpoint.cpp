#include "kernel_debugger/breakpoint.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>

namespace {
    // Helper function to create readable breakpoint descriptions
    std::string createBreakpointDescription(const BreakpointManager::Breakpoint& bp) {
        std::stringstream ss;
        ss << "Type: ";
        switch (bp.type) {
            case BreakpointManager::Type::INSTRUCTION:
                ss << "Instruction";
                break;
            case BreakpointManager::Type::MEMORY:
                ss << "Memory";
                break;
            case BreakpointManager::Type::CONDITION:
                ss << "Condition";
                break;
        }
        ss << ", Location: " << bp.location;
        if (bp.condition) {
            ss << " (Conditional)";
        }
        ss << ", Status: " << (bp.enabled ? "Enabled" : "Disabled");
        return ss.str();
    }
}

int BreakpointManager::addInstructionBreakpoint(const std::string& location) {
    int id = nextBreakpointId++;
    breakpoints.emplace(id, Breakpoint(Type::INSTRUCTION, location));
    std::cout << "Added instruction breakpoint [" << id << "] at " << location << std::endl;
    return id;
}

int BreakpointManager::addMemoryBreakpoint(const std::string& address, 
                                         MemoryOperation op,
                                         size_t size) {
    std::stringstream ss;
    ss << "addr:" << address << ";op:" << static_cast<int>(op) << ";size:" << size;
    
    int id = nextBreakpointId++;
    breakpoints.emplace(id, Breakpoint(Type::MEMORY, ss.str()));
    
    std::cout << "Added memory breakpoint [" << id << "] for ";
    switch (op) {
        case MemoryOperation::READ:
            std::cout << "read";
            break;
        case MemoryOperation::WRITE:
            std::cout << "write";
            break;
        case MemoryOperation::ACCESS:
            std::cout << "access";
            break;
    }
    std::cout << " at " << address << " (size: " << size << ")" << std::endl;
    
    return id;
}

int BreakpointManager::addConditionalBreakpoint(const std::string& location,
                                              std::function<bool()> condition) {
    int id = nextBreakpointId++;
    breakpoints.emplace(id, Breakpoint(Type::CONDITION, location, condition));
    std::cout << "Added conditional breakpoint [" << id << "] at " << location << std::endl;
    return id;
}

bool BreakpointManager::checkBreakpoint(const std::string& location,
                                      void* address,
                                      MemoryOperation op,
                                      size_t size) {
    bool shouldBreak = false;
    
    for (const auto& [id, bp] : breakpoints) {
        if (!bp.enabled) continue;

        switch (bp.type) {
            case Type::INSTRUCTION:
                if (bp.location == location) {
                    shouldBreak = true;
                    std::cout << "Hit instruction breakpoint [" << id << "] at " 
                             << location << std::endl;
                }
                break;

            case Type::MEMORY:
                if (checkMemoryBreakpoint(bp, address, op, size)) {
                    shouldBreak = true;
                    std::cout << "Hit memory breakpoint [" << id << "] at "
                             << std::hex << address << std::dec << std::endl;
                }
                break;

            case Type::CONDITION:
                if (bp.location == location && bp.condition && bp.condition()) {
                    shouldBreak = true;
                    std::cout << "Hit conditional breakpoint [" << id << "] at "
                             << location << std::endl;
                }
                break;
        }
    }

    return shouldBreak;
}

void BreakpointManager::listBreakpoints() const {
    if (breakpoints.empty()) {
        std::cout << "No breakpoints set" << std::endl;
        return;
    }

    std::cout << "\nActive Breakpoints:\n";
    std::cout << std::setfill('-') << std::setw(80) << "-" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << std::setw(4) << "ID" << " | " 
              << std::setw(60) << "Description" << " | "
              << std::setw(8) << "Status" << std::endl;
    std::cout << std::setfill('-') << std::setw(80) << "-" << std::endl;
    std::cout << std::setfill(' ');

    for (const auto& [id, bp] : breakpoints) {
        std::cout << std::setw(4) << id << " | "
                  << std::setw(60) << createBreakpointDescription(bp) << " | "
                  << std::setw(8) << (bp.enabled ? "Enabled" : "Disabled")
                  << std::endl;
    }
    std::cout << std::setfill('-') << std::setw(80) << "-" << std::endl;
}

bool BreakpointManager::checkMemoryBreakpoint(const Breakpoint& bp,
                                            void* address,
                                            MemoryOperation op,
                                            size_t size) {
    std::stringstream ss(bp.location);
    std::string token;
    void* bpAddress = nullptr;
    MemoryOperation bpOp = MemoryOperation::ACCESS;
    size_t bpSize = 0;

    // Parse breakpoint location string
    while (std::getline(ss, token, ';')) {
        size_t pos = token.find(':');
        if (pos == std::string::npos) continue;

        std::string key = token.substr(0, pos);
        std::string value = token.substr(pos + 1);

        if (key == "addr") {
            bpAddress = reinterpret_cast<void*>(std::stoull(value, nullptr, 16));
        } else if (key == "op") {
            bpOp = static_cast<MemoryOperation>(std::stoi(value));
        } else if (key == "size") {
            bpSize = std::stoull(value);
        }
    }

    // Check if memory operation matches breakpoint conditions
    if (bpAddress == address && bpSize == size) {
        if (bpOp == MemoryOperation::ACCESS ||
            bpOp == op) {
            return true;
        }
    }

    return false;
}