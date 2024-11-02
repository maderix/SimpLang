#ifndef KERNEL_DEBUGGER_BREAKPOINT_HPP
#define KERNEL_DEBUGGER_BREAKPOINT_HPP

#include <string>
#include <vector>
#include <map>
#include <functional>

class BreakpointManager {
public:
    // Breakpoint types
    enum class Type {
        INSTRUCTION,    // Break at specific instruction
        MEMORY,        // Break on memory access
        CONDITION      // Break when condition is met
    };

    // Memory operation types
    enum class MemoryOperation {
        READ,
        WRITE,
        ACCESS
    };

    struct Breakpoint {
        Type type;
        std::string location;
        std::function<bool()> condition;
        bool enabled;
        
        Breakpoint(Type t, const std::string& loc, 
                  std::function<bool()> cond = nullptr)
            : type(t), location(loc), condition(cond), enabled(true) {}
    };

private:
    std::map<int, Breakpoint> breakpoints;
    int nextBreakpointId = 1;

public:
    // Add specialized breakpoints
    int addInstructionBreakpoint(const std::string& location);
    int addMemoryBreakpoint(const std::string& address, 
                           MemoryOperation op,
                           size_t size);
    int addConditionalBreakpoint(const std::string& location,
                                std::function<bool()> condition);

    // Add a generic breakpoint
    int addBreakpoint(Type type, const std::string& location,
                     std::function<bool()> condition = nullptr) {
        int id = nextBreakpointId++;
        breakpoints.emplace(id, Breakpoint(type, location, condition));
        return id;
    }

    // Remove a breakpoint
    bool removeBreakpoint(int id) {
        return breakpoints.erase(id) > 0;
    }

    // Enable/disable a breakpoint
    bool enableBreakpoint(int id, bool enable = true) {
        auto it = breakpoints.find(id);
        if (it != breakpoints.end()) {
            it->second.enabled = enable;
            return true;
        }
        return false;
    }

    // Check if we should break
    bool checkBreakpoint(const std::string& location,
                        void* address = nullptr,
                        MemoryOperation op = MemoryOperation::ACCESS,
                        size_t size = 0);

    // List breakpoints
    void listBreakpoints() const;

    // Get breakpoints
    const std::map<int, Breakpoint>& getBreakpoints() const {
        return breakpoints;
    }

private:
    // Helper method for checking memory breakpoints
    bool checkMemoryBreakpoint(const Breakpoint& bp,
                              void* address,
                              MemoryOperation op,
                              size_t size);
};

#endif // KERNEL_DEBUGGER_BREAKPOINT_HPP