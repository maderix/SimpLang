#ifndef KERNEL_DEBUGGER_BREAKPOINT_HPP
#define KERNEL_DEBUGGER_BREAKPOINT_HPP

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>

class BreakpointManager {
public:
    struct Breakpoint {
        int id;
        std::string file;
        int line;
        bool enabled{true};
        std::string condition;
        size_t hitCount{0};
        
        Breakpoint(int i, const std::string& f, int l, const std::string& cond = "")
            : id(i), file(f), line(l), condition(cond) {}
    };

    BreakpointManager() = default;

    // Breakpoint management
    int addBreakpoint(const std::string& file, int line, const std::string& condition = "");
    bool removeBreakpoint(int id);
    void enableBreakpoint(int id, bool enable = true);
    void clearAllBreakpoints();

    // Breakpoint queries
    bool hasBreakpoint(const std::string& file, int line) const;
    bool shouldBreak(const std::string& file, int line) const;
    const Breakpoint* getBreakpoint(int id) const;
    std::vector<Breakpoint> getAllBreakpoints() const;

    // Condition evaluation
    using ConditionEvaluator = std::function<bool(const std::string&)>;
    void setConditionEvaluator(ConditionEvaluator evaluator);

private:
    std::map<int, Breakpoint> breakpoints;
    int nextBreakpointId{1};
    ConditionEvaluator conditionEvaluator;

    bool evaluateCondition(const std::string& condition) const;
};

#endif // KERNEL_DEBUGGER_BREAKPOINT_HPP