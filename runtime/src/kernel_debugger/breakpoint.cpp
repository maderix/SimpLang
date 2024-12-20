#include "kernel_debugger/breakpoint.hpp"
#include <iostream>
#include <algorithm>

int BreakpointManager::addBreakpoint(const std::string& file, int line, const std::string& condition) {
    int id = nextBreakpointId++;
    auto [it, inserted] = breakpoints.emplace(id, Breakpoint(id, file, line, condition));
    
    if (inserted) {
        std::cout << "Breakpoint " << id << " set at " << file << ":" << line;
        if (!condition.empty()) {
            std::cout << " when " << condition;
        }
        std::cout << std::endl;
        return id;
    }
    return -1;
}

bool BreakpointManager::removeBreakpoint(int id) {
    auto it = breakpoints.find(id);
    if (it != breakpoints.end()) {
        breakpoints.erase(it);
        std::cout << "Breakpoint " << id << " removed\n";
        return true;
    }
    return false;
}

void BreakpointManager::enableBreakpoint(int id, bool enable) {
    auto it = breakpoints.find(id);
    if (it != breakpoints.end()) {
        it->second.enabled = enable;
        std::cout << "Breakpoint " << id << (enable ? " enabled\n" : " disabled\n");
    }
}

void BreakpointManager::clearAllBreakpoints() {
    breakpoints.clear();
    std::cout << "All breakpoints cleared\n";
}

bool BreakpointManager::hasBreakpoint(const std::string& file, int line) const {
    return std::any_of(breakpoints.begin(), breakpoints.end(),
        [&](const auto& pair) {
            const auto& bp = pair.second;
            return bp.enabled && bp.file == file && bp.line == line;
        });
}

bool BreakpointManager::shouldBreak(const std::string& file, int line) const {
    for (const auto& [id, bp] : breakpoints) {
        if (bp.enabled && bp.file == file && bp.line == line) {
            if (bp.condition.empty() || evaluateCondition(bp.condition)) {
                const_cast<Breakpoint&>(bp).hitCount++;
                return true;
            }
        }
    }
    return false;
}

const BreakpointManager::Breakpoint* BreakpointManager::getBreakpoint(int id) const {
    auto it = breakpoints.find(id);
    return it != breakpoints.end() ? &it->second : nullptr;
}

std::vector<BreakpointManager::Breakpoint> BreakpointManager::getAllBreakpoints() const {
    std::vector<Breakpoint> result;
    result.reserve(breakpoints.size());
    for (const auto& [_, bp] : breakpoints) {
        result.push_back(bp);
    }
    return result;
}

void BreakpointManager::setConditionEvaluator(ConditionEvaluator evaluator) {
    conditionEvaluator = std::move(evaluator);
}

bool BreakpointManager::evaluateCondition(const std::string& condition) const {
    if (condition.empty()) return true;
    if (!conditionEvaluator) return true;
    
    try {
        return conditionEvaluator(condition);
    } catch (const std::exception& e) {
        std::cerr << "Error evaluating breakpoint condition: " << e.what() << std::endl;
        return true; // Break anyway on evaluation error
    }
}