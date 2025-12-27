#pragma once
/*
 * Lum Schedule AST Nodes
 * Nodes for schedule declarations and program structure
 */

#include "base.hpp"
#include "pattern.hpp"
#include "transform.hpp"
#include <string>
#include <vector>
#include <memory>
#include <map>

namespace lum {

// Schedule declaration: schedule OptMatmul(linalg.matmul) { ... }
class ScheduleDecl : public Node {
    std::string name_;
    std::string target_;  // Pattern name or op type
    std::vector<std::unique_ptr<Transform>> transforms_;

public:
    ScheduleDecl(const std::string& name,
                 const std::string& target,
                 const std::vector<Transform*>& transforms)
        : name_(name), target_(target) {
        for (auto* t : transforms) {
            transforms_.emplace_back(t);
        }
    }

    NodeKind getKind() const override { return NodeKind::ScheduleDecl; }

    const std::string& getName() const { return name_; }
    const std::string& getTarget() const { return target_; }
    const std::vector<std::unique_ptr<Transform>>& getTransforms() const { return transforms_; }

    // Check if target is a pattern name (simple identifier) or op type (dialect.op)
    bool isPatternTarget() const {
        return target_.find('.') == std::string::npos;
    }

    void dump(std::ostream& os, int level = 0) const override {
        indent(os, level);
        os << "ScheduleDecl: " << name_ << "(" << target_ << ") {\n";
        for (const auto& t : transforms_) {
            t->dump(os, level + 1);
        }
        indent(os, level);
        os << "}\n";
    }
};

// Top-level program containing all declarations
class Program : public Node {
    std::vector<std::unique_ptr<PatternDecl>> patterns_;
    std::vector<std::unique_ptr<ScheduleDecl>> schedules_;

public:
    Program() = default;

    void addPattern(PatternDecl* pattern) {
        patterns_.emplace_back(pattern);
    }

    void addSchedule(ScheduleDecl* schedule) {
        schedules_.emplace_back(schedule);
    }

    NodeKind getKind() const override { return NodeKind::Program; }

    const std::vector<std::unique_ptr<PatternDecl>>& getPatterns() const { return patterns_; }
    const std::vector<std::unique_ptr<ScheduleDecl>>& getSchedules() const { return schedules_; }

    // Find pattern by name
    const PatternDecl* findPattern(const std::string& name) const {
        for (const auto& p : patterns_) {
            if (p->getName() == name) return p.get();
        }
        return nullptr;
    }

    // Find schedule by name
    const ScheduleDecl* findSchedule(const std::string& name) const {
        for (const auto& s : schedules_) {
            if (s->getName() == name) return s.get();
        }
        return nullptr;
    }

    void dump(std::ostream& os, int level = 0) const override {
        os << "Program {\n";
        for (const auto& p : patterns_) {
            p->dump(os, level + 1);
        }
        for (const auto& s : schedules_) {
            s->dump(os, level + 1);
        }
        os << "}\n";
    }
};

} // namespace lum
