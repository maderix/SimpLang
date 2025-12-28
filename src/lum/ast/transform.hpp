#pragma once
/*
 * Lum Transform AST Nodes
 * Nodes for schedule transforms (tile, fuse, vec, etc.)
 */

#include "base.hpp"
#include <string>
#include <vector>
#include <memory>

namespace lum {

// Base class for all transforms
class Transform : public Node {
protected:
    std::string target_;  // Optional explicit target handle

public:
    virtual ~Transform() = default;

    void setTarget(const std::string& target) { target_ = target; }
    const std::string& getTarget() const { return target_; }
    bool hasExplicitTarget() const { return !target_.empty(); }
};

// Tile transform: tile [64, 64, 32] => m, n, k
class TileTransform : public Transform {
    std::vector<int64_t> sizes_;
    std::vector<std::string> bindings_;  // Loop handle names

public:
    TileTransform(const std::vector<int64_t>& sizes,
                  const std::vector<std::string>& bindings = {})
        : sizes_(sizes), bindings_(bindings) {}

    NodeKind getKind() const override { return NodeKind::TileTransform; }

    const std::vector<int64_t>& getSizes() const { return sizes_; }
    const std::vector<std::string>& getBindings() const { return bindings_; }
    bool hasBindings() const { return !bindings_.empty(); }

    void dump(std::ostream& os, int level = 0) const override {
        indent(os, level);
        os << "TileTransform: [";
        for (size_t i = 0; i < sizes_.size(); ++i) {
            if (i > 0) os << ", ";
            os << sizes_[i];
        }
        os << "]";
        if (!bindings_.empty()) {
            os << " => ";
            for (size_t i = 0; i < bindings_.size(); ++i) {
                if (i > 0) os << ", ";
                os << bindings_[i];
            }
        }
        if (hasExplicitTarget()) {
            os << " (target: " << target_ << ")";
        }
        os << "\n";
    }
};

// Fuse transform: fuse bias into m
class FuseTransform : public Transform {
    std::vector<std::string> producers_;  // Handles to fuse
    std::string loop_;                     // Loop to fuse into

public:
    // Single producer
    FuseTransform(const std::string& producer, const std::string& loop)
        : loop_(loop) {
        producers_.push_back(producer);
    }

    // Multiple producers
    FuseTransform(const std::vector<std::string>& producers, const std::string& loop)
        : producers_(producers), loop_(loop) {}

    NodeKind getKind() const override { return NodeKind::FuseTransform; }

    const std::vector<std::string>& getProducers() const { return producers_; }
    const std::string& getLoop() const { return loop_; }

    void dump(std::ostream& os, int level = 0) const override {
        indent(os, level);
        os << "FuseTransform: ";
        if (producers_.size() == 1) {
            os << producers_[0];
        } else {
            os << "[";
            for (size_t i = 0; i < producers_.size(); ++i) {
                if (i > 0) os << ", ";
                os << producers_[i];
            }
            os << "]";
        }
        os << " into " << loop_ << "\n";
    }
};

// FuseChain transform: fuse_chain [op1, op2, op3] or fuse_chain [...] with custom.op
// Fuses a chain of element-wise operations together, optionally replacing with custom op
class FuseChainTransform : public Transform {
    std::vector<std::string> ops_;  // Handles of ops to fuse
    std::string replacement_;       // Optional: custom op to replace with

public:
    FuseChainTransform(const std::vector<std::string>& ops,
                       const std::string& replacement = "")
        : ops_(ops), replacement_(replacement) {}

    NodeKind getKind() const override { return NodeKind::FuseChainTransform; }

    const std::vector<std::string>& getOps() const { return ops_; }
    const std::string& getReplacement() const { return replacement_; }
    bool hasReplacement() const { return !replacement_.empty(); }

    void dump(std::ostream& os, int level = 0) const override {
        indent(os, level);
        os << "FuseChainTransform: [";
        for (size_t i = 0; i < ops_.size(); ++i) {
            if (i > 0) os << ", ";
            os << ops_[i];
        }
        os << "]";
        if (hasReplacement()) {
            os << " with " << replacement_;
        }
        os << "\n";
    }
};

// Vec transform: vec [16] or vec [16] vnni
class VecTransform : public Transform {
    std::vector<int64_t> sizes_;
    bool useVNNI_;

public:
    VecTransform(const std::vector<int64_t>& sizes, bool useVNNI = false)
        : sizes_(sizes), useVNNI_(useVNNI) {}

    NodeKind getKind() const override { return NodeKind::VecTransform; }

    const std::vector<int64_t>& getSizes() const { return sizes_; }
    bool useVNNI() const { return useVNNI_; }

    void dump(std::ostream& os, int level = 0) const override {
        indent(os, level);
        os << "VecTransform: [";
        for (size_t i = 0; i < sizes_.size(); ++i) {
            if (i > 0) os << ", ";
            os << sizes_[i];
        }
        os << "]";
        if (useVNNI_) os << " vnni";
        if (hasExplicitTarget()) {
            os << " (target: " << target_ << ")";
        }
        os << "\n";
    }
};

// Check transform: check accuracy
class CheckTransform : public Transform {
    std::string checkType_;  // "accuracy", "valid", "perf", "memory"

public:
    CheckTransform(const std::string& checkType)
        : checkType_(checkType) {}

    NodeKind getKind() const override { return NodeKind::CheckTransform; }

    const std::string& getCheckType() const { return checkType_; }

    void dump(std::ostream& os, int level = 0) const override {
        indent(os, level);
        os << "CheckTransform: " << checkType_ << "\n";
    }
};

// Unroll transform: unroll k 4
class UnrollTransform : public Transform {
    std::string loop_;
    int64_t factor_;

public:
    UnrollTransform(const std::string& loop, int64_t factor)
        : loop_(loop), factor_(factor) {}

    NodeKind getKind() const override { return NodeKind::UnrollTransform; }

    const std::string& getLoop() const { return loop_; }
    int64_t getFactor() const { return factor_; }

    void dump(std::ostream& os, int level = 0) const override {
        indent(os, level);
        os << "UnrollTransform: " << loop_ << " " << factor_ << "\n";
    }
};

// Interchange transform: interchange [i, j, k]
class InterchangeTransform : public Transform {
    std::vector<std::string> order_;

public:
    InterchangeTransform(const std::vector<std::string>& order)
        : order_(order) {}

    NodeKind getKind() const override { return NodeKind::InterchangeTransform; }

    const std::vector<std::string>& getOrder() const { return order_; }

    void dump(std::ostream& os, int level = 0) const override {
        indent(os, level);
        os << "InterchangeTransform: [";
        for (size_t i = 0; i < order_.size(); ++i) {
            if (i > 0) os << ", ";
            os << order_[i];
        }
        os << "]\n";
    }
};

// Parallel transform: parallel m
class ParallelTransform : public Transform {
    std::string loop_;
    bool simd_;

public:
    ParallelTransform(const std::string& loop, bool simd = false)
        : loop_(loop), simd_(simd) {}

    NodeKind getKind() const override { return NodeKind::ParallelTransform; }

    const std::string& getLoop() const { return loop_; }
    bool isSIMD() const { return simd_; }

    void dump(std::ostream& os, int level = 0) const override {
        indent(os, level);
        os << "ParallelTransform: " << loop_;
        if (simd_) os << " simd";
        os << "\n";
    }
};

} // namespace lum
