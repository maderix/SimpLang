#pragma once
/*
 * Lum Pattern AST Nodes
 * Nodes for pattern matching declarations
 */

#include "base.hpp"
#include <string>
#include <vector>
#include <memory>

namespace lum {

// Represents a matched operation: mm: linalg.matmul
class OpPattern : public Node {
    std::string handle_;    // The name to bind (e.g., "mm")
    std::string dialect_;   // Dialect name (e.g., "linalg")
    std::string opName_;    // Operation name (e.g., "matmul")

public:
    OpPattern(const std::string& handle, const std::string& qualifiedOp)
        : handle_(handle) {
        // Parse "dialect.op" format
        size_t dot = qualifiedOp.find('.');
        if (dot != std::string::npos) {
            dialect_ = qualifiedOp.substr(0, dot);
            opName_ = qualifiedOp.substr(dot + 1);
        } else {
            // Pattern name reference (not a dialect.op)
            dialect_ = "";
            opName_ = qualifiedOp;
        }
    }

    NodeKind getKind() const override { return NodeKind::OpPattern; }

    const std::string& getHandle() const { return handle_; }
    const std::string& getDialect() const { return dialect_; }
    const std::string& getOpName() const { return opName_; }

    // Returns qualified name like "linalg.matmul"
    std::string getQualifiedName() const {
        if (dialect_.empty()) return opName_;
        return dialect_ + "." + opName_;
    }

    void dump(std::ostream& os, int level = 0) const override {
        indent(os, level);
        os << "OpPattern: " << handle_ << " : " << getQualifiedName() << "\n";
    }
};

// Represents dataflow: mm -> bias
class DataflowEdge : public Node {
    std::string from_;
    std::string to_;

public:
    DataflowEdge(const std::string& from, const std::string& to)
        : from_(from), to_(to) {}

    NodeKind getKind() const override { return NodeKind::DataflowEdge; }

    const std::string& getFrom() const { return from_; }
    const std::string& getTo() const { return to_; }

    void dump(std::ostream& os, int level = 0) const override {
        indent(os, level);
        os << "DataflowEdge: " << from_ << " -> " << to_ << "\n";
    }
};

// Pattern declaration: pattern MatmulBias { ... }
class PatternDecl : public Node {
    std::string name_;
    std::vector<std::unique_ptr<OpPattern>> ops_;
    std::vector<std::unique_ptr<DataflowEdge>> edges_;

public:
    PatternDecl(const std::string& name,
                const std::vector<OpPattern*>& ops)
        : name_(name) {
        for (auto* op : ops) {
            ops_.emplace_back(op);
        }
    }

    void addEdge(DataflowEdge* edge) {
        edges_.emplace_back(edge);
    }

    NodeKind getKind() const override { return NodeKind::PatternDecl; }

    const std::string& getName() const { return name_; }
    const std::vector<std::unique_ptr<OpPattern>>& getOps() const { return ops_; }
    const std::vector<std::unique_ptr<DataflowEdge>>& getEdges() const { return edges_; }

    // Find an op by handle name
    const OpPattern* findOp(const std::string& handle) const {
        for (const auto& op : ops_) {
            if (op->getHandle() == handle) return op.get();
        }
        return nullptr;
    }

    void dump(std::ostream& os, int level = 0) const override {
        indent(os, level);
        os << "PatternDecl: " << name_ << " {\n";
        for (const auto& op : ops_) {
            op->dump(os, level + 1);
        }
        for (const auto& edge : edges_) {
            edge->dump(os, level + 1);
        }
        indent(os, level);
        os << "}\n";
    }
};

} // namespace lum
