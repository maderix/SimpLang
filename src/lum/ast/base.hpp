#pragma once
/*
 * Lum AST Base Classes
 * Base types for all Lum AST nodes
 */

#include <string>
#include <vector>
#include <memory>
#include <iostream>

namespace lum {

// Source location for error reporting
struct SourceLocation {
    unsigned line = 0;
    unsigned column = 0;

    SourceLocation() = default;
    SourceLocation(unsigned l, unsigned c) : line(l), column(c) {}
};

// AST node kinds for type checking without RTTI
enum class NodeKind {
    // Top-level
    Program,
    PatternDecl,
    ScheduleDecl,
    PipelineDecl,

    // Pattern elements
    OpPattern,
    DataflowEdge,
    Constraint,

    // Transforms
    TileTransform,
    FuseTransform,
    VecTransform,
    UnrollTransform,
    InterchangeTransform,
    ParallelTransform,
    PackTransform,
    PeelTransform,
    PromoteTransform,
    MapTransform,
    CoopTransform,
    SyncTransform,
    PrefetchTransform,

    // Verification
    CheckTransform,
    TraceStmt,
    BreakStmt,
    AssertStmt,
    SnapshotStmt,
    DiffStmt,

    // Control flow
    LetBinding,
    IfStmt,
    ForLoop,

    // Expressions
    BinaryExpr,
    UnaryExpr,
    Identifier,
    IntegerLiteral,
    FloatLiteral,
    StringLiteral,
    PropertyAccess,
};

// Base class for all AST nodes
class Node {
protected:
    SourceLocation loc_;

public:
    virtual ~Node() = default;
    virtual NodeKind getKind() const = 0;
    virtual void dump(std::ostream& os, int indent = 0) const = 0;

    void setLocation(unsigned line, unsigned col = 0) {
        loc_ = SourceLocation(line, col);
    }

    void setLocation(const SourceLocation& loc) {
        loc_ = loc;
    }

    const SourceLocation& getLocation() const { return loc_; }
    unsigned getLine() const { return loc_.line; }
    unsigned getColumn() const { return loc_.column; }

protected:
    void indent(std::ostream& os, int level) const {
        for (int i = 0; i < level; ++i) os << "  ";
    }
};

} // namespace lum
