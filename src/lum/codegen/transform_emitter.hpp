#pragma once
/*
 * Lum Transform Dialect Emitter
 * Generates MLIR Transform Dialect from Lum AST
 *
 * MVP: Generates text output (Transform Dialect MLIR as string)
 * Future: Use MLIR C++ API for proper IR construction
 */

#include "../ast/base.hpp"
#include "../ast/pattern.hpp"
#include "../ast/schedule.hpp"
#include "../ast/transform.hpp"

#include <string>
#include <sstream>
#include <map>
#include <set>
#include <vector>

namespace lum {

// Tracks SSA handle state during emission
struct HandleInfo {
    std::string ssaName;     // MLIR SSA name (e.g., "%mm", "%tiled")
    bool consumed = false;   // Has this handle been consumed?
    std::string consumedBy;  // Name of consuming operation
};

class TransformEmitter {
    std::stringstream output_;
    int indentLevel_ = 0;
    int ssaCounter_ = 0;

    // Handle tracking
    std::map<std::string, HandleInfo> handles_;

    // Error state
    std::vector<std::string> errors_;

public:
    TransformEmitter() = default;

    // Main entry point: emit entire program
    std::string emit(const Program& program);

    // Check for errors
    bool hasErrors() const { return !errors_.empty(); }
    const std::vector<std::string>& getErrors() const { return errors_; }

private:
    // Emit helpers
    void emitSchedule(const ScheduleDecl& schedule, const PatternDecl* pattern);
    void emitPattern(const PatternDecl& pattern);
    void emitTransform(const Transform& transform);

    // Transform-specific emitters
    void emitTile(const TileTransform& tile);
    void emitFuse(const FuseTransform& fuse);
    void emitFuseChain(const FuseChainTransform& fuseChain);
    void emitVec(const VecTransform& vec);
    void emitCheck(const CheckTransform& check);
    void emitUnroll(const UnrollTransform& unroll);
    void emitInterchange(const InterchangeTransform& interchange);
    void emitParallel(const ParallelTransform& parallel);

    // Handle management
    std::string newSSAName(const std::string& hint = "");
    void registerHandle(const std::string& lumName, const std::string& ssaName);
    std::string getHandle(const std::string& lumName);
    void consumeHandle(const std::string& lumName, const std::string& consumer);
    void rebindHandle(const std::string& lumName, const std::string& newSSA);

    // Output helpers
    void indent();
    void line(const std::string& s = "");
    void emit(const std::string& s);
};

} // namespace lum
