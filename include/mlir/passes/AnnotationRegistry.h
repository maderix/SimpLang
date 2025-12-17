//===- AnnotationRegistry.h - Backend-agnostic annotation registry --------===//
//
// Part of the SimpLang Project
//
// This header provides a shared registry for communication between the
// MLIR-level AnnotationLoweringPass and backend-specific LLVM passes.
//
// Design: Backend-agnostic registry that stores annotation information.
// - AnnotationLoweringPass registers functions with their annotation info
// - Backend passes (VNNI, GPU, etc.) query the registry and act on patterns
//   they recognize (e.g., "vnni.i8_matmul", "cuda.tensor_core", etc.)
//
//===----------------------------------------------------------------------===//

#ifndef SIMPLANG_MLIR_PASSES_ANNOTATION_REGISTRY_H
#define SIMPLANG_MLIR_PASSES_ANNOTATION_REGISTRY_H

#include <map>
#include <string>
#include <vector>
#include <mutex>

namespace mlir {
namespace simp {

/// Backend-agnostic annotation information for a function
struct AnnotationInfo {
  std::string lowerPattern;        // e.g., "vnni.i8_matmul", "cuda.tensor_core"
  std::vector<int64_t> tileSizes;  // Tile sizes from @tile annotation
  int64_t alignment = 0;           // Memory alignment from @align annotation

  bool hasPattern() const { return !lowerPattern.empty(); }
  bool hasTileSizes() const { return !tileSizes.empty(); }
  bool hasAlignment() const { return alignment > 0; }

  /// Check if pattern starts with a given prefix (for backend filtering)
  bool patternStartsWith(const std::string& prefix) const {
    return lowerPattern.compare(0, prefix.size(), prefix) == 0;
  }
};

/// Singleton registry for function annotation information
/// Thread-safe for concurrent compilation scenarios
class AnnotationRegistry {
public:
  static AnnotationRegistry& instance() {
    static AnnotationRegistry registry;
    return registry;
  }

  /// Register a function with its annotation info
  void registerFunction(const std::string& funcName, const AnnotationInfo& info) {
    std::lock_guard<std::mutex> lock(mutex_);
    functions_[funcName] = info;
  }

  /// Check if a function has annotation info
  bool hasAnnotation(const std::string& funcName) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return functions_.find(funcName) != functions_.end();
  }

  /// Get annotation info for a function (returns empty info if not found)
  AnnotationInfo getInfo(const std::string& funcName) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = functions_.find(funcName);
    if (it != functions_.end()) {
      return it->second;
    }
    return AnnotationInfo{};
  }

  /// Check if function has a pattern matching the given prefix
  /// Useful for backend-specific filtering (e.g., "vnni.", "cuda.")
  bool hasPatternWithPrefix(const std::string& funcName, const std::string& prefix) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = functions_.find(funcName);
    if (it != functions_.end()) {
      return it->second.patternStartsWith(prefix);
    }
    return false;
  }

  /// Clear the registry (call between compilation units)
  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    functions_.clear();
  }

  /// Get all registered function names (for debugging)
  std::vector<std::string> getRegisteredFunctions() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> names;
    for (const auto& [name, _] : functions_) {
      names.push_back(name);
    }
    return names;
  }

private:
  AnnotationRegistry() = default;
  ~AnnotationRegistry() = default;
  AnnotationRegistry(const AnnotationRegistry&) = delete;
  AnnotationRegistry& operator=(const AnnotationRegistry&) = delete;

  mutable std::mutex mutex_;
  std::map<std::string, AnnotationInfo> functions_;
};

} // namespace simp
} // namespace mlir

#endif // SIMPLANG_MLIR_PASSES_ANNOTATION_REGISTRY_H
