#pragma once

#include "diagnostics/span.hpp"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace simp {
namespace diag {

/// Represents a loaded source file with cached line offset information
class SourceFile {
public:
    SourceFile(uint32_t id, std::string name, std::string content);

    uint32_t id() const { return id_; }
    const std::string& name() const { return name_; }
    const std::string& content() const { return content_; }
    size_t size() const { return content_.size(); }

    /// Get the number of lines in this file
    uint32_t lineCount() const;

    /// Get a specific line (1-indexed), returns empty if out of bounds
    std::string_view getLine(uint32_t lineNumber) const;

    /// Get the byte offset for a given line/column (1-indexed)
    uint32_t getOffset(uint32_t line, uint32_t column) const;

    /// Get the line/column position for a byte offset
    Position getPosition(uint32_t offset) const;

    /// Get the starting offset of a line (1-indexed)
    uint32_t getLineStartOffset(uint32_t lineNumber) const;

    /// Get the ending offset of a line (1-indexed, exclusive)
    uint32_t getLineEndOffset(uint32_t lineNumber) const;

private:
    void buildLineOffsets();

    uint32_t id_;
    std::string name_;
    std::string content_;
    std::vector<uint32_t> lineOffsets_; // Offset of start of each line
};

/// Manages all source files loaded during compilation
class SourceManager {
public:
    SourceManager() = default;
    ~SourceManager() = default;

    // Non-copyable but movable
    SourceManager(const SourceManager&) = delete;
    SourceManager& operator=(const SourceManager&) = delete;
    SourceManager(SourceManager&&) = default;
    SourceManager& operator=(SourceManager&&) = default;

    /// Load a file from disk, returns file ID or 0 on failure
    uint32_t loadFile(const std::string& path);

    /// Add source content directly (e.g., from stdin or REPL)
    uint32_t addSource(const std::string& name, std::string content);

    /// Get a file by ID, returns nullptr if not found
    const SourceFile* getFile(uint32_t fileId) const;

    /// Get the filename for a file ID
    std::string_view getFileName(uint32_t fileId) const;

    /// Get a line from a file (1-indexed)
    std::string_view getLine(uint32_t fileId, uint32_t lineNumber) const;

    /// Get the source text for a span
    std::string_view getSpanText(const Span& span) const;

    /// Get offset from line/column (delegates to SourceFile)
    uint32_t getOffset(uint32_t fileId, uint32_t line, uint32_t column) const;

    /// Get position from offset (delegates to SourceFile)
    Position getPosition(uint32_t fileId, uint32_t offset) const;

    /// Get the main file ID (first file loaded, typically the entry point)
    uint32_t mainFileId() const { return mainFileId_; }

    /// Set the main file ID
    void setMainFileId(uint32_t id) { mainFileId_ = id; }

    /// Get total number of files loaded
    size_t fileCount() const { return files_.size(); }

private:
    std::vector<std::unique_ptr<SourceFile>> files_;
    uint32_t nextFileId_ = 1; // 0 is reserved for "no file"
    uint32_t mainFileId_ = 0;
};

} // namespace diag
} // namespace simp
