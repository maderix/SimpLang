#pragma once

#include <cstdint>
#include <string>

namespace simp {
namespace diag {

// Forward declaration
class SourceManager;

/// Position in source file (1-indexed line and column)
struct Position {
    uint32_t line = 1;
    uint32_t column = 1;

    bool operator==(const Position& other) const {
        return line == other.line && column == other.column;
    }

    bool operator<(const Position& other) const {
        if (line != other.line) return line < other.line;
        return column < other.column;
    }
};

/// A span represents a range in a source file using byte offsets.
/// Line/column positions are computed on-demand from SourceManager.
struct Span {
    uint32_t fileId = 0;        // Index into SourceManager's file list
    uint32_t startOffset = 0;   // Byte offset from start of file
    uint32_t endOffset = 0;     // Exclusive end offset

    /// Create an empty span
    Span() = default;

    /// Create a span with the given offsets
    Span(uint32_t fileId, uint32_t start, uint32_t end)
        : fileId(fileId), startOffset(start), endOffset(end) {}

    /// Check if this span is valid (has non-zero length)
    bool isValid() const { return endOffset > startOffset; }

    /// Get the length of this span in bytes
    uint32_t length() const { return endOffset - startOffset; }

    /// Check if an offset is contained within this span
    bool contains(uint32_t offset) const {
        return offset >= startOffset && offset < endOffset;
    }

    /// Check if this span overlaps with another
    bool overlaps(const Span& other) const {
        if (fileId != other.fileId) return false;
        return startOffset < other.endOffset && other.startOffset < endOffset;
    }

    /// Merge two spans (assumes same file)
    Span merge(const Span& other) const {
        if (fileId != other.fileId) return *this;
        return Span(fileId,
                    std::min(startOffset, other.startOffset),
                    std::max(endOffset, other.endOffset));
    }

    /// Shrink to just the start position (single character)
    Span shrinkToStart() const {
        return Span(fileId, startOffset, startOffset + 1);
    }

    /// Shrink to just the end position (single character)
    Span shrinkToEnd() const {
        return Span(fileId, endOffset > 0 ? endOffset - 1 : 0, endOffset);
    }

    /// Create a span from line/column positions (requires SourceManager)
    static Span fromPositions(const SourceManager& sm, uint32_t fileId,
                              Position start, Position end);

    /// Get start position (requires SourceManager)
    Position startPosition(const SourceManager& sm) const;

    /// Get end position (requires SourceManager)
    Position endPosition(const SourceManager& sm) const;

    bool operator==(const Span& other) const {
        return fileId == other.fileId &&
               startOffset == other.startOffset &&
               endOffset == other.endOffset;
    }
};

} // namespace diag
} // namespace simp
