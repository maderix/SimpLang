#include "diagnostics/source_manager.hpp"
#include <algorithm>
#include <fstream>
#include <sstream>

namespace simp {
namespace diag {

// ============================================================================
// SourceFile Implementation
// ============================================================================

SourceFile::SourceFile(uint32_t id, std::string name, std::string content)
    : id_(id), name_(std::move(name)), content_(std::move(content)) {
    buildLineOffsets();
}

void SourceFile::buildLineOffsets() {
    lineOffsets_.clear();
    lineOffsets_.push_back(0); // Line 1 starts at offset 0

    for (size_t i = 0; i < content_.size(); ++i) {
        if (content_[i] == '\n') {
            lineOffsets_.push_back(static_cast<uint32_t>(i + 1));
        }
    }
}

uint32_t SourceFile::lineCount() const {
    return static_cast<uint32_t>(lineOffsets_.size());
}

std::string_view SourceFile::getLine(uint32_t lineNumber) const {
    if (lineNumber == 0 || lineNumber > lineOffsets_.size()) {
        return {};
    }

    uint32_t lineIndex = lineNumber - 1;
    uint32_t startOffset = lineOffsets_[lineIndex];
    uint32_t endOffset;

    if (lineIndex + 1 < lineOffsets_.size()) {
        endOffset = lineOffsets_[lineIndex + 1];
        // Don't include the newline character
        if (endOffset > 0 && endOffset <= content_.size() &&
            content_[endOffset - 1] == '\n') {
            endOffset--;
        }
        // Handle Windows line endings
        if (endOffset > 0 && endOffset <= content_.size() &&
            content_[endOffset - 1] == '\r') {
            endOffset--;
        }
    } else {
        endOffset = static_cast<uint32_t>(content_.size());
    }

    if (startOffset >= content_.size()) {
        return {};
    }

    return std::string_view(content_.data() + startOffset, endOffset - startOffset);
}

uint32_t SourceFile::getOffset(uint32_t line, uint32_t column) const {
    if (line == 0 || line > lineOffsets_.size()) {
        return 0;
    }

    uint32_t lineStart = lineOffsets_[line - 1];
    uint32_t offset = lineStart + (column > 0 ? column - 1 : 0);

    return std::min(offset, static_cast<uint32_t>(content_.size()));
}

Position SourceFile::getPosition(uint32_t offset) const {
    Position pos;

    if (lineOffsets_.empty()) {
        return pos;
    }

    // Binary search for the line containing this offset
    auto it = std::upper_bound(lineOffsets_.begin(), lineOffsets_.end(), offset);
    if (it == lineOffsets_.begin()) {
        pos.line = 1;
    } else {
        --it;
        pos.line = static_cast<uint32_t>(std::distance(lineOffsets_.begin(), it)) + 1;
    }

    uint32_t lineStart = lineOffsets_[pos.line - 1];
    pos.column = offset - lineStart + 1;

    return pos;
}

uint32_t SourceFile::getLineStartOffset(uint32_t lineNumber) const {
    if (lineNumber == 0 || lineNumber > lineOffsets_.size()) {
        return 0;
    }
    return lineOffsets_[lineNumber - 1];
}

uint32_t SourceFile::getLineEndOffset(uint32_t lineNumber) const {
    if (lineNumber == 0 || lineNumber > lineOffsets_.size()) {
        return 0;
    }

    if (lineNumber < lineOffsets_.size()) {
        return lineOffsets_[lineNumber];
    }
    return static_cast<uint32_t>(content_.size());
}

// ============================================================================
// SourceManager Implementation
// ============================================================================

uint32_t SourceManager::loadFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return 0;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    return addSource(path, std::move(content));
}

uint32_t SourceManager::addSource(const std::string& name, std::string content) {
    uint32_t id = nextFileId_++;
    files_.push_back(std::make_unique<SourceFile>(id, name, std::move(content)));

    // Set the first file as the main file
    if (mainFileId_ == 0) {
        mainFileId_ = id;
    }

    return id;
}

const SourceFile* SourceManager::getFile(uint32_t fileId) const {
    for (const auto& file : files_) {
        if (file->id() == fileId) {
            return file.get();
        }
    }
    return nullptr;
}

std::string_view SourceManager::getFileName(uint32_t fileId) const {
    const SourceFile* file = getFile(fileId);
    if (file) {
        return file->name();
    }
    return "<unknown>";
}

std::string_view SourceManager::getLine(uint32_t fileId, uint32_t lineNumber) const {
    const SourceFile* file = getFile(fileId);
    if (file) {
        return file->getLine(lineNumber);
    }
    return {};
}

std::string_view SourceManager::getSpanText(const Span& span) const {
    const SourceFile* file = getFile(span.fileId);
    if (!file || span.startOffset >= file->size()) {
        return {};
    }

    uint32_t endOffset = std::min(span.endOffset, static_cast<uint32_t>(file->size()));
    return std::string_view(file->content().data() + span.startOffset,
                            endOffset - span.startOffset);
}

uint32_t SourceManager::getOffset(uint32_t fileId, uint32_t line, uint32_t column) const {
    const SourceFile* file = getFile(fileId);
    if (file) {
        return file->getOffset(line, column);
    }
    return 0;
}

Position SourceManager::getPosition(uint32_t fileId, uint32_t offset) const {
    const SourceFile* file = getFile(fileId);
    if (file) {
        return file->getPosition(offset);
    }
    return Position{};
}

} // namespace diag
} // namespace simp
