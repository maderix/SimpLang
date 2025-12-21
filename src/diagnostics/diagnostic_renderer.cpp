#include "diagnostics/diagnostic_renderer.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>

namespace simp {
namespace diag {

DiagnosticRenderer::DiagnosticRenderer(const SourceManager& sourceManager)
    : sourceManager_(sourceManager), out_(&std::cerr), useColors_(true) {}

void DiagnosticRenderer::setOutputStream(std::ostream& os) {
    out_ = &os;
}

const char* DiagnosticRenderer::severityColor(Severity severity) const {
    switch (severity) {
        case Severity::Error:
        case Severity::Fatal:
            return Colors::Error;
        case Severity::Warning:
            return Colors::Warning;
        case Severity::Note:
            return Colors::Note;
    }
    return Colors::Reset;
}

const char* DiagnosticRenderer::severityName(Severity severity) const {
    switch (severity) {
        case Severity::Error:
            return "error";
        case Severity::Fatal:
            return "fatal";
        case Severity::Warning:
            return "warning";
        case Severity::Note:
            return "note";
    }
    return "unknown";
}

const char* DiagnosticRenderer::color(const char* code) const {
    return useColors_ ? code : "";
}

void DiagnosticRenderer::render(const Diagnostic& diag) {
    renderHeader(diag);
    renderLocation(diag);
    renderSnippet(diag);
    renderNotes(diag);
    renderSuggestions(diag);
    *out_ << "\n";
}

void DiagnosticRenderer::renderHeader(const Diagnostic& diag) {
    // error[E0200]: type mismatch in assignment
    *out_ << color(severityColor(diag.severity()))
          << color(Colors::Bold)
          << severityName(diag.severity())
          << "[" << formatErrorCode(diag.code()) << "]"
          << color(Colors::Reset)
          << ": "
          << color(Colors::Bold)
          << diag.message()
          << color(Colors::Reset)
          << "\n";
}

void DiagnosticRenderer::renderLocation(const Diagnostic& diag) {
    auto primarySpan = diag.primarySpan();
    if (!primarySpan) return;

    Position pos = sourceManager_.getPosition(primarySpan->fileId, primarySpan->startOffset);
    std::string_view fileName = sourceManager_.getFileName(primarySpan->fileId);

    // --> examples/test.sl:5:12
    *out_ << color(Colors::LineNumber)
          << " --> "
          << color(Colors::Reset)
          << fileName << ":" << pos.line << ":" << pos.column
          << "\n";
}

void DiagnosticRenderer::renderSnippet(const Diagnostic& diag) {
    if (diag.labels().empty()) return;

    uint32_t maxLineWidth = computeMaxLineNumberWidth(diag);

    // Collect all lines we need to display
    std::set<std::pair<uint32_t, uint32_t>> linesToShow; // (fileId, lineNumber)
    for (const auto& label : diag.labels()) {
        Position startPos = sourceManager_.getPosition(label.span.fileId, label.span.startOffset);
        Position endPos = sourceManager_.getPosition(label.span.fileId, label.span.endOffset);
        for (uint32_t line = startPos.line; line <= endPos.line; ++line) {
            linesToShow.insert({label.span.fileId, line});
        }
    }

    if (linesToShow.empty()) return;

    // Empty line with just the pipe
    *out_ << std::string(maxLineWidth + 1, ' ')
          << color(Colors::LineNumber) << "|" << color(Colors::Reset) << "\n";

    // Track the previous line number for gap detection
    uint32_t prevFileId = 0;
    uint32_t prevLine = 0;

    for (const auto& [fileId, lineNum] : linesToShow) {
        // Show gap indicator if lines are not consecutive
        if (prevFileId == fileId && lineNum > prevLine + 1) {
            *out_ << color(Colors::LineNumber)
                  << std::string(maxLineWidth, '.')
                  << color(Colors::Reset) << "\n";
        }

        std::string_view lineContent = sourceManager_.getLine(fileId, lineNum);
        renderLine(lineNum, lineContent, maxLineWidth);

        // Collect labels that apply to this line
        std::vector<Label> lineLabels;
        for (const auto& label : diag.labels()) {
            if (label.span.fileId != fileId) continue;
            Position startPos = sourceManager_.getPosition(label.span.fileId, label.span.startOffset);
            Position endPos = sourceManager_.getPosition(label.span.fileId, label.span.endOffset);
            if (lineNum >= startPos.line && lineNum <= endPos.line) {
                lineLabels.push_back(label);
            }
        }

        if (!lineLabels.empty()) {
            renderUnderlines(lineLabels, lineNum, lineContent, maxLineWidth);
        }

        prevFileId = fileId;
        prevLine = lineNum;
    }

    // Final empty line with pipe
    *out_ << std::string(maxLineWidth + 1, ' ')
          << color(Colors::LineNumber) << "|" << color(Colors::Reset) << "\n";
}

void DiagnosticRenderer::renderLine(uint32_t lineNumber, std::string_view content,
                                     uint32_t maxLineNumWidth) {
    // 5 |     var x: i32 = "hello";
    *out_ << color(Colors::LineNumber)
          << std::setw(maxLineNumWidth) << lineNumber
          << " | "
          << color(Colors::Reset)
          << content
          << "\n";
}

void DiagnosticRenderer::renderUnderlines(const std::vector<Label>& labels,
                                           uint32_t lineNumber,
                                           std::string_view lineContent,
                                           uint32_t maxLineNumWidth) {
    // Build the underline string
    std::string underline(lineContent.size(), ' ');
    std::vector<std::pair<uint32_t, const Label*>> messagePositions;

    for (const auto& label : labels) {
        Position startPos = sourceManager_.getPosition(label.span.fileId, label.span.startOffset);
        Position endPos = sourceManager_.getPosition(label.span.fileId, label.span.endOffset);

        uint32_t startCol, endCol;

        if (startPos.line == lineNumber) {
            startCol = startPos.column - 1;
        } else {
            startCol = 0;
        }

        if (endPos.line == lineNumber) {
            endCol = std::min(endPos.column - 1, static_cast<uint32_t>(lineContent.size()));
        } else {
            endCol = static_cast<uint32_t>(lineContent.size());
        }

        if (startCol >= lineContent.size()) startCol = lineContent.size() > 0 ? lineContent.size() - 1 : 0;
        if (endCol <= startCol) endCol = startCol + 1;
        if (endCol > lineContent.size()) endCol = lineContent.size();

        char underChar = label.isPrimary ? '^' : '-';
        for (uint32_t i = startCol; i < endCol && i < underline.size(); ++i) {
            underline[i] = underChar;
        }

        if (!label.message.empty()) {
            messagePositions.push_back({startCol, &label});
        }
    }

    // Print the underline
    *out_ << std::string(maxLineNumWidth, ' ')
          << color(Colors::LineNumber) << " | " << color(Colors::Reset);

    // Print underlines with color
    for (size_t i = 0; i < underline.size(); ++i) {
        if (underline[i] == '^') {
            *out_ << color(Colors::Primary) << '^' << color(Colors::Reset);
        } else if (underline[i] == '-') {
            *out_ << color(Colors::Secondary) << '-' << color(Colors::Reset);
        } else {
            *out_ << ' ';
        }
    }

    // Print first message inline if there's only one
    if (messagePositions.size() == 1) {
        *out_ << " " << messagePositions[0].second->message;
    }
    *out_ << "\n";

    // Print additional messages on separate lines with vertical bars
    if (messagePositions.size() > 1) {
        // Sort by column position (reverse to print rightmost first in the vertical lines)
        std::sort(messagePositions.begin(), messagePositions.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        for (size_t msgIdx = 0; msgIdx < messagePositions.size(); ++msgIdx) {
            *out_ << std::string(maxLineNumWidth, ' ')
                  << color(Colors::LineNumber) << " | " << color(Colors::Reset);

            // Draw vertical bars and the message
            for (size_t i = 0; i < messagePositions[msgIdx].first; ++i) {
                bool hasBar = false;
                for (size_t j = msgIdx; j < messagePositions.size(); ++j) {
                    if (messagePositions[j].first == i) {
                        hasBar = true;
                        break;
                    }
                }
                if (hasBar) {
                    const char* c = messagePositions[msgIdx].second->isPrimary ?
                                    Colors::Primary : Colors::Secondary;
                    *out_ << color(c) << "|" << color(Colors::Reset);
                } else {
                    *out_ << " ";
                }
            }

            // Print the message
            const char* c = messagePositions[msgIdx].second->isPrimary ?
                            Colors::Primary : Colors::Secondary;
            *out_ << color(c) << "|" << color(Colors::Reset)
                  << " " << messagePositions[msgIdx].second->message << "\n";
        }
    }
}

void DiagnosticRenderer::renderNotes(const Diagnostic& diag) {
    for (const auto& note : diag.notes()) {
        *out_ << color(Colors::Note)
              << " = note: "
              << color(Colors::Reset)
              << note << "\n";
    }
}

void DiagnosticRenderer::renderSuggestions(const Diagnostic& diag) {
    for (const auto& suggestion : diag.suggestions()) {
        *out_ << color(Colors::Help)
              << "help: "
              << color(Colors::Reset)
              << suggestion.message << "\n";

        if (suggestion.span.isValid()) {
            Position pos = sourceManager_.getPosition(suggestion.span.fileId,
                                                       suggestion.span.startOffset);
            std::string_view line = sourceManager_.getLine(suggestion.span.fileId, pos.line);

            // Build the suggested line
            std::string newLine;
            uint32_t startCol = pos.column - 1;
            uint32_t endCol = startCol + suggestion.span.length();

            if (startCol < line.size()) {
                newLine = std::string(line.substr(0, startCol));
                newLine += suggestion.newText;
                if (endCol < line.size()) {
                    newLine += std::string(line.substr(endCol));
                }
            } else {
                newLine = std::string(line) + suggestion.newText;
            }

            uint32_t maxLineWidth = std::to_string(pos.line).length();

            *out_ << std::string(maxLineWidth + 1, ' ')
                  << color(Colors::LineNumber) << "|" << color(Colors::Reset) << "\n";

            *out_ << color(Colors::LineNumber)
                  << std::setw(maxLineWidth) << pos.line
                  << " | "
                  << color(Colors::Reset)
                  << newLine << "\n";

            // Underline the change with tildes
            *out_ << std::string(maxLineWidth, ' ')
                  << color(Colors::LineNumber) << " | " << color(Colors::Reset)
                  << std::string(startCol, ' ')
                  << color(Colors::Help)
                  << std::string(suggestion.newText.length(), '~')
                  << color(Colors::Reset) << "\n";
        }
    }
}

uint32_t DiagnosticRenderer::computeMaxLineNumberWidth(const Diagnostic& diag) const {
    uint32_t maxLine = 1;
    for (const auto& label : diag.labels()) {
        Position endPos = sourceManager_.getPosition(label.span.fileId, label.span.endOffset);
        maxLine = std::max(maxLine, endPos.line);
    }
    return std::to_string(maxLine).length();
}

std::string DiagnosticRenderer::escapeJSON(const std::string& str) const {
    std::ostringstream oss;
    for (char c : str) {
        switch (c) {
            case '"': oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if (c >= 0 && c < 32) {
                    oss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c;
                } else {
                    oss << c;
                }
        }
    }
    return oss.str();
}

void DiagnosticRenderer::renderJSON(const Diagnostic& diag) {
    *out_ << "{";
    *out_ << "\"code\":\"" << formatErrorCode(diag.code()) << "\",";
    *out_ << "\"severity\":\"" << severityName(diag.severity()) << "\",";
    *out_ << "\"message\":\"" << escapeJSON(diag.message()) << "\",";

    // Primary span location
    auto primarySpan = diag.primarySpan();
    if (primarySpan) {
        Position start = sourceManager_.getPosition(primarySpan->fileId, primarySpan->startOffset);
        Position end = sourceManager_.getPosition(primarySpan->fileId, primarySpan->endOffset);
        std::string_view fileName = sourceManager_.getFileName(primarySpan->fileId);

        *out_ << "\"location\":{";
        *out_ << "\"file\":\"" << escapeJSON(std::string(fileName)) << "\",";
        *out_ << "\"start\":{\"line\":" << start.line << ",\"column\":" << start.column << "},";
        *out_ << "\"end\":{\"line\":" << end.line << ",\"column\":" << end.column << "}";
        *out_ << "},";
    }

    // Labels
    *out_ << "\"labels\":[";
    bool first = true;
    for (const auto& label : diag.labels()) {
        if (!first) *out_ << ",";
        first = false;

        Position start = sourceManager_.getPosition(label.span.fileId, label.span.startOffset);
        Position end = sourceManager_.getPosition(label.span.fileId, label.span.endOffset);
        std::string_view fileName = sourceManager_.getFileName(label.span.fileId);

        *out_ << "{";
        *out_ << "\"file\":\"" << escapeJSON(std::string(fileName)) << "\",";
        *out_ << "\"start\":{\"line\":" << start.line << ",\"column\":" << start.column << "},";
        *out_ << "\"end\":{\"line\":" << end.line << ",\"column\":" << end.column << "},";
        *out_ << "\"message\":\"" << escapeJSON(label.message) << "\",";
        *out_ << "\"primary\":" << (label.isPrimary ? "true" : "false");
        *out_ << "}";
    }
    *out_ << "],";

    // Notes
    *out_ << "\"notes\":[";
    first = true;
    for (const auto& note : diag.notes()) {
        if (!first) *out_ << ",";
        first = false;
        *out_ << "\"" << escapeJSON(note) << "\"";
    }
    *out_ << "],";

    // Suggestions
    *out_ << "\"suggestions\":[";
    first = true;
    for (const auto& suggestion : diag.suggestions()) {
        if (!first) *out_ << ",";
        first = false;

        *out_ << "{";
        if (suggestion.span.isValid()) {
            Position start = sourceManager_.getPosition(suggestion.span.fileId,
                                                         suggestion.span.startOffset);
            Position end = sourceManager_.getPosition(suggestion.span.fileId,
                                                       suggestion.span.endOffset);
            *out_ << "\"start\":{\"line\":" << start.line << ",\"column\":" << start.column << "},";
            *out_ << "\"end\":{\"line\":" << end.line << ",\"column\":" << end.column << "},";
        }
        *out_ << "\"replacement\":\"" << escapeJSON(suggestion.newText) << "\",";
        *out_ << "\"message\":\"" << escapeJSON(suggestion.message) << "\"";
        *out_ << "}";
    }
    *out_ << "]";

    *out_ << "}\n";
}

} // namespace diag
} // namespace simp
