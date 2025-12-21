#pragma once

#include "diagnostics/error_codes.hpp"
#include "diagnostics/span.hpp"
#include <optional>
#include <string>
#include <vector>

namespace simp {
namespace diag {

/// A label attached to a diagnostic, highlighting a specific span of source code
struct Label {
    Span span;
    std::string message;
    bool isPrimary; // Primary labels use ^^^ style, secondary use ---

    Label(Span span, std::string message, bool isPrimary = false)
        : span(span), message(std::move(message)), isPrimary(isPrimary) {}
};

/// A suggestion for how to fix an error
struct Suggestion {
    Span span;           // What to replace
    std::string newText; // What to replace it with
    std::string message; // Human-readable explanation

    Suggestion(Span span, std::string newText, std::string message)
        : span(span), newText(std::move(newText)), message(std::move(message)) {}
};

/// A complete diagnostic message
class Diagnostic {
public:
    Diagnostic(ErrorCode code, Severity severity, std::string message)
        : code_(code), severity_(severity), message_(std::move(message)) {}

    // Getters
    ErrorCode code() const { return code_; }
    Severity severity() const { return severity_; }
    const std::string& message() const { return message_; }
    const std::vector<Label>& labels() const { return labels_; }
    const std::vector<std::string>& notes() const { return notes_; }
    const std::vector<Suggestion>& suggestions() const { return suggestions_; }

    /// Get the primary span (first primary label's span, or first label's span)
    std::optional<Span> primarySpan() const {
        for (const auto& label : labels_) {
            if (label.isPrimary) {
                return label.span;
            }
        }
        if (!labels_.empty()) {
            return labels_.front().span;
        }
        return std::nullopt;
    }

    // Fluent API for building diagnostics
    Diagnostic& withPrimaryLabel(Span span, std::string message) {
        labels_.emplace_back(span, std::move(message), true);
        return *this;
    }

    Diagnostic& withSecondaryLabel(Span span, std::string message) {
        labels_.emplace_back(span, std::move(message), false);
        return *this;
    }

    Diagnostic& withNote(std::string note) {
        notes_.push_back(std::move(note));
        return *this;
    }

    Diagnostic& withSuggestion(Span span, std::string newText, std::string message) {
        suggestions_.emplace_back(span, std::move(newText), std::move(message));
        return *this;
    }

    // Sort labels by file, then by span start offset
    void sortLabels();

private:
    ErrorCode code_;
    Severity severity_;
    std::string message_;
    std::vector<Label> labels_;
    std::vector<std::string> notes_;
    std::vector<Suggestion> suggestions_;
};

/// Statistics about diagnostics emitted
struct DiagnosticStats {
    uint32_t errors = 0;
    uint32_t warnings = 0;
    uint32_t notes = 0;

    bool hasErrors() const { return errors > 0; }

    void record(Severity severity) {
        switch (severity) {
            case Severity::Error:
            case Severity::Fatal:
                ++errors;
                break;
            case Severity::Warning:
                ++warnings;
                break;
            case Severity::Note:
                ++notes;
                break;
        }
    }

    void reset() {
        errors = 0;
        warnings = 0;
        notes = 0;
    }
};

} // namespace diag
} // namespace simp
