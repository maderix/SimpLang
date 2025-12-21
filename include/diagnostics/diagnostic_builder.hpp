#pragma once

#include "diagnostics/diagnostic.hpp"
#include "diagnostics/error_codes.hpp"
#include "diagnostics/span.hpp"
#include <memory>
#include <string>

namespace simp {
namespace diag {

// Forward declaration
class DiagnosticEngine;

/// Fluent API for building and emitting diagnostics
///
/// Example usage:
///   engine.error(ErrorCode::E0300)
///       .withMessage("undefined variable")
///       .at(span, "`foo` not found in this scope")
///       .noteAt(defSpan, "did you mean `food`?")
///       .emit();
///
class DiagnosticBuilder {
public:
    DiagnosticBuilder(DiagnosticEngine& engine, ErrorCode code, Severity severity);

    // Move-only (prevents accidental copying and double-emit)
    DiagnosticBuilder(const DiagnosticBuilder&) = delete;
    DiagnosticBuilder& operator=(const DiagnosticBuilder&) = delete;
    DiagnosticBuilder(DiagnosticBuilder&& other) noexcept;
    DiagnosticBuilder& operator=(DiagnosticBuilder&& other) noexcept;

    ~DiagnosticBuilder();

    /// Set the main message (overrides default from error code)
    DiagnosticBuilder& withMessage(std::string message);

    /// Add a primary label at the given span
    /// Primary labels are highlighted with ^^^ and show the main error location
    DiagnosticBuilder& at(Span span, std::string message = "");

    /// Add a secondary label at the given span
    /// Secondary labels are highlighted with --- and provide additional context
    DiagnosticBuilder& noteAt(Span span, std::string message);

    /// Add a note without a source location
    DiagnosticBuilder& note(std::string message);

    /// Add a suggestion for fixing the error
    DiagnosticBuilder& suggest(Span span, std::string replacement, std::string message);

    /// Add a help message (alias for note with "help: " prefix)
    DiagnosticBuilder& help(std::string message);

    /// Emit the diagnostic to the engine
    void emit();

    /// Cancel the diagnostic without emitting
    void cancel();

    /// Check if this builder has been consumed (emitted or cancelled)
    bool isConsumed() const { return consumed_; }

private:
    DiagnosticEngine* engine_;
    std::unique_ptr<Diagnostic> diagnostic_;
    bool consumed_;
};

} // namespace diag
} // namespace simp
