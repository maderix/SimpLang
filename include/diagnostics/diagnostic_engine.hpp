#pragma once

#include "diagnostics/diagnostic.hpp"
#include "diagnostics/diagnostic_builder.hpp"
#include "diagnostics/error_codes.hpp"
#include "diagnostics/source_manager.hpp"
#include <functional>
#include <memory>
#include <ostream>
#include <vector>

namespace simp {
namespace diag {

// Forward declaration
class DiagnosticRenderer;

/// Output format for diagnostics
enum class OutputFormat {
    Terminal,  // Human-readable colored terminal output
    JSON       // Machine-readable JSON output
};

/// Configuration for the diagnostic engine
struct DiagnosticConfig {
    OutputFormat format = OutputFormat::Terminal;
    bool useColors = true;          // Use ANSI colors (terminal only)
    bool showNotes = true;          // Show note-level diagnostics
    uint32_t maxErrors = 50;        // Stop after this many errors (0 = unlimited)
    bool treatWarningsAsErrors = false;
};

/// Handler callback for diagnostics (for custom handling)
using DiagnosticHandler = std::function<void(const Diagnostic&)>;

/// The main diagnostic engine
///
/// This is the central coordinator for all diagnostic output. It:
/// - Provides factory methods for creating DiagnosticBuilders
/// - Tracks diagnostic statistics
/// - Renders diagnostics via the DiagnosticRenderer
/// - Supports custom diagnostic handlers
///
class DiagnosticEngine {
public:
    explicit DiagnosticEngine(SourceManager& sourceManager);
    ~DiagnosticEngine();

    // Non-copyable, non-movable (holds reference to SourceManager)
    DiagnosticEngine(const DiagnosticEngine&) = delete;
    DiagnosticEngine& operator=(const DiagnosticEngine&) = delete;
    DiagnosticEngine(DiagnosticEngine&&) = delete;
    DiagnosticEngine& operator=(DiagnosticEngine&&) = delete;

    // ========================================================================
    // Configuration
    // ========================================================================

    /// Set the configuration
    void setConfig(DiagnosticConfig config);

    /// Get the current configuration
    const DiagnosticConfig& config() const { return config_; }

    /// Set the output stream (default is std::cerr)
    void setOutputStream(std::ostream& os);

    /// Add a custom diagnostic handler
    void addHandler(DiagnosticHandler handler);

    // ========================================================================
    // Diagnostic Creation (Fluent API)
    // ========================================================================

    /// Create an error diagnostic
    DiagnosticBuilder error(ErrorCode code);

    /// Create a warning diagnostic
    DiagnosticBuilder warning(ErrorCode code);

    /// Create a note diagnostic
    DiagnosticBuilder note(ErrorCode code);

    /// Create a fatal error diagnostic
    DiagnosticBuilder fatal(ErrorCode code);

    /// Create a diagnostic with explicit severity
    DiagnosticBuilder diagnostic(ErrorCode code, Severity severity);

    // ========================================================================
    // Direct Emission (for simple cases)
    // ========================================================================

    /// Emit a simple error with just a message
    void emitError(ErrorCode code, Span span, const std::string& message);

    /// Emit a simple warning with just a message
    void emitWarning(ErrorCode code, Span span, const std::string& message);

    // ========================================================================
    // Statistics and State
    // ========================================================================

    /// Get current diagnostic statistics
    const DiagnosticStats& stats() const { return stats_; }

    /// Check if any errors have been emitted
    bool hasErrors() const { return stats_.hasErrors(); }

    /// Check if error limit has been reached
    bool errorLimitReached() const;

    /// Reset statistics
    void resetStats() { stats_.reset(); }

    // ========================================================================
    // Source Manager Access
    // ========================================================================

    /// Get the associated source manager
    SourceManager& sourceManager() { return sourceManager_; }
    const SourceManager& sourceManager() const { return sourceManager_; }

    // ========================================================================
    // Explain Mode
    // ========================================================================

    /// Print detailed explanation for an error code
    void explain(ErrorCode code);

    /// Print detailed explanation for an error code string (e.g., "E0200")
    bool explain(const std::string& codeStr);

    // ========================================================================
    // Internal Methods (used by DiagnosticBuilder)
    // ========================================================================

    /// Emit a constructed diagnostic (called by DiagnosticBuilder)
    void emit(Diagnostic diagnostic);

private:
    SourceManager& sourceManager_;
    DiagnosticConfig config_;
    DiagnosticStats stats_;
    std::unique_ptr<DiagnosticRenderer> renderer_;
    std::vector<DiagnosticHandler> handlers_;
    std::ostream* outputStream_;
};

} // namespace diag
} // namespace simp
