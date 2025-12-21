#pragma once

#include "diagnostics/diagnostic.hpp"
#include "diagnostics/source_manager.hpp"
#include <ostream>
#include <string>

namespace simp {
namespace diag {

/// ANSI color codes for terminal output
struct Colors {
    // Basic colors
    static constexpr const char* Reset = "\033[0m";
    static constexpr const char* Bold = "\033[1m";

    // Severity colors
    static constexpr const char* Error = "\033[1;31m";     // Bold red
    static constexpr const char* Warning = "\033[1;33m";   // Bold yellow
    static constexpr const char* Note = "\033[1;36m";      // Bold cyan
    static constexpr const char* Help = "\033[1;32m";      // Bold green

    // Source display
    static constexpr const char* LineNumber = "\033[1;34m"; // Bold blue
    static constexpr const char* Primary = "\033[1;31m";    // Bold red (for ^^^)
    static constexpr const char* Secondary = "\033[1;34m";  // Bold blue (for ---)
};

/// Renders diagnostics in a Rust-style format
///
/// Example output:
/// ```
/// error[E0200]: type mismatch in assignment
///  --> examples/test.sl:5:12
///   |
/// 5 |     var x: i32 = "hello";
///   |            ^^^   ^^^^^^^ expected `i32`, found `string`
///   |            |
///   |            expected due to this type annotation
///   |
/// help: consider using an integer literal
///   |
/// 5 |     var x: i32 = 42;
///   |                  ~~
/// ```
///
class DiagnosticRenderer {
public:
    explicit DiagnosticRenderer(const SourceManager& sourceManager);
    ~DiagnosticRenderer() = default;

    /// Set the output stream
    void setOutputStream(std::ostream& os);

    /// Enable or disable colors
    void setUseColors(bool useColors) { useColors_ = useColors; }

    /// Render a diagnostic in human-readable format
    void render(const Diagnostic& diag);

    /// Render a diagnostic as JSON
    void renderJSON(const Diagnostic& diag);

private:
    // Color helpers
    const char* severityColor(Severity severity) const;
    const char* severityName(Severity severity) const;
    const char* color(const char* code) const;

    // Rendering helpers
    void renderHeader(const Diagnostic& diag);
    void renderLocation(const Diagnostic& diag);
    void renderSnippet(const Diagnostic& diag);
    void renderNotes(const Diagnostic& diag);
    void renderSuggestions(const Diagnostic& diag);

    // Line rendering
    void renderLine(uint32_t lineNumber, std::string_view content,
                    uint32_t maxLineNumWidth);
    void renderUnderlines(const std::vector<Label>& labels, uint32_t lineNumber,
                          std::string_view lineContent, uint32_t maxLineNumWidth);

    // Utility
    uint32_t computeMaxLineNumberWidth(const Diagnostic& diag) const;
    std::string escapeJSON(const std::string& str) const;

    const SourceManager& sourceManager_;
    std::ostream* out_;
    bool useColors_;
};

} // namespace diag
} // namespace simp
