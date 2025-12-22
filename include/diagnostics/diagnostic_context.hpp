#pragma once

/// @file diagnostic_context.hpp
/// @brief Global diagnostic context for lexer/parser integration
///
/// Flex and Bison require global state for error reporting. This header
/// provides the global pointers and helper functions needed to emit
/// diagnostics from the lexer and parser.

#include "diagnostics/diagnostics.hpp"

namespace simp {
namespace diag {

/// Global diagnostic context - must be initialized before parsing
struct DiagnosticContext {
    SourceManager* sourceManager = nullptr;
    DiagnosticEngine* engine = nullptr;
    uint32_t currentFileId = 0;

    bool isValid() const {
        return sourceManager != nullptr && engine != nullptr;
    }
};

/// Get the global diagnostic context
DiagnosticContext& getDiagnosticContext();

/// Initialize the global diagnostic context
void initDiagnosticContext(SourceManager& sm, DiagnosticEngine& engine, uint32_t fileId);

/// Reset the global diagnostic context
void resetDiagnosticContext();

/// Create a span from lexer/parser location info
/// @param startLine Starting line (1-indexed)
/// @param startCol Starting column (1-indexed)
/// @param endLine Ending line (1-indexed)
/// @param endCol Ending column (1-indexed)
Span makeSpan(int startLine, int startCol, int endLine, int endCol);

/// Create a span for a single position
Span makeSpan(int line, int col);

} // namespace diag
} // namespace simp

// C-compatible interface for flex/bison
#ifdef __cplusplus
extern "C" {
#endif

/// Emit a lexer error with the current token location
/// @param line Current line number
/// @param col Current column number
/// @param message Error message
void simp_lexer_error(int line, int col, const char* message);

/// Emit a parser error with location info
/// @param first_line Start line
/// @param first_col Start column
/// @param last_line End line
/// @param last_col End column
/// @param message Error message
void simp_parser_error(int first_line, int first_col,
                       int last_line, int last_col,
                       const char* message);

/// Check if any errors have been emitted
int simp_has_errors(void);

#ifdef __cplusplus
}
#endif
