#include "diagnostics/diagnostic_context.hpp"

namespace simp {
namespace diag {

// Global diagnostic context instance
static DiagnosticContext g_diagContext;

DiagnosticContext& getDiagnosticContext() {
    return g_diagContext;
}

void initDiagnosticContext(SourceManager& sm, DiagnosticEngine& engine, uint32_t fileId) {
    g_diagContext.sourceManager = &sm;
    g_diagContext.engine = &engine;
    g_diagContext.currentFileId = fileId;
}

void resetDiagnosticContext() {
    g_diagContext.sourceManager = nullptr;
    g_diagContext.engine = nullptr;
    g_diagContext.currentFileId = 0;
}

Span makeSpan(int startLine, int startCol, int endLine, int endCol) {
    auto& ctx = getDiagnosticContext();
    if (!ctx.isValid() || ctx.currentFileId == 0) {
        return Span();
    }

    uint32_t startOffset = ctx.sourceManager->getOffset(ctx.currentFileId, startLine, startCol);
    uint32_t endOffset = ctx.sourceManager->getOffset(ctx.currentFileId, endLine, endCol);

    return Span(ctx.currentFileId, startOffset, endOffset);
}

Span makeSpan(int line, int col) {
    return makeSpan(line, col, line, col + 1);
}

} // namespace diag
} // namespace simp

// C-compatible interface implementation
extern "C" {

void simp_lexer_error(int line, int col, const char* message) {
    auto& ctx = simp::diag::getDiagnosticContext();
    if (!ctx.isValid()) {
        // Fallback to stderr if diagnostics not initialized
        fprintf(stderr, "Lexer error at line %d, col %d: %s\n", line, col, message);
        return;
    }

    auto span = simp::diag::makeSpan(line, col);
    ctx.engine->error(simp::diag::ErrorCode::E0001)
        .withMessage(message)
        .at(span)
        .emit();
}

void simp_parser_error(int first_line, int first_col,
                       int last_line, int last_col,
                       const char* message) {
    auto& ctx = simp::diag::getDiagnosticContext();
    if (!ctx.isValid()) {
        // Fallback to stderr if diagnostics not initialized
        fprintf(stderr, "Parse error at line %d:%d-%d:%d: %s\n",
                first_line, first_col, last_line, last_col, message);
        return;
    }

    auto span = simp::diag::makeSpan(first_line, first_col, last_line, last_col);
    ctx.engine->error(simp::diag::ErrorCode::E0100)
        .withMessage(message)
        .at(span)
        .emit();
}

int simp_has_errors(void) {
    auto& ctx = simp::diag::getDiagnosticContext();
    if (!ctx.isValid()) {
        return 0;
    }
    return ctx.engine->hasErrors() ? 1 : 0;
}

} // extern "C"
