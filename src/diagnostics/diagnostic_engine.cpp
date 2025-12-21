#include "diagnostics/diagnostic_engine.hpp"
#include "diagnostics/diagnostic_renderer.hpp"
#include <iostream>
#include <regex>

namespace simp {
namespace diag {

DiagnosticEngine::DiagnosticEngine(SourceManager& sourceManager)
    : sourceManager_(sourceManager),
      renderer_(std::make_unique<DiagnosticRenderer>(sourceManager)),
      outputStream_(&std::cerr) {}

DiagnosticEngine::~DiagnosticEngine() = default;

void DiagnosticEngine::setConfig(DiagnosticConfig config) {
    config_ = config;
    renderer_->setUseColors(config.useColors);
}

void DiagnosticEngine::setOutputStream(std::ostream& os) {
    outputStream_ = &os;
    renderer_->setOutputStream(os);
}

void DiagnosticEngine::addHandler(DiagnosticHandler handler) {
    handlers_.push_back(std::move(handler));
}

DiagnosticBuilder DiagnosticEngine::error(ErrorCode code) {
    return DiagnosticBuilder(*this, code, Severity::Error);
}

DiagnosticBuilder DiagnosticEngine::warning(ErrorCode code) {
    return DiagnosticBuilder(*this, code, Severity::Warning);
}

DiagnosticBuilder DiagnosticEngine::note(ErrorCode code) {
    return DiagnosticBuilder(*this, code, Severity::Note);
}

DiagnosticBuilder DiagnosticEngine::fatal(ErrorCode code) {
    return DiagnosticBuilder(*this, code, Severity::Fatal);
}

DiagnosticBuilder DiagnosticEngine::diagnostic(ErrorCode code, Severity severity) {
    return DiagnosticBuilder(*this, code, severity);
}

void DiagnosticEngine::emitError(ErrorCode code, Span span, const std::string& message) {
    error(code).withMessage(message).at(span).emit();
}

void DiagnosticEngine::emitWarning(ErrorCode code, Span span, const std::string& message) {
    warning(code).withMessage(message).at(span).emit();
}

bool DiagnosticEngine::errorLimitReached() const {
    return config_.maxErrors > 0 && stats_.errors >= config_.maxErrors;
}

void DiagnosticEngine::emit(Diagnostic diagnostic) {
    // Adjust severity if treating warnings as errors
    Severity severity = diagnostic.severity();
    if (config_.treatWarningsAsErrors && severity == Severity::Warning) {
        severity = Severity::Error;
    }

    // Skip notes if disabled
    if (!config_.showNotes && severity == Severity::Note) {
        return;
    }

    // Record statistics
    stats_.record(severity);

    // Check error limit - only skip rendering AFTER we exceed the limit
    if (config_.maxErrors > 0 && stats_.errors > config_.maxErrors) {
        // We're past the limit, don't render more errors
        // Print "too many errors" exactly once (when we first exceed)
        if (stats_.errors == config_.maxErrors + 1) {
            *outputStream_ << "error: too many errors emitted, stopping now\n";
        }
        return;
    }

    // Call custom handlers
    for (const auto& handler : handlers_) {
        handler(diagnostic);
    }

    // Render the diagnostic
    if (config_.format == OutputFormat::Terminal) {
        renderer_->render(diagnostic);
    } else {
        renderer_->renderJSON(diagnostic);
    }
}

void DiagnosticEngine::explain(ErrorCode code) {
    const auto& info = getErrorInfo(code);
    *outputStream_ << "\n";
    *outputStream_ << info.name << ": " << info.title << "\n";
    *outputStream_ << std::string(70, '-') << "\n";
    *outputStream_ << info.explanation << "\n";
    *outputStream_ << "\n";
}

bool DiagnosticEngine::explain(const std::string& codeStr) {
    // Parse error code string like "E0200"
    std::regex pattern("E(\\d{4})");
    std::smatch match;
    if (std::regex_match(codeStr, match, pattern)) {
        uint32_t codeNum = std::stoul(match[1].str());
        explain(static_cast<ErrorCode>(codeNum));
        return true;
    }
    *outputStream_ << "error: invalid error code format '" << codeStr
                   << "' (expected format: E0001)\n";
    return false;
}

} // namespace diag
} // namespace simp
