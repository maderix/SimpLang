#include "diagnostics/diagnostic_builder.hpp"
#include "diagnostics/diagnostic_engine.hpp"

namespace simp {
namespace diag {

DiagnosticBuilder::DiagnosticBuilder(DiagnosticEngine& engine, ErrorCode code, Severity severity)
    : engine_(&engine), consumed_(false) {
    const auto& info = getErrorInfo(code);
    diagnostic_ = std::make_unique<Diagnostic>(code, severity, info.title);
}

DiagnosticBuilder::DiagnosticBuilder(DiagnosticBuilder&& other) noexcept
    : engine_(other.engine_),
      diagnostic_(std::move(other.diagnostic_)),
      consumed_(other.consumed_) {
    other.consumed_ = true;
}

DiagnosticBuilder& DiagnosticBuilder::operator=(DiagnosticBuilder&& other) noexcept {
    if (this != &other) {
        // Emit current diagnostic if not consumed
        if (!consumed_ && diagnostic_) {
            emit();
        }
        engine_ = other.engine_;
        diagnostic_ = std::move(other.diagnostic_);
        consumed_ = other.consumed_;
        other.consumed_ = true;
    }
    return *this;
}

DiagnosticBuilder::~DiagnosticBuilder() {
    // Auto-emit if not explicitly emitted or cancelled
    if (!consumed_ && diagnostic_) {
        emit();
    }
}

DiagnosticBuilder& DiagnosticBuilder::withMessage(std::string message) {
    if (diagnostic_) {
        diagnostic_ = std::make_unique<Diagnostic>(
            diagnostic_->code(),
            diagnostic_->severity(),
            std::move(message));
        // Preserve existing labels and notes
        for (const auto& label : diagnostic_->labels()) {
            if (label.isPrimary) {
                diagnostic_->withPrimaryLabel(label.span, label.message);
            } else {
                diagnostic_->withSecondaryLabel(label.span, label.message);
            }
        }
    }
    return *this;
}

DiagnosticBuilder& DiagnosticBuilder::at(Span span, std::string message) {
    if (diagnostic_) {
        diagnostic_->withPrimaryLabel(span, std::move(message));
    }
    return *this;
}

DiagnosticBuilder& DiagnosticBuilder::noteAt(Span span, std::string message) {
    if (diagnostic_) {
        diagnostic_->withSecondaryLabel(span, std::move(message));
    }
    return *this;
}

DiagnosticBuilder& DiagnosticBuilder::note(std::string message) {
    if (diagnostic_) {
        diagnostic_->withNote(std::move(message));
    }
    return *this;
}

DiagnosticBuilder& DiagnosticBuilder::suggest(Span span, std::string replacement,
                                               std::string message) {
    if (diagnostic_) {
        diagnostic_->withSuggestion(span, std::move(replacement), std::move(message));
    }
    return *this;
}

DiagnosticBuilder& DiagnosticBuilder::help(std::string message) {
    if (diagnostic_) {
        diagnostic_->withNote("help: " + std::move(message));
    }
    return *this;
}

void DiagnosticBuilder::emit() {
    if (!consumed_ && diagnostic_ && engine_) {
        diagnostic_->sortLabels();
        engine_->emit(std::move(*diagnostic_));
        consumed_ = true;
    }
}

void DiagnosticBuilder::cancel() {
    consumed_ = true;
    diagnostic_.reset();
}

} // namespace diag
} // namespace simp
