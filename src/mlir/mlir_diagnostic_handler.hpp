//===- mlir_diagnostic_handler.hpp - ICE Handler for MLIR Errors ---------===//
//
// Part of the SimpLang Project
//
// This file provides a diagnostic handler that intercepts MLIR errors and
// presents them as user-friendly Internal Compiler Errors (ICE).
//
//===----------------------------------------------------------------------===//

#pragma once

#include "diagnostics/error_codes.hpp"
#include "diagnostics/diagnostic_context.hpp"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"

#include <string>
#include <vector>
#include <regex>

namespace mlir {
namespace simp {

/// Captured MLIR diagnostic information
struct CapturedDiagnostic {
    std::string message;
    std::string filename;
    unsigned line = 0;
    unsigned column = 0;
    mlir::DiagnosticSeverity severity;
};

/// Handler that intercepts MLIR diagnostics and converts them to ICE format
class ICEDiagnosticHandler {
public:
    explicit ICEDiagnosticHandler(mlir::MLIRContext* context, bool verbose = false);
    ~ICEDiagnosticHandler();

    /// Set the current compilation phase (for error messages)
    void setPhase(const std::string& phaseName) { currentPhase = phaseName; }

    /// Check if any errors were captured
    bool hasErrors() const { return !capturedDiags.empty(); }

    /// Get number of captured errors
    size_t errorCount() const { return capturedDiags.size(); }

    /// Clear captured diagnostics (call before each phase)
    void clear() { capturedDiags.clear(); }

    /// Emit user-friendly ICE for all captured errors
    void emitICE();

    /// Set verbose mode (show raw MLIR diagnostics)
    void setVerbose(bool v) { verbose = v; }

    /// Get the captured diagnostics (for testing)
    const std::vector<CapturedDiagnostic>& getDiagnostics() const {
        return capturedDiags;
    }

private:
    /// Handle a single MLIR diagnostic
    mlir::LogicalResult handleDiagnostic(mlir::Diagnostic& diag);

    /// Extract source location from MLIR Location
    void extractLocation(mlir::Location loc, std::string& filename,
                        unsigned& line, unsigned& column);

    /// Classify MLIR error into our ICE error code
    ::simp::diag::ErrorCode classifyError(const std::string& message);

    /// Format the user-friendly ICE message
    std::string formatICEMessage(const CapturedDiagnostic& diag);

    mlir::MLIRContext* context;
    mlir::DiagnosticEngine::HandlerID handlerId;
    std::string currentPhase;
    std::vector<CapturedDiagnostic> capturedDiags;
    bool verbose;
};

/// RAII helper to set phase and clear diagnostics
class PhaseScope {
public:
    PhaseScope(ICEDiagnosticHandler& handler, const std::string& phase)
        : handler(handler) {
        handler.clear();
        handler.setPhase(phase);
    }

    ~PhaseScope() = default;

private:
    ICEDiagnosticHandler& handler;
};

} // namespace simp
} // namespace mlir
