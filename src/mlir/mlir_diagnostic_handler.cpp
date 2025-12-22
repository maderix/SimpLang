//===- mlir_diagnostic_handler.cpp - ICE Handler Implementation ----------===//
//
// Part of the SimpLang Project
//
//===----------------------------------------------------------------------===//

#include "mlir_diagnostic_handler.hpp"

#include "mlir/IR/Location.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Regex.h"

#include <iostream>
#include <sstream>

namespace mlir {
namespace simp {

ICEDiagnosticHandler::ICEDiagnosticHandler(mlir::MLIRContext* context, bool verbose)
    : context(context), currentPhase("unknown"), verbose(verbose) {

    // Register our diagnostic handler with MLIR
    auto& diagEngine = context->getDiagEngine();
    handlerId = diagEngine.registerHandler(
        [this](mlir::Diagnostic& diag) -> mlir::LogicalResult {
            return this->handleDiagnostic(diag);
        });
}

ICEDiagnosticHandler::~ICEDiagnosticHandler() {
    // Unregister the handler
    context->getDiagEngine().eraseHandler(handlerId);
}

mlir::LogicalResult ICEDiagnosticHandler::handleDiagnostic(mlir::Diagnostic& diag) {
    // Only capture errors (not warnings or notes)
    if (diag.getSeverity() != mlir::DiagnosticSeverity::Error) {
        // For verbose mode, print warnings/notes
        if (verbose) {
            llvm::errs() << "[MLIR] " << diag << "\n";
        }
        return mlir::success(); // Handled (suppressed)
    }

    // Capture the diagnostic
    CapturedDiagnostic captured;
    captured.message = diag.str();
    captured.severity = diag.getSeverity();
    extractLocation(diag.getLocation(), captured.filename, captured.line, captured.column);

    capturedDiags.push_back(std::move(captured));

    // Return success to indicate we handled it (prevents default printing)
    return mlir::success();
}

void ICEDiagnosticHandler::extractLocation(mlir::Location loc, std::string& filename,
                                           unsigned& line, unsigned& column) {
    // Default values
    filename = "";
    line = 0;
    column = 0;

    // Handle FileLineColLoc directly
    if (auto fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>(loc)) {
        filename = fileLoc.getFilename().str();
        line = fileLoc.getLine();
        column = fileLoc.getColumn();
        return;
    }

    // Handle NameLoc (wraps another location with a name)
    // Pattern: loc("var:result"("test.sl":9:0))
    if (auto nameLoc = mlir::dyn_cast<mlir::NameLoc>(loc)) {
        // Recursively extract from the child location
        extractLocation(nameLoc.getChildLoc(), filename, line, column);
        return;
    }

    // Handle FusedLoc (multiple locations fused together)
    if (auto fusedLoc = mlir::dyn_cast<mlir::FusedLoc>(loc)) {
        // Use the first non-unknown location
        for (auto innerLoc : fusedLoc.getLocations()) {
            extractLocation(innerLoc, filename, line, column);
            if (!filename.empty() && line > 0) {
                return;
            }
        }
    }

    // Handle CallSiteLoc
    if (auto callSiteLoc = mlir::dyn_cast<mlir::CallSiteLoc>(loc)) {
        extractLocation(callSiteLoc.getCallee(), filename, line, column);
        return;
    }

    // For unknown locations, try to parse from string representation
    // This is a fallback for complex nested locations
    std::string locStr;
    llvm::raw_string_ostream os(locStr);
    loc.print(os);
    os.flush();

    // Try to parse patterns like: "test.sl":9:0 or ("test.sl":9:0)
    // Using LLVM's Regex (no exceptions)
    llvm::Regex fileLocPattern("\"([^\"]+)\":([0-9]+):([0-9]+)");
    llvm::SmallVector<llvm::StringRef, 4> matches;
    if (fileLocPattern.match(locStr, &matches) && matches.size() >= 4) {
        filename = matches[1].str();
        line = std::stoul(matches[2].str());
        column = std::stoul(matches[3].str());
    }
}

::simp::diag::ErrorCode ICEDiagnosticHandler::classifyError(const std::string& message) {
    using EC = ::simp::diag::ErrorCode;

    // Pattern matching for known MLIR errors
    if (message.find("memref.alloc") != std::string::npos &&
        message.find("symbol operand count") != std::string::npos) {
        return EC::E0704;  // Memory allocation lowering failed
    }

    if (message.find("memref.collapse_shape") != std::string::npos ||
        message.find("memref.expand_shape") != std::string::npos) {
        return EC::E0705;  // Tensor reshape failed
    }

    if (message.find("unknown memory side effects") != std::string::npos ||
        message.find("bufferization") != std::string::npos) {
        return EC::E0703;  // Bufferization failed
    }

    if (message.find("verification failed") != std::string::npos) {
        return EC::E0701;  // MLIR verification failed
    }

    // Default to generic ICE
    return EC::E0700;
}

std::string ICEDiagnosticHandler::formatICEMessage(const CapturedDiagnostic& diag) {
    std::ostringstream ss;

    // Location header
    if (!diag.filename.empty() && diag.line > 0) {
        ss << "  --> " << diag.filename << ":" << diag.line;
        if (diag.column > 0) {
            ss << ":" << diag.column;
        }
        ss << "\n";
    }

    // Phase info
    ss << "   = note: failed during " << currentPhase << "\n";

    // Verbose mode: show raw MLIR error
    if (verbose) {
        ss << "\nMLIR Diagnostic:\n";
        ss << "  " << diag.message << "\n";
    }

    return ss.str();
}

void ICEDiagnosticHandler::emitICE() {
    if (capturedDiags.empty()) {
        return;
    }

    // Get the diagnostic context
    auto& ctx = ::simp::diag::getDiagnosticContext();

    // Group errors by classification
    std::map<::simp::diag::ErrorCode, std::vector<const CapturedDiagnostic*>> grouped;
    for (const auto& diag : capturedDiags) {
        auto code = classifyError(diag.message);
        grouped[code].push_back(&diag);
    }

    // Emit ICE for each group
    for (const auto& [code, diags] : grouped) {
        const auto& info = ::simp::diag::getErrorInfo(code);

        // Print header
        llvm::errs() << "\n";
        llvm::errs() << "\033[1;31merror[" << info.name << "]: " << info.title << "\033[0m\n";
        llvm::errs() << "\n";
        llvm::errs() << "The compiler encountered an unexpected internal error.\n";
        llvm::errs() << "This is a bug in the SimpLang compiler, not in your code.\n";
        llvm::errs() << "\n";

        // Print locations
        for (const auto* diag : diags) {
            llvm::errs() << formatICEMessage(*diag);
        }

        // Help text
        llvm::errs() << "   = help: please report this at https://github.com/simplang/simplang/issues\n";

        if (!verbose) {
            llvm::errs() << "\nRun with --verbose for full MLIR diagnostic details.\n";
        } else {
            // In verbose mode, show all raw MLIR messages
            llvm::errs() << "\nRaw MLIR diagnostics:\n";
            for (const auto* diag : diags) {
                llvm::errs() << "  " << diag->message << "\n";
            }
        }
    }

    llvm::errs() << "\n";
}

} // namespace simp
} // namespace mlir
