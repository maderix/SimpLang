#pragma once

/// @file diagnostics.hpp
/// @brief Convenience header that includes all diagnostic system components
///
/// Usage:
/// @code
/// #include "diagnostics/diagnostics.hpp"
///
/// simp::diag::SourceManager sourceManager;
/// sourceManager.loadFile("example.sl");
///
/// simp::diag::DiagnosticEngine engine(sourceManager);
///
/// engine.error(simp::diag::ErrorCode::E0300)
///     .withMessage("undefined variable")
///     .at(span, "`foo` not found in this scope")
///     .noteAt(similarSpan, "did you mean `food`?")
///     .emit();
/// @endcode

#include "diagnostics/span.hpp"
#include "diagnostics/error_codes.hpp"
#include "diagnostics/source_manager.hpp"
#include "diagnostics/diagnostic.hpp"
#include "diagnostics/diagnostic_builder.hpp"
#include "diagnostics/diagnostic_engine.hpp"
#include "diagnostics/diagnostic_renderer.hpp"
