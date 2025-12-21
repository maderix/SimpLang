/// @file test_diagnostics.cpp
/// @brief Comprehensive tests for the diagnostics system
///
/// Build and run:
///   cd build_mlir && make test_diagnostics && ./tests/test_diagnostics

#include "diagnostics/diagnostics.hpp"
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

using namespace simp::diag;

// Test helper to capture output
class OutputCapture {
public:
    std::ostringstream ss;
    std::string get() const { return ss.str(); }
    void clear() { ss.str(""); ss.clear(); }
};

// Test source file content (5 lines, no trailing newline after last line)
const char* testSource = "fn kernel_main() {\n"
                          "    var x: i32 = \"hello\";\n"
                          "    var y = foo + bar;\n"
                          "    return x;\n"
                          "}";

void testSourceManager() {
    std::cout << "Testing SourceManager..." << std::endl;

    SourceManager sm;

    // Test adding source directly
    uint32_t fileId = sm.addSource("test.sl", std::string(testSource));
    assert(fileId == 1);
    assert(sm.fileCount() == 1);
    assert(sm.mainFileId() == fileId);

    // Test getting file
    const SourceFile* file = sm.getFile(fileId);
    assert(file != nullptr);
    assert(file->name() == "test.sl");
    assert(file->lineCount() == 5);

    // Test getting lines
    std::string_view line1 = sm.getLine(fileId, 1);
    assert(line1 == "fn kernel_main() {");

    std::string_view line2 = sm.getLine(fileId, 2);
    assert(line2 == "    var x: i32 = \"hello\";");

    // Test position to offset conversion
    uint32_t offset = sm.getOffset(fileId, 2, 5); // line 2, column 5
    assert(offset > 0);

    // Test offset to position conversion
    Position pos = sm.getPosition(fileId, offset);
    assert(pos.line == 2);
    assert(pos.column == 5);

    // Test span text
    Span span(fileId, sm.getOffset(fileId, 2, 5), sm.getOffset(fileId, 2, 8));
    std::string_view spanText = sm.getSpanText(span);
    assert(spanText == "var");

    std::cout << "  SourceManager tests passed!" << std::endl;
}

void testSpan() {
    std::cout << "Testing Span..." << std::endl;

    // Basic span
    Span s1(1, 10, 20);
    assert(s1.isValid());
    assert(s1.length() == 10);
    assert(s1.contains(15));
    assert(!s1.contains(5));
    assert(!s1.contains(25));

    // Overlapping spans
    Span s2(1, 15, 25);
    assert(s1.overlaps(s2));

    // Non-overlapping spans
    Span s3(1, 30, 40);
    assert(!s1.overlaps(s3));

    // Different file IDs
    Span s4(2, 10, 20);
    assert(!s1.overlaps(s4));

    // Merge spans
    Span merged = s1.merge(s2);
    assert(merged.startOffset == 10);
    assert(merged.endOffset == 25);

    // Shrink operations
    Span shrunkStart = s1.shrinkToStart();
    assert(shrunkStart.startOffset == 10);
    assert(shrunkStart.endOffset == 11);

    Span shrunkEnd = s1.shrinkToEnd();
    assert(shrunkEnd.startOffset == 19);
    assert(shrunkEnd.endOffset == 20);

    std::cout << "  Span tests passed!" << std::endl;
}

void testErrorCodes() {
    std::cout << "Testing ErrorCodes..." << std::endl;

    // Test error info lookup
    const ErrorInfo& info = getErrorInfo(ErrorCode::E0200);
    assert(std::string(info.name) == "E0200");
    assert(std::string(info.title) == "type mismatch in assignment");

    // Test category lookup
    assert(getErrorCategory(ErrorCode::E0001) == "lexer");
    assert(getErrorCategory(ErrorCode::E0100) == "parser");
    assert(getErrorCategory(ErrorCode::E0200) == "type");
    assert(getErrorCategory(ErrorCode::E0300) == "semantic");
    assert(getErrorCategory(ErrorCode::E0400) == "tensor");
    assert(getErrorCategory(ErrorCode::E0500) == "annotation");
    assert(getErrorCategory(ErrorCode::E0600) == "codegen");

    // Test formatting
    assert(formatErrorCode(ErrorCode::E0001) == "E0001");
    assert(formatErrorCode(ErrorCode::E0200) == "E0200");

    // Test category predicates
    assert(isLexerError(ErrorCode::E0001));
    assert(!isLexerError(ErrorCode::E0100));
    assert(isParserError(ErrorCode::E0100));
    assert(isTypeError(ErrorCode::E0200));
    assert(isSemanticError(ErrorCode::E0300));
    assert(isTensorError(ErrorCode::E0400));
    assert(isAnnotationError(ErrorCode::E0500));
    assert(isCodegenError(ErrorCode::E0600));

    std::cout << "  ErrorCodes tests passed!" << std::endl;
}

void testDiagnostic() {
    std::cout << "Testing Diagnostic..." << std::endl;

    Diagnostic diag(ErrorCode::E0200, Severity::Error, "type mismatch");

    assert(diag.code() == ErrorCode::E0200);
    assert(diag.severity() == Severity::Error);
    assert(diag.message() == "type mismatch");

    // Test labels
    Span span1(1, 10, 20);
    Span span2(1, 25, 35);

    diag.withPrimaryLabel(span1, "expected `i32`");
    diag.withSecondaryLabel(span2, "found `string`");

    assert(diag.labels().size() == 2);
    assert(diag.labels()[0].isPrimary);
    assert(!diag.labels()[1].isPrimary);

    // Test primary span
    auto primarySpan = diag.primarySpan();
    assert(primarySpan.has_value());
    assert(primarySpan->startOffset == 10);

    // Test notes
    diag.withNote("consider using a type conversion");
    assert(diag.notes().size() == 1);

    // Test suggestions
    diag.withSuggestion(span2, "42", "use an integer literal");
    assert(diag.suggestions().size() == 1);

    std::cout << "  Diagnostic tests passed!" << std::endl;
}

void testDiagnosticStats() {
    std::cout << "Testing DiagnosticStats..." << std::endl;

    DiagnosticStats stats;
    assert(!stats.hasErrors());
    assert(stats.errors == 0);
    assert(stats.warnings == 0);
    assert(stats.notes == 0);

    stats.record(Severity::Error);
    assert(stats.hasErrors());
    assert(stats.errors == 1);

    stats.record(Severity::Warning);
    assert(stats.warnings == 1);

    stats.record(Severity::Note);
    assert(stats.notes == 1);

    stats.reset();
    assert(!stats.hasErrors());
    assert(stats.errors == 0);

    std::cout << "  DiagnosticStats tests passed!" << std::endl;
}

void testDiagnosticEngine() {
    std::cout << "Testing DiagnosticEngine..." << std::endl;

    SourceManager sm;
    uint32_t fileId = sm.addSource("test.sl", std::string(testSource));

    DiagnosticEngine engine(sm);
    OutputCapture capture;
    engine.setOutputStream(capture.ss);

    // Disable colors for easier testing
    DiagnosticConfig config;
    config.useColors = false;
    engine.setConfig(config);

    // Test error emission
    Span span(fileId, sm.getOffset(fileId, 2, 12), sm.getOffset(fileId, 2, 15));
    engine.error(ErrorCode::E0200)
        .withMessage("type mismatch in assignment")
        .at(span, "expected `i32`, found `string`")
        .emit();

    assert(engine.hasErrors());
    assert(engine.stats().errors == 1);

    // Check output contains expected elements
    std::string output = capture.get();
    assert(output.find("error[E0200]") != std::string::npos);
    assert(output.find("type mismatch in assignment") != std::string::npos);
    assert(output.find("test.sl:2:12") != std::string::npos);

    // Test warning
    capture.clear();
    engine.warning(ErrorCode::E0311)
        .withMessage("unreachable code")
        .at(span)
        .emit();

    assert(engine.stats().warnings == 1);
    output = capture.get();
    assert(output.find("warning[E0311]") != std::string::npos);

    std::cout << "  DiagnosticEngine tests passed!" << std::endl;
}

void testDiagnosticRenderer() {
    std::cout << "Testing DiagnosticRenderer..." << std::endl;

    SourceManager sm;
    uint32_t fileId = sm.addSource("test.sl", std::string(testSource));

    OutputCapture capture;
    DiagnosticRenderer renderer(sm);
    renderer.setOutputStream(capture.ss);
    renderer.setUseColors(false);

    // Create a diagnostic with multiple labels
    Diagnostic diag(ErrorCode::E0200, Severity::Error, "type mismatch in assignment");

    Span typeSpan(fileId, sm.getOffset(fileId, 2, 12), sm.getOffset(fileId, 2, 15));
    Span valueSpan(fileId, sm.getOffset(fileId, 2, 18), sm.getOffset(fileId, 2, 25));

    diag.withPrimaryLabel(typeSpan, "expected due to this type annotation");
    diag.withSecondaryLabel(valueSpan, "found `string`");

    renderer.render(diag);

    std::string output = capture.get();

    // Verify output structure
    assert(output.find("error[E0200]") != std::string::npos);
    assert(output.find("--> test.sl:2:12") != std::string::npos);
    assert(output.find("var x: i32") != std::string::npos);

    std::cout << "  DiagnosticRenderer tests passed!" << std::endl;
}

void testJSONOutput() {
    std::cout << "Testing JSON output..." << std::endl;

    SourceManager sm;
    uint32_t fileId = sm.addSource("test.sl", std::string(testSource));

    OutputCapture capture;
    DiagnosticRenderer renderer(sm);
    renderer.setOutputStream(capture.ss);

    Diagnostic diag(ErrorCode::E0300, Severity::Error, "undefined variable");
    Span span(fileId, sm.getOffset(fileId, 3, 13), sm.getOffset(fileId, 3, 16));
    diag.withPrimaryLabel(span, "`foo` not found in this scope");

    renderer.renderJSON(diag);

    std::string output = capture.get();

    // Verify JSON structure
    assert(output.find("\"code\":\"E0300\"") != std::string::npos);
    assert(output.find("\"severity\":\"error\"") != std::string::npos);
    assert(output.find("\"message\":\"undefined variable\"") != std::string::npos);
    assert(output.find("\"file\":\"test.sl\"") != std::string::npos);
    assert(output.find("\"labels\":[") != std::string::npos);

    std::cout << "  JSON output tests passed!" << std::endl;
}

void testExplainMode() {
    std::cout << "Testing explain mode..." << std::endl;

    SourceManager sm;
    DiagnosticEngine engine(sm);

    OutputCapture capture;
    engine.setOutputStream(capture.ss);

    // Test explain with ErrorCode
    engine.explain(ErrorCode::E0200);
    std::string output = capture.get();
    assert(output.find("E0200") != std::string::npos);
    assert(output.find("type mismatch") != std::string::npos);

    // Test explain with string
    capture.clear();
    bool result = engine.explain("E0300");
    assert(result);
    output = capture.get();
    assert(output.find("E0300") != std::string::npos);

    // Test invalid code string
    capture.clear();
    result = engine.explain("invalid");
    assert(!result);

    std::cout << "  Explain mode tests passed!" << std::endl;
}

void testFluentAPI() {
    std::cout << "Testing fluent API..." << std::endl;

    SourceManager sm;
    uint32_t fileId = sm.addSource("test.sl", std::string(testSource));

    DiagnosticEngine engine(sm);
    OutputCapture capture;
    engine.setOutputStream(capture.ss);

    DiagnosticConfig config;
    config.useColors = false;
    engine.setConfig(config);

    // Test chaining
    Span span1(fileId, sm.getOffset(fileId, 3, 13), sm.getOffset(fileId, 3, 16));
    Span span2(fileId, sm.getOffset(fileId, 1, 1), sm.getOffset(fileId, 1, 3));

    engine.error(ErrorCode::E0300)
        .withMessage("undefined variable `foo`")
        .at(span1, "not found in this scope")
        .noteAt(span2, "similar name exists")
        .note("did you mean `food`?")
        .help("check your variable declarations")
        .suggest(span1, "food", "replace with similar name")
        .emit();

    assert(engine.hasErrors());

    std::string output = capture.get();
    assert(output.find("undefined variable `foo`") != std::string::npos);
    assert(output.find("help:") != std::string::npos);

    std::cout << "  Fluent API tests passed!" << std::endl;
}

void testErrorLimit() {
    std::cout << "Testing error limit..." << std::endl;

    SourceManager sm;
    uint32_t fileId = sm.addSource("test.sl", std::string(testSource));

    DiagnosticEngine engine(sm);
    OutputCapture capture;
    engine.setOutputStream(capture.ss);

    DiagnosticConfig config;
    config.useColors = false;
    config.maxErrors = 3;
    engine.setConfig(config);

    Span span(fileId, 10, 20);

    // Emit more errors than limit
    for (int i = 0; i < 10; i++) {
        engine.emitError(ErrorCode::E0001, span, "test error");
    }

    // Error limit should be reached, stats continue to count all errors
    assert(engine.errorLimitReached());
    assert(engine.stats().errors >= config.maxErrors);

    // Output should contain the "too many errors" message
    std::string output = capture.get();
    assert(output.find("too many errors") != std::string::npos);

    // Verify only maxErrors error messages were rendered (count occurrences)
    size_t count = 0;
    size_t pos = 0;
    while ((pos = output.find("error[E0001]", pos)) != std::string::npos) {
        ++count;
        ++pos;
    }
    assert(count == config.maxErrors);

    std::cout << "  Error limit tests passed!" << std::endl;
}

void demonstrateOutput() {
    std::cout << "\n=== Demonstration of Rust-Style Error Output ===\n" << std::endl;

    SourceManager sm;
    uint32_t fileId = sm.addSource("examples/test.sl", std::string(testSource));

    DiagnosticEngine engine(sm);

    // Demo 1: Type mismatch error
    Span typeSpan(fileId, sm.getOffset(fileId, 2, 12), sm.getOffset(fileId, 2, 15));
    Span valueSpan(fileId, sm.getOffset(fileId, 2, 18), sm.getOffset(fileId, 2, 25));

    engine.error(ErrorCode::E0200)
        .withMessage("type mismatch in assignment")
        .at(typeSpan, "expected due to this type annotation")
        .noteAt(valueSpan, "found `string`")
        .note("expected `i32` but found `string`")
        .suggest(valueSpan, "42", "consider using an integer literal")
        .emit();

    // Demo 2: Undefined variable error
    Span fooSpan(fileId, sm.getOffset(fileId, 3, 13), sm.getOffset(fileId, 3, 16));
    Span barSpan(fileId, sm.getOffset(fileId, 3, 19), sm.getOffset(fileId, 3, 22));

    engine.error(ErrorCode::E0300)
        .withMessage("undefined variable `foo`")
        .at(fooSpan, "not found in this scope")
        .emit();

    engine.error(ErrorCode::E0300)
        .withMessage("undefined variable `bar`")
        .at(barSpan, "not found in this scope")
        .emit();

    std::cout << "\n=== End of Demonstration ===\n" << std::endl;
}

int main() {
    std::cout << "\n=== SimpLang Diagnostics Test Suite ===\n" << std::endl;

    testSourceManager();
    testSpan();
    testErrorCodes();
    testDiagnostic();
    testDiagnosticStats();
    testDiagnosticEngine();
    testDiagnosticRenderer();
    testJSONOutput();
    testExplainMode();
    testFluentAPI();
    testErrorLimit();

    std::cout << "\n=== All tests passed! ===\n" << std::endl;

    // Demonstrate the output
    demonstrateOutput();

    return 0;
}
