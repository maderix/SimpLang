# Rust-Style Error Reporting for SimpLang Compiler

## Overview
Overhaul error reporting to provide Rust-like helpful error messages with source snippets, precise span highlighting, colored output, and actionable suggestions.

## Target Output Format
```
error[E0200]: type mismatch in assignment
 --> examples/test.sl:5:12
  |
5 |     var x: i32 = "hello";
  |            ^^^   ^^^^^^^ expected `i32`, found `string`
  |            |
  |            expected due to this type annotation
  |
help: consider using an integer literal
  |
5 |     var x: i32 = 42;
  |                  ~~
```

## Current State

| Component | Current Error Handling | Location Tracking |
|-----------|----------------------|-------------------|
| Lexer (src/lexer.l) | `printf("Unknown token")` | Has yycolumn but unused |
| Parser (src/parser.y) | `yyerror()` - line only | Full yylloc available |
| LLVM Codegen | `std::cerr`, `LOG_ERROR` | No source locations |
| MLIR Codegen | `llvm::errs()`, `op.emitError()` | FileLineColLoc per op |
| Logger | 5-level, color-coded | N/A |

## Architecture

```
SourceManager <-- DiagnosticEngine --> DiagnosticRenderer
     ^                   |                    |
     |                   v                    v
     +----------- DiagnosticBuilder -----> Terminal Output
                         ^
                         |
              Span / ErrorCode Catalog
```

## New Files to Create

```
include/diagnostics/
├── span.hpp              # Span struct (fileId, startOffset, endOffset)
├── error_codes.hpp       # ErrorCode enum + catalog
├── diagnostic.hpp        # Diagnostic, Label, Suggestion classes
├── diagnostic_builder.hpp # Fluent API builder
├── diagnostic_engine.hpp  # Main engine (emit, stats, config)
├── diagnostic_renderer.hpp # Pretty-printer with colors
├── source_manager.hpp     # File content + line mapping
└── diagnostics.hpp        # Convenience header

src/diagnostics/
├── span.cpp
├── error_codes.cpp
├── diagnostic.cpp
├── diagnostic_builder.cpp
├── diagnostic_engine.cpp
├── diagnostic_renderer.cpp
├── source_manager.cpp
└── CMakeLists.txt
```

## Files to Modify

| File | Changes |
|------|---------|
| `src/lexer.l` | Replace printf with DiagnosticEngine, add Span creation |
| `src/parser.y` | Replace yyerror(), add error recovery sync points |
| `include/ast/base/ast_base.hpp` | Extend SourceLocation with endLine/endColumn, add toSpan() |
| `src/mlir/mlir_codegen.cpp` | Add SimpDiagnosticHandler, replace llvm::errs() calls |
| `src/mlir/mlir_pipeline.cpp` | Integrate diagnostics for pass failures |
| `src/main.cpp` | Initialize SourceManager + DiagnosticEngine, add --explain, --error-format |
| `CMakeLists.txt` | Add diagnostics subdirectory |

*Note: src/codegen.cpp (LLVM AVX/SSE backend) is deprecated - no changes needed*

## Core Data Structures

### Span (byte-offset based for efficiency)
```cpp
struct Span {
    uint32_t fileId;
    uint32_t startOffset;
    uint32_t endOffset;
    // Line/column computed on-demand from SourceManager
};
```

### Error Codes
```cpp
enum class ErrorCode : uint32_t {
    // Lexer (E0001-E0099)
    E0001,  // Unknown token
    E0002,  // Unterminated string
    E0003,  // Invalid numeric literal

    // Parser (E0100-E0199)
    E0100,  // Unexpected token
    E0101,  // Expected expression
    E0102,  // Expected statement
    E0103,  // Missing semicolon
    E0104,  // Missing closing delimiter

    // Type (E0200-E0299)
    E0200,  // Type mismatch in assignment
    E0201,  // Type mismatch in operation
    E0202,  // Type mismatch in function arg
    E0206,  // Unknown type

    // Semantic (E0300-E0399)
    E0300,  // Undefined variable
    E0301,  // Undefined function
    E0302,  // Variable redefinition
    E0304,  // Wrong number of arguments

    // Tensor (E0400-E0499) - replaces deprecated SIMD
    E0400,  // Invalid tensor shape
    E0401,  // Tensor dimension mismatch
    E0402,  // Axis out of bounds
    E0403,  // Invalid tensor operation

    // Annotation (E0500-E0599)
    E0500,  // Unknown annotation
    E0501,  // Invalid annotation parameter
    E0502,  // Conflicting annotations

    // Codegen (E0600-E0699)
    E0600,  // Failed to generate code
    E0601,  // Unsupported feature
};
```

### DiagnosticBuilder (Fluent API)
```cpp
diagnostics->error(ErrorCode::E0300)
    .withMessage("undefined variable")
    .at(span, "`foo` not found in this scope")
    .noteAt(similarSpan, "did you mean `food`?")
    .emit();
```

## Implementation Phases

### Phase 1: Core Infrastructure
1. Create `include/diagnostics/` and `src/diagnostics/` directories
2. Implement Span, SourceManager (file loading + line mapping)
3. Implement ErrorCode catalog (~50 codes covering all categories)
4. Implement Diagnostic, Label, Suggestion classes
5. Basic DiagnosticEngine and DiagnosticBuilder

### Phase 2: Renderer + Output Formats
1. Implement DiagnosticRenderer with Rust-style terminal output
2. Add ANSI color support (reuse existing Logger colors)
3. Multi-line snippet rendering with line numbers
4. Underline/caret rendering for spans
5. Suggestion rendering with diff-style
6. **Add JSON output format for IDE/LSP integration**
7. Add `--error-format={terminal,json}` CLI flag

### Phase 3: Lexer/Parser Integration
1. Add global DiagnosticEngine/SourceManager pointers for flex/bison
2. Replace lexer printf with proper diagnostics
3. Replace parser yyerror with diagnostics
4. Extend SourceLocation in AST with full span info
5. **Add error recovery with sync tokens (`;`, `}`, `fn`)**
6. Report multiple errors per compilation

### Phase 4: MLIR Backend Integration
*Note: LLVM AVX/SSE backend (AVXSlice/SSESlice) is deprecated - focus on MLIR only*

1. Create SimpDiagnosticHandler for MLIR → our diagnostics
2. Hook into mlir::DiagnosticEngine
3. Convert FileLineColLoc to our Span format
4. Replace llvm::errs() calls in mlir_codegen.cpp
5. Add tensor operation error diagnostics
6. Add semantic error diagnostics (undefined var, type mismatch)

### Phase 5: Polish + Testing
1. Add `--explain E0001` flag for detailed explanations
2. Performance: lazy line offset computation
3. Comprehensive tests for error message formatting
4. Test JSON output parsing
5. Integration tests with intentionally broken .sl files

## Key Design Decisions

1. **Byte offsets internally**: More efficient for span operations, line/column computed lazily
2. **Global pointers for lex/parse**: Required by flex/bison architecture
3. **Reuse Logger colors**: Consistent color scheme across compiler
4. **Error recovery**: Sync at `;`, `}`, `fn` to report multiple errors
5. **MLIR backend focus**: LLVM AVX/SSE backend deprecated, diagnostics for MLIR only
6. **JSON output**: For IDE/LSP integration from the start
7. **~50 error codes**: Comprehensive coverage (lexer, parser, type, semantic, tensor, annotation, codegen)

## Example Error Messages

```
error[E0001]: unknown token
 --> test.sl:3:5
  |
3 |     @#$ invalid
  |     ^ unexpected character '@'

error[E0300]: undefined variable `foo`
  --> test.sl:10:12
   |
10 |     return foo + bar;
   |            ^^^ not found in this scope
   |
note: a variable with a similar name exists
   |
8  |     var food = 10.0;
   |         ^^^^ did you mean `food`?

error[E0401]: tensor dimension mismatch in matmul
 --> test.sl:5:20
  |
5 |     var C = tensor_matmul(A, B);
  |                           ^  ^ B has shape <64,128>
  |                           |
  |                           A has shape <32,64>
  |
  = note: matmul requires A.cols == B.rows (64 == 64 here, but A.rows=32)
```

## Success Criteria
- [ ] All errors show file:line:column
- [ ] Source snippets with line numbers
- [ ] Underlines/carets pointing to error location
- [ ] Color-coded output (error=red, warning=yellow, help=cyan)
- [ ] Suggestions for common mistakes
- [ ] ~50 error codes with --explain support
- [ ] Integrated with MLIR backend (lexer/parser/codegen)
- [ ] JSON output with --error-format=json
- [ ] Parser error recovery (multiple errors per run)
- [ ] Comprehensive tests with intentionally broken .sl files
