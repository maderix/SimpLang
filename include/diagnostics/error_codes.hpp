#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace simp {
namespace diag {

/// Error codes for the SimpLang compiler
/// Organized by category:
///   E0001-E0099: Lexer errors
///   E0100-E0199: Parser errors
///   E0200-E0299: Type errors
///   E0300-E0399: Semantic errors
///   E0400-E0499: Tensor errors
///   E0500-E0599: Annotation errors
///   E0600-E0699: Codegen errors
enum class ErrorCode : uint32_t {
    // ========================================================================
    // Generic/Internal (E0000)
    // ========================================================================
    E0000 = 0,    // Generic warning/note (no specific code)

    // ========================================================================
    // Lexer Errors (E0001-E0099)
    // ========================================================================
    E0001 = 1,    // Unknown token
    E0002 = 2,    // Unterminated string literal
    E0003 = 3,    // Invalid numeric literal
    E0004 = 4,    // Invalid character in identifier
    E0005 = 5,    // Unterminated block comment
    E0006 = 6,    // Invalid escape sequence
    E0007 = 7,    // Numeric literal overflow
    E0008 = 8,    // Invalid Unicode character

    // ========================================================================
    // Parser Errors (E0100-E0199)
    // ========================================================================
    E0100 = 100,  // Unexpected token
    E0101 = 101,  // Expected expression
    E0102 = 102,  // Expected statement
    E0103 = 103,  // Missing semicolon
    E0104 = 104,  // Missing closing delimiter (paren, bracket, brace)
    E0105 = 105,  // Expected identifier
    E0106 = 106,  // Expected type annotation
    E0107 = 107,  // Invalid function declaration
    E0108 = 108,  // Invalid parameter list
    E0109 = 109,  // Expected 'fn' keyword
    E0110 = 110,  // Expected block after function signature
    E0111 = 111,  // Unexpected end of file
    E0112 = 112,  // Invalid tensor shape syntax
    E0113 = 113,  // Expected comma or closing bracket
    E0114 = 114,  // Empty block not allowed
    E0115 = 115,  // Invalid assignment target

    // ========================================================================
    // Type Errors (E0200-E0299)
    // ========================================================================
    E0200 = 200,  // Type mismatch in assignment
    E0201 = 201,  // Type mismatch in binary operation
    E0202 = 202,  // Type mismatch in function argument
    E0203 = 203,  // Type mismatch in return statement
    E0204 = 204,  // Cannot infer type
    E0205 = 205,  // Incompatible types in conditional
    E0206 = 206,  // Unknown type name
    E0207 = 207,  // Invalid type for operation
    E0208 = 208,  // Cannot convert between types
    E0209 = 209,  // Array element type mismatch
    E0210 = 210,  // Invalid index type (expected integer)

    // ========================================================================
    // Semantic Errors (E0300-E0399)
    // ========================================================================
    E0300 = 300,  // Undefined variable
    E0301 = 301,  // Undefined function
    E0302 = 302,  // Variable redefinition in same scope
    E0303 = 303,  // Function redefinition
    E0304 = 304,  // Wrong number of function arguments
    E0305 = 305,  // Cannot assign to immutable variable
    E0306 = 306,  // Variable used before initialization
    E0307 = 307,  // Break outside of loop
    E0308 = 308,  // Continue outside of loop
    E0309 = 309,  // Return outside of function
    E0310 = 310,  // Missing return in function
    E0311 = 311,  // Unreachable code
    E0312 = 312,  // Division by zero (compile-time constant)

    // ========================================================================
    // Tensor Errors (E0400-E0499)
    // ========================================================================
    E0400 = 400,  // Invalid tensor shape
    E0401 = 401,  // Tensor dimension mismatch
    E0402 = 402,  // Tensor axis out of bounds
    E0403 = 403,  // Invalid tensor operation
    E0404 = 404,  // Tensor shape mismatch in matmul
    E0405 = 405,  // Invalid tensor element type
    E0406 = 406,  // Tensor index out of bounds
    E0407 = 407,  // Invalid tensor slice
    E0408 = 408,  // Cannot broadcast tensors
    E0409 = 409,  // Tensor memory alignment error
    E0410 = 410,  // Invalid tensor reduction axis

    // ========================================================================
    // Annotation Errors (E0500-E0599)
    // ========================================================================
    E0500 = 500,  // Unknown annotation
    E0501 = 501,  // Invalid annotation parameter
    E0502 = 502,  // Conflicting annotations
    E0503 = 503,  // Annotation not allowed here
    E0504 = 504,  // Missing required annotation parameter
    E0505 = 505,  // Invalid tile size
    E0506 = 506,  // Invalid parallel specification
    E0507 = 507,  // Invalid unroll factor
    E0508 = 508,  // Annotation requires loop context

    // ========================================================================
    // Codegen Errors (E0600-E0699)
    // ========================================================================
    E0600 = 600,  // Failed to generate code
    E0601 = 601,  // Unsupported feature in target
    E0602 = 602,  // MLIR lowering failed
    E0603 = 603,  // LLVM compilation failed
    E0604 = 604,  // Linking failed
    E0605 = 605,  // Invalid memory access pattern
    E0606 = 606,  // Vectorization failed
    E0607 = 607,  // Pass execution failed
    E0608 = 608,  // Invalid IR state

    // ========================================================================
    // Internal Compiler Errors (E0700-E0799)
    // ========================================================================
    E0700 = 700,  // Internal compiler error (ICE)
    E0701 = 701,  // ICE: MLIR verification failed
    E0702 = 702,  // ICE: MLIR pass failed
    E0703 = 703,  // ICE: Bufferization failed
    E0704 = 704,  // ICE: Memory allocation lowering failed
    E0705 = 705,  // ICE: Tensor reshape failed
};

/// Severity level for diagnostics
enum class Severity {
    Note,     // Informational note, often accompanies other diagnostics
    Warning,  // Warning that doesn't prevent compilation
    Error,    // Error that prevents successful compilation
    Fatal     // Fatal error that stops compilation immediately
};

/// Information about an error code
struct ErrorInfo {
    ErrorCode code;
    const char* name;        // Short identifier like "E0200"
    const char* title;       // Short title like "type mismatch in assignment"
    const char* explanation; // Detailed explanation for --explain
};

/// Get information about an error code
const ErrorInfo& getErrorInfo(ErrorCode code);

/// Get the category name for an error code
std::string_view getErrorCategory(ErrorCode code);

/// Format an error code as a string (e.g., "E0200")
std::string formatErrorCode(ErrorCode code);

/// Check if an error code is in a specific range
inline bool isLexerError(ErrorCode code) {
    uint32_t v = static_cast<uint32_t>(code);
    return v >= 1 && v < 100;
}

inline bool isParserError(ErrorCode code) {
    uint32_t v = static_cast<uint32_t>(code);
    return v >= 100 && v < 200;
}

inline bool isTypeError(ErrorCode code) {
    uint32_t v = static_cast<uint32_t>(code);
    return v >= 200 && v < 300;
}

inline bool isSemanticError(ErrorCode code) {
    uint32_t v = static_cast<uint32_t>(code);
    return v >= 300 && v < 400;
}

inline bool isTensorError(ErrorCode code) {
    uint32_t v = static_cast<uint32_t>(code);
    return v >= 400 && v < 500;
}

inline bool isAnnotationError(ErrorCode code) {
    uint32_t v = static_cast<uint32_t>(code);
    return v >= 500 && v < 600;
}

inline bool isCodegenError(ErrorCode code) {
    uint32_t v = static_cast<uint32_t>(code);
    return v >= 600 && v < 700;
}

inline bool isInternalError(ErrorCode code) {
    uint32_t v = static_cast<uint32_t>(code);
    return v >= 700 && v < 800;
}

} // namespace diag
} // namespace simp
