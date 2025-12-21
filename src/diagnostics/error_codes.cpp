#include "diagnostics/error_codes.hpp"
#include <cstdio>
#include <unordered_map>

namespace simp {
namespace diag {

namespace {

// Static error info table
const std::unordered_map<uint32_t, ErrorInfo> errorInfoTable = {
    // ========================================================================
    // Lexer Errors
    // ========================================================================
    {1, {ErrorCode::E0001, "E0001", "unknown token",
         "The lexer encountered a character or sequence of characters that "
         "it doesn't recognize as a valid token in SimpLang."}},

    {2, {ErrorCode::E0002, "E0002", "unterminated string literal",
         "A string literal was started with a quote but never closed. "
         "Make sure all string literals have matching opening and closing quotes."}},

    {3, {ErrorCode::E0003, "E0003", "invalid numeric literal",
         "The number literal is malformed. Check for invalid characters, "
         "multiple decimal points, or invalid exponent notation."}},

    {4, {ErrorCode::E0004, "E0004", "invalid character in identifier",
         "Identifiers can only contain letters, digits, and underscores, "
         "and must start with a letter or underscore."}},

    {5, {ErrorCode::E0005, "E0005", "unterminated block comment",
         "A block comment starting with /* was never closed with */. "
         "Block comments can be nested, so make sure all are properly closed."}},

    {6, {ErrorCode::E0006, "E0006", "invalid escape sequence",
         "The escape sequence in this string is not recognized. "
         "Valid escape sequences include: \\n, \\t, \\r, \\\\, \\\", \\'."}},

    {7, {ErrorCode::E0007, "E0007", "numeric literal overflow",
         "The numeric literal is too large to be represented. "
         "Consider using a smaller value or a different numeric type."}},

    {8, {ErrorCode::E0008, "E0008", "invalid Unicode character",
         "The source file contains an invalid Unicode character that "
         "cannot be processed."}},

    // ========================================================================
    // Parser Errors
    // ========================================================================
    {100, {ErrorCode::E0100, "E0100", "unexpected token",
           "The parser encountered a token it wasn't expecting at this position. "
           "Check the surrounding syntax for missing or extra tokens."}},

    {101, {ErrorCode::E0101, "E0101", "expected expression",
           "An expression was expected here but something else was found. "
           "Expressions include literals, variables, function calls, and operators."}},

    {102, {ErrorCode::E0102, "E0102", "expected statement",
           "A statement was expected here. Statements include variable declarations, "
           "assignments, function calls, if/while/for constructs, and return statements."}},

    {103, {ErrorCode::E0103, "E0103", "missing semicolon",
           "Statements in SimpLang must end with a semicolon. "
           "Add a semicolon at the end of the statement."}},

    {104, {ErrorCode::E0104, "E0104", "missing closing delimiter",
           "A closing parenthesis, bracket, or brace is missing. "
           "Make sure all opening delimiters have matching closing ones."}},

    {105, {ErrorCode::E0105, "E0105", "expected identifier",
           "An identifier (name) was expected here. Identifiers are used for "
           "variable names, function names, and type names."}},

    {106, {ErrorCode::E0106, "E0106", "expected type annotation",
           "A type annotation was expected after the colon. "
           "Types include i32, i64, f32, f64, and tensor types."}},

    {107, {ErrorCode::E0107, "E0107", "invalid function declaration",
           "The function declaration is malformed. Functions should be declared as: "
           "fn name(params) { body } or fn name(params) -> type { body }"}},

    {108, {ErrorCode::E0108, "E0108", "invalid parameter list",
           "The function parameter list is malformed. Parameters should be "
           "comma-separated and may include type annotations."}},

    {109, {ErrorCode::E0109, "E0109", "expected 'fn' keyword",
           "Top-level declarations in SimpLang must start with 'fn' for functions."}},

    {110, {ErrorCode::E0110, "E0110", "expected block after function signature",
           "A function body enclosed in braces {{ }} was expected after the "
           "function signature."}},

    {111, {ErrorCode::E0111, "E0111", "unexpected end of file",
           "The file ended unexpectedly. There may be unclosed braces, "
           "incomplete statements, or missing function bodies."}},

    {112, {ErrorCode::E0112, "E0112", "invalid tensor shape syntax",
           "Tensor shapes should be specified as type<dim1,dim2,...>. "
           "For example: f32<32,64> or i8<128,256,3>."}},

    {113, {ErrorCode::E0113, "E0113", "expected comma or closing bracket",
           "In a list or array, expected either a comma to continue the list "
           "or a closing bracket to end it."}},

    {114, {ErrorCode::E0114, "E0114", "empty block not allowed",
           "Empty blocks are not allowed here. Add at least one statement "
           "or use a placeholder like 'return 0;'."}},

    {115, {ErrorCode::E0115, "E0115", "invalid assignment target",
           "The left side of an assignment must be a valid target like a variable "
           "name or an indexing expression."}},

    // ========================================================================
    // Type Errors
    // ========================================================================
    {200, {ErrorCode::E0200, "E0200", "type mismatch in assignment",
           "The type of the value being assigned doesn't match the type of the "
           "variable. Either change the value or use an explicit type conversion."}},

    {201, {ErrorCode::E0201, "E0201", "type mismatch in binary operation",
           "The operands of this binary operation have incompatible types. "
           "Both operands should typically have the same type."}},

    {202, {ErrorCode::E0202, "E0202", "type mismatch in function argument",
           "The type of the argument doesn't match the expected parameter type. "
           "Check the function signature and provide a value of the correct type."}},

    {203, {ErrorCode::E0203, "E0203", "type mismatch in return statement",
           "The type of the returned value doesn't match the function's return type. "
           "Either change the return value or update the function's return type."}},

    {204, {ErrorCode::E0204, "E0204", "cannot infer type",
           "The type of this expression cannot be determined. Add an explicit type "
           "annotation to help the compiler understand your intent."}},

    {205, {ErrorCode::E0205, "E0205", "incompatible types in conditional",
           "The 'then' and 'else' branches of a conditional must have compatible types, "
           "or the result must not be used."}},

    {206, {ErrorCode::E0206, "E0206", "unknown type name",
           "The type name used here is not recognized. Check spelling and ensure "
           "the type is defined or imported."}},

    {207, {ErrorCode::E0207, "E0207", "invalid type for operation",
           "This operation cannot be applied to values of this type. Check the "
           "documentation for which types support this operation."}},

    {208, {ErrorCode::E0208, "E0208", "cannot convert between types",
           "There is no valid conversion between these types. You may need to use "
           "an explicit cast or a different approach."}},

    {209, {ErrorCode::E0209, "E0209", "array element type mismatch",
           "All elements in an array literal must have the same type, or types "
           "that can be implicitly converted to a common type."}},

    {210, {ErrorCode::E0210, "E0210", "invalid index type",
           "Array and tensor indices must be integers (i32 or i64). "
           "Use an integer type or convert your index value."}},

    // ========================================================================
    // Semantic Errors
    // ========================================================================
    {300, {ErrorCode::E0300, "E0300", "undefined variable",
           "This variable has not been declared in the current scope or any "
           "enclosing scope. Check spelling or add a declaration."}},

    {301, {ErrorCode::E0301, "E0301", "undefined function",
           "No function with this name exists. Check spelling and ensure "
           "the function is defined before it's called."}},

    {302, {ErrorCode::E0302, "E0302", "variable redefinition in same scope",
           "A variable with this name already exists in the current scope. "
           "Use a different name or remove the duplicate declaration."}},

    {303, {ErrorCode::E0303, "E0303", "function redefinition",
           "A function with this name is already defined. "
           "Use a different name or remove the duplicate definition."}},

    {304, {ErrorCode::E0304, "E0304", "wrong number of function arguments",
           "The number of arguments provided doesn't match the number of parameters "
           "in the function definition."}},

    {305, {ErrorCode::E0305, "E0305", "cannot assign to immutable variable",
           "This variable was declared as immutable and cannot be reassigned. "
           "Use 'var' instead of 'let' if you need to modify the variable."}},

    {306, {ErrorCode::E0306, "E0306", "variable used before initialization",
           "This variable is being used before it has been assigned a value. "
           "Initialize the variable before using it."}},

    {307, {ErrorCode::E0307, "E0307", "break outside of loop",
           "The 'break' statement can only be used inside a loop (while or for)."}},

    {308, {ErrorCode::E0308, "E0308", "continue outside of loop",
           "The 'continue' statement can only be used inside a loop (while or for)."}},

    {309, {ErrorCode::E0309, "E0309", "return outside of function",
           "The 'return' statement can only be used inside a function body."}},

    {310, {ErrorCode::E0310, "E0310", "missing return in function",
           "This function is expected to return a value, but not all code paths "
           "contain a return statement."}},

    {311, {ErrorCode::E0311, "E0311", "unreachable code",
           "This code will never be executed because it comes after a return, "
           "break, or continue statement."}},

    {312, {ErrorCode::E0312, "E0312", "division by zero",
           "This expression divides by zero, which is undefined behavior. "
           "Check the divisor or add a runtime check."}},

    // ========================================================================
    // Tensor Errors
    // ========================================================================
    {400, {ErrorCode::E0400, "E0400", "invalid tensor shape",
           "The tensor shape is invalid. Shapes must have positive integer dimensions."}},

    {401, {ErrorCode::E0401, "E0401", "tensor dimension mismatch",
           "The tensors have incompatible dimensions for this operation. "
           "Check that the tensor shapes are compatible."}},

    {402, {ErrorCode::E0402, "E0402", "tensor axis out of bounds",
           "The axis specified is out of bounds for this tensor. "
           "Axes are 0-indexed and must be less than the tensor's rank."}},

    {403, {ErrorCode::E0403, "E0403", "invalid tensor operation",
           "This operation is not valid for tensors of these shapes or types."}},

    {404, {ErrorCode::E0404, "E0404", "tensor shape mismatch in matmul",
           "Matrix multiplication requires the inner dimensions to match: "
           "A<M,K> @ B<K,N> -> C<M,N>. Check your tensor shapes."}},

    {405, {ErrorCode::E0405, "E0405", "invalid tensor element type",
           "Tensors of this element type cannot be used in this operation. "
           "Supported types: i8, i32, i64, f32, f64."}},

    {406, {ErrorCode::E0406, "E0406", "tensor index out of bounds",
           "The index is outside the valid range for this tensor dimension."}},

    {407, {ErrorCode::E0407, "E0407", "invalid tensor slice",
           "The slice parameters are invalid. Check start, stop, and step values."}},

    {408, {ErrorCode::E0408, "E0408", "cannot broadcast tensors",
           "The tensors cannot be broadcast to a common shape. "
           "Review broadcasting rules and tensor dimensions."}},

    {409, {ErrorCode::E0409, "E0409", "tensor memory alignment error",
           "The tensor memory is not properly aligned for this operation. "
           "This may be an internal compiler error."}},

    {410, {ErrorCode::E0410, "E0410", "invalid tensor reduction axis",
           "The reduction axis is invalid for this tensor shape."}},

    // ========================================================================
    // Annotation Errors
    // ========================================================================
    {500, {ErrorCode::E0500, "E0500", "unknown annotation",
           "This annotation is not recognized. Valid annotations include: "
           "@parallel, @tile, @unroll, @vectorize, @lower."}},

    {501, {ErrorCode::E0501, "E0501", "invalid annotation parameter",
           "The parameter value for this annotation is invalid. "
           "Check the annotation documentation for valid values."}},

    {502, {ErrorCode::E0502, "E0502", "conflicting annotations",
           "These annotations cannot be used together as they conflict. "
           "Remove one of the conflicting annotations."}},

    {503, {ErrorCode::E0503, "E0503", "annotation not allowed here",
           "This annotation cannot be applied to this construct. "
           "Check where this annotation is allowed."}},

    {504, {ErrorCode::E0504, "E0504", "missing required annotation parameter",
           "This annotation requires a parameter that was not provided."}},

    {505, {ErrorCode::E0505, "E0505", "invalid tile size",
           "The tile size is invalid. Tile sizes must be positive integers "
           "and typically powers of 2."}},

    {506, {ErrorCode::E0506, "E0506", "invalid parallel specification",
           "The @parallel annotation has invalid parameters."}},

    {507, {ErrorCode::E0507, "E0507", "invalid unroll factor",
           "The unroll factor must be a positive integer. "
           "Common values are 2, 4, 8, or 16."}},

    {508, {ErrorCode::E0508, "E0508", "annotation requires loop context",
           "This annotation can only be applied to loops (while or for)."}},

    // ========================================================================
    // Codegen Errors
    // ========================================================================
    {600, {ErrorCode::E0600, "E0600", "failed to generate code",
           "An error occurred during code generation. This may be an internal "
           "compiler error or an unsupported language construct."}},

    {601, {ErrorCode::E0601, "E0601", "unsupported feature in target",
           "This feature is not supported by the current compilation target."}},

    {602, {ErrorCode::E0602, "E0602", "MLIR lowering failed",
           "Failed to lower the IR to a lower-level representation. "
           "This may indicate an unsupported operation or internal error."}},

    {603, {ErrorCode::E0603, "E0603", "LLVM compilation failed",
           "The LLVM backend failed to compile the generated IR. "
           "This is usually an internal compiler error."}},

    {604, {ErrorCode::E0604, "E0604", "linking failed",
           "Failed to link the compiled object files. "
           "Check for missing symbols or incompatible object files."}},

    {605, {ErrorCode::E0605, "E0605", "invalid memory access pattern",
           "The memory access pattern is invalid and cannot be lowered to "
           "the target architecture."}},

    {606, {ErrorCode::E0606, "E0606", "vectorization failed",
           "The loop or operation could not be vectorized. "
           "Check for data dependencies or unsupported operations."}},

    {607, {ErrorCode::E0607, "E0607", "pass execution failed",
           "An optimization or transformation pass failed to execute."}},

    {608, {ErrorCode::E0608, "E0608", "invalid IR state",
           "The intermediate representation is in an invalid state. "
           "This is usually an internal compiler error."}},
};

// Fallback error info for unknown codes
const ErrorInfo unknownErrorInfo = {
    static_cast<ErrorCode>(0), "E????", "unknown error",
    "This error code is not recognized. Please report this as a bug."
};

} // anonymous namespace

const ErrorInfo& getErrorInfo(ErrorCode code) {
    auto it = errorInfoTable.find(static_cast<uint32_t>(code));
    if (it != errorInfoTable.end()) {
        return it->second;
    }
    return unknownErrorInfo;
}

std::string_view getErrorCategory(ErrorCode code) {
    if (isLexerError(code)) return "lexer";
    if (isParserError(code)) return "parser";
    if (isTypeError(code)) return "type";
    if (isSemanticError(code)) return "semantic";
    if (isTensorError(code)) return "tensor";
    if (isAnnotationError(code)) return "annotation";
    if (isCodegenError(code)) return "codegen";
    return "unknown";
}

std::string formatErrorCode(ErrorCode code) {
    char buffer[8];
    std::snprintf(buffer, sizeof(buffer), "E%04u", static_cast<uint32_t>(code));
    return std::string(buffer);
}

} // namespace diag
} // namespace simp
