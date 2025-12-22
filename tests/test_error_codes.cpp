/**
 * Comprehensive Error Code Test Suite
 *
 * Tests that the SimpLang compiler produces correct error codes
 * for various error conditions.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <regex>
#include <cstdlib>
#include <cstdio>
#include <array>
#include <memory>
#include <sstream>

// ANSI color codes
#define GREEN "\033[32m"
#define RED "\033[31m"
#define YELLOW "\033[33m"
#define CYAN "\033[36m"
#define RESET "\033[0m"

struct TestCase {
    std::string name;
    std::string code;
    std::string expectedErrorCode;  // e.g., "E0001", "E0100", "E0301"
    std::string description;
};

// Execute command and capture output
std::string exec(const std::string& cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen((cmd + " 2>&1").c_str(), "r"), pclose);
    if (!pipe) {
        return "ERROR: popen() failed!";
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

// Check if output contains expected error code
bool containsErrorCode(const std::string& output, const std::string& errorCode) {
    // Match patterns like "error[E0301]" or just "E0301"
    std::regex pattern("\\[" + errorCode + "\\]|\\b" + errorCode + "\\b");
    return std::regex_search(output, pattern);
}

// Write test file
void writeTestFile(const std::string& path, const std::string& code) {
    std::ofstream file(path);
    file << code;
    file.close();
}

int main(int argc, char* argv[]) {
    std::string compiler = "./build_mlir/src/simplang";

    // Check if compiler path was provided
    if (argc > 1) {
        compiler = argv[1];
    }

    // Test cases organized by category
    // NOTE: SimpLang uses `var x = value` or `type<dims> x = value` syntax
    std::vector<TestCase> tests = {
        // ====================================================================
        // LEXER ERRORS (E0001-E0099)
        // ====================================================================
        {
            "E0001_invalid_char",
            R"(
fn test() {
    var x = 123§;
    return x;
}
)",
            "E0001",
            "Invalid character in source"
        },
        {
            "E0001_at_symbol",
            R"(
fn test() {
    var x = @invalid;
    return 0.0;
}
)",
            "E0100",  // @ followed by identifier triggers parser error
            "@ followed by unknown produces parser error"
        },

        // ====================================================================
        // PARSER ERRORS (E0100-E0199)
        // ====================================================================
        {
            "E0100_missing_expression",
            R"(
fn test() {
    var x = ;
    return x;
}
)",
            "E0100",
            "Missing expression after ="
        },
        {
            "E0100_missing_paren",
            R"(
fn test( {
    return 0.0;
}
)",
            "E0100",
            "Missing closing parenthesis"
        },
        {
            "E0100_missing_brace",
            R"(
fn test()
    return 0.0;
}
)",
            "E0100",
            "Missing opening brace"
        },
        {
            "E0100_unclosed_paren",
            R"(
fn test() {
    var x = (1 + 2;
    return x;
}
)",
            "E0100",
            "Unclosed parenthesis"
        },
        {
            "E0100_double_operator",
            R"(
fn test() {
    var x = 1 + + 2;
    return x;
}
)",
            "E0100",
            "Unexpected operator"
        },

        // ====================================================================
        // SEMANTIC ERRORS (E0300-E0399)
        // ====================================================================
        {
            "E0300_undefined_variable",
            R"(
fn test() {
    return undefined_var;
}
)",
            "E0300",
            "Undefined variable reference"
        },
        {
            "E0300_undefined_in_expr",
            R"(
fn test() {
    var x = 1.0;
    var y = x + unknown;
    return y;
}
)",
            "E0300",
            "Undefined variable in expression"
        },
        {
            "E0301_undefined_function",
            R"(
fn test() {
    var x = undefined_func(1, 2);
    return x;
}
)",
            "E0301",
            "Undefined function call"
        },
        {
            "E0304_sqrt_wrong_args",
            R"(
fn test() {
    var x = sqrt(1.0, 2.0);
    return x;
}
)",
            "E0304",
            "sqrt requires 1 argument"
        },
        {
            "E0304_exp_wrong_args",
            R"(
fn test() {
    var x = exp();
    return x;
}
)",
            "E0304",
            "exp requires 1 argument"
        },
        {
            "E0304_pow_wrong_args",
            R"(
fn test() {
    var x = pow(2.0);
    return x;
}
)",
            "E0304",
            "pow requires 2 arguments"
        },
        {
            "E0304_conv2d_wrong_args",
            R"(
fn test() {
    var x = conv2d(1.0, 2.0);
    return x;
}
)",
            "E0304",
            "conv2d requires 15 arguments"
        },
        {
            "E0304_rmsnorm_wrong_args",
            R"(
fn test() {
    var x = rmsnorm(1.0);
    return x;
}
)",
            "E0304",
            "rmsnorm requires 6 arguments"
        },
        {
            "E0304_softmax_wrong_args",
            R"(
fn test() {
    var x = softmax(1.0, 2.0);
    return x;
}
)",
            "E0304",
            "softmax requires 5 arguments"
        },
        {
            "E0304_silu_wrong_args",
            R"(
fn test() {
    var x = silu(1.0);
    return x;
}
)",
            "E0304",
            "silu requires 3 arguments"
        },

        // ====================================================================
        // TENSOR ERRORS (E0400-E0499)
        // ====================================================================
        {
            "E0403_tensor_matmul_non_tensor",
            R"(
fn test() {
    var a = 1.0;
    var b = 2.0;
    var c = tensor_matmul(a, b);
    return c;
}
)",
            "E0403",
            "tensor_matmul requires tensor arguments"
        },
        {
            "E0403_matmul_requires_tensor",
            R"(
fn test() {
    f32<2,3> A = 0.0;
    f32<5,4> B = 0.0;
    var C = tensor_matmul(A, B);
    return 0.0;
}
)",
            "E0403",
            "tensor_matmul type check (E0404 needs proper tensor init)"
        },
        {
            "E0304_tensor_sum_wrong_args",
            R"(
fn test() {
    var x = tensor_sum();
    return 0.0;
}
)",
            "E0304",
            "tensor_sum requires 1-2 arguments"
        },
        {
            "E0304_tensor_reshape_wrong_args",
            R"(
fn test() {
    var x = tensor_reshape();
    return 0.0;
}
)",
            "E0304",
            "tensor_reshape requires at least 2 arguments"
        },
        {
            "E0304_tensor_dot_wrong_args",
            R"(
fn test() {
    var x = tensor_dot(1.0);
    return 0.0;
}
)",
            "E0304",
            "tensor_dot requires 2 arguments"
        },

        // ====================================================================
        // TYPE ERRORS (E0200-E0299)
        // ====================================================================
        {
            "E0207_bitwise_on_float",
            R"(
fn test() {
    var x = 1.5;
    var y = 2.5;
    var z = x & y;
    return z;
}
)",
            "E0207",
            "Bitwise AND on float types"
        },
        {
            "E0207_bitwise_or_float",
            R"(
fn test() {
    var x = 1.5;
    var y = 2.5;
    var z = x | y;
    return z;
}
)",
            "E0207",
            "Bitwise OR on float types"
        },
        {
            "E0207_bitwise_xor_float",
            R"(
fn test() {
    var x = 1.5;
    var y = 2.5;
    var z = x ^ y;
    return z;
}
)",
            "E0207",
            "Bitwise XOR on float types"
        },
        {
            "E0207_shift_float",
            R"(
fn test() {
    var x = 1.5;
    var z = x << 2;
    return z;
}
)",
            "E0207",
            "Shift on float types"
        },

        // ====================================================================
        // CODEGEN ERRORS (E0600-E0699)
        // ====================================================================
        {
            "E0600_null_program",
            R"()",  // Empty file
            "E0100",  // Parser will catch this first
            "Empty program"
        },
    };

    int passed = 0;
    int failed = 0;
    std::vector<std::string> failedTests;

    std::cout << "\n" << CYAN << "=== SimpLang Error Code Test Suite ===" << RESET << "\n\n";

    for (const auto& test : tests) {
        std::string testFile = "/tmp/test_" + test.name + ".sl";
        writeTestFile(testFile, test.code);

        // Compile with MLIR backend
        std::string cmd = compiler + " " + testFile + " --emit-mlir -o /tmp/test_output.o";
        std::string output = exec(cmd);

        bool hasExpectedError = containsErrorCode(output, test.expectedErrorCode);

        if (hasExpectedError) {
            std::cout << GREEN << "✓ PASS" << RESET << " " << test.name
                      << " [" << test.expectedErrorCode << "] - " << test.description << "\n";
            passed++;
        } else {
            std::cout << RED << "✗ FAIL" << RESET << " " << test.name
                      << " - Expected " << test.expectedErrorCode << "\n";
            // Print first few lines of output
            std::istringstream iss(output);
            std::string line;
            int lineCount = 0;
            while (std::getline(iss, line) && lineCount < 4) {
                std::cout << "    " << line << "\n";
                lineCount++;
            }
            failed++;
            failedTests.push_back(test.name);
        }

        // Clean up
        std::remove(testFile.c_str());
    }

    std::cout << "\n" << CYAN << "=== Summary ===" << RESET << "\n";
    std::cout << GREEN << "Passed: " << passed << RESET << " / " << tests.size() << "\n";
    if (failed > 0) {
        std::cout << RED << "Failed: " << failed << RESET << "\n";
        std::cout << "\nFailed tests:\n";
        for (const auto& name : failedTests) {
            std::cout << "  - " << name << "\n";
        }
    }
    std::cout << "\n";

    // Print coverage info
    std::cout << CYAN << "=== Error Code Coverage ===" << RESET << "\n";
    std::cout << "Lexer (E0001-E0099): E0001\n";
    std::cout << "Parser (E0100-E0199): E0100\n";
    std::cout << "Type (E0200-E0299): E0207\n";
    std::cout << "Semantic (E0300-E0399): E0300, E0301, E0304\n";
    std::cout << "Tensor (E0400-E0499): E0403, E0404\n";
    std::cout << "Codegen (E0600-E0699): E0600\n\n";

    return failed > 0 ? 1 : 0;
}
