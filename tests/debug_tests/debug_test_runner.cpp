#include "kernel_debugger/debugger.hpp"
#include "kernel.h"
#include <iostream>
#include <cstdlib>
#include <string>

// Forward declaration - using same signature as our test_simd.sl
extern "C" {
    double kernel_main();
}

class SimpleLangDebugger {
private:
    KernelDebugger* debugger;

public:
    SimpleLangDebugger() {
        debugger = KernelDebugger::getInstance();
        debugger->start();
        std::cout << "SimpleLang debugger initialized\n";
    }

    ~SimpleLangDebugger() {
        debugger->stop();
    }

    void enableStepMode() {
        debugger->setMode(KernelDebugger::Mode::STEP);
        std::cout << "Step mode enabled\n";
    }

    void enableBreakpointMode() {
        debugger->setMode(KernelDebugger::Mode::BREAKPOINT);
        std::cout << "Breakpoint mode enabled\n";
    }

    void addBreakpoint(const std::string& operation) {
        debugger->addBreakpoint(operation);
        std::cout << "Breakpoint added for " << operation << "\n";
    }

    void runTests() {
        std::cout << "\n=== Running SimpleLang SIMD Debug Tests ===\n";

        try {
            // Run test with debugger in step mode
            std::cout << "\nRunning with step mode...\n";
            debugger->setMode(KernelDebugger::Mode::STEP);
            double result = kernel_main();
            std::cout << "Test completed with result: " << result << std::endl;

            // Test with breakpoints
            std::cout << "\nTesting with breakpoints...\n";
            debugger->setMode(KernelDebugger::Mode::BREAKPOINT);
            debugger->addBreakpoint("simd_mul");  // Break on multiplication operations
            result = kernel_main();
            std::cout << "Breakpoint test completed with result: " << result << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Error during test execution: " << e.what() << std::endl;
            throw;
        }
    }

    void showHelp() {
        std::cout << "\nSimpleLang SIMD Debugger Commands:\n";
        std::cout << "  s: Enable step mode\n";
        std::cout << "  b: Add breakpoint\n";
        std::cout << "  r: Run tests\n";
        std::cout << "  h: Show this help\n";
        std::cout << "  q: Quit\n";
        std::cout << "\nBreakpoint Operations:\n";
        std::cout << "  simd_add     - SSE addition\n";
        std::cout << "  simd_mul     - SSE multiplication\n";
        std::cout << "  simd_add_avx - AVX addition\n";
        std::cout << "  simd_mul_avx - AVX multiplication\n";
        std::cout << "  slice_get_sse - SSE slice read\n";
        std::cout << "  slice_set_sse - SSE slice write\n";
        std::cout << "  slice_get_avx - AVX slice read\n";
        std::cout << "  slice_set_avx - AVX slice write\n";
    }
};

int main() {
    try {
        SimpleLangDebugger debugger;
        debugger.showHelp();

        char cmd;
        while (true) {
            std::cout << "\nCommand: ";
            std::cin >> cmd;

            switch (cmd) {
                case 's':
                    debugger.enableStepMode();
                    break;

                case 'b': {
                    std::string op;
                    std::cout << "Enter operation to break on: ";
                    std::cin >> op;
                    debugger.addBreakpoint(op);
                    break;
                }

                case 'r':
                    debugger.runTests();
                    break;

                case 'h':
                    debugger.showHelp();
                    break;

                case 'q':
                    std::cout << "Exiting debugger...\n";
                    return 0;

                default:
                    std::cout << "Unknown command. Type 'h' for help.\n";
            }
        }

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}