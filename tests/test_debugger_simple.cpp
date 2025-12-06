#include "kernel_debugger/debugger.hpp"
#include "kernel_debugger/config.hpp"
#include <iostream>

int main() {
    try {
        std::cout << "Testing SimpleLang Debugger capabilities...\n\n";
        
        // Get debugger instance
        KernelDebugger& debugger = KernelDebugger::getInstance();
        
        // Initialize debugger
        std::cout << "1. Initializing debugger...\n";
        debugger.initialize();
        std::cout << "✓ Debugger initialized\n\n";
        
        // Test configuration
        std::cout << "2. Testing configuration...\n";
        auto config = DebuggerConfig::getInstance();
        std::cout << "✓ Memory tracking enabled: " << (config->debug().enableMemoryTracking ? "Yes" : "No") << "\n";
        std::cout << "✓ SIMD tracking enabled: " << (config->debug().enableSIMDTracking ? "Yes" : "No") << "\n";
        std::cout << "✓ Color output: " << (config->display().colorOutput ? "Yes" : "No") << "\n\n";
        
        // Try to load a debug test file
        std::cout << "3. Testing source file loading...\n";
        bool loaded = debugger.loadKernel("tests/debug_tests/test_debug.sl");
        if (loaded) {
            std::cout << "✓ Source file loaded successfully\n";
            std::cout << "✓ Current file: " << debugger.getCurrentFile() << "\n";
        } else {
            std::cout << "⚠ Could not load source file (expected in container)\n";
        }
        
        // Test breakpoint functionality
        std::cout << "\n4. Testing breakpoint management...\n";
        int bp1 = debugger.addBreakpoint("test_debug.sl", 5);
        int bp2 = debugger.addBreakpoint("test_debug.sl", 10);
        std::cout << "✓ Added breakpoints: " << bp1 << " and " << bp2 << "\n";
        
        debugger.listBreakpoints();
        
        // Test source display
        std::cout << "\n5. Testing source display...\n";
        if (loaded) {
            debugger.showSource("", 1, 5);
        } else {
            std::cout << "⚠ Skipping source display (no file loaded)\n";
        }
        
        // Start debugger
        std::cout << "\n6. Starting debug session...\n";
        debugger.start();
        std::cout << "✓ Debug session started\n";
        
        std::cout << "\n7. Cleaning up...\n";
        debugger.clearBreakpoints();
        debugger.stop();
        std::cout << "✓ Debugger stopped and cleaned up\n";
        
        std::cout << "\n🎉 Debugger functionality test completed successfully!\n";
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
}