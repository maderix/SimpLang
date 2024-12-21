SimpleLang Design Document (Updated)
1. Overview
SimpleLang is a domain-specific language (DSL) designed for SIMD hardware optimization, particularly for deep learning. It provides high-level abstractions for SIMD operations, along with powerful debugging and runtime infrastructure.

Key highlights include:

A robust compiler pipeline using Flex/Bison and LLVM
Expanded runtime modules for debugging (breakpoints, memory tracking, step/continue, etc.)
A command-line debugger with a KernelDebugger and CommandProcessor system
Integration of SSE and AVX vector operations for performance-critical workloads
2. Current State
2.1 Core Language Features
Strong Type System: Native scalar, vector (SSE/AVX), tensor (WIP)
Functions: Definition and calling mechanisms
Control Flow: If-else, while loops
Variables: Declarations and assignments
Basic Arithmetic: Extended to handle modulo and vector operations
2.2 SIMD Support
SSE and AVX vector operations
SIMD slice types for SSE/AVX (e.g., SSESlice, AVXSlice)
Vector intrinsics: Add, multiply, subtract, and divide
Alignment Handling: SSE- and AVX-specific alignment logic
Vector Register Management: Integration with debugger and memory tracker
2.3 Development Infrastructure
Flex / Bison for lexical/syntax analysis
LLVM-based IR generation and machine code emission
CMake build system with modular subdirectories
Runtime subdirectory for kernel runners, memory tracking, and debugging
Host Runner libraries for dynamic loading and testing
Comprehensive Tests across multiple .sl programs (arithmetic, loops, Fibonacci, etc.)
2.4 Debugging Support
Interactive Debugger:
Breakpoint support (line-based)
Single-step (stepIn, stepOver, stepOut)
Condition-based breakpoints
Call stack inspection
Local variables printing
Source-level stepping via source_manager
SIMD Register Inspection: printVectorState() and per-register display
Memory Tracking: MemoryTracker to detect out-of-bounds and leaks
Command-Line Debugger:
CommandProcessor and UIHelper for a GDB-like interface
Readline-based input with history and tab completion
Event-Based Debugging: DebugEventQueue, EventLogger
3. Architecture
3.1 Compiler Pipeline
css
Copy code
Source Code → Lexer → Parser → AST → LLVM IR → Machine Code
                                  ↓
                            Type Checker
                                  ↓
                         SIMD Optimization
Lexer (Flex) → Parser (Bison) → AST → CodeGen → LLVM IR → LLVM-based back-end → Machine code
SIMD Optimization: Uses custom abstractions for SSE/AVX
3.2 Runtime Components
Kernel Runner

Dynamically loads the compiled .so or object file
Executes kernel_main or specialized entry points for SIMD tests
Kernel Debugger

KernelDebugger: Global debugging engine with breakpoints, stepping, memory checks
CallStack / CallStackManager: Tracks function calls, local variables, and SIMD ops
MemoryTracker: Tracks allocations, deallocations, out-of-bounds, alignment, leaks
EventLogger: Logs debug events (SIMD ops, memory ops, breakpoints) with timestamps
CommandProcessor + UIHelper: Provide a command-line debugging interface
Memory Management

SSE/AVX slice creation (make_sse_slice, make_avx_slice)
Automatic alignment (aligned_alloc) for vectors
Lifetime tracking in MemoryTracker
Debugger UI

Command-line interface using Readline (UIHelper)
Commands: break, continue, step, print, list, info, etc.
3.3 Key Components
Frontend

Lexer (lexer.l) and Parser (parser.y)
AST (in ast.hpp/cpp)
Middle-end

Type Checking (partially integrated in ast nodes)
SIMD Optimizations (intrinsic insertion, vector type detection)
IR Generation (in codegen.cpp)
Backend

LLVM integration for codegen
TargetMachine creation and object file emission
Pass Manager for final optimizations
Platform-Specific intrinsics for SSE/AVX
Runtime

SIMD Operation libraries (simd_ops.cpp, simd_interface.cpp)
Kernel Debugger code (kernel_debugger/*.cpp)
Host Runner libraries (host_runner, kernel_runner)
MemoryTracking (memory_tracker.hpp/cpp)
4. Roadmap
4.1 Phase 1: Language Enhancement (Current Priority)
Tensor Data Types and operations
Advanced Type Inference (in-progress)
Improved Error Handling in parser and codegen
Documentation System (ongoing)
4.2 Phase 2: SIMD Optimization
Auto-vectorization framework (beyond manual intrinsics)
Loop Optimization for unrolled vector loops
Platform-Specific intrinsics expansions
Performance Profiling
4.3 Phase 3: Deep Learning Support
Neural Network Layer Primitives
Automatic Differentiation
Training Support in SimpleLang
Model Serialization
4.4 Phase 4: Tooling and Infrastructure
Enhanced Debugging: Already partially completed with line-step, breakpoints
Visual Profiling Tools: Graphical representation of memory usage, SIMD ops
Integration with Existing ML Frameworks
Package Management
5. Implementation Details
5.1 Type System
cpp
Copy code
// Example from ast.hpp
class Type {
    enum class TypeKind {
        Scalar,
        Vector,
        Tensor,
        Function
    };
};
TypeChecker integrated in ast nodes
Slices (SSE/AVX) with specialized struct types in codegen.cpp
5.2 SIMD Abstraction
cpp
Copy code
// SSE / AVX vector abstractions
class SIMDOperation {
    enum class OpType {
        Add,
        Multiply,
        Subtract,
        Divide
    };
};
SIMDInterface (simd_interface.cpp) for SSE/AVX bridging
simd_ops.cpp: Utility methods for vector creation, broadcast, etc.
5.3 Memory & Debugging Infrastructure
Memory Management
cpp
Copy code
class MemoryManager {
    void* allocateAligned(size_t size, size_t alignment);
    void copyVectorAligned(void* dest, const void* src, size_t size);
    void trackAllocation(void* ptr, size_t size);
};
MemoryTracker: More advanced, includes alignment checks and leak detection
Slices use aligned_alloc() under the hood for SSE/AVX arrays
Debugger Components
KernelDebugger

Singleton controlling breakpoints, stepping, event queue
onSIMDOperation(), onMemoryAccess(), etc.
MemoryTracker

Tracks active allocations (allocations)
Out-of-bounds checks on every memory access
Logs usage in operationHistory
EventLogger

Maintains ring buffer of debug events
Real-time printing or post-run summary
UIHelper + CommandProcessor

Readline-based input
Commands: run, step, continue, print <var>, break <line>, etc.
Tab completion
CallStack

Maintains call frames for function boundaries
Tracks local variables (double, SSE vectors, AVX vectors, slices)
6. Testing Strategy
6.1 Unit Testing
Compiler Components: AST node checks, parser correctness
SIMD Ops: Validate SSE/AVX creation, addition, multiplication
Memory Tracking: Verify out-of-bounds detection, alignment checks
Debugging: Breakpoint creation, stepping logic
6.2 Integration Testing
End-to-End compilation of sample .sl programs
SIMD Performance tests (comparing SSE vs. AVX)
Cross-Platform checks on Windows/Linux (where possible)
6.3 Benchmarking
SIMD Operation Performance
Memory Access Patterns
Compilation Time
Generated Code Quality
perf_test.sl: Compares SimpleLang vs. native C++ implementations
7. Future Directions
7.1 Language Extensions
Custom SIMD Intrinsics via DSL-level definitions
Advanced Control Flow (break, continue, for loops, etc.)
Metaprogramming / compile-time expansions
7.2 Tool Ecosystem
Language Server Protocol (LSP) integration for autocompletion
IDE Integration with debugging UI
Performance Analysis Tools with visual timelines
Documentation Generation from source
7.3 Integration
Python Bindings for script-level integration
C++ Template Library bridging native and DSL code
Deep Learning Framework integration (PyTorch, TensorFlow ops)
8. Contributing
8.1 Development Process
Git Workflow with feature branches & PRs
Code Review guidelines: ensure coverage & readability
Testing Requirements: Must add relevant unit or integration tests
Documentation: Update relevant .md files and inline docs
8.2 Building and Testing
CMake build system: cmake .. && make -j4
Test Suite: ctest or make test
Performance Benchmarking: perf_test_host
Cross-Platform: Building on Linux/Windows with minimal changes
9. References
LLVM Documentation
SIMD Instruction Sets (Intel SSE/AVX)
Compiler Design Resources
Deep Learning Optimization Techniques
10. Next Steps
10.1 Immediate Priorities
Complete Debugging Infrastructure for advanced memory checks, stepping, and breakpoints
Implement Tensor Types and operations (phase 1 target)
Enhance SIMD Optimization for auto-vectorization
Improve Error Handling and user-facing diagnostics
10.2 Technical Debt
Refactor Compiler Pipeline for improved modularity (separate passes)
Improve Test Coverage for new debugger features
Complete Documentation for new runtime modules
Enhance Build System (potential multi-configuration support, Windows nuance)
Note: The expanded runtime/debugger infrastructure now plays a crucial role in SimpleLang’s development and usage. The runtime/kernel_debugger directory and related classes (KernelDebugger, MemoryTracker, CallStack, CommandProcessor, UIHelper, etc.) are critical to providing advanced debugging capabilities, including breakpoints, stepping, memory tracking, and interactive command input. This significantly improves developer productivity and diagnostic capabilities for both compiler and runtime issues.