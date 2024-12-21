# SimpleLang

A domain-specific language (DSL) designed for SIMD hardware optimization, particularly focused on deep learning applications. SimpleLang provides high-level abstractions for SIMD operations with powerful debugging and runtime infrastructure.

## Architecture Overview

### Host-Kernel Interface

The way SimpleLang talks to your main program (the "host") is designed to be straightforward and reliable. Think of it like having a specialized worker (the kernel) that your main program can call on to do specific tasks.

### How It Works

When you write a program using SimpleLang, it creates two main parts:

1. **Your Main Program (The Host)**
   This is where you control everything. It's like the manager of your application. You write this in regular C++ and it handles things like:
   - Loading your SimpleLang code
   - Sending it data to work on
   - Getting results back
   - Handling any errors that might happen

2. **The SimpleLang Code (The Kernel)**
   This is your specialized worker. It's compiled into a separate file (a .so file on Linux) that your main program can load when needed. This separation means you can:
   - Change the kernel without recompiling your whole program
   - Load different kernels for different tasks
   - Keep your main program safe if the kernel crashes

### Using It In Practice

Here's what it looks like in real use:

1. First, you write your SimpleLang code:
```simplang
fn kernel_main() {
    // Your specialized code here
    return result;
}
```

2. Then, in your main program, you can use it like this:
```cpp
KernelRunner runner;
runner.loadLibrary("./my_kernel.so");
double result = runner.runKernel();
```

### Safety Features

We've built in several safety measures:

- **Error Handling**: If something goes wrong, you get clear error messages, not mysterious crashes
- **Resource Management**: Memory and other resources are automatically cleaned up
- **Type Safety**: The interface ensures data is passed correctly between your program and the kernel

### Debugging Support

When you're developing, you can:
- Step through your kernel code
- Inspect variables
- Set breakpoints
- Monitor performance

This makes it much easier to find and fix problems.

### Performance Considerations

The interface is designed to be lightweight. There's very little overhead when:
- Loading kernels
- Passing data back and forth
- Running kernel code

This means you can focus on making your actual computations fast without worrying about the communication between your program and the kernel slowing things down.

### Compiler Pipeline

When you write SimpleLang code, it goes through several steps to turn into something your computer can run. Let's break down this journey:

### From Text to Tokens

First, your code gets broken down into meaningful pieces (we call these tokens). It's like taking a sentence and identifying the nouns, verbs, and other parts of speech. For example:

```simplang
fn add(var x, var y) {
    return x + y;
}
```

Gets broken into pieces like:
- `fn` (this tells us it's a function)
- `add` (the function's name)
- `var` (tells us what kind of things we're working with)
- `x`, `y` (the names we gave our variables)
- `+` (the operation we want to do)

This step catches basic errors like misspelled keywords or missing punctuation.

### Understanding Structure

Next, SimpleLang figures out how all these pieces fit together. It's like understanding that in "The cat sat on the mat", "the cat" is the subject and "sat" is what it's doing.

The compiler builds what we call an Abstract Syntax Tree (AST). Think of it like an outline:
- Function `add`
  - Parameters: `x`, `y`
  - Body:
    - Return statement
      - Addition operation
        - Left side: `x`
        - Right side: `y`

This helps catch logical errors, like trying to add a number to a function.

### Making It Fast

Once SimpleLang understands your code, it works on making it run efficiently. This happens in several stages:

1. **Basic Improvements**
   - Combining simple operations
   - Removing unnecessary steps
   - Organizing calculations better

2. **SIMD Optimization**
   When possible, SimpleLang converts your code to use special CPU instructions that can do multiple calculations at once. For example, instead of:
   ```
   a + b
   c + d
   ```
   It might do both additions simultaneously.

3. **Memory Optimization**
   The compiler tries to arrange your data in ways that work best with modern computers:
   - Keeping related data close together
   - Aligning data properly for fast access
   - Minimizing memory moves

### Final Output

The end result is a shared library (.so file) that contains your optimized code. This file:
- Can be loaded by your main program
- Contains debug information to help you fix problems
- Is ready for the CPU to execute efficiently

### Development Experience

We've made sure the compiler gives helpful feedback:

- **Clear Error Messages**
  Instead of cryptic errors, you get messages that point to the problem and suggest fixes.

- **Warning System**
  The compiler warns you about potential problems before they become bugs.

- **Debug Information**
  The compiler includes information that helps you track down problems when they occur.

### Performance Focus

The compiler is designed to create fast code:
- It knows about modern CPU features
- It can use special instructions for math operations
- It organizes your code to work well with the CPU's cache
- It can often spot and fix inefficient patterns automatically

### Debug Infrastructure

SimpleLang's debugging infrastructure provides comprehensive introspection capabilities while maintaining performance. The system is designed for minimal overhead when disabled and detailed inspection when needed.

### Core Architecture

```
Host Program <-> Debug Interface <-> Kernel Runtime
     |               |                    |
     +-> Commands ---+-> Runtime Hooks    |
     |               |                    |
     +-- State   <---+-- Event Queue     |
     |               |                    |
     +-- Control <---+-- Breakpoints     |
```

The debugger integrates directly with the kernel runtime, providing:
- Hardware breakpoint support with minimal overhead
- Real-time memory tracking and leak detection
- Full call stack inspection with variable state
- Asynchronous event handling for UI responsiveness

### Breakpoint Implementation

The breakpoint system uses hardware debugging registers when available, falling back to software interrupts when needed:

```
Code Execution    Breakpoint Handler    Debug Interface
    |                    |                    |
    |   [INT3/DR0-3]    |                    |
    |------------------->|                    |
    |                    |    [State Sync]    |
    |                    |------------------->|
    |   [Resume/Step]    |                    |
    |<-------------------|                    |
```

Key features:
- Zero-overhead when disabled
- Conditional breakpoints with expression evaluation
- Non-blocking UI updates during long operations
- Source-level to machine code mapping

### Memory Tracking System

Memory tracking is implemented through runtime hooks with configurable granularity:

```
Allocation Path    Tracking System    Analysis
    |                    |               |
    +-> malloc/free     [|]  Stats      |
    |                   [|]  Leaks      |
    +-> SIMD alloc      [|]  Patterns   |
    |                   [|]  Alignment  |
    +-> Pool alloc      [|]  Usage      |
```

Features targeted at SIMD development:
- SIMD alignment verification
- Vector operation tracking
- Memory access pattern analysis
- Pool allocation statistics

### Call Stack Integration

The call stack system provides both runtime inspection and post-mortem analysis:

```
[Top] kernel_main()
      ├── vector_multiply()  // Args: (float*, float*, size_t)
      │   └── simd_mul_ps() // Vec registers: xmm0-xmm7
      └── reduce_sum()      // Alignment: 32-byte
```

Developers can:
- Inspect SIMD register state
- Track vector operation flow
- Analyze memory access patterns
- Profile hot paths in real-time

### Event Processing

The event system is designed for high-throughput debugging scenarios:

```
Producer                Consumer
   |                       |
   |  [Lock-Free Queue]    |
   +---------------------->|
   |   Debug Events       [|]
   |   Memory Ops         [|]
   |   Profile Data       [|]
   |                       |
   +---------------------->|
```

Optimized for:
- Lock-free event processing
- Minimal impact on kernel execution
- Configurable buffering strategies
- Real-time event filtering

### Integration Example

```cpp
// Attach debugger with custom configuration
DebugConfig config;
config.enableMemoryTracking()
     .setBreakpointMode(HardwareBreakpoints)
     .setEventBuffering(1024);

runner.attachDebugger(config);

// Register custom event handlers
runner.debugger().onMemoryLeak([](const LeakInfo& info) {
    std::cout << "Leak detected: " << info.size << " bytes\n";
});
```

### Performance Characteristics

With selective feature enabling:
- Breakpoints: ~1 cycle when not hit
- Memory tracking: 2-5% overhead
- Call stack: 1-3% overhead
- Event system: <1% with buffering

### SIMD Operations

Native support for SIMD operations:

1. **SSE Support**
   - 128-bit vector operations
   - Aligned memory access
   - Optimized math functions

2. **AVX Support**
   - 256-bit vector operations
   - Advanced vector extensions
   - Hardware-specific optimizations

### Runtime System

The runtime provides essential services:

1. **Memory Management**
   - SIMD-aligned allocations
   - Memory pool optimization
   - Garbage collection (optional)

2. **Error Handling**
   - Exception propagation
   - Error recovery
   - Debug information

3. **Performance Monitoring**
   - Operation timing
   - Memory usage tracking
   - Optimization hints

## Implementation Example

### Writing a Kernel

```simplang
fn bounded_sum(var n) {
    var sum = 0.0;
    var i = 1.0;
    
    while (i <= n) {
        sum = (sum + i) % 10000.0;
        i = i + 1.0;
    }
    return sum;
}

fn kernel_main() {
    var n = 100000.0;
    return bounded_sum(n);
}
```

### Host Program Integration

```cpp
#include "kernel_runner.hpp"

int main() {
    try {
        KernelRunner runner;
        runner.loadLibrary("./kernel.so");
        
        // Optional: Enable debugging
        runner.attachDebugger();
        
        // Run kernel and get result
        double result = runner.runKernel();
        
        std::cout << "Result: " << result << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
```

## Performance Characteristics

Recent benchmarks show:
- JIT compilation overhead: < 1ms
- SIMD operation performance: ~1.3x slower than optimized C++
- Memory overhead: ~2MB per kernel instance
- Debug mode overhead: ~5% in typical usage

## Development Workflow

1. Write SimpleLang kernel code
2. Compile to shared library
3. Integrate with host program
4. Debug using built-in tools
5. Profile and optimize

## Future Directions

1. **Language Extensions**
   - Advanced type system
   - Template support
   - Meta-programming capabilities

2. **Optimization Improvements**
   - Auto-vectorization
   - Pattern-based optimizations
   - Hardware-specific tuning

3. **Tooling**
   - IDE integration
   - Visual debugger
   - Performance analyzer

## Contributing

See CONTRIBUTING.md for guidelines on:
- Code style
- Testing requirements
- Pull request process
- Documentation standards 