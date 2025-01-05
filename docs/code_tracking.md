**SimpLang Requirement Tracking Document (Revised)**

**I. Project Overview**

*   **Project Name:** SimpLang (Domain-Specific Language for SIMD Optimization)
*   **Project Goal:** Develop a DSL with robust SIMD support, debugging infrastructure, and a clear path toward deep learning integration.
*   **Tracking Document Version:** 2.0
*   **Last Updated:** 2024-12-30

**II. Core Modules & Components**

This section describes the major modules of the project, each with a unique ID for tracking. Includes class diagrams and test cases.

    +-----------------------+
    |    Project: SimpLang   |
    +-----------------------+
            |
      +-----------+------------+
      |  Compiler   |   Runtime  |
      +-----------+------------+
           |           |
      +-----+---+    +-----+-----+
      | Front | Mid |  Back | Core| Debug |
      +-----+---+    +-----+-----+

*   **A. Compiler (ID: CMP)**
    *   **A.1 Frontend (ID: CMP_FNT)**: Responsible for lexical and syntactical analysis.
        *   **A.1.1 Lexer (ID: CMP_FNT_LEX)** (lexer.l)
            *   **Class Diagram:**
            ```
             +------------+
             |   Lexer    |
             +------------+
             | -yyin      |
             | -yylineno  |
             | -yytext    |
             +------------+
             | +yylex()   |
             | +yywrap()  |
             +------------+
             ```
            *   **State:** Scans source code, generates tokens.
            *   **Output:** Stream of tokens (TIDENTIFIER, TINTEGER, etc.)
            *   **Core Logic:** Uses Flex regex rules to match and extract tokens, maintains line number information (`yylineno`).
            *   **Test Cases:**
                *   **TC_LEX_1:** Correct tokenization of single-line comments (`// comment`).
                *   **TC_LEX_2:** Correct tokenization of multi-line comments (`/* comment */`).
                *   **TC_LEX_3:** Proper handling of whitespace, skipping spaces, tabs, newlines.
                *   **TC_LEX_4:** Recognition of various numeric types (TINTEGER, TFLOAT, TINTLIT).
                *   **TC_LEX_5:** Recognition of keywords (TFUNC, TRETURN, etc).
                *   **TC_LEX_6:** Recognition of SIMD tokens (TSSE, TAVX, TSSESLICE, TAVXSLICE, slice_get_sse etc)
        *   **A.1.2 Parser (ID: CMP_FNT_PAR)** (parser.y)
            *   **Class Diagram:**
            ```
            +------------+
            |   Parser   |
            +------------+
            | -programBlock|
            | -yylex()    |
            | -yylineno  |
            | -yytext    |
            +------------+
            | +yyparse() |
            | +yyerror() |
            +------------+
            ```
            *   **State:** Analyzes token stream, produces Abstract Syntax Tree (AST).
            *   **Input:** Token stream from the Lexer.
            *   **Output:** Abstract Syntax Tree (e.g., `programBlock`).
            *   **Core Logic:** Uses Bison grammar rules to parse token sequence; builds AST nodes based on recognized language constructs.
            *   **Test Cases:**
                *   **TC_PAR_1:** Correct parsing and AST creation for basic arithmetic expressions.
                *   **TC_PAR_2:** Correct parsing and AST creation for variable declaration and assignments (with and without initializers).
                *   **TC_PAR_3:** Parsing for different combinations of binary and unary operators.
                *   **TC_PAR_4:** Parsing function declarations, handling arguments, function body.
                *   **TC_PAR_5:** Correct parsing of control flow statement like `if`, `if-else` and `while` blocks.
                *   **TC_PAR_6:** Parsing SIMD vector operations and slice creation/access expressions.
                *   **TC_PAR_7:** Correct parsing for return statements.
    *   **A.2 Middle-end (ID: CMP_MID)**
        *   **A.2.1 Type Checker (ID: CMP_MID_TYP)**:
            *   **State:** Inspects AST, annotates types, reports type errors
            *    **Core Logic:** Traverses the AST and ensures that all nodes abide by the type system's rules, emitting errors for inconsistencies.
            *   **Test Cases:**
                *   **TC_TYP_1:** Type checking for basic arithmetic operations on scalars.
                *   **TC_TYP_2:** Type checking variable assignments.
                *   **TC_TYP_3:** Type mismatches in binary operations.
                *   **TC_TYP_4:** Function arguments type checks.
                *   **TC_TYP_5:** Type checking when using slices.
                *   **TC_TYP_6:** Type checking when using SIMD vector types.
        *   **A.2.2 SIMD Optimization (ID: CMP_MID_SIMD)**:
             *   **State:** Modifies AST, inserts intrinsics, marks vector types.
            *  **Core Logic:**  Identifies opportunities to use SIMD instructions and transforms nodes to call the appropriate intrinsic functions or typecasts.
             *   **Test Cases:**
                *   **TC_SIMD_OPT_1:**  Detection of simple vector operations (addition, multiplication etc) and mapping them to SSE/AVX intrinsics based on architecture.
                *  **TC_SIMD_OPT_2:** Correct handling for type-casting operations before applying vector operations
    *   **A.3 Backend (ID: CMP_BCK)**
        *   **A.3.1 IR Generation (ID: CMP_BCK_IR)**:
              *  **Class Diagram:**
               ```
                 +----------------+
                 | CodeGenContext  |
                 +----------------+
                 | -builder       |
                 | -module        |
                 | -symbolTable    |
                 | ...            |
                 +----------------+
                 | +generateCode()|
                 | +createValue() |
                 | ...            |
                 +----------------+
                 ```
            *   **State:** Traverses AST, generates IR instructions using LLVM builder
            *   **Core Logic:** Traverses the AST, generates corresponding LLVM instructions. It maintains a context (`CodeGenContext`) containing builder, module, and symbol information, using this to create LLVM values, functions, and instructions.
            *   **Test Cases:**
                *   **TC_IR_1:** IR generation for different types of arithmetic operations
                *   **TC_IR_2:** IR generation for variable declarations and usage
                *   **TC_IR_3:** IR generation for basic control flow structures such as `if`, `if-else` and `while` statements
                *   **TC_IR_4:** IR generation for function declarations and calls (including parameters and return value handling).
                *   **TC_IR_5:** Correct handling of SIMD intrinsics for vector operations
                *   **TC_IR_6:** Generation of slice creation, access, and store operations.
        *  **A.3.2 Code Emission (ID: CMP_BCK_EMIT)**:
            *   **State:** Runs LLVM optimization passes, generates object code
            *    **Core Logic:** Sets up LLVM target machine, runs the LLVM passes, and emits the object file.
            *   **Test Cases:**
                *   **TC_EMIT_1:** Generate correct object files from the IR with no errors.
                *   **TC_EMIT_2:** Generates an optimized object file after running optimization passes.
                *   **TC_EMIT_3:** Verify that generated object file can be dynamically linked with host runner.

    * **A.4 CodeGen Context (ID: CMP_CTX)**
            *  **State:** LLVM Context and Builder object, Module, Memory tracking information.
            *   **Core Logic:** Acts as central component for LLVM code generation, providing a consistent environment for code gen methods and maintaining state and tracking objects.
            *    **Test Cases:**
                *   **TC_CTX_1:** Test to check if the required types for LLVM are initialized.
                *   **TC_CTX_2:** Test to verify that the memory tracker object is set correctly in the codegen context.
                *   **TC_CTX_3:** Test for the functionality of SIMD interface object to correctly generate vector types.
                *    **TC_CTX_4:** Test for correctness of pushing and popping basic blocks and maintaining the local symbol tables.

*   **B. Runtime (ID: RUN)**
    *   **B.1 Core (ID: RUN_COR)**
        *   **B.1.1 SIMD Operations (ID: RUN_COR_SIMD)** (simd_ops.cpp, simd_interface.cpp)
             *   **Class Diagram:**

               ```
                +---------------------+    +---------------------+
                |   SIMDInterface     |    |   SIMDHelper       |
                +---------------------+    +---------------------+
                | +createVector()     |    | +performOp()       |
                | +add()              |    | +createVector()    |
                | +sub()              |    | +broadcastScalar()|
                | +mul()              |    |                    |
                | +div()              |    |                    |
                | ...                 |    +---------------------+
                | +getVectorType()   |
                +---------^----------+
                            |
             +------------+   +------------+
             | SSEInterface |  |AVXInterface |
             +------------+   +------------+

            ```
           *   **State:** Implements SSE and AVX operations using intrinsic instructions.
            *   **Core Logic:** `SIMDInterface` is an abstract class with concrete implementations for SSE and AVX operations (e.g. `add`, `sub`, `mul`, `div`, `createVector`, and `broadcast`). The `SIMDHelper` class helps to perform generic operations with different SIMD widths.
            *  **Test Cases:**
                 *    **TC_SIMD_CORE_1:** Verification of SIMD vector addition/subtraction for both SSE and AVX.
                 *    **TC_SIMD_CORE_2:** Verification of SIMD vector multiplication/division for both SSE and AVX.
                 *    **TC_SIMD_CORE_3:** Verify broadcasting of scalar values to vectors.
                 *   **TC_SIMD_CORE_4:** Verify creation of SIMD vector based on different input element size and data types.
        *   **B.1.2 SIMD Types (ID: RUN_COR_TYPES)** (simd_types.hpp)
             *   **State:** Structure definitions and contant sizes related to SSE and AVX.
             *    **Core Logic:**  Simple struct definitions with pre-defined sizes for SIMD slice variables, specifically `SSESlice` and `AVXSlice`.
            *   **Test Cases:**
                 *   **TC_TYPES_1:** Verifying the `VECTOR_SIZE` of `SSESlice` to 2.
                 *   **TC_TYPES_2:** Verifying the `VECTOR_SIZE` of `AVXSlice` to 8.
                 *   **TC_TYPES_3:** Verify size of `SSESlice` and `AVXSlice` is equal to the size of required data types.
        *   **B.1.3 Memory Management:**
             *   **State:** Manages dynamic memory allocation and deallocation for various purposes.
             *   **Core Logic:**  Uses aligned allocation and free system call (`aligned_alloc`), for various sizes and requirements, which handles memory alignment.
             *   **Test Cases:**
                *    **TC_MEM_1:** Memory Allocation and deallocation with valid pointer.
                *    **TC_MEM_2:** Test for different alignment requirements for vector creation.
                *   **TC_MEM_3:** Test for leak free behavior while allocating and deallocating memory using utility functions.
    *  **B.2 Debug (ID: RUN_DBG)**
         *   **B.2.1 Debugger Engine (ID: RUN_DBG_ENG)**
             *    **Class Diagram:**
               ```
               +------------------+
               |  KernelDebugger  |
               +------------------+
               | -memoryTracker   |
               | -eventLogger    |
               | -cmdProcessor   |
               | -sourceManager   |
               | -breakpointMgr   |
               | -callStack        |
               | ...              |
               +------------------+
               | +initialize()    |
               | +loadKernel()    |
               | +start()         |
               | ...              |
               +------------------+
            ```
              *   **State:** Controls program execution, breakpoint management, and memory state.
            *    **Core Logic:** The `KernelDebugger` class is a singleton that manages the core debugging process, including breakpoints, step execution, memory access tracking and maintaining the program state.
            *  **Test Cases:**
                 *   **TC_DBG_ENG_1:** Start, Stop, Initialize, and Reset of the debugger.
                 *   **TC_DBG_ENG_2:** Testing set and remove breakpoint functionalities.
                 *   **TC_DBG_ENG_3:** Testing the various step execution methods stepIn, stepOut, stepOver.
                 *   **TC_DBG_ENG_4:** Testing of memory tracking during run-time.
                 *  **TC_DBG_ENG_5:** Testing the various SIMD register inspection and printing methods.
         *  **B.2.2 Source Manager (ID: RUN_DBG_SRC)**
              *   **Class Diagram:**
              ```
              +-----------------+
              |  SourceManager  |
              +-----------------+
              | -sourceFiles   |
              | -currentFile   |
              | -currentLine  |
              | ...             |
              +-----------------+
              | +loadSource()  |
              | +setLocation() |
              | +getLine()     |
              | ...             |
              +-----------------+

              ```
              *   **State:** Manages source files, tracks execution position.
              *    **Core Logic:** Keeps track of loaded source files, current kernel, location of execution and various helper methods for accessing/printing the source file.
              *  **Test Cases:**
                  *   **TC_DBG_SRC_1:** Loading source file from various path formats and checking file availability.
                  *   **TC_DBG_SRC_2:** Set, update, and retrieve the source location.
                  *  **TC_DBG_SRC_3:** Retrieve lines from a given range from any of the source file.
         *   **B.2.3 Memory Tracker (ID: RUN_DBG_MEM)**
              *   **Class Diagram:**
               ```
                  +-----------------+
                  | MemoryTracker    |
                  +-----------------+
                  | -allocations     |
                  | -variableStates |
                  | -operationHistory|
                  | -stats           |
                  | ...            |
                  +-----------------+
                  | +trackAllocation()|
                  | +trackDeallocation()|
                  | +trackAccess()    |
                  | ...            |
                  +-----------------+
               ```
              *   **State:** Manages memory allocation and access.
              *  **Core Logic:** Tracks all allocation and deallocation operations to detect potential memory leaks, out-of-bound access and SIMD alignment issues during the program execution.
              *   **Test Cases:**
                  *  **TC_MEM_TRK_1:** Allocation of different data types using track and deallocation using `trackDeallocation`.
                  *  **TC_MEM_TRK_2:** Track SIMD specific aligned allocation methods.
                  *  **TC_MEM_TRK_3:** Verify validity of the memory locations through various pointer types, and throw exceptions if required.
                  *    **TC_MEM_TRK_4:** Verifying correct detection of out-of-bounds access.
                  *   **TC_MEM_TRK_5:** Verify memory leak detection.
         *   **B.2.4 UI Helper (ID: RUN_DBG_UI)**
             *  **Class Diagram:**
            ```
                  +-----------------+
                  |   UIHelper     |
                  +-----------------+
                  | -options       |
                  | -history       |
                  | ...            |
                  +-----------------+
                  | +getInput()     |
                  | +printSourceLine()|
                  | +printError()    |
                  | ...            |
                  +-----------------+
            ```
              *   **State:** Manages command-line UI interactions.
              *  **Core Logic:** Uses readline to get the user input, supports history and completion, provides methods for styled terminal output.
              *   **Test Cases:**
                   *    **TC_UI_1:** Verify prompt generation using defined string.
                   *    **TC_UI_2:** Verify the ability to read user input with readline and store in history.
                   *    **TC_UI_3:** Test the callback functionality for command completion.
                   *   **TC_UI_4:** Verify the usage of ANSI color code for formatted output.
         *  **B.2.5 Call Stack Manager (ID: RUN_DBG_CALL)**
              *  **Class Diagram:**
               ```
                 +-------------------+
                 |  CallStack        |
                 +-------------------+
                 | -frames         |
                 | -inSimdOp        |
                 | -currentSimdOp   |
                 | ...               |
                 +-------------------+
                 | +pushFrame()      |
                 | +popFrame()       |
                 | +addLocal()      |
                 | ...              |
                 +-------------------+

            ```
             *   **State:** Keeps track of function calls.
             *    **Core Logic:** Manages a stack of function frames to track function entry, exit and also SIMD op execution with local variables associated with each function.
             *   **Test Cases:**
                *  **TC_CALL_1:** Test that the stack is maintaining accurate count of the function calls.
                 *  **TC_CALL_2:** Test for maintaining SIMD operation boundaries correctly.
                 *   **TC_CALL_3:** Test to correctly identify local variables and their state.
                 *  **TC_CALL_4:** Verify stack-trace using `printBacktrace`.
         * **B.2.6 Command Processor (ID: RUN_DBG_CMD)**
              *  **Class Diagram:**
            ```
              +---------------------+
              | CommandProcessor   |
              +---------------------+
              | -debugger        |
              | -commands         |
              | -ui               |
              | ...              |
              +---------------------+
              | +processCommand()  |
              | +showHelp()       |
              | +registerCommand()|
              | ...              |
              +---------------------+

            ```
             *   **State:** Processes command-line inputs for the debugger.
             *    **Core Logic:** Parses user inputs, validates syntax, executes command handlers, and displays output to the terminal. Manages a map of supported commands, with handler methods.
             *  **Test Cases:**
                *  **TC_CMD_1:** Test the execution of debugger commands like `run`, `step` and `continue` from user input.
                 *  **TC_CMD_2:** Verify correct registration and usage of command aliases.
                 *   **TC_CMD_3:** Verify proper parsing of inputs with different argument patterns including quoted string and spaces.
                 *  **TC_CMD_4:** Check proper help text generation and handling of unknown commands.


**V. Change Tracking & LLM Implementation**

*   **LLM Role:** LLM will monitor the source code changes based on the identifiers within this document. It will also manage the addition of new identifiers.

*   **Change Detection:**
    *   **File Hash:** LLM will keep track of the hash for each file and detect when a file has changed.
    *   **Identifier Based:** Every class, method, function, and major logical block should be referenced by the IDs mentioned in this document. LLM will detect if the implementation associated with those identifiers has changed.
    *   **New Component Detection:** LLM will detect addition of new files, classes, methods or any other components which is not part of tracking and will assign a new identifier to it.

*   **Change Recording:**
    *   LLM will record all changes to a document named ".codetracking.txt".
    *   Upon detecting a code change, LLM will:
        *   Record the date/time of the change.
        *   Identify the file and specific component (ID) that was modified.
        *   Summarize the nature of the change (bug fix, new feature, refactor, etc.)
        *   Record the code diff between the original and changed code.
        *   Track the test status of the changes after building the project using `ctest`.
        *   Track the performance of changed module by running the required test if performance parameters are changed.
        *   **Crucially:** Evaluate the incoming code against the current code using diff analysis and *only* implement code changes that are different or necessary. If the same logic is implemented already, code must be kept unchanged and this should be explicitly recorded in "code_tracking.txt".

    *   **New Component Handling:**
        *  Upon detecting any new files/classes/methods, or logic, the LLM will assign a new unique ID. The numbering should follow the pattern as described in this document (e.g `CMP_FNT_LEX_NEW_01` if a new item is added in Lexer)
        *  The new components should be added to the correct section in this document.
        *  The code for adding the new component should be recorded in "code_tracking.txt" with the newly generated id.
        *  The changes made to this tracking document should also be recorded in "code_tracking.txt" with a special id `TRACKING_DOC_UPDATE`.

    *   **"code\_tracking.txt" Format:** Each change entry in this file will follow a consistent, structured format:

        ```
        [CHANGE_ID]  <timestamp>
        File: <file path>
        Component: <component ID>
        Type: <change type - new | modified | deleted>
        Description: <summary of the change>
        Diff:
        <code diff>
        Test Status: <test status - pass/fail>
        Performance Impact: <impact on performance, if any>
        Existing Logic Maintained: <true/false - if an existing code was replaced>
        ------------------------
        ```
        Where:
          * `[CHANGE_ID]`: Unique ID for this change event (incrementing counter).
          *  `<timestamp>`: Time when the code change was detected and recorded.
           * `<file path>`: Path of the file that was changed.
           * `<component ID>`: ID of the component affected by the change.
           * `<change type>`: Indicates whether the component is new, modified or deleted.
            *  `<summary of the change>`: Brief explanation of why the change was necessary or what was modified.
           *  `<code diff>`: The output of `git diff` (or a similar tool), showing the differences between versions of the file.
           * `<test status - pass/fail>`: output from `ctest` after building. If the test status is `fail` then the code must be reverted.
           * `<performance Impact>`: Performance metrics of the changed module.
           * `Existing Logic Maintained`: Whether existing logic was kept untouched.

*   **Version Control:** This document should be updated with versioning information and the change history should be recorded for every update to the document itself by adding this entry to code_tracking.txt with ID `TRACKING_DOC_UPDATE` and diff of the original vs the current tracking document.



**VI. ASCII Diagram (Updated)**

```
+---------------------------+
|    SimpLang Project       |
+---------------------------+
         /        |        \
       /          |          \
+-----+----+  +-------+  +--------+
| Compiler  |  Runtime |   Tests  |
+-----+----+  +-------+  +--------+
     |          /      \     |
   +---+       /        \  +------+
   |Lex|     +-----+   +---+ Bench |
   |Par|     | Core|  |Debug|    |
   |AST|     +-----+   +---+ +----+
   |TYP|
   |SIMD|
   |IRGen|
   |Emitt|
   +----+
```

