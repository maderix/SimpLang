set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate lexer and parser
BISON_TARGET(Parser parser.y ${CMAKE_CURRENT_BINARY_DIR}/parser.cpp
             DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/parser.hpp)
FLEX_TARGET(Scanner lexer.l ${CMAKE_CURRENT_BINARY_DIR}/lexer.cpp)
ADD_FLEX_BISON_DEPENDENCY(Scanner Parser)

# Collect AST source files
file(GLOB_RECURSE AST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/ast/*.cpp
)

# Common sources
set(SIMPLANG_SOURCES
    main.cpp
    ${AST_SOURCES}
    codegen.cpp
    simd_ops.cpp
    simd_interface.cpp
    logger.cpp
    ${BISON_Parser_OUTPUTS}
    ${FLEX_Scanner_OUTPUTS}
)

# Add x86-specific sources
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686")
    list(APPEND SIMPLANG_SOURCES avx512_backend.cpp)
endif()

# Add executable
add_executable(simplang ${SIMPLANG_SOURCES})

# Include directories
target_include_directories(simplang PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/ast
    ${CMAKE_CURRENT_BINARY_DIR}
    ${LLVM_INCLUDE_DIRS}
)

# Link against LLVM libraries
set(LLVM_COMMON_COMPONENTS
    Core
    Support
    TransformUtils
    ScalarOpts
    InstCombine
    Analysis
    Target
    MC
    Object
    BitWriter
    Vectorize
)

# Add architecture-specific components for cross-compilation support
# Only include targets that are available in the LLVM build
set(LLVM_ARCH_COMPONENTS)

# X86 target (almost always available)
if("X86" IN_LIST LLVM_TARGETS_TO_BUILD)
    list(APPEND LLVM_ARCH_COMPONENTS X86CodeGen X86AsmParser X86Desc X86Info)
endif()

# ARM 64-bit target
if("AArch64" IN_LIST LLVM_TARGETS_TO_BUILD)
    list(APPEND LLVM_ARCH_COMPONENTS AArch64CodeGen AArch64AsmParser AArch64Desc AArch64Info)
endif()

# ARM 32-bit target (optional)
if("ARM" IN_LIST LLVM_TARGETS_TO_BUILD)
    list(APPEND LLVM_ARCH_COMPONENTS ARMCodeGen ARMAsmParser ARMDesc ARMInfo)
endif()

llvm_map_components_to_libnames(llvm_libs
    ${LLVM_COMMON_COMPONENTS}
    ${LLVM_ARCH_COMPONENTS}
)

# Link tcmalloc first to override malloc (must be before other libs)
if(TCMALLOC_FOUND)
    target_link_libraries(simplang ${TCMALLOC_LIBRARY})
endif()

target_link_libraries(simplang ${llvm_libs} simplang_runtime)

# Add MLIR support if enabled
if(USE_MLIR)
    target_compile_definitions(simplang PRIVATE USE_MLIR)
    target_include_directories(simplang PRIVATE
        ${CMAKE_SOURCE_DIR}/include/mlir
        ${CMAKE_BINARY_DIR}/src/mlir  # For generated .inc files
        ${MLIR_INCLUDE_DIRS}
    )
    target_link_libraries(simplang
        simp_dialect
        MLIRExecutionEngine  # For makeOptimizingTransformer and initializeLLVMPasses
    )
endif()
