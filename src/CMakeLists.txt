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

# Add executable
add_executable(simplang
    main.cpp
    ${AST_SOURCES}
    codegen.cpp
    simd_ops.cpp
    simd_interface.cpp
    avx512_backend.cpp
    logger.cpp

    ${BISON_Parser_OUTPUTS}
    ${FLEX_Scanner_OUTPUTS}
)

# Include directories
target_include_directories(simplang PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/ast
    ${CMAKE_CURRENT_BINARY_DIR}
    ${LLVM_INCLUDE_DIRS}
)

# Link against LLVM libraries
llvm_map_components_to_libnames(llvm_libs
    Core
    Support
    native
    nativecodegen
    x86codegen
    x86asmparser
    x86info
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
