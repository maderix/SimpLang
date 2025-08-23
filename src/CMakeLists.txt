set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate lexer and parser
BISON_TARGET(Parser parser.y ${CMAKE_CURRENT_BINARY_DIR}/parser.cpp
             DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/parser.hpp)
FLEX_TARGET(Scanner lexer.l ${CMAKE_CURRENT_BINARY_DIR}/lexer.cpp)
ADD_FLEX_BISON_DEPENDENCY(Scanner Parser)

# Add executable
add_executable(simplang
    main.cpp
    ast.cpp
    codegen.cpp
    simd_ops.cpp
    simd_interface.cpp
    logger.cpp
    
    ${BISON_Parser_OUTPUTS}
    ${FLEX_Scanner_OUTPUTS}
)

# Include directories
target_include_directories(simplang PRIVATE
    ${CMAKE_SOURCE_DIR}/include
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
)

target_link_libraries(simplang ${llvm_libs} simplang_runtime)
