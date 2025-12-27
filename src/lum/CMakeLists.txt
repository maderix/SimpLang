# Lum DSL Compiler - Transform Dialect Code Generator
# Generates MLIR Transform Dialect from .lum schedule files

find_package(BISON REQUIRED)
find_package(FLEX REQUIRED)

# Generate lexer and parser
BISON_TARGET(LumParser
    ${CMAKE_CURRENT_SOURCE_DIR}/parser.y
    ${CMAKE_CURRENT_BINARY_DIR}/parser.cpp
    DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/parser.hpp
)

FLEX_TARGET(LumLexer
    ${CMAKE_CURRENT_SOURCE_DIR}/lexer.l
    ${CMAKE_CURRENT_BINARY_DIR}/lexer.cpp
)

ADD_FLEX_BISON_DEPENDENCY(LumLexer LumParser)

# Collect AST sources
set(LUM_AST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/ast/pattern.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ast/schedule.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ast/transform.cpp
)

# Collect codegen sources
set(LUM_CODEGEN_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen/transform_emitter.cpp
)

# All sources
set(LUM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${LUM_AST_SOURCES}
    ${LUM_CODEGEN_SOURCES}
    ${BISON_LumParser_OUTPUTS}
    ${FLEX_LumLexer_OUTPUTS}
)

# Create lum executable
add_executable(lum ${LUM_SOURCES})

# Include directories
target_include_directories(lum PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${LLVM_INCLUDE_DIRS}
)

# Add MLIR include directories if available
if(MLIR_INCLUDE_DIRS)
    target_include_directories(lum PRIVATE ${MLIR_INCLUDE_DIRS})
endif()

# Link MLIR libraries for Transform Dialect
# Note: For MVP, we generate text output. Full MLIR linking is for future phases.
target_link_libraries(lum PRIVATE
    ${llvm_libs}
)

# Compiler flags
target_compile_features(lum PRIVATE cxx_std_17)
target_compile_options(lum PRIVATE -Wall -Wextra)

# Install
install(TARGETS lum DESTINATION bin)
