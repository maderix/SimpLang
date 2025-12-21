# Diagnostics library for SimpLang compiler
# Provides Rust-style error reporting with source snippets, colored output, and suggestions

add_library(simplang_diagnostics STATIC
    span.cpp
    source_manager.cpp
    error_codes.cpp
    diagnostic.cpp
    diagnostic_builder.cpp
    diagnostic_engine.cpp
    diagnostic_renderer.cpp
)

target_include_directories(simplang_diagnostics
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

target_compile_features(simplang_diagnostics PUBLIC cxx_std_17)

# Enable warnings
target_compile_options(simplang_diagnostics PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)
