# Runtime CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find required packages
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(READLINE readline REQUIRED)

# Source files
set(RUNTIME_SOURCES
    src/host.cpp
    src/kernel_runtime.cpp
    src/kernel_debugger/kernel_debugger.cpp
    src/kernel_debugger/command_processor.cpp
    src/kernel_debugger/memory_tracker.cpp
    src/kernel_debugger/event_logger.cpp
    src/kernel_debugger/ui_helper.cpp
    src/kernel_debugger/source_manager.cpp
    src/kernel_debugger/call_stack.cpp
    src/kernel_debugger/breakpoint.cpp
    src/kernel_debugger/debug_events.cpp
    src/kernel_debugger/config.cpp
)

# Add runner source
set(RUNNER_SOURCES
    src/host.cpp
)

# Create runtime library first
add_library(simplang_runtime SHARED ${RUNTIME_SOURCES} ${SIMD_SOURCES} ${DEBUG_SOURCES})

# Include directories
target_include_directories(simplang_runtime
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include/kernel_debugger
        ${READLINE_INCLUDE_DIRS}
)

# Compiler definitions
target_compile_definitions(simplang_runtime
    PRIVATE
        # Debug-related definitions
        $<$<CONFIG:Debug>:SIMD_DEBUG=1>
        $<$<BOOL:${ENABLE_DEBUGGER}>:ENABLE_DEBUGGER=1>
        $<$<BOOL:${ENABLE_PROFILING}>:ENABLE_PROFILING=1>
)

# Compiler options
target_compile_options(simplang_runtime
    PRIVATE
        -Wall
        -Wextra
        -mavx512f
        -msse4.2
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Debug>:-O0>
)

# Link libraries
target_link_libraries(simplang_runtime
    PRIVATE
        ${READLINE_LIBRARIES}
        readline
        history
        Threads::Threads
        ${CMAKE_DL_LIBS}
)

# Create host runner library (used by tests)
add_library(host_runner SHARED src/host.cpp)
target_link_libraries(host_runner
    PRIVATE
        simplang_runtime
        ${CMAKE_DL_LIBS}
        ${READLINE_LIBRARIES}
        readline
        history
)

target_include_directories(host_runner
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include
        ${READLINE_INCLUDE_DIRS}
)

target_compile_options(host_runner
    PRIVATE
        -Wall
        -Wextra
        -mavx512f
        -msse4.2
)

# Install rules
install(TARGETS simplang_runtime host_runner
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)