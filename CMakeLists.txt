# Root CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(SimpleLang)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

# Add cmake modules path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# Options
option(SIMD_DEBUG "Enable SIMD debugging" OFF)
option(ENABLE_DEBUGGER "Enable kernel debugger" ON)

# LLVM setup
find_package(LLVM REQUIRED CONFIG)
find_package(BISON REQUIRED)
find_package(FLEX REQUIRED)
find_package(Threads REQUIRED)

# Handle readline dependency
include(FindPkgConfig)
pkg_check_modules(READLINE readline)
if(NOT READLINE_FOUND)
    find_path(READLINE_INCLUDE_DIR readline/readline.h)
    find_library(READLINE_LIBRARY NAMES readline)
    if(READLINE_LIBRARY AND READLINE_INCLUDE_DIR)
        set(READLINE_FOUND TRUE)
    endif()
endif()

if(NOT READLINE_FOUND)
    message(FATAL_ERROR "readline library not found. Install libreadline-dev or equivalent.")
endif()

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Found readline: ${READLINE_LIBRARY}")

include_directories(
    ${LLVM_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include/kernel_debugger
    ${CMAKE_SOURCE_DIR}/runtime/include
    ${READLINE_INCLUDE_DIR}
)

add_definitions(${LLVM_DEFINITIONS})

# Create build_all target first
add_custom_target(build_all
    COMMENT "Building all targets"
)

# Add subdirectories
add_subdirectory(runtime)    # Runtime library and utilities first
add_subdirectory(src)        # SimpleLang compiler
add_subdirectory(tests)      # Tests directory
add_subdirectory(tests/benchmark)  # Benchmarks

# Update build_all dependencies
add_dependencies(build_all 
    simplang 
    simplang_runtime 
    host_runner 
    test_simd_runner 
    debug_test_runner
    simplang_runner
)

# Add the runner executable
add_executable(simplang_runner
    runtime/src/host.cpp
)

# Link against required libraries
target_link_libraries(simplang_runner
    PRIVATE 
        simplang_runtime
        ${CMAKE_DL_LIBS}
        ${READLINE_LIBRARIES}
)

# Add compile definitions
target_compile_definitions(simplang_runner
    PRIVATE
        ENABLE_PROFILING=1
)

# Set include directories
target_include_directories(simplang_runner
    PRIVATE
        ${CMAKE_SOURCE_DIR}/runtime/include
        ${CMAKE_SOURCE_DIR}/src
)

# Install the runner
install(TARGETS simplang_runner
    RUNTIME DESTINATION bin
)

# Optional: Add test that uses the runner
add_test(
    NAME runner_test
    COMMAND simplang_runner ${CMAKE_BINARY_DIR}/tests/obj/test_arithmetic.so
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)