# Root CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(SimpleLang)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

# LLVM setup
find_package(LLVM REQUIRED CONFIG)
find_package(BISON REQUIRED)
find_package(FLEX REQUIRED)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(
    ${LLVM_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include/kernel_debugger
    ${CMAKE_SOURCE_DIR}/runtime/include
)
add_definitions(${LLVM_DEFINITIONS})

# Add subdirectories
add_subdirectory(runtime)    # Runtime library and utilities first
add_subdirectory(src)        # SimpleLang compiler
add_subdirectory(tests)      # Tests directory

# Convenience target to build everything
add_custom_target(build_all
    DEPENDS simplang simplang_runtime host_runner test_simd_runner debug_test_runner
    COMMENT "Building all targets"
)