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

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Add subdirectories
add_subdirectory(src)        # SimpleLang compiler
add_subdirectory(runtime)    # Runtime library and utilities

# Custom targets for easy building
add_custom_target(build_kernel
    COMMAND ${CMAKE_BINARY_DIR}/src/simplang ${CMAKE_SOURCE_DIR}/tests/test_simd.sl -o ${CMAKE_BINARY_DIR}/kernel.o
    DEPENDS simplang
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Building SIMD kernel"
)

add_custom_target(build_test
    COMMAND ${CMAKE_CXX_COMPILER} 
        -o ${CMAKE_BINARY_DIR}/simd_test
        ${CMAKE_BINARY_DIR}/kernel.o
        $<TARGET_FILE:simplang_runtime>
        $<TARGET_FILE:host_runner>
        -mavx512f -msse4.2
    DEPENDS build_kernel simplang_runtime host_runner
    COMMENT "Building final test executable"
)

# Convenience target to build everything
add_custom_target(build_all
    DEPENDS build_test
    COMMENT "Building all targets"
)