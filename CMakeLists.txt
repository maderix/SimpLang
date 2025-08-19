# Root CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(SimpleLang)

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

# Add cmake modules path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# Options
option(SIMD_DEBUG "Enable SIMD debugging" OFF)
option(ENABLE_DEBUGGER "Enable kernel debugger" ON)
option(SB_ENABLE_AVX512 "Enable AVX-512 kernels" OFF)
option(SB_ENABLE_NEON "Enable NEON kernels" ON)
option(SB_BUILD_SHARED "Build shared library" OFF)

# LLVM setup
find_package(LLVM REQUIRED CONFIG)
find_package(BISON REQUIRED)
find_package(FLEX REQUIRED)
find_package(Threads REQUIRED)

# Handle readline dependency
include(FindPkgConfig)
pkg_check_modules(READLINE readline)
if(NOT READLINE_FOUND)
    find_path(READLINE_INCLUDE_DIR readline/readline.h
        PATHS /usr/include /usr/local/include)
    find_library(READLINE_LIBRARY NAMES readline
        PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)
    if(READLINE_LIBRARY AND READLINE_INCLUDE_DIR)
        set(READLINE_FOUND TRUE)
        set(READLINE_LIBRARIES ${READLINE_LIBRARY})
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
    ${CMAKE_SOURCE_DIR}/include
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

# Build simpblas library
set(SIMPBLAS_SOURCES
    src/init.c
    src/common/cpuid.c
    src/scalar/kernels_f32.c
)

# Add AVX2 sources if supported
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    list(APPEND SIMPBLAS_SOURCES src/x86/avx2/kernels_f32.c)
endif()

# Create static library
add_library(simpblas STATIC ${SIMPBLAS_SOURCES})
target_include_directories(simpblas 
    PUBLIC 
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

# Add compiler-specific optimizations
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(simpblas PRIVATE -O3 -march=native -mavx2 -mfma)
endif()

# Conditionally enable SIMD
if(SB_ENABLE_AVX512)
    target_compile_definitions(simpblas PRIVATE -DSB_ENABLE_AVX512)
endif()

if(SB_ENABLE_NEON AND (CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64"))
    target_compile_definitions(simpblas PRIVATE -DSB_ENABLE_NEON)
endif()

# Create shared library if requested
if(SB_BUILD_SHARED)
    add_library(simpblas_shared SHARED ${SIMPBLAS_SOURCES})
    target_include_directories(simpblas_shared 
        PUBLIC 
            ${CMAKE_SOURCE_DIR}/include
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )
    set_target_properties(simpblas_shared PROPERTIES OUTPUT_NAME simpblas)
    
    if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(simpblas_shared PRIVATE -O3 -march=native -fPIC)
    endif()
endif()

# Update build_all dependencies
add_dependencies(build_all 
    simplang 
    simplang_runtime 
    host_runner 
    test_simd_runner 
    debug_test_runner
    simplang_runner
    simpblas
)

# Add the runner executable
add_executable(simplang_runner
    runtime/src/host.cpp
)

# Link against required libraries
target_link_libraries(simplang_runner
    PRIVATE 
        simplang_runtime
        simpblas
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
        ${CMAKE_SOURCE_DIR}/include
)

# Install the runner
install(TARGETS simplang_runner
    RUNTIME DESTINATION bin
)

# Install simpblas
install(TARGETS simpblas
    ARCHIVE DESTINATION lib
)

install(FILES include/simpblas.h
    DESTINATION include
)

if(SB_BUILD_SHARED)
    install(TARGETS simpblas_shared
        LIBRARY DESTINATION lib
    )
endif()

# Optional: Add test that uses the runner
add_test(
    NAME runner_test
    COMMAND simplang_runner ${CMAKE_BINARY_DIR}/tests/obj/test_arithmetic.so
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)