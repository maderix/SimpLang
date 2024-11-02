cmake_minimum_required(VERSION 3.10)

# Debug option
option(SIMD_DEBUG "Enable SIMD debug output" OFF)

if(SIMD_DEBUG)
    add_definitions(-DSIMD_DEBUG)
endif()

# Runtime library
add_library(simplang_runtime STATIC
    src/kernel_runtime.cpp
)

target_include_directories(simplang_runtime 
    PUBLIC include
)

target_compile_options(simplang_runtime
    PRIVATE -mavx512f -msse4.2
)

# Host runner library (instead of executable)
add_library(host_runner STATIC
    src/host.cpp
)

target_link_libraries(host_runner
    PUBLIC simplang_runtime
)

target_compile_options(host_runner
    PRIVATE -mavx512f -msse4.2
)