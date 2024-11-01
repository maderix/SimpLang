cmake_minimum_required(VERSION 3.10)

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

# Host runner executable
add_executable(host_runner
    src/host.cpp
)

target_link_libraries(host_runner
    PRIVATE simplang_runtime
)

target_compile_options(host_runner
    PRIVATE -mavx512f -msse4.2
)