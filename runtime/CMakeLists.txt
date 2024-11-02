cmake_minimum_required(VERSION 3.10)

# Debug options
option(SIMD_DEBUG "Enable SIMD debug output" OFF)
option(ENABLE_DEBUGGER "Enable kernel debugger" ON)

# Configure debug definitions
if(SIMD_DEBUG)
    add_definitions(-DSIMD_DEBUG)
endif()

if(ENABLE_DEBUGGER)
    add_definitions(-DENABLE_DEBUGGER)
endif()

# Define source files for debugger if enabled
if(ENABLE_DEBUGGER)
    set(DEBUGGER_SOURCES
        src/kernel_debugger/breakpoint.cpp
        src/kernel_debugger/event_logger.cpp
        src/kernel_debugger/memory_tracker.cpp
        src/kernel_debugger/kernel_debugger.cpp
    )
endif()

# Runtime library
add_library(simplang_runtime STATIC
    src/kernel_runtime.cpp
    ${DEBUGGER_SOURCES}
)

# Configure include directories
target_include_directories(simplang_runtime 
    PUBLIC 
        include
        ${CMAKE_SOURCE_DIR}/include
        $<$<BOOL:${ENABLE_DEBUGGER}>:${CMAKE_SOURCE_DIR}/include/kernel_debugger>
)

# Configure compile options
target_compile_options(simplang_runtime
    PRIVATE 
        -mavx512f 
        -msse4.2
        $<$<CONFIG:Debug>:-g -O0>
        $<$<CONFIG:Release>:-O3>
)

# Host runner library
add_library(host_runner STATIC
    src/host.cpp
)

target_link_libraries(host_runner
    PUBLIC simplang_runtime
)

target_include_directories(host_runner
    PUBLIC 
        include
        ${CMAKE_SOURCE_DIR}/include
        $<$<BOOL:${ENABLE_DEBUGGER}>:${CMAKE_SOURCE_DIR}/include/kernel_debugger>
)

target_compile_options(host_runner
    PRIVATE 
        -mavx512f 
        -msse4.2
        $<$<CONFIG:Debug>:-g -O0>
        $<$<CONFIG:Release>:-O3>
)

# Install rules
install(TARGETS simplang_runtime host_runner
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

if(ENABLE_DEBUGGER)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/kernel_debugger/
        DESTINATION include/kernel_debugger
        FILES_MATCHING PATTERN "*.hpp"
    )
endif()