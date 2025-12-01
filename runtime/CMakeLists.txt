# Runtime CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find required packages
find_package(Threads REQUIRED)

# Readline only needed for debugger (x86 only)
if(NOT CMAKE_CROSSCOMPILING)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(READLINE readline REQUIRED)
endif()

# Source files
set(RUNTIME_SOURCES
    src/host.cpp
    src/kernel_runtime.cpp
)

# Add debugger sources only if enabled
if(ENABLE_DEBUGGER)
    list(APPEND RUNTIME_SOURCES
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
endif()

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
)

# Add readline include dirs only if not cross-compiling
if(NOT CMAKE_CROSSCOMPILING)
    target_include_directories(simplang_runtime PRIVATE ${READLINE_INCLUDE_DIRS})
endif()

# Add debugger include directory only if enabled
if(ENABLE_DEBUGGER)
    target_include_directories(simplang_runtime
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include/kernel_debugger
    )
endif()

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
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Debug>:-O0>
)

# Add architecture-specific SIMD flags
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686")
    # x86 SIMD flags
    target_compile_options(simplang_runtime PRIVATE -mavx512f -msse4.2)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
    # ARM64 NEON flags (set by toolchain, just verify)
    target_compile_options(simplang_runtime PRIVATE -march=armv8-a)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7")
    # ARM32 NEON flags
    target_compile_options(simplang_runtime PRIVATE -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard)
endif()

# Link libraries (tcmalloc first to override malloc)
if(TCMALLOC_FOUND)
    target_link_libraries(simplang_runtime PRIVATE ${TCMALLOC_LIBRARY})
endif()

target_link_libraries(simplang_runtime
    PRIVATE
        Threads::Threads
        ${CMAKE_DL_LIBS}
)

# Add readline libraries only if not cross-compiling
if(NOT CMAKE_CROSSCOMPILING)
    target_link_libraries(simplang_runtime PRIVATE ${READLINE_LIBRARIES} readline history)
endif()

# Create host runner library (used by tests)
add_library(host_runner SHARED src/host.cpp)

# Link tcmalloc first to override malloc
if(TCMALLOC_FOUND)
    target_link_libraries(host_runner PRIVATE ${TCMALLOC_LIBRARY})
endif()

target_link_libraries(host_runner
    PRIVATE
        simplang_runtime
        ${CMAKE_DL_LIBS}
)

# Add readline libraries only if not cross-compiling
if(NOT CMAKE_CROSSCOMPILING)
    target_link_libraries(host_runner PRIVATE ${READLINE_LIBRARIES} readline history)
endif()

target_include_directories(host_runner
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include
)

# Add readline include dirs only if not cross-compiling
if(NOT CMAKE_CROSSCOMPILING)
    target_include_directories(host_runner PRIVATE ${READLINE_INCLUDE_DIRS})
endif()

target_compile_options(host_runner
    PRIVATE
        -Wall
        -Wextra
)

# Add architecture-specific SIMD flags
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686")
    # x86 SIMD flags
    target_compile_options(host_runner PRIVATE -mavx512f -msse4.2)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
    # ARM64 NEON flags (set by toolchain, just verify)
    target_compile_options(host_runner PRIVATE -march=armv8-a)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7")
    # ARM32 NEON flags
    target_compile_options(host_runner PRIVATE -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard)
endif()

# Install rules
install(TARGETS simplang_runtime host_runner
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)