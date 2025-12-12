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
option(USE_MLIR "Enable MLIR integration" OFF)
option(USE_TCMALLOC "Use tcmalloc as default allocator" ON)

# LLVM setup
# LLVM_DIR should be set via -DLLVM_DIR=... when running cmake
# build.sh will set this to the correct location
# Force static linking to avoid command-line option conflicts with MLIR static libs
set(LLVM_LINK_LLVM_DYLIB OFF CACHE BOOL "Link against shared LLVM library" FORCE)
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

# tcmalloc setup (for better memory allocation performance)
if(USE_TCMALLOC AND NOT CMAKE_CROSSCOMPILING)
    find_library(TCMALLOC_LIBRARY NAMES tcmalloc_minimal tcmalloc)
    if(TCMALLOC_LIBRARY)
        set(TCMALLOC_FOUND TRUE)
        message(STATUS "Found tcmalloc: ${TCMALLOC_LIBRARY}")
    else()
        set(TCMALLOC_FOUND FALSE)
        message(WARNING "tcmalloc not found, using system malloc. Install with: apt install libtcmalloc-minimal4 libgoogle-perftools-dev")
    endif()
else()
    set(TCMALLOC_FOUND FALSE)
    if(CMAKE_CROSSCOMPILING)
        message(STATUS "tcmalloc disabled for cross-compilation")
    endif()
endif()

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Found readline: ${READLINE_LIBRARY}")

# MLIR setup (optional, controlled by USE_MLIR option)
if(USE_MLIR)
    # MLIR_DIR should be set via -DMLIR_DIR=... when running cmake
    # build.sh will set this to the correct location
    find_package(MLIR REQUIRED CONFIG)

    message(STATUS "MLIR integration ENABLED")
    message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

    # Add MLIR include directories
    include_directories(
        ${MLIR_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include/mlir
        ${CMAKE_BINARY_DIR}/src/mlir  # For generated .inc files
    )

    # MLIR requires these definitions
    add_definitions(${MLIR_DEFINITIONS})

    # Include MLIR's CMake modules for mlir_tablegen()
    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
    include(TableGen)
    include(AddLLVM)
    include(AddMLIR)
    include(HandleLLVMOptions)

    # For cross-compilation, create dummy targets for missing dependencies
    if(CMAKE_CROSSCOMPILING)
        include(${CMAKE_SOURCE_DIR}/cmake/CrossCompileDeps.cmake)
    endif()
else()
    message(STATUS "MLIR integration DISABLED (use -DUSE_MLIR=ON to enable)")
endif()

# CUDA GPU backend setup (optional, requires USE_MLIR=ON)
option(USE_CUDA "Enable CUDA GPU backend" OFF)
set(CUDA_ARCH "80" CACHE STRING "CUDA compute capability (e.g., 70 for V100, 80 for A100, 90 for H100)")

if(USE_CUDA)
    if(NOT USE_MLIR)
        message(FATAL_ERROR "USE_CUDA requires USE_MLIR=ON. CUDA backend uses MLIR GPU dialect.")
    endif()

    # Find CUDA Toolkit
    find_package(CUDAToolkit REQUIRED)

    message(STATUS "CUDA GPU backend ENABLED")
    message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA Toolkit path: ${CUDAToolkit_LIBRARY_DIR}")
    message(STATUS "Target CUDA architecture: sm_${CUDA_ARCH}")

    # Add CUDA include directories
    include_directories(${CUDAToolkit_INCLUDE_DIRS})

    # Define CUDA-related macros
    add_definitions(-DUSE_CUDA)
    add_definitions(-DCUDA_ARCH=${CUDA_ARCH})

    # MLIR GPU dialect libraries (will be linked in src/mlir/CMakeLists.txt)
    set(MLIR_GPU_LIBS
        MLIRGPUDialect
        MLIRGPUTransforms
        MLIRGPUToNVVMTransforms
        MLIRGPUToGPURuntimeTransforms
        MLIRNVVMIR
        MLIRNVVMToLLVMIRTranslation
        CACHE INTERNAL "MLIR GPU libraries"
    )
else()
    message(STATUS "CUDA GPU backend DISABLED (use -DUSE_CUDA=ON to enable)")
endif()

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
add_subdirectory(simpblas)   # SimpBLAS optimized kernels library
add_subdirectory(runtime)    # Runtime library and utilities first
add_subdirectory(src)        # SimpleLang compiler

# Add MLIR dialect if enabled
if(USE_MLIR)
    add_subdirectory(src/mlir)  # MLIR dialect implementation
endif()

add_subdirectory(tests)      # Tests directory
add_subdirectory(tests/benchmark)  # Benchmarks
add_subdirectory(simptensor) # SimpTensor library tests
add_subdirectory(mobilenet_poc) # MobileNet POC demonstration

# Update build_all dependencies
add_dependencies(build_all 
    simpblas         # SimpBLAS optimized kernels library
    simplang 
    simplang_runtime 
    host_runner 
    # test_simd_runner  # DEPRECATED: old SIMD implementation
    # debug_test_runner # DEPRECATED: old SIMD implementation
    simplang_runner
)

# Add the runner executable
add_executable(simplang_runner
    runtime/src/host.cpp
)

# Link tcmalloc first to override malloc
if(TCMALLOC_FOUND)
    target_link_libraries(simplang_runner PRIVATE ${TCMALLOC_LIBRARY})
endif()

# Link against required libraries
target_link_libraries(simplang_runner
    PRIVATE
        simplang_runtime
        ${CMAKE_DL_LIBS}
)

# Add readline libraries only if not cross-compiling
if(NOT CMAKE_CROSSCOMPILING)
    target_link_libraries(simplang_runner PRIVATE ${READLINE_LIBRARIES})
endif()

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