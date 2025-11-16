# ARM Cross-Compilation Toolchain for Raspberry Pi
# Usage: cmake -DCMAKE_TOOLCHAIN_FILE=cmake/arm-toolchain.cmake ..

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

# Detect which ARM variant to use
if(NOT DEFINED ARM_ARCH)
    # Default to ARMv8 64-bit (Raspberry Pi 3/4/5)
    set(ARM_ARCH "aarch64" CACHE STRING "ARM architecture: armv7l or aarch64")
endif()

if(ARM_ARCH STREQUAL "aarch64")
    # ARM 64-bit (Raspberry Pi 3/4/5 in 64-bit mode)
    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
    set(CMAKE_SYSTEM_PROCESSOR aarch64)
    set(TRIPLE aarch64-linux-gnu)
elseif(ARM_ARCH STREQUAL "armv7l")
    # ARM 32-bit (Raspberry Pi 2/3/4 in 32-bit mode)
    set(CMAKE_C_COMPILER arm-linux-gnueabihf-gcc)
    set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++)
    set(CMAKE_SYSTEM_PROCESSOR armv7l)
    set(TRIPLE arm-linux-gnueabihf)
else()
    message(FATAL_ERROR "Unknown ARM_ARCH: ${ARM_ARCH}. Use 'aarch64' or 'armv7l'")
endif()

# Where is the target environment
set(CMAKE_FIND_ROOT_PATH /usr/${TRIPLE})
set(CMAKE_SYSROOT /usr/${TRIPLE})

# Search for programs in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# For libraries and headers in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Compiler flags for ARM optimization
if(ARM_ARCH STREQUAL "aarch64")
    # ARMv8 64-bit flags
    set(CMAKE_C_FLAGS_INIT "-march=armv8-a -mtune=cortex-a53")
    set(CMAKE_CXX_FLAGS_INIT "-march=armv8-a -mtune=cortex-a53")
else()
    # ARMv7 32-bit flags
    set(CMAKE_C_FLAGS_INIT "-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard")
    set(CMAKE_CXX_FLAGS_INIT "-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard")
endif()

# Enable NEON SIMD
add_definitions(-DARM_NEON_ENABLED)

message(STATUS "ARM Cross-Compilation Toolchain Configured")
message(STATUS "  Architecture: ${ARM_ARCH}")
message(STATUS "  Triple: ${TRIPLE}")
message(STATUS "  C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "  CXX Compiler: ${CMAKE_CXX_COMPILER}")
