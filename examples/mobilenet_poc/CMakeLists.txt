# MobileNet POC CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

# Enable testing
enable_testing()

# Create version script for kernel symbol exports
file(WRITE ${CMAKE_BINARY_DIR}/mobilenet_poc/kernel.version
"{\n    global:\n        kernel_main;\n    local:\n        *;\n};\n")

# Create mobilenet_poc output directories
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/mobilenet_poc/obj)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/mobilenet_poc/ir)

# Function to add a mobilenet_poc test
function(add_mobilenet_test name sl_file)
    # Get filename without .sl extension
    get_filename_component(base_name ${sl_file} NAME_WE)
    
    # Define output files
    set(obj_file "${CMAKE_BINARY_DIR}/mobilenet_poc/obj/${base_name}.o")
    set(so_file "${CMAKE_BINARY_DIR}/mobilenet_poc/obj/${base_name}.so")
    
    message(STATUS "Processing MobileNet test ${name}:")
    message(STATUS "  Source: ${sl_file}")
    message(STATUS "  Object: ${obj_file}")
    message(STATUS "  Shared: ${so_file}")
    
    # First generate a temporary object file
    set(temp_obj "${CMAKE_BINARY_DIR}/mobilenet_poc/obj/${base_name}_temp.o")
    add_custom_command(
        OUTPUT ${temp_obj}
        COMMAND ${CMAKE_BINARY_DIR}/src/simplang 
                ${CMAKE_CURRENT_SOURCE_DIR}/${sl_file}
                -o ${temp_obj}
                -v
                -fPIC
        DEPENDS simplang ${CMAKE_CURRENT_SOURCE_DIR}/${sl_file}
        COMMENT "Compiling ${sl_file} for MobileNet POC"
        VERBATIM
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # Create object file with proper symbol visibility
    add_custom_command(
        OUTPUT ${obj_file}
        COMMAND ${CMAKE_C_COMPILER} -r ${temp_obj} -o ${obj_file}
                -Wl,--version-script=${CMAKE_BINARY_DIR}/mobilenet_poc/kernel.version
                -Wl,--export-dynamic-symbol=kernel_main
                -fvisibility=default
        DEPENDS ${temp_obj} ${CMAKE_BINARY_DIR}/mobilenet_poc/kernel.version
        COMMENT "Creating object file for ${base_name}"
        VERBATIM
    )

    # Create shared library with explicit exports
    add_custom_command(
        OUTPUT ${so_file}
        COMMAND ${CMAKE_C_COMPILER} -shared ${obj_file} -o ${so_file} 
                -Wl,--export-dynamic
                -Wl,--version-script=${CMAKE_BINARY_DIR}/mobilenet_poc/kernel.version
                -Wl,--export-dynamic-symbol=kernel_main
                -fvisibility=default
        DEPENDS ${obj_file} ${CMAKE_BINARY_DIR}/mobilenet_poc/kernel.version
        COMMENT "Creating shared library for ${base_name}"
        VERBATIM
    )

    add_custom_target(${name}_obj DEPENDS ${so_file})

    # Create MobileNet host runner executable
    add_executable(${name}_runner 
        ${CMAKE_CURRENT_SOURCE_DIR}/${name}_host.cpp
    )

    # Link against required dependencies
    target_link_libraries(${name}_runner
        PRIVATE 
            ${CMAKE_DL_LIBS}
            simpblas  # Required SimpBLAS library
    )
    
    message(STATUS "  Linking with SimpBLAS library (required)")

    # Set include directories
    target_include_directories(${name}_runner
        PRIVATE
            ${CMAKE_SOURCE_DIR}/runtime/include
            ${CMAKE_SOURCE_DIR}/simpblas/include
    )

    # Add dependencies
    add_dependencies(${name}_runner ${name}_obj)

    # Add CTest test
    add_test(
        NAME ${name}
        COMMAND $<TARGET_FILE:${name}_runner> ${so_file}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/mobilenet_poc
    )
    
endfunction()

# Add the MobileNet POC test - DISABLED: needs nhwc_offset function implementation
# add_mobilenet_test(mobilenet_inference mobilenet_inference.sl)

# Create a combined target for all mobilenet_poc components - DISABLED
# add_custom_target(mobilenet_poc_all
#     DEPENDS mobilenet_inference_runner mobilenet_inference_obj
#     COMMENT "Building complete MobileNet POC"
# )