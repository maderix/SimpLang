# SimpTensor Tests CMakeLists.txt
enable_testing()

# Create version script for kernel symbol exports  
file(WRITE ${CMAKE_BINARY_DIR}/simptensor/kernel.version
"{\n    global:\n        kernel_main;\n    local:\n        *;\n};\n")

# Create simptensor output directories
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/simptensor/obj)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/simptensor/ir)

# Create all_simptensor_runners target
add_custom_target(all_simptensor_runners)

# Function to add a simptensor test
function(add_simptensor_test name sl_file expected_result)
    # Get filename without .sl extension
    get_filename_component(base_name ${sl_file} NAME_WE)
    
    # Define output files in organized directories
    set(obj_file "${CMAKE_BINARY_DIR}/simptensor/obj/${base_name}.o")
    set(so_file "${CMAKE_BINARY_DIR}/simptensor/obj/${base_name}.so")
    
    message(STATUS "Processing simptensor test ${name}:")
    message(STATUS "  Source: ${sl_file}")
    message(STATUS "  Object: ${obj_file}")
    message(STATUS "  Shared: ${so_file}")
    message(STATUS "  Expected Result: ${expected_result}")
    
    # First generate a temporary object file
    set(temp_obj "${CMAKE_BINARY_DIR}/simptensor/obj/${base_name}_temp.o")
    add_custom_command(
        OUTPUT ${temp_obj}
        COMMAND ${CMAKE_BINARY_DIR}/src/simplang 
                ${CMAKE_CURRENT_SOURCE_DIR}/${sl_file}
                -o ${temp_obj}
                -v
                -fPIC
        DEPENDS simplang ${CMAKE_CURRENT_SOURCE_DIR}/${sl_file}
        COMMENT "Compiling ${sl_file}"
        VERBATIM
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # Then create object file with proper symbol visibility
    add_custom_command(
        OUTPUT ${obj_file}
        COMMAND ${CMAKE_C_COMPILER} -r ${temp_obj} -o ${obj_file}
                -Wl,--version-script=${CMAKE_BINARY_DIR}/simptensor/kernel.version
                -Wl,--export-dynamic-symbol=kernel_main
                -fvisibility=default
        DEPENDS ${temp_obj} ${CMAKE_BINARY_DIR}/simptensor/kernel.version
        COMMENT "Creating object file for ${base_name}"
        VERBATIM
    )

    # Create shared library with explicit exports
    add_custom_command(
        OUTPUT ${so_file}
        COMMAND ${CMAKE_C_COMPILER} -shared ${obj_file} -o ${so_file} 
                -Wl,--export-dynamic
                -Wl,--version-script=${CMAKE_BINARY_DIR}/simptensor/kernel.version
                -Wl,--export-dynamic-symbol=kernel_main
                -fvisibility=default
        DEPENDS ${obj_file} ${CMAKE_BINARY_DIR}/simptensor/kernel.version
        COMMENT "Creating shared library for ${base_name}"
        VERBATIM
    )

    add_custom_target(${name}_obj DEPENDS ${so_file})

    # Create test runner executable
    add_executable(${name}_runner 
        ${CMAKE_CURRENT_SOURCE_DIR}/${name}_host.cpp
    )

    # Link against kernel runner and other dependencies
    target_link_libraries(${name}_runner
        PRIVATE 
            ${CMAKE_DL_LIBS}
    )

    # Set include directories
    target_include_directories(${name}_runner
        PRIVATE
            ${CMAKE_SOURCE_DIR}/runtime/include
    )

    # Add the runner to the all_simptensor_runners target
    add_dependencies(all_simptensor_runners ${name}_runner)
    add_dependencies(${name}_runner ${name}_obj)

    # Add CTest test
    add_test(
        NAME ${name}
        COMMAND ${name}_runner ${so_file}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/simptensor
    )
    
    # Dump symbols for debugging
    add_custom_command(TARGET ${name}_obj POST_BUILD
        COMMAND echo "Dumping symbols for ${obj_file}"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/simptensor
    )
    
endfunction()

# Add simptensor tests
add_simptensor_test(test_tensor_basic test_tensor_basic.sl 28.0)
add_simptensor_test(tensor_perf tensor_perf.sl 0.0) # Performance test - result varies