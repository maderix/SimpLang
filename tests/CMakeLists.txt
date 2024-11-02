# Tests CMakeLists.txt
enable_testing()

# Function to add a test
function(add_sl_test name sl_file expected_result)
    # Get filename without .sl extension
    get_filename_component(base_name ${sl_file} NAME_WE)
    
    # Define output files
    set(obj_file "${CMAKE_CURRENT_BINARY_DIR}/${base_name}.o")
    set(ll_file "${CMAKE_CURRENT_BINARY_DIR}/${base_name}.ll")
    
    message(STATUS "Processing test ${name}:")
    message(STATUS "  Source: ${sl_file}")
    message(STATUS "  Object: ${obj_file}")
    message(STATUS "  LLVM IR: ${ll_file}")
    message(STATUS "  Expected Result: ${expected_result}")
    
    # Generate LLVM IR and object file
    add_custom_command(
        OUTPUT ${obj_file}
        COMMAND ${CMAKE_BINARY_DIR}/src/simplang 
                ${CMAKE_CURRENT_SOURCE_DIR}/${sl_file}
                -o ${obj_file}
                -v  # Add verbose output
        DEPENDS simplang ${CMAKE_CURRENT_SOURCE_DIR}/${sl_file}
        COMMENT "Compiling ${sl_file} to ${base_name}.o"
        VERBATIM
    )

    # Create object target
    add_custom_target(${name}_obj DEPENDS ${obj_file})

    # Create test runner executable
    add_executable(${name}_runner 
        ${CMAKE_SOURCE_DIR}/runtime/src/host.cpp
    )

    # Add the object file to linker command
    target_link_options(${name}_runner
        PRIVATE ${obj_file}
    )

    # Add dependency on object file
    add_dependencies(${name}_runner ${name}_obj)

    # Link with runtime library and dynamic linker
    target_link_libraries(${name}_runner
        PRIVATE simplang_runtime
        PRIVATE ${CMAKE_DL_LIBS}
    )

    # Set include directories
    target_include_directories(${name}_runner
        PRIVATE 
            ${CMAKE_SOURCE_DIR}/runtime/include
            ${CMAKE_SOURCE_DIR}/include
    )

    # Set compile definitions
    target_compile_definitions(${name}_runner
        PRIVATE 
            SIMD_DEBUG=1
            EXPECTED_RESULT=${expected_result}  # Pass expected result to the runner
    )

    # Set compile options
    target_compile_options(${name}_runner
        PRIVATE 
            -g  # Add debug symbols
            -O0 # No optimization
            -mavx512f 
            -msse4.2
    )

    # Add symbol check command
    add_custom_command(
        TARGET ${name}_runner POST_BUILD
        COMMAND objdump -t ${obj_file} > ${name}_symbols.txt
        COMMAND nm ${obj_file} >> ${name}_symbols.txt
        COMMENT "Dumping symbols for ${obj_file}"
    )

    # Add test to CTest with a reasonable tolerance for floating point comparison
    add_test(
        NAME ${name}
        COMMAND ${name}_runner
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    
    # Set SIMD_DEBUG environment variable and test properties
    set_tests_properties(${name} PROPERTIES
        ENVIRONMENT "SIMD_DEBUG=1"
        PASS_REGULAR_EXPRESSION "Test passed with result: ${expected_result}"
    )
endfunction()

# Add regular tests with their expected results
add_sl_test(test_arithmetic test_arithmetic.sl "130.14")
add_sl_test(test_conditions test_conditions.sl "55.0")
add_sl_test(test_loop test_loop.sl "147.0")
add_sl_test(test_fibonacci test_fibonacci.sl "110.0")
add_sl_test(test_return test_return.sl "80.0")

# Handle SIMD test separately since it has a different structure
function(add_simd_test)
    set(name "test_simd")
    set(sl_file "${CMAKE_CURRENT_SOURCE_DIR}/${name}.sl")
    set(obj_file "${CMAKE_CURRENT_BINARY_DIR}/${name}.o")
    
    # Generate object file
    add_custom_command(
        OUTPUT ${obj_file}
        COMMAND ${CMAKE_BINARY_DIR}/src/simplang 
                ${sl_file}
                -o ${obj_file}
                -v
        DEPENDS simplang ${sl_file}
        COMMENT "Compiling SIMD test"
    )

    add_custom_target(${name}_obj DEPENDS ${obj_file})

    # Create SIMD test executable
    add_executable(${name}_runner
        ${CMAKE_SOURCE_DIR}/runtime/src/host.cpp
    )

    target_link_options(${name}_runner
        PRIVATE ${obj_file}
    )

    add_dependencies(${name}_runner ${name}_obj)

    # Configure SIMD test
    target_compile_definitions(${name}_runner
        PRIVATE 
            TEST_SIMD
            SIMD_DEBUG=1
    )

    target_link_libraries(${name}_runner
        PRIVATE simplang_runtime
        PRIVATE ${CMAKE_DL_LIBS}
    )

    target_include_directories(${name}_runner
        PRIVATE 
            ${CMAKE_SOURCE_DIR}/runtime/include
            ${CMAKE_SOURCE_DIR}/include
    )

    target_compile_options(${name}_runner
        PRIVATE 
            -g
            -O0
            -mavx512f 
            -msse4.2
    )

    # Add test to CTest
    add_test(
        NAME ${name}
        COMMAND ${name}_runner
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    set_tests_properties(${name} PROPERTIES
        ENVIRONMENT "SIMD_DEBUG=1"
    )
endfunction()

# Add SIMD test
add_simd_test()

# Debug test specific section
if(ENABLE_DEBUGGER)
    # Compile the debug SimpleLang file
    set(DEBUG_SL_FILE "${CMAKE_CURRENT_SOURCE_DIR}/debug_tests/test_simd_debug.sl")
    set(DEBUG_OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/test_simd_debug.o")
    
    message(STATUS "Configuring SIMD debug test")
    message(STATUS "  Source: ${DEBUG_SL_FILE}")
    message(STATUS "  Object: ${DEBUG_OBJ_FILE}")
    
    add_custom_command(
        OUTPUT ${DEBUG_OBJ_FILE}
        COMMAND ${CMAKE_BINARY_DIR}/src/simplang 
                ${DEBUG_SL_FILE}
                -o ${DEBUG_OBJ_FILE}
                -v
        DEPENDS simplang ${DEBUG_SL_FILE}
        COMMENT "Compiling debug SimpleLang file"
        VERBATIM
    )
    
    add_custom_target(debug_test_obj DEPENDS ${DEBUG_OBJ_FILE})

    # Create debug test executable
    add_executable(debug_test_runner
        debug_tests/debug_test_runner.cpp
        ${DEBUG_OBJ_FILE}
    )

    target_compile_definitions(debug_test_runner
        PRIVATE
            SIMD_DEBUG=1
            ENABLE_DEBUGGER=1
    )

    target_include_directories(debug_test_runner
        PRIVATE 
            ${CMAKE_SOURCE_DIR}/runtime/include
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/include/kernel_debugger
    )

    target_link_libraries(debug_test_runner
        PRIVATE 
            simplang_runtime
            ${CMAKE_DL_LIBS}
    )

    target_compile_options(debug_test_runner
        PRIVATE 
            -g
            -O0
            -mavx512f 
            -msse4.2
    )

    add_dependencies(debug_test_runner 
        debug_test_obj 
        simplang_runtime
    )

    # Add to CTest
    add_test(
        NAME debug_test
        COMMAND debug_test_runner
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    
    set_tests_properties(debug_test PROPERTIES
        ENVIRONMENT "SIMD_DEBUG=1"
    )

    message(STATUS "SIMD debug test configured")
endif()