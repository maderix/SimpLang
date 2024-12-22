# Tests CMakeLists.txt
enable_testing()

# Create version script for kernel symbol exports
file(WRITE ${CMAKE_BINARY_DIR}/tests/kernel.version
"{\n    global:\n        kernel_main;\n    local:\n        *;\n};\n")

# Find readline package
find_package(PkgConfig REQUIRED)
pkg_check_modules(READLINE readline REQUIRED)

# Create test output directories
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/tests/obj)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/tests/ir)

# Create all_test_runners target
add_custom_target(all_test_runners)

# Function to add a test
function(add_sl_test name sl_file expected_result)
    # Get filename without .sl extension
    get_filename_component(base_name ${sl_file} NAME_WE)
    
    # Define output files in organized directories
    set(obj_file "${CMAKE_BINARY_DIR}/tests/obj/${base_name}.o")
    set(so_file "${CMAKE_BINARY_DIR}/tests/obj/${base_name}.so")
    
    message(STATUS "Processing test ${name}:")
    message(STATUS "  Source: ${sl_file}")
    message(STATUS "  Object: ${obj_file}")
    message(STATUS "  Shared: ${so_file}")
    message(STATUS "  Expected Result: ${expected_result}")
    
    # First generate a temporary object file
    set(temp_obj "${CMAKE_BINARY_DIR}/tests/obj/${base_name}_temp.o")
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
    )

    # Then create object file with proper symbol visibility
    add_custom_command(
        OUTPUT ${obj_file}
        COMMAND ${CMAKE_C_COMPILER} -r ${temp_obj} -o ${obj_file}
                -Wl,--version-script=${CMAKE_BINARY_DIR}/tests/kernel.version
                -Wl,--export-dynamic-symbol=kernel_main
                -fvisibility=default
        DEPENDS ${temp_obj} ${CMAKE_BINARY_DIR}/tests/kernel.version
        COMMENT "Creating object file for ${base_name}"
        VERBATIM
    )

    # Create shared library with explicit exports
    add_custom_command(
        OUTPUT ${so_file}
        COMMAND ${CMAKE_C_COMPILER} -shared ${obj_file} -o ${so_file} 
                -Wl,--export-dynamic
                -Wl,--version-script=${CMAKE_BINARY_DIR}/tests/kernel.version
                -Wl,--export-dynamic-symbol=kernel_main
                -fvisibility=default
        DEPENDS ${obj_file} ${CMAKE_BINARY_DIR}/tests/kernel.version
        COMMENT "Creating shared library for ${base_name}"
        VERBATIM
    )

    add_custom_target(${name}_obj DEPENDS ${so_file})

    # Create test runner executable
    add_executable(${name}_runner 
        ${CMAKE_CURRENT_SOURCE_DIR}/${name}_host.cpp
    )

    add_dependencies(${name}_runner ${name}_obj)

    # Add to all_test_runners target
    add_dependencies(all_test_runners ${name}_runner)

    # Link with runtime library and dynamic linker
    target_link_libraries(${name}_runner
        PRIVATE 
            simplang_runtime
            ${CMAKE_DL_LIBS}
            ${READLINE_LIBRARIES}
            readline
            history
    )

    # Add the object file to linker command
    target_link_options(${name}_runner
        PRIVATE ${obj_file}
        -rdynamic
    )

    # Set include directories
    target_include_directories(${name}_runner
        PRIVATE 
            ${CMAKE_SOURCE_DIR}/runtime/include
            ${CMAKE_SOURCE_DIR}/include
            ${READLINE_INCLUDE_DIRS}
    )

    # Set compile definitions
    target_compile_definitions(${name}_runner
        PRIVATE 
            SIMD_DEBUG=1
            EXPECTED_RESULT=${expected_result}
    )

    # Set compile options
    target_compile_options(${name}_runner
        PRIVATE 
            -g
            -O0
            -mavx512f 
            -msse4.2
    )

    # Add symbol check command - output to obj directory
    add_custom_command(
        TARGET ${name}_runner POST_BUILD
        COMMAND objdump -t ${obj_file} > ${CMAKE_BINARY_DIR}/tests/obj/${name}_symbols.txt
        COMMAND nm ${obj_file} >> ${CMAKE_BINARY_DIR}/tests/obj/${name}_symbols.txt
        COMMENT "Dumping symbols for ${obj_file}"
    )

    add_test(
        NAME ${name}
        COMMAND ${name}_runner
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    )
    
    set_tests_properties(${name} PROPERTIES
        ENVIRONMENT "SIMD_DEBUG=1"
        PASS_REGULAR_EXPRESSION "Test passed with result: ${expected_result}"
    )
endfunction()

# Handle SIMD test similarly
function(add_simd_test)
    set(name "test_simd")
    set(sl_file "${CMAKE_CURRENT_SOURCE_DIR}/${name}.sl")
    set(obj_file "${CMAKE_BINARY_DIR}/tests/obj/${name}.o")
    set(so_file "${CMAKE_BINARY_DIR}/tests/obj/${name}.so")
    
    # First compile to object file
    add_custom_command(
        OUTPUT ${obj_file}
        COMMAND ${CMAKE_BINARY_DIR}/src/simplang 
                ${sl_file}
                -o ${obj_file}
                -v
                -fPIC
        DEPENDS simplang ${sl_file}
        COMMENT "Compiling SIMD test"
    )

    # Then create shared library
    add_custom_command(
        OUTPUT ${so_file}
        COMMAND ${CMAKE_C_COMPILER} -shared ${obj_file} -o ${so_file}
                -Wl,--version-script=${CMAKE_BINARY_DIR}/tests/kernel.version
                -Wl,--export-dynamic-symbol=kernel_main
        DEPENDS ${obj_file} ${CMAKE_BINARY_DIR}/tests/kernel.version
        COMMENT "Creating shared library for SIMD test"
        VERBATIM
    )

    add_custom_target(${name}_obj DEPENDS ${so_file})

    # Create SIMD test executable with SIMD-specific host
    add_executable(${name}_runner
        ${CMAKE_CURRENT_SOURCE_DIR}/test_simd_host.cpp
    )

    target_link_options(${name}_runner
        PRIVATE ${obj_file}
        -rdynamic  # Ensures symbols are exported
    )

    add_dependencies(${name}_runner ${name}_obj)

    # Add to all_test_runners target
    add_dependencies(all_test_runners ${name}_runner)

    # Configure SIMD test
    target_compile_definitions(${name}_runner
        PRIVATE 
            TEST_SIMD
            SIMD_DEBUG=1
    )

    target_link_libraries(${name}_runner
        PRIVATE 
            simplang_runtime
            ${CMAKE_DL_LIBS}
            ${READLINE_LIBRARIES}
            readline
            history
    )

    target_include_directories(${name}_runner
        PRIVATE 
            ${CMAKE_SOURCE_DIR}/runtime/include
            ${CMAKE_SOURCE_DIR}/include
            ${READLINE_INCLUDE_DIRS}
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

# Add regular tests
add_sl_test(test_arithmetic test_arithmetic.sl "72.0")
add_sl_test(test_main test_main.sl "30.0")
add_sl_test(test_loop test_loop.sl "135.0")
add_sl_test(test_fibonacci test_fibonacci.sl "110.0")
add_sl_test(test_return test_return.sl "85.0")

# Add SIMD test
add_simd_test()

# Add performance test runner to all_test_runners
add_dependencies(all_test_runners perf_test_runner)

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
            ${READLINE_INCLUDE_DIRS}
    )

    target_link_libraries(debug_test_runner
        PRIVATE 
            simplang_runtime
            ${CMAKE_DL_LIBS}
            ${READLINE_LIBRARIES}
            readline
            history
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

    # Add to all_test_runners target
    add_dependencies(all_test_runners debug_test_runner)
endif()

# Compile performance test kernel
set(PERF_OBJ_FILE "${CMAKE_BINARY_DIR}/tests/obj/perf_test.o")
add_custom_command(
    OUTPUT ${PERF_OBJ_FILE}
    COMMAND ${CMAKE_BINARY_DIR}/src/simplang 
            ${CMAKE_CURRENT_SOURCE_DIR}/perf_test.sl
            -o ${PERF_OBJ_FILE}
            -v
    DEPENDS simplang ${CMAKE_CURRENT_SOURCE_DIR}/perf_test.sl
    COMMENT "Compiling performance test kernel"
    VERBATIM
)

# Create shared library for perf test
set(PERF_SO_FILE "${CMAKE_BINARY_DIR}/tests/obj/perf_test.so")
add_custom_command(
    OUTPUT ${PERF_SO_FILE}
    COMMAND ${CMAKE_C_COMPILER} -shared ${PERF_OBJ_FILE} -o ${PERF_SO_FILE}
            -Wl,--version-script=${CMAKE_BINARY_DIR}/tests/kernel.version
            -Wl,--export-dynamic-symbol=kernel_main
    DEPENDS ${PERF_OBJ_FILE} ${CMAKE_BINARY_DIR}/tests/kernel.version
    COMMENT "Creating shared library for perf test"
    VERBATIM
)

add_custom_target(perf_test_obj DEPENDS ${PERF_SO_FILE})

# Add performance test executable
add_executable(perf_test_runner 
    ${CMAKE_CURRENT_SOURCE_DIR}/perf_test_host.cpp
)

target_link_libraries(perf_test_runner
    PRIVATE 
        simplang_runtime
        ${CMAKE_DL_LIBS}
        ${READLINE_LIBRARIES}
)

target_include_directories(perf_test_runner
    PRIVATE 
        ${CMAKE_SOURCE_DIR}/runtime/include
        ${CMAKE_SOURCE_DIR}/include
        ${READLINE_INCLUDE_DIRS}
)

target_compile_options(perf_test_runner
    PRIVATE 
        -O3  # Use optimization for fair comparison
        -g
        -mavx512f 
        -msse4.2
)

add_dependencies(perf_test_runner perf_test_obj)