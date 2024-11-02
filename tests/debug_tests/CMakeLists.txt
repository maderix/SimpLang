# Debug tests CMakeLists.txt

# Add SimpleLang debug test
add_sl_test(debug_simd test_simd_debug.sl)

# Create debug test runner
add_executable(debug_test_runner
    debug_test_runner.cpp
)

target_link_libraries(debug_test_runner
    PRIVATE 
        simplang_runtime
        debug_simd_obj
)

target_include_directories(debug_test_runner
    PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/kernel_debugger
        ${CMAKE_SOURCE_DIR}/runtime/include
)

target_compile_options(debug_test_runner
    PRIVATE 
        -mavx512f 
        -msse4.2
        $<$<CONFIG:Debug>:-g -O0>
        $<$<CONFIG:Release>:-O3>
)

# Add test to CTest
add_test(NAME debug_test 
         COMMAND debug_test_runner
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# Create custom target to run debug tests
add_custom_target(run_debug_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -R "debug_test" --output-on-failure
    DEPENDS debug_test_runner
    COMMENT "Running SimpleLang debug tests"
)