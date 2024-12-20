# Debug test specific section
if(ENABLE_DEBUGGER)
    # Find readline
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(READLINE readline REQUIRED)

    # Create debug test executable
    add_executable(debug_test_runner
        debug_tests/debug_test_runner.cpp
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
            ${READLINE_LIBRARIES}  # Add pkg-config libraries
            readline               # Explicitly link readline
            history               # Explicitly link history
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

    message(STATUS "SIMD debug test configured with readline: ${READLINE_LIBRARIES}")
endif()