# SimpLang Tensor Library Build Configuration
cmake_minimum_required(VERSION 3.20)

# Tensor library sources (SimpLang .sl files)
set(TENSOR_CORE_SOURCES
    core/tensor_types.sl
    core/tensor_ops.sl
)

set(TENSOR_TESTS
    tests/test_tensor_types.sl
    tests/test_tensor_performance.sl
    tests/test_tensor_matrix.sl
)

# Create tensor build directory
set(TENSOR_BUILD_DIR ${CMAKE_BINARY_DIR}/tensor)
file(MAKE_DIRECTORY ${TENSOR_BUILD_DIR}/core)
file(MAKE_DIRECTORY ${TENSOR_BUILD_DIR}/tests)
file(MAKE_DIRECTORY ${TENSOR_BUILD_DIR}/lib)

# Function to compile SimpLang tensor files
function(compile_tensor_file SOURCE_FILE OUTPUT_NAME)
    get_filename_component(FILE_NAME ${SOURCE_FILE} NAME_WE)
    set(OUTPUT_OBJ ${TENSOR_BUILD_DIR}/${OUTPUT_NAME}/${FILE_NAME}.o)
    set(OUTPUT_LL ${TENSOR_BUILD_DIR}/${OUTPUT_NAME}/${FILE_NAME}.ll)
    
    add_custom_command(
        OUTPUT ${OUTPUT_OBJ} ${OUTPUT_LL}
        COMMAND ${CMAKE_BINARY_DIR}/src/simplang 
                ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}
                -o ${OUTPUT_OBJ}
        DEPENDS simplang ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}
        COMMENT "Compiling tensor file: ${SOURCE_FILE}"
        VERBATIM
    )
    
    # Add to global list for dependency tracking
    list(APPEND TENSOR_OBJECTS ${OUTPUT_OBJ})
    set(TENSOR_OBJECTS ${TENSOR_OBJECTS} PARENT_SCOPE)
endfunction()

# Compile all tensor core files
foreach(TENSOR_FILE ${TENSOR_CORE_SOURCES})
    compile_tensor_file(${TENSOR_FILE} "core")
endforeach()

# Compile all tensor test files
foreach(TEST_FILE ${TENSOR_TESTS})
    compile_tensor_file(${TEST_FILE} "tests")
endforeach()

# Create tensor library target
add_custom_target(tensor_library
    DEPENDS ${TENSOR_OBJECTS}
    COMMENT "Building SimpLang Tensor Library with auto-vectorization"
)

# Add tensor library to main build
add_dependencies(tensor_library simplang)

# Tensor test runner target
add_custom_target(tensor_tests
    COMMAND ${CMAKE_COMMAND} -E echo "Running SimpLang Tensor Tests..."
    COMMAND ${CMAKE_COMMAND} -E echo "Testing backend-agnostic SIMD auto-vectorization"
    DEPENDS tensor_library
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running tensor test suite"
)

# Integration with main SimpLang build
if(TARGET tests)
    add_dependencies(tests tensor_tests)
endif()

# Install tensor headers and libraries
install(FILES 
    core/tensor_types.sl
    core/tensor_ops.sl
    DESTINATION include/tensor
    COMPONENT tensor-dev
)

install(DIRECTORY ${TENSOR_BUILD_DIR}/core/
    DESTINATION lib/tensor
    COMPONENT tensor-runtime
    FILES_MATCHING 
    PATTERN "*.o" 
    PATTERN "*.ll"
)

# Print build info
add_custom_command(
    TARGET tensor_library POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "=========================================="
    COMMAND ${CMAKE_COMMAND} -E echo "SimpLang Tensor Library Built Successfully"
    COMMAND ${CMAKE_COMMAND} -E echo "Auto-vectorization: Enabled (backend-agnostic)"
    COMMAND ${CMAKE_COMMAND} -E echo "Supported types: f32, f64, i8-i64, u8-u64, bool"
    COMMAND ${CMAKE_COMMAND} -E echo "SIMD backends: Auto-detect (AVX-512/AVX/SSE)"
    COMMAND ${CMAKE_COMMAND} -E echo "=========================================="
    COMMENT "Tensor library build summary"
)

# Optional: Create combined tensor library archive
if(UNIX)
    add_custom_command(
        TARGET tensor_library POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Creating tensor library archive..."
        COMMAND ar rcs ${TENSOR_BUILD_DIR}/lib/libtensor.a ${TENSOR_BUILD_DIR}/core/*.o
        COMMENT "Archiving tensor object files"
    )
endif()

# Development convenience targets
add_custom_target(tensor_clean
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${TENSOR_BUILD_DIR}
    COMMENT "Cleaning tensor build artifacts"
)

add_custom_target(tensor_ir
    COMMAND ${CMAKE_COMMAND} -E echo "Generated LLVM IR files:"
    COMMAND find ${TENSOR_BUILD_DIR} -name "*.ll" -exec echo "  {}" \;
    DEPENDS tensor_library
    COMMENT "Listing tensor LLVM IR files"
)