# Benchmark CMakeLists.txt

# Create output directories
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/benchmark/obj)

# Get the base name for our benchmark
set(BENCH_NAME "test_scalar_benchmark")
set(BENCH_SL_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${BENCH_NAME}.sl")
set(BENCH_OBJ_FILE "${CMAKE_BINARY_DIR}/benchmark/obj/${BENCH_NAME}.o")
set(BENCH_SO_FILE "${CMAKE_BINARY_DIR}/benchmark/obj/${BENCH_NAME}.so")

# First compile to object file
add_custom_command(
    OUTPUT ${BENCH_OBJ_FILE}
    COMMAND ${CMAKE_BINARY_DIR}/src/simplang 
            ${BENCH_SL_FILE}
            -o ${BENCH_OBJ_FILE}
            -v
            -fPIC
    DEPENDS simplang ${BENCH_SL_FILE}
    COMMENT "Compiling scalar benchmark"
    VERBATIM
)

# Create shared library
add_custom_command(
    OUTPUT ${BENCH_SO_FILE}
    COMMAND ${CMAKE_C_COMPILER} -shared ${BENCH_OBJ_FILE} -o ${BENCH_SO_FILE}
            -Wl,--version-script=${CMAKE_BINARY_DIR}/tests/kernel.version
            -Wl,--export-dynamic-symbol=kernel_main
            -Wl,--export-dynamic
    DEPENDS ${BENCH_OBJ_FILE} ${CMAKE_BINARY_DIR}/tests/kernel.version
    COMMENT "Creating shared library for scalar benchmark"
    VERBATIM
)

# Create target for the benchmark
add_custom_target(scalar_benchmark_obj 
    DEPENDS ${BENCH_SO_FILE}
)

# Add benchmark runner target
add_custom_target(run_scalar_benchmark
    COMMAND ${CMAKE_BINARY_DIR}/simplang_runner ${BENCH_SO_FILE} 
            --warmup 1000 
            --iterations 10000 
            --verbose
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/${BENCH_NAME}.py
    DEPENDS scalar_benchmark_obj simplang_runner
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/benchmark/obj
    COMMENT "Running scalar benchmark comparison"
)

# Add to build_all target
add_dependencies(build_all scalar_benchmark_obj)
