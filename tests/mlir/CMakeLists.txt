# MLIR Dialect Tests

if(BUILD_TESTING)
  # Get MLIR dialect libraries (includes base Dialect class)
  get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

  # Dialect types test
  add_executable(test_simp_dialect
    test_simp_dialect.cpp
  )

target_compile_options(test_simp_dialect PRIVATE -fno-rtti -fno-exceptions)

target_include_directories(test_simp_dialect PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include/mlir
  ${CMAKE_BINARY_DIR}/src/mlir
  ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(test_simp_dialect
  PRIVATE
    simp_dialect
    ${dialect_libs}
)

# Add test to CTest
add_test(NAME SimpDialectTypes COMMAND test_simp_dialect)

# Lowering pass integration test
add_executable(test_lowering
  test_lowering.cpp
)

target_compile_options(test_lowering PRIVATE -fno-rtti -fno-exceptions)

target_include_directories(test_lowering PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include/mlir
  ${CMAKE_BINARY_DIR}/src/mlir
  ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(test_lowering
  PRIVATE
    simp_dialect
    MLIRPass
    MLIRMemRef
    MLIRArithmetic
    MLIRSCF
    MLIRParser
    MLIRSupport
    ${dialect_libs}
)

add_test(NAME SimpLoweringPass COMMAND test_lowering)

endif() # BUILD_TESTING
