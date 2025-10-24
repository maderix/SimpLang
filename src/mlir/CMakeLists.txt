# MLIR Dialect CMakeLists.txt
# This file configures TableGen to generate C++ code from .td files

cmake_minimum_required(VERSION 3.13.4)

# Set the correct mlir-tblgen executable from our MLIR build
# Use the mlir-tblgen from the MLIR build directory, not system LLVM
if(DEFINED ENV{HOME})
  set(MLIR_TABLEGEN_EXE "$ENV{HOME}/llvm-project/build/bin/mlir-tblgen")
else()
  set(MLIR_TABLEGEN_EXE "/home/maderix/llvm-project/build/bin/mlir-tblgen")
endif()

# MLIR TableGen commands for generating operation definitions

# Generate operation declarations (.h.inc)
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/mlir/Dialects/Simp/SimpOps.td)
mlir_tablegen(SimpOps.h.inc -gen-op-decls -I ${CMAKE_SOURCE_DIR}/include/mlir/Dialects/Simp)
mlir_tablegen(SimpOps.cpp.inc -gen-op-defs -I ${CMAKE_SOURCE_DIR}/include/mlir/Dialects/Simp)
add_public_tablegen_target(MLIRSimpOpsIncGen)

# Generate type declarations (.h.inc)
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/mlir/Dialects/Simp/SimpTypes.td)
mlir_tablegen(SimpTypes.h.inc -gen-typedef-decls -I ${CMAKE_SOURCE_DIR}/include/mlir/Dialects/Simp)
mlir_tablegen(SimpTypes.cpp.inc -gen-typedef-defs -I ${CMAKE_SOURCE_DIR}/include/mlir/Dialects/Simp)
add_public_tablegen_target(MLIRSimpTypesIncGen)

# Generate dialect declarations (.h.inc)
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/mlir/Dialects/Simp/SimpBase.td)
mlir_tablegen(SimpDialect.h.inc -gen-dialect-decls -I ${CMAKE_SOURCE_DIR}/include/mlir/Dialects/Simp)
mlir_tablegen(SimpDialect.cpp.inc -gen-dialect-defs -I ${CMAKE_SOURCE_DIR}/include/mlir/Dialects/Simp)
add_public_tablegen_target(MLIRSimpDialectIncGen)

#===----------------------------------------------------------------------===//
# Simp Dialect Library
#===----------------------------------------------------------------------===//

add_mlir_dialect_library(simp_dialect
  simp_dialect.cpp
  simp_types.cpp
  simp_ops.cpp
  mlir_codegen.cpp
  mlir_pipeline.cpp
  lowering/ConvertSimpToMemRef.cpp
  lowering/ConvertSimpToLinalg.cpp
  lowering/SpecializeShapes.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/mlir

  DEPENDS
  MLIRSimpOpsIncGen
  MLIRSimpTypesIncGen
  MLIRSimpDialectIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRStandard
  MLIRSCF
  MLIRAffine
  MLIRAffineTransforms
  MLIRArithmetic
  MLIRMath
  MLIRMemRef
  MLIRLinalg
  MLIRLinalgTransforms
  MLIRBufferizationTransforms
  MLIRBufferizationToMemRef
  MLIRTransforms
  MLIRLLVMIR
  MLIRLLVMCommonConversion
  MLIRArithmeticToLLVM
  MLIRMathToLLVM
  MLIRLinalgToLLVM
  MLIRMemRefToLLVM
  MLIRSCFToStandard
  MLIRStandardToLLVM
  MLIRTargetLLVMIRExport
  MLIRLLVMToLLVMIRTranslation
)

# Disable RTTI and exceptions to match LLVM/MLIR build
target_compile_options(simp_dialect PRIVATE -fno-rtti -fno-exceptions)
target_compile_options(obj.simp_dialect PRIVATE -fno-rtti -fno-exceptions)

# Define USE_MLIR so AST headers can conditionally disable exception-throwing code
target_compile_definitions(simp_dialect PRIVATE USE_MLIR)
target_compile_definitions(obj.simp_dialect PRIVATE USE_MLIR)

# Set include directories
target_include_directories(simp_dialect
  PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/mlir
    ${CMAKE_BINARY_DIR}/src/mlir  # For generated .inc files
    ${MLIR_INCLUDE_DIRS}
)

# Set C++ standard
set_target_properties(simp_dialect PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)
