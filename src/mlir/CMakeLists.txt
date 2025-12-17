# MLIR Dialect CMakeLists.txt
# This file configures TableGen to generate C++ code from .td files

cmake_minimum_required(VERSION 3.13.4)

# Set the correct mlir-tblgen executable from our MLIR build
# Derive from MLIR_DIR which is set by build.sh
get_filename_component(MLIR_BUILD_DIR "${MLIR_DIR}/../../.." ABSOLUTE)

# When cross-compiling, use native x86 mlir-tblgen from Stage 1 build
if(CMAKE_CROSSCOMPILING)
    get_filename_component(NATIVE_MLIR_BUILD "${MLIR_BUILD_DIR}/../build-native" ABSOLUTE)
    set(MLIR_TABLEGEN_EXE "${NATIVE_MLIR_BUILD}/bin/mlir-tblgen")
    message(STATUS "Cross-compiling: Using native mlir-tblgen from ${MLIR_TABLEGEN_EXE}")
else()
    set(MLIR_TABLEGEN_EXE "${MLIR_BUILD_DIR}/bin/mlir-tblgen")
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
  cache_info.cpp
  Passes.cpp
  lowering/ConvertSimpToMemRef.cpp
  lowering/ConvertSimpToLinalg.cpp
  lowering/SpecializeShapes.cpp
  passes/LinalgTiling.cpp
  passes/Pipelines.cpp
  passes/InsertPrefetch.cpp
  passes/VNNIPass.cpp
  passes/AnnotationLoweringPass.cpp
  PromoteLargeAllocaToHeap.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/mlir

  DEPENDS
  MLIRSimpOpsIncGen
  MLIRSimpTypesIncGen
  MLIRSimpDialectIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect                 # Replaced MLIRStandard in LLVM 15+
  MLIRSCFDialect
  MLIRAffineDialect
  MLIRAffineTransforms
  MLIRAffineToStandard            # For createLowerAffinePass
  MLIRArithDialect                # Renamed from MLIRArithmetic
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRVectorDialect               # Vector dialect
  MLIRVectorTransforms            # Vector transforms
  MLIRVectorToLLVM                # Vector to LLVM conversion patterns
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRBufferizationTransforms
  MLIRBufferizationToMemRef
  MLIRTransforms
  MLIRLLVMDialect                 # Renamed from MLIRLLVMIR
  MLIRLLVMCommonConversion
  MLIRControlFlowDialect          # New: needed for SCF lowering
  MLIRControlFlowToLLVM           # New: ControlFlow to LLVM
  MLIRIndexDialect                # Index dialect
  MLIRIndexToLLVM                 # Index to LLVM conversion
  MLIRArithToLLVM                 # Renamed from MLIRArithmeticToLLVM
  MLIRMathToLLVM
  MLIRLinalgToStandard            # Replaced MLIRLinalgToLLVM
  MLIRMemRefToLLVM
  MLIRSCFToControlFlow            # Renamed from MLIRSCFToStandard
  MLIRFuncToLLVM                  # Replaced MLIRStandardToLLVM
  MLIROpenMPDialect               # OpenMP dialect (renamed)
  MLIRSCFToOpenMP                 # SCF to OpenMP conversion
  MLIROpenMPToLLVM                # OpenMP to LLVM conversion
  MLIROpenMPToLLVMIRTranslation   # OpenMP LLVM IR translation
  MLIRReconcileUnrealizedCasts
  MLIRTargetLLVMIRExport
  MLIRLLVMToLLVMIRTranslation
  mlir_c_runner_utils
  mlir_runner_utils
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
