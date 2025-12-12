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

# Build the list of MLIR libraries to link based on USE_CUDA
# For CUDA: Use shared libMLIR.so (linked later) to avoid symbol conflicts
# For non-CUDA: Use static MLIR libraries
if(USE_CUDA)
    # Empty for CUDA - we'll link libMLIR.so separately after the target is created
    set(SIMP_DIALECT_LINK_LIBS)
else()
    # Static MLIR libraries for non-CUDA builds
    set(SIMP_DIALECT_LINK_LIBS
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
        MLIROpenMP
        MLIRSCFToOpenMP
        MLIROpenMPToLLVM
        MLIROpenMPToLLVMIRTranslation
        MLIRReconcileUnrealizedCasts
        MLIRTargetLLVMIRExport
        MLIRLLVMToLLVMIRTranslation
        mlir_c_runner_utils
        mlir_runner_utils
    )
endif()

# Collect GPU-specific sources when CUDA is enabled
if(USE_CUDA)
    set(GPU_SOURCES
        gpu/GPUMemoryManager.cpp
        lowering/ConvertSimpToGPU.cpp
    )
else()
    set(GPU_SOURCES)
endif()

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
  passes/EmulateNarrowType.cpp
  passes/VNNIPass.cpp
  PromoteLargeAllocaToHeap.cpp
  ${GPU_SOURCES}

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/mlir

  DEPENDS
  MLIRSimpOpsIncGen
  MLIRSimpTypesIncGen
  MLIRSimpDialectIncGen

  LINK_LIBS PUBLIC
  ${SIMP_DIALECT_LINK_LIBS}
)

# Link GPU/CUDA libraries when CUDA is enabled
if(USE_CUDA)
  # Find nvptxcompiler static library
  find_library(NVPTXCOMPILER_LIBRARY
    NAMES nvptxcompiler_static
    PATHS ${CUDAToolkit_LIBRARY_DIR}
    NO_DEFAULT_PATH
  )

  # Use shared MLIR library to avoid command-line option conflicts
  # The shared libMLIR.so includes all dialects including GPU/NVVM
  find_library(MLIR_SHARED_LIBRARY
    NAMES MLIR
    PATHS ${LLVM_LIBRARY_DIR}
    NO_DEFAULT_PATH
  )

  if(MLIR_SHARED_LIBRARY)
    message(STATUS "Using shared MLIR library: ${MLIR_SHARED_LIBRARY}")
    target_link_libraries(simp_dialect
      PUBLIC
        ${MLIR_SHARED_LIBRARY}
        CUDA::cudart
        CUDA::cublas
        ${NVPTXCOMPILER_LIBRARY}
    )
  else()
    message(STATUS "Shared MLIR not found, using static GPU libraries")
    # Fall back to static libs + individual GPU libraries
    target_link_libraries(simp_dialect
      PUBLIC
        # Core MLIR static libs
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
        MLIROpenMP
        MLIRSCFToOpenMP
        MLIROpenMPToLLVM
        MLIROpenMPToLLVMIRTranslation
        MLIRReconcileUnrealizedCasts
        MLIRTargetLLVMIRExport
        MLIRLLVMToLLVMIRTranslation
        mlir_c_runner_utils
        mlir_runner_utils
        # MLIR GPU dialect libraries (Ubuntu LLVM 14 naming)
        MLIRGPUOps
        MLIRGPUTransforms
        MLIRGPUToNVVMTransforms
        MLIRGPUToGPURuntimeTransforms
        MLIRNVVMIR
        MLIRNVVMToLLVMIRTranslation
        # CUDA runtime libraries
        CUDA::cudart
        CUDA::cublas
        ${NVPTXCOMPILER_LIBRARY}
    )
  endif()

  message(STATUS "MLIR GPU dialect libraries linked for CUDA backend")
  message(STATUS "Using nvptxcompiler: ${NVPTXCOMPILER_LIBRARY}")
endif()

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
