//===- SimpBase.td - Simp dialect definition ---------------*- tablegen -*-===//
//
// Part of the SimpLang Project
//
// This file defines the base Simp dialect and common utilities.
//
//===----------------------------------------------------------------------===//

#ifndef SIMP_BASE
#define SIMP_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Simp Dialect Definition
//===----------------------------------------------------------------------===//

def Simp_Dialect : Dialect {
  let name = "simp";

  let summary = "SimpLang high-level operations dialect";

  let description = [{
    The Simp dialect represents high-level SimpLang semantics before
    progressive lowering to LLVM IR. It preserves array and matrix
    operation semantics for optimization.

    Key features:
    - SSA-pure operations (functional style, no mutations)
    - Array types with element type parameterization
    - Progressive lowering path: simp → memref → linalg → llvm
    - Integration with SimpBLAS for optimized BLAS operations

    Design principles:
    1. All operations return new SSA values (no in-place mutations)
    2. Memref-first lowering strategy for stable compilation
    3. Late SimpBLAS lowering after fusion/tiling optimizations
  }];

  let cppNamespace = "::mlir::simp";

  // Use automatic type parsing/printing from TableGen (MLIR 15+ handles this)
  let useDefaultTypePrinterParser = 1;

  // Enable constant materialization for constant folding
  let hasConstantMaterializer = 1;

  // Note: parseType/printType are auto-generated when useDefaultTypePrinterParser = 1
  // Do NOT declare them manually in extraClassDeclaration
}

//===----------------------------------------------------------------------===//
// Base Simp Operation Definition
//===----------------------------------------------------------------------===//

// Base class for all operations in the Simp dialect
class Simp_Op<string mnemonic, list<Trait> traits = []> :
    Op<Simp_Dialect, mnemonic, traits>;

#endif // SIMP_BASE
