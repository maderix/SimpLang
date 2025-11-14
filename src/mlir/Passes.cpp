//===- Passes.cpp - Pass Registration for Simp Dialect -------------------===//
//
// Part of the SimpLang Project
//
// This file implements pass registration for command-line usage.
//
//===----------------------------------------------------------------------===//

#include "mlir/Passes.h"

namespace mlir {
namespace simp {

/// Register all Simp dialect passes for command-line usage
void registerSimpPasses() {
  // Register lowering passes
  registerConvertSimpToMemRefPass();
  registerConvertSimpToLinalgPass();
  registerSpecializeShapesPass();

  // Register optimization passes
  registerSimpLinalgTilingPass();

  // Future passes will be registered here as needed
}

} // namespace simp
} // namespace mlir
