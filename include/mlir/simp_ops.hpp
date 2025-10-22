//===- simp_ops.hpp - Simp dialect operations ------------------*- C++ -*-===//
//
// Part of the SimpLang Project
//
// This file declares the operations for the Simp dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SIMP_OPS_HPP
#define MLIR_SIMP_OPS_HPP

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "simp_dialect.hpp"
#include "simp_types.hpp"

// Include generated operation class declarations
#define GET_OP_CLASSES
#include "SimpOps.h.inc"

#endif // MLIR_SIMP_OPS_HPP
