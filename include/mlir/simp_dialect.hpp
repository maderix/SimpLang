//===- simp_dialect.hpp - Simp dialect definition ---------------*- C++ -*-===//
//
// Part of the SimpLang Project
//
// This file declares the Simp dialect and includes generated code.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SIMP_DIALECT_HPP
#define MLIR_SIMP_DIALECT_HPP

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Include generated dialect class with methods we need to implement
#include "SimpDialect.h.inc"

#endif // MLIR_SIMP_DIALECT_HPP
