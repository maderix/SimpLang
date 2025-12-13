//===- test_lowering.cpp - Test Simp to MemRef lowering --------*- C++ -*-===//
//
// Part of the SimpLang Project
//
// This file tests the lowering pass from Simp dialect to MemRef dialect.
//
//===----------------------------------------------------------------------===//

// LLVM 21: Updated header paths
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/simp_dialect.hpp"
#include "mlir/simp_ops.hpp"
#include "mlir/Passes.h"
#include "llvm/Support/SourceMgr.h"
#include <iostream>

using namespace mlir;

// Helper function to load and verify MLIR module
static OwningOpRef<ModuleOp> loadMLIRModule(const std::string& filename, MLIRContext& context) {
  // Try multiple paths
  std::vector<std::string> paths = {
    filename,
    "../../" + filename,
    "../../../" + filename
  };

  // LLVM 21: parseSourceFile now requires ParserConfig
  ParserConfig config(&context);

  for (const auto& path : paths) {
    auto module = parseSourceFile<ModuleOp>(path, config);
    if (module) {
      // Verify before lowering
      if (failed(verify(*module))) {
        llvm::errs() << "Module verification failed before lowering\n";
        return nullptr;
      }
      return module;
    }
  }

  llvm::errs() << "Failed to parse MLIR file (tried multiple paths): " << filename << "\n";
  return nullptr;
}

// Test basic constant lowering
static bool testConstantLowering() {
  MLIRContext context;
  context.getOrLoadDialect<simp::SimpDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<func::FuncDialect>();

  // Create a simple module with a constant
  OpBuilder builder(&context);
  auto module = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module.getBody());

  // Create function with constant
  auto funcType = builder.getFunctionType({}, {builder.getF64Type()});
  auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "test_constant", funcType);
  auto* entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Create simp.constant
  auto constOp = builder.create<simp::ConstantOp>(
      builder.getUnknownLoc(), builder.getF64Type(),
      builder.getF64FloatAttr(42.0));

  builder.create<func::ReturnOp>(builder.getUnknownLoc(), constOp.getResult());

  // Run lowering pass
  PassManager pm(&context);
  pm.addPass(simp::createConvertSimpToMemRefPass());

  if (failed(pm.run(module))) {
    llvm::errs() << "❌ Test 1 Failed: Constant lowering pass failed\n";
    return false;
  }

  // Verify after lowering
  if (failed(verify(module))) {
    llvm::errs() << "❌ Test 1 Failed: Module verification failed after lowering\n";
    module.dump();
    return false;
  }

  std::cout << "✅ Test 1 Passed: Constant lowering\n";
  return true;
}

// Test array operations lowering
static bool testArrayLowering() {
  MLIRContext context;
  context.getOrLoadDialect<simp::SimpDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<func::FuncDialect>();

  OpBuilder builder(&context);
  auto module = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module.getBody());

  // Create function with array operations
  auto funcType = builder.getFunctionType({}, {builder.getF64Type()});
  auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "test_arrays", funcType);
  auto* entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Create array with size 10
  auto arrayType = simp::ArrayType::get(&context, builder.getF64Type());
  auto sizeConst = builder.create<simp::ConstantOp>(
      builder.getUnknownLoc(), builder.getI64Type(),
      builder.getI64IntegerAttr(10));
  auto arrayOp = builder.create<simp::ArrayCreateOp>(
      builder.getUnknownLoc(), arrayType, sizeConst.getResult());

  // Create index 5 and value 3.14
  auto idxConst = builder.create<simp::ConstantOp>(
      builder.getUnknownLoc(), builder.getI64Type(),
      builder.getI64IntegerAttr(5));
  auto valConst = builder.create<simp::ConstantOp>(
      builder.getUnknownLoc(), builder.getF64Type(),
      builder.getF64FloatAttr(3.14));

  // Set array[5] = 3.14
  auto setOp = builder.create<simp::ArraySetOp>(
      builder.getUnknownLoc(), arrayType, arrayOp.getResult(),
      idxConst.getResult(), valConst.getResult());

  // Get array[5]
  auto getOp = builder.create<simp::ArrayGetOp>(
      builder.getUnknownLoc(), builder.getF64Type(),
      setOp.getResult(), idxConst.getResult());

  builder.create<func::ReturnOp>(builder.getUnknownLoc(), getOp.getResult());

  // Run lowering pass
  PassManager pm(&context);
  pm.addPass(simp::createConvertSimpToMemRefPass());

  if (failed(pm.run(module))) {
    llvm::errs() << "❌ Test 2 Failed: Array lowering pass failed\n";
    module.dump();
    return false;
  }

  // Verify after lowering
  if (failed(verify(module))) {
    llvm::errs() << "❌ Test 2 Failed: Module verification failed after lowering\n";
    module.dump();
    return false;
  }

  // Check that simp operations are gone
  bool hasSimpOps = false;
  module.walk([&](Operation* op) {
    if (op->getDialect() && op->getDialect()->getNamespace() == "simp") {
      hasSimpOps = true;
      llvm::errs() << "Found remaining simp op: " << op->getName() << "\n";
    }
  });

  if (hasSimpOps) {
    llvm::errs() << "❌ Test 2 Failed: Simp operations still present after lowering\n";
    return false;
  }

  std::cout << "✅ Test 2 Passed: Array operations lowering\n";
  return true;
}

// Test arithmetic operations lowering
static bool testArithmeticLowering() {
  MLIRContext context;
  context.getOrLoadDialect<simp::SimpDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<func::FuncDialect>();

  auto module = loadMLIRModule("tests/mlir/integration/test_arithmetic.mlir", context);
  if (!module) {
    llvm::errs() << "❌ Test 3 Failed: Could not load test_arithmetic.mlir\n";
    return false;
  }

  // Run lowering pass
  PassManager pm(&context);
  pm.addPass(simp::createConvertSimpToMemRefPass());

  if (failed(pm.run(*module))) {
    llvm::errs() << "❌ Test 3 Failed: Arithmetic lowering pass failed\n";
    return false;
  }

  // Verify after lowering
  if (failed(verify(*module))) {
    llvm::errs() << "❌ Test 3 Failed: Module verification failed after lowering\n";
    module->dump();
    return false;
  }

  std::cout << "✅ Test 3 Passed: Arithmetic operations lowering\n";
  return true;
}

// Test control flow with lowering
static bool testControlFlowLowering() {
  MLIRContext context;
  context.getOrLoadDialect<simp::SimpDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<scf::SCFDialect>();

  auto module = loadMLIRModule("tests/mlir/integration/test_value_passing.mlir", context);
  if (!module) {
    llvm::errs() << "❌ Test 4 Failed: Could not load test_value_passing.mlir\n";
    return false;
  }

  // Run lowering pass
  PassManager pm(&context);
  pm.addPass(simp::createConvertSimpToMemRefPass());

  if (failed(pm.run(*module))) {
    llvm::errs() << "❌ Test 4 Failed: Control flow lowering pass failed\n";
    return false;
  }

  // Verify after lowering
  if (failed(verify(*module))) {
    llvm::errs() << "❌ Test 4 Failed: Module verification failed after lowering\n";
    module->dump();
    return false;
  }

  std::cout << "✅ Test 4 Passed: Control flow with lowering\n";
  return true;
}

// Test type conversion in chained operations
static bool testTypeConversion() {
  MLIRContext context;
  context.getOrLoadDialect<simp::SimpDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<func::FuncDialect>();

  OpBuilder builder(&context);
  auto module = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module.getBody());

  // Create function with chained array operations
  auto funcType = builder.getFunctionType({builder.getI64Type()}, {builder.getF64Type()});
  auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "test_type_chain", funcType);
  auto* entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Create array, set element, then get element (tests type propagation)
  auto arrayType = simp::ArrayType::get(&context, builder.getF64Type());
  auto sizeArg = entryBlock->getArgument(0);

  // Create array
  auto arrayOp = builder.create<simp::ArrayCreateOp>(
      builder.getUnknownLoc(), arrayType, sizeArg);

  // Create index and value
  auto idx = builder.create<simp::ConstantOp>(
      builder.getUnknownLoc(), builder.getI64Type(),
      builder.getI64IntegerAttr(0));
  auto val = builder.create<simp::ConstantOp>(
      builder.getUnknownLoc(), builder.getF64Type(),
      builder.getF64FloatAttr(42.0));

  // Set element (array_set returns updated array - tests SSA with type conversion)
  auto setOp = builder.create<simp::ArraySetOp>(
      builder.getUnknownLoc(), arrayType, arrayOp.getResult(),
      idx.getResult(), val.getResult());

  // Get element from updated array
  auto getOp = builder.create<simp::ArrayGetOp>(
      builder.getUnknownLoc(), builder.getF64Type(),
      setOp.getResult(), idx.getResult());

  builder.create<func::ReturnOp>(builder.getUnknownLoc(), getOp.getResult());

  // Dump IR before conversion
  llvm::errs() << "\n=== Test 5: IR before conversion ===\n";
  module.dump();

  // Run lowering pass
  PassManager pm(&context);
  pm.addPass(simp::createConvertSimpToMemRefPass());

  if (failed(pm.run(module))) {
    llvm::errs() << "\n=== Test 5: IR after FAILED conversion ===\n";
    module.dump();
    llvm::errs() << "❌ Test 5 Failed: Type conversion pass failed\n";
    return false;
  }

  // Dump IR after conversion
  llvm::errs() << "\n=== Test 5: IR after successful conversion ===\n";
  module.dump();

  // Verify after lowering
  if (failed(verify(module))) {
    llvm::errs() << "❌ Test 5 Failed: Module verification failed after lowering\n";
    module.dump();
    return false;
  }

  // Check that no simp operations remain
  bool hasSimpOps = false;
  module.walk([&](Operation* op) {
    if (op->getDialect() && op->getDialect()->getNamespace() == "simp") {
      hasSimpOps = true;
      llvm::errs() << "Found remaining simp op: " << op->getName() << "\n";
    }
  });

  if (hasSimpOps) {
    llvm::errs() << "❌ Test 5 Failed: Simp operations still present after lowering\n";
    return false;
  }

  std::cout << "✅ Test 5 Passed: Type conversion in chained operations\n";
  return true;
}

// Test modulo operation lowering
static bool testModuloLowering() {
  MLIRContext context;
  context.getOrLoadDialect<simp::SimpDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<func::FuncDialect>();

  OpBuilder builder(&context);
  auto module = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module.getBody());

  // Create function with modulo operations (both float and integer)
  auto funcType = builder.getFunctionType({}, {builder.getF64Type()});
  auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "test_modulo", funcType);
  auto* entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Float modulo: 10.0 % 3.0
  auto lhsF = builder.create<simp::ConstantOp>(
      builder.getUnknownLoc(), builder.getF64Type(),
      builder.getF64FloatAttr(10.0));
  auto rhsF = builder.create<simp::ConstantOp>(
      builder.getUnknownLoc(), builder.getF64Type(),
      builder.getF64FloatAttr(3.0));
  auto modF = builder.create<simp::ModOp>(
      builder.getUnknownLoc(), builder.getF64Type(),
      lhsF.getResult(), rhsF.getResult());

  // Integer modulo: 17 % 5
  auto lhsI = builder.create<simp::ConstantOp>(
      builder.getUnknownLoc(), builder.getI64Type(),
      builder.getI64IntegerAttr(17));
  auto rhsI = builder.create<simp::ConstantOp>(
      builder.getUnknownLoc(), builder.getI64Type(),
      builder.getI64IntegerAttr(5));
  auto modI = builder.create<simp::ModOp>(
      builder.getUnknownLoc(), builder.getI64Type(),
      lhsI.getResult(), rhsI.getResult());

  // Convert integer result to float for return
  // (We'll just return the float modulo result for simplicity)
  builder.create<func::ReturnOp>(builder.getUnknownLoc(), modF.getResult());

  // Run lowering pass
  PassManager pm(&context);
  pm.addPass(simp::createConvertSimpToMemRefPass());

  if (failed(pm.run(module))) {
    llvm::errs() << "❌ Test 6 Failed: Modulo lowering pass failed\n";
    module.dump();
    return false;
  }

  // Verify after lowering
  if (failed(verify(module))) {
    llvm::errs() << "❌ Test 6 Failed: Module verification failed after lowering\n";
    module.dump();
    return false;
  }

  // Check that simp.mod operations are gone
  bool hasModOps = false;
  module.walk([&](Operation* op) {
    if (auto modOp = dyn_cast<simp::ModOp>(op)) {
      hasModOps = true;
      llvm::errs() << "Found remaining simp.mod op\n";
    }
  });

  if (hasModOps) {
    llvm::errs() << "❌ Test 6 Failed: simp.mod operations still present after lowering\n";
    return false;
  }

  // Verify that arith.remf and arith.remsi operations exist
  bool hasRemF = false;
  bool hasRemSI = false;
  module.walk([&](Operation* op) {
    if (isa<arith::RemFOp>(op)) hasRemF = true;
    if (isa<arith::RemSIOp>(op)) hasRemSI = true;
  });

  if (!hasRemF || !hasRemSI) {
    llvm::errs() << "❌ Test 6 Failed: Expected arith.remf and arith.remsi operations not found\n";
    llvm::errs() << "hasRemF: " << hasRemF << ", hasRemSI: " << hasRemSI << "\n";
    module.dump();
    return false;
  }

  std::cout << "✅ Test 6 Passed: Modulo operation lowering (float and integer)\n";
  return true;
}

int main() {
  std::cout << "=== Running MLIR Lowering Pass Integration Tests ===\n\n";

  int passed = 0;
  int failed = 0;

  // Run all tests
  if (testConstantLowering()) passed++; else failed++;
  if (testArrayLowering()) passed++; else failed++;
  if (testArithmeticLowering()) passed++; else failed++;
  if (testControlFlowLowering()) passed++; else failed++;
  if (testTypeConversion()) passed++; else failed++;
  if (testModuloLowering()) passed++; else failed++;

  // Summary
  std::cout << "\n=== Test Summary ===\n";
  std::cout << "Passed: " << passed << "/" << (passed + failed) << "\n";
  std::cout << "Failed: " << failed << "/" << (passed + failed) << "\n";

  if (failed == 0) {
    std::cout << "\n🎉 All lowering pass tests passed!\n";
    return 0;
  } else {
    std::cout << "\n❌ Some tests failed\n";
    return 1;
  }
}
