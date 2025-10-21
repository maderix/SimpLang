//===- test_simp_dialect.cpp - Test Simp dialect types --------------------===//
//
// Simple test to verify SimpDialect types are working correctly
//
//===----------------------------------------------------------------------===//

#include "mlir/simp_dialect.hpp"
#include "mlir/simp_types.hpp"
#include "mlir/simp_ops.hpp"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/raw_ostream.h"

int main() {
  mlir::MLIRContext context;

  // Load the Simp dialect
  context.getOrLoadDialect<mlir::simp::SimpDialect>();
  llvm::outs() << "✅ Test 1: SimpDialect loaded successfully!\n";

  // Test ArrayType creation
  auto f64Type = mlir::Float64Type::get(&context);
  auto arrayType = mlir::simp::ArrayType::get(&context, f64Type);
  llvm::outs() << "✅ Test 2: Created ArrayType with f64 element type\n";

  // Verify ArrayType element type
  if (arrayType.getElementType() == f64Type) {
    llvm::outs() << "✅ Test 3: ArrayType.getElementType() works correctly\n";
  } else {
    llvm::errs() << "❌ Test 3 FAILED: Element type mismatch\n";
    return 1;
  }

  // Test SimpTensorType with static shape
  llvm::SmallVector<int64_t, 2> shape = {10, 20};
  auto tensorType = mlir::simp::SimpTensorType::get(&context, shape, f64Type);
  llvm::outs() << "✅ Test 4: Created SimpTensorType with shape [10, 20]\n";

  // Verify tensor shape
  auto retrievedShape = tensorType.getShape();
  if (retrievedShape.size() == 2 && retrievedShape[0] == 10 && retrievedShape[1] == 20) {
    llvm::outs() << "✅ Test 5: SimpTensorType.getShape() works correctly\n";
  } else {
    llvm::errs() << "❌ Test 5 FAILED: Shape mismatch\n";
    return 1;
  }

  // Verify tensor rank
  if (tensorType.getRank() == 2) {
    llvm::outs() << "✅ Test 6: SimpTensorType.getRank() works correctly\n";
  } else {
    llvm::errs() << "❌ Test 6 FAILED: Rank mismatch\n";
    return 1;
  }

  // Verify tensor element type
  if (tensorType.getElementType() == f64Type) {
    llvm::outs() << "✅ Test 7: SimpTensorType.getElementType() works correctly\n";
  } else {
    llvm::errs() << "❌ Test 7 FAILED: Element type mismatch\n";
    return 1;
  }

  // Test hasStaticShape
  if (tensorType.hasStaticShape()) {
    llvm::outs() << "✅ Test 8: SimpTensorType.hasStaticShape() returns true for static shape\n";
  } else {
    llvm::errs() << "❌ Test 8 FAILED: Should have static shape\n";
    return 1;
  }

  // Test dynamic tensor type
  llvm::SmallVector<int64_t, 2> dynamicShape = {-1, 20};
  auto dynamicTensorType = mlir::simp::SimpTensorType::get(&context, dynamicShape, f64Type);
  llvm::outs() << "✅ Test 9: Created SimpTensorType with dynamic shape [?, 20]\n";

  // Verify dynamic dimension
  if (dynamicTensorType.isDynamicDim(0) && !dynamicTensorType.isDynamicDim(1)) {
    llvm::outs() << "✅ Test 10: SimpTensorType.isDynamicDim() works correctly\n";
  } else {
    llvm::errs() << "❌ Test 10 FAILED: Dynamic dimension check failed\n";
    return 1;
  }

  // Verify hasStaticShape for dynamic tensor
  if (!dynamicTensorType.hasStaticShape()) {
    llvm::outs() << "✅ Test 11: SimpTensorType.hasStaticShape() returns false for dynamic shape\n";
  } else {
    llvm::errs() << "❌ Test 11 FAILED: Should not have static shape\n";
    return 1;
  }

  // Test type uniquing (same parameters should return same type instance)
  auto arrayType2 = mlir::simp::ArrayType::get(&context, f64Type);
  if (arrayType == arrayType2) {
    llvm::outs() << "✅ Test 12: Type uniquing works - identical ArrayTypes are same instance\n";
  } else {
    llvm::errs() << "❌ Test 12 FAILED: Type uniquing not working\n";
    return 1;
  }

  auto tensorType2 = mlir::simp::SimpTensorType::get(&context, shape, f64Type);
  if (tensorType == tensorType2) {
    llvm::outs() << "✅ Test 13: Type uniquing works - identical SimpTensorTypes are same instance\n";
  } else {
    llvm::errs() << "❌ Test 13 FAILED: Type uniquing not working\n";
    return 1;
  }

  // Test with different element types
  auto i32Type = mlir::IntegerType::get(&context, 32);
  auto intArrayType = mlir::simp::ArrayType::get(&context, i32Type);
  llvm::outs() << "✅ Test 14: Created ArrayType with i32 element type\n";

  auto f32Type = mlir::Float32Type::get(&context);
  llvm::SmallVector<int64_t, 3> shape3d = {3, 224, 224};
  auto imageTensorType = mlir::simp::SimpTensorType::get(&context, shape3d, f32Type);
  if (imageTensorType.getRank() == 3) {
    llvm::outs() << "✅ Test 15: Created 3D SimpTensorType (3x224x224xf32) for image data\n";
  } else {
    llvm::errs() << "❌ Test 15 FAILED: 3D tensor creation failed\n";
    return 1;
  }

  llvm::outs() << "\n================================================\n";
  llvm::outs() << "🎉 All 15 tests passed!\n";
  llvm::outs() << "================================================\n";
  llvm::outs() << "\nSummary:\n";
  llvm::outs() << "  - ArrayType: ✅ Working\n";
  llvm::outs() << "  - SimpTensorType: ✅ Working\n";
  llvm::outs() << "  - Static shapes: ✅ Working\n";
  llvm::outs() << "  - Dynamic shapes: ✅ Working\n";
  llvm::outs() << "  - Type uniquing: ✅ Working\n";
  llvm::outs() << "  - Multiple element types: ✅ Working\n";

  return 0;
}
