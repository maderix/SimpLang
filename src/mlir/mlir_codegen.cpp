//===- mlir_codegen.cpp - MLIR Code Generation Implementation ------------===//
//
// Part of the SimpLang Project
//
// This file implements the MLIRCodeGenContext class for lowering SimpLang AST
// to MLIR Simp dialect operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/mlir_codegen.hpp"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/tensor_layout.hpp"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"
#include "logger.hpp"
#include <cmath>

// Include full AST definitions here in the .cpp file
// This way the header doesn't pull in exception-throwing code
#include "ast/ast.hpp"

using namespace mlir;
using namespace mlir::simp;

//===----------------------------------------------------------------------===//
// Constructor & Initialization
//===----------------------------------------------------------------------===//

MLIRCodeGenContext::MLIRCodeGenContext(const std::string& moduleName)
    : builder(&mlirContext), sourceFileName(moduleName) {
  initializeMLIRContext();
  createModule(moduleName);

  // Push global scope
  pushScope();
}

void MLIRCodeGenContext::initializeMLIRContext() {
  // Load required dialects
  mlirContext.getOrLoadDialect<mlir::simp::SimpDialect>();
  mlirContext.getOrLoadDialect<mlir::StandardOpsDialect>();
  mlirContext.getOrLoadDialect<mlir::scf::SCFDialect>();
  mlirContext.getOrLoadDialect<mlir::arith::ArithmeticDialect>();
  mlirContext.getOrLoadDialect<mlir::math::MathDialect>();
  mlirContext.getOrLoadDialect<mlir::vector::VectorDialect>();  // For vectorization
  mlirContext.getOrLoadDialect<mlir::memref::MemRefDialect>();  // For memref ops
  mlirContext.getOrLoadDialect<mlir::linalg::LinalgDialect>();  // For linalg ops
}

void MLIRCodeGenContext::createModule(const std::string& moduleName) {
  // Create the MLIR module
  // ModuleOp::create expects Optional<StringRef>
  module = mlir::ModuleOp::create(builder.getUnknownLoc(), llvm::StringRef(moduleName));
}

//===----------------------------------------------------------------------===//
// Symbol Table Management
//===----------------------------------------------------------------------===//

void MLIRCodeGenContext::declareVariable(const std::string& name, mlir::Value value) {
  if (symbolTable.empty()) {
    llvm::errs() << "Error: No scope to declare variable '" << name << "'\n";
    return;
  }

  // Add to current scope (top of stack)
  symbolTable.back()[name] = value;
}

mlir::Value MLIRCodeGenContext::lookupVariable(const std::string& name) {
  // Search from innermost to outermost scope
  for (auto it = symbolTable.rbegin(); it != symbolTable.rend(); ++it) {
    auto found = it->find(name);
    if (found != it->end()) {
      return found->second;
    }
  }

  // Not found
  llvm::errs() << "Error: Undefined variable '" << name << "'\n";
  return nullptr;
}

bool MLIRCodeGenContext::variableExists(const std::string& name) {
  if (symbolTable.empty()) {
    return false;
  }

  // Check current scope only
  return symbolTable.back().find(name) != symbolTable.back().end();
}

void MLIRCodeGenContext::pushScope() {
  symbolTable.push_back(std::map<std::string, mlir::Value>());
}

void MLIRCodeGenContext::popScope() {
  if (!symbolTable.empty()) {
    symbolTable.pop_back();
  }
}

//===----------------------------------------------------------------------===//
// Type Conversion Helper
//===----------------------------------------------------------------------===//

namespace {
/// Helper class for converting SimpLang types to MLIR types
/// Supports all TypeKind values including F16, BF16, and full integer range
class TypeConverter {
public:
  /// Convert TypeKind enum to MLIR type
  static mlir::Type toMLIRType(TypeKind kind, mlir::OpBuilder& builder) {
    switch (kind) {
      case TypeKind::F16:     return builder.getF16Type();
      case TypeKind::BF16:    return builder.getBF16Type();
      case TypeKind::F32:     return builder.getF32Type();
      case TypeKind::F64:     return builder.getF64Type();
      case TypeKind::I8:      return builder.getIntegerType(8);
      case TypeKind::I16:     return builder.getIntegerType(16);
      case TypeKind::I32:     return builder.getI32Type();
      case TypeKind::I64:     return builder.getI64Type();
      case TypeKind::U8:      return builder.getIntegerType(8, /*isSigned=*/false);
      case TypeKind::U16:     return builder.getIntegerType(16, /*isSigned=*/false);
      case TypeKind::U32:     return builder.getIntegerType(32, /*isSigned=*/false);
      case TypeKind::U64:     return builder.getIntegerType(64, /*isSigned=*/false);
      case TypeKind::Bool:    return builder.getI1Type();
      case TypeKind::Void:    return builder.getNoneType();
      case TypeKind::Dynamic: return builder.getF32Type(); // Default for var
      default:                return builder.getF32Type();
    }
  }

  /// Convert string type name to MLIR type
  static mlir::Type fromString(const std::string& typeStr, mlir::OpBuilder& builder) {
    // Floating point types
    if (typeStr == "f16")                     return builder.getF16Type();
    if (typeStr == "bf16")                    return builder.getBF16Type();
    if (typeStr == "f32" || typeStr == "float") return builder.getF32Type();
    if (typeStr == "f64" || typeStr == "double") return builder.getF64Type();

    // Signed integer types
    if (typeStr == "i8")                      return builder.getIntegerType(8);
    if (typeStr == "i16")                     return builder.getIntegerType(16);
    if (typeStr == "i32" || typeStr == "int") return builder.getI32Type();
    if (typeStr == "i64")                     return builder.getI64Type();

    // Unsigned integer types
    if (typeStr == "u8")                      return builder.getIntegerType(8, false);
    if (typeStr == "u16")                     return builder.getIntegerType(16, false);
    if (typeStr == "u32")                     return builder.getIntegerType(32, false);
    if (typeStr == "u64")                     return builder.getIntegerType(64, false);

    // Boolean
    if (typeStr == "bool" || typeStr == "i1") return builder.getI1Type();

    // Void
    if (typeStr == "void")                    return builder.getNoneType();

    // Unknown - default to f32
    return builder.getF32Type();
  }

  /// Check if type supports arithmetic operations
  static bool supportsArithmetic(mlir::Type type) {
    return type.isa<mlir::FloatType>() || type.isa<mlir::IntegerType>();
  }

  /// Get type size in bits
  static unsigned getTypeSizeInBits(mlir::Type type) {
    if (auto floatTy = type.dyn_cast<mlir::FloatType>()) {
      return floatTy.getWidth();
    }
    if (auto intTy = type.dyn_cast<mlir::IntegerType>()) {
      return intTy.getWidth();
    }
    return 32; // Default
  }
};

//===----------------------------------------------------------------------===//
// Tensor Layout and Affine Map Helpers
//===----------------------------------------------------------------------===//

/// Create affine map for NHWC layout
/// Logical indices: (N, C, H, W) → Physical memory: (N, H, W, C)
mlir::AffineMap createNHWCAffineMap(mlir::MLIRContext* ctx) {
  using namespace mlir;
  auto d0 = getAffineDimExpr(0, ctx);  // N
  auto d1 = getAffineDimExpr(1, ctx);  // C
  auto d2 = getAffineDimExpr(2, ctx);  // H
  auto d3 = getAffineDimExpr(3, ctx);  // W

  // Permute: NCHW → NHWC
  return AffineMap::get(4, 0, {d0, d2, d3, d1}, ctx);
}

/// Create affine map for NCHW layout (identity)
mlir::AffineMap createNCHWAffineMap(mlir::MLIRContext* ctx) {
  return mlir::AffineMap::getMultiDimIdentityMap(4, ctx);
}

/// Create affine map for given layout
mlir::AffineMap createAffineMapForLayout(::simp::TensorLayout layout,
                                          unsigned rank,
                                          mlir::MLIRContext* ctx) {
  switch (layout) {
    case ::simp::TensorLayout::NHWC:
      if (rank == 4) return createNHWCAffineMap(ctx);
      break;
    case ::simp::TensorLayout::NCHW:
      if (rank == 4) return createNCHWAffineMap(ctx);
      break;
    case ::simp::TensorLayout::RowMajor:
      return mlir::AffineMap::getMultiDimIdentityMap(rank, ctx);
    case ::simp::TensorLayout::Custom:
      // TODO: Parse custom affine map from string
      break;
  }
  // Default: identity map
  return mlir::AffineMap::getMultiDimIdentityMap(rank, ctx);
}

/// Compute strides for NHWC layout: [N, H, W, C]
/// Memory layout: [N][H][W][C]
/// Strides: [H*W*C, W*C, C, 1]
mlir::SmallVector<mlir::Value> computeNHWCStrides(mlir::Value H, mlir::Value W, mlir::Value C,
                                                   mlir::OpBuilder& builder, mlir::Location loc) {
  mlir::Value one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  mlir::Value WC = builder.create<mlir::arith::MulIOp>(loc, W, C);
  mlir::Value HWC = builder.create<mlir::arith::MulIOp>(loc, H, WC);

  return {HWC, WC, C, one};
}

/// Compute strides for NCHW layout: [N, C, H, W]
/// Memory layout: [N][C][H][W]
/// Strides: [C*H*W, H*W, W, 1]
mlir::SmallVector<mlir::Value> computeNCHWStrides(mlir::Value C, mlir::Value H, mlir::Value W,
                                                   mlir::OpBuilder& builder, mlir::Location loc) {
  mlir::Value one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  mlir::Value HW = builder.create<mlir::arith::MulIOp>(loc, H, W);
  mlir::Value CHW = builder.create<mlir::arith::MulIOp>(loc, C, HW);

  return {CHW, HW, W, one};
}

/// Helper to cast i64 to index type (forward declare before use)
mlir::Value castToIndex(mlir::Value val, mlir::OpBuilder& builder, mlir::Location loc);

/// Compute row-major strides for arbitrary rank
/// For shape [d0, d1, ..., dn], strides are [d1*d2*...*dn, d2*d3*...*dn, ..., 1]
mlir::SmallVector<mlir::Value> computeRowMajorStrides(mlir::ArrayRef<mlir::Value> dims,
                                                       mlir::OpBuilder& builder, mlir::Location loc) {
  mlir::SmallVector<mlir::Value> strides;
  mlir::Value one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);

  for (size_t i = 0; i < dims.size(); ++i) {
    mlir::Value stride = one;
    for (size_t j = i + 1; j < dims.size(); ++j) {
      // Cast dimension to index type if needed
      mlir::Value dim = castToIndex(dims[j], builder, loc);
      stride = builder.create<mlir::arith::MulIOp>(loc, stride, dim);
    }
    strides.push_back(stride);
  }

  return strides;
}

/// Create memref type with layout
mlir::MemRefType createMemRefWithLayout(mlir::ArrayRef<int64_t> shape,
                                         mlir::Type elemType,
                                         ::simp::TensorLayout layout,
                                         mlir::MLIRContext* ctx) {
  mlir::AffineMap layoutMap = createAffineMapForLayout(layout, shape.size(), ctx);
  return mlir::MemRefType::get(shape, elemType, layoutMap);
}

/// Helper to cast i64 to index type
mlir::Value castToIndex(mlir::Value val, mlir::OpBuilder& builder, mlir::Location loc) {
  if (val.getType().isa<mlir::IndexType>()) {
    return val;
  }
  return builder.create<mlir::arith::IndexCastOp>(loc, builder.getIndexType(), val);
}

//===----------------------------------------------------------------------===//
// Type Promotion (C++ style)
//===----------------------------------------------------------------------===//

/// Get type precedence for C++ style promotion (higher = wider type)
/// Float types always outrank integer types in mixed arithmetic
int getTypePrecedence(mlir::Type type) {
  if (auto floatTy = type.dyn_cast<mlir::FloatType>()) {
    // Float precedence: f16=10, bf16=11, f32=12, f64=13
    switch (floatTy.getWidth()) {
      case 16: return floatTy.isBF16() ? 11 : 10;
      case 32: return 12;
      case 64: return 13;
      default: return 10;
    }
  }
  if (auto intTy = type.dyn_cast<mlir::IntegerType>()) {
    if (intTy.getWidth() == 1) return 0;  // bool (lowest)
    // Integer precedence: i8=1, i16=2, i32=3, i64=4 (below all floats)
    return (intTy.getWidth() / 16) + 1;  // i8=1, i16=2, i32=3, i64=5
  }
  return 0;
}

/// Promote a value to target type following C++ rules
mlir::Value promoteType(mlir::Value val, mlir::Type targetType,
                        mlir::OpBuilder& builder, mlir::Location loc) {
  mlir::Type srcType = val.getType();
  if (srcType == targetType) return val;

  // Float to float (extend or truncate)
  if (srcType.isa<mlir::FloatType>() && targetType.isa<mlir::FloatType>()) {
    auto srcFloat = srcType.cast<mlir::FloatType>();
    auto targetFloat = targetType.cast<mlir::FloatType>();
    if (targetFloat.getWidth() > srcFloat.getWidth()) {
      return builder.create<mlir::arith::ExtFOp>(loc, targetType, val);
    } else if (targetFloat.getWidth() < srcFloat.getWidth()) {
      return builder.create<mlir::arith::TruncFOp>(loc, targetType, val);
    }
  }

  // Int to int (extend or truncate)
  if (srcType.isa<mlir::IntegerType>() && targetType.isa<mlir::IntegerType>()) {
    auto srcInt = srcType.cast<mlir::IntegerType>();
    auto targetInt = targetType.cast<mlir::IntegerType>();
    if (srcInt.getWidth() == 1) return val;  // Don't convert bool
    if (targetInt.getWidth() > srcInt.getWidth()) {
      return builder.create<mlir::arith::ExtSIOp>(loc, targetType, val);
    } else if (targetInt.getWidth() < srcInt.getWidth()) {
      return builder.create<mlir::arith::TruncIOp>(loc, targetType, val);
    }
  }

  // Int to float (C++ style: always promote int to float in mixed arithmetic)
  if (srcType.isa<mlir::IntegerType>() && targetType.isa<mlir::FloatType>()) {
    return builder.create<mlir::arith::SIToFPOp>(loc, targetType, val);
  }

  // Float to int (rarely needed, but support it)
  if (srcType.isa<mlir::FloatType>() && targetType.isa<mlir::IntegerType>()) {
    return builder.create<mlir::arith::FPToSIOp>(loc, targetType, val);
  }

  return val;  // No conversion possible
}

/// Apply C++ usual arithmetic conversions to binary operands
/// Returns promoted operands with common type
std::pair<mlir::Value, mlir::Value> applyUsualArithmeticConversions(
    mlir::Value lhs, mlir::Value rhs, mlir::OpBuilder& builder, mlir::Location loc) {

  mlir::Type lhsType = lhs.getType();
  mlir::Type rhsType = rhs.getType();

  if (lhsType == rhsType) {
    return {lhs, rhs};
  }

  // Apply C++ usual arithmetic conversions
  // Rule: promote to the "wider" type
  int lhsPrec = getTypePrecedence(lhsType);
  int rhsPrec = getTypePrecedence(rhsType);

  if (lhsPrec > rhsPrec) {
    // Promote rhs to lhs type
    return {lhs, promoteType(rhs, lhsType, builder, loc)};
  } else {
    // Promote lhs to rhs type
    return {promoteType(lhs, rhsType, builder, loc), rhs};
  }
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

mlir::Type MLIRCodeGenContext::convertType(const std::string& simpType) {
  // Handle basic types using TypeConverter
  if (simpType.find('<') == std::string::npos && simpType.find('[') == std::string::npos) {
    // Simple type (not array or tensor)
    return TypeConverter::fromString(simpType, builder);
  }

  // Handle array types: "array<T>"
  if (simpType.find("array<") == 0) {
    // Extract element type from "array<f64>"
    size_t start = simpType.find('<') + 1;
    size_t end = simpType.find('>');
    std::string elemTypeStr = simpType.substr(start, end - start);
    mlir::Type elemType = convertType(elemTypeStr);

    return mlir::simp::ArrayType::get(&mlirContext, elemType);
  }

  // Handle tensor types: "tensor<10x20xf64>" or "tensor<?x?xf32>"
  if (simpType.find("tensor<") == 0) {
    // Parse shape and element type from "tensor<2x3x4xf32>"
    size_t start = simpType.find('<') + 1;
    size_t end = simpType.find('>');
    std::string shapeAndType = simpType.substr(start, end - start);

    // Split by 'x' to get dimensions and element type
    std::vector<int64_t> shape;
    std::string elemTypeStr;
    size_t pos = 0;
    size_t lastX = 0;

    while ((pos = shapeAndType.find('x', lastX)) != std::string::npos) {
      std::string token = shapeAndType.substr(lastX, pos - lastX);
      // Try to parse as integer (dimension)
      char* endptr;
      long dim = std::strtol(token.c_str(), &endptr, 10);
      if (*endptr == '\0') {
        // Successfully parsed as integer
        shape.push_back(dim);
      } else {
        // Not a number, must be the element type
        elemTypeStr = shapeAndType.substr(lastX);
        break;
      }
      lastX = pos + 1;
    }

    // If we didn't find element type yet, the remainder is the element type
    if (elemTypeStr.empty()) {
      elemTypeStr = shapeAndType.substr(lastX);
    }

    mlir::Type elemType = convertType(elemTypeStr);
    return mlir::simp::SimpTensorType::get(&mlirContext, shape, elemType);
  }

  // Default to f32 (matches existing SimpLang compiler)
  llvm::errs() << "Warning: Unknown type '" << simpType << "', defaulting to f32\n";
  return builder.getF32Type();
}

mlir::Type MLIRCodeGenContext::getMLIRType(TypeInfo* typeInfo) {
  if (!typeInfo) {
    return builder.getF32Type(); // Default (matches existing compiler)
  }

  // For non-array/tensor types, use TypeConverter directly for better performance
  if (!typeInfo->isArray() && !typeInfo->isTensor()) {
    return TypeConverter::toMLIRType(typeInfo->kind, builder);
  }

  // Use the toString() method from TypeInfo for arrays and tensors
  return convertType(typeInfo->toString());
}

mlir::Type MLIRCodeGenContext::getElementType(const std::string& typeStr) {
  // Extract element type from array/tensor string
  if (typeStr.find('<') != std::string::npos) {
    size_t start = typeStr.rfind('<') + 1;
    size_t end = typeStr.rfind('>');
    if (end != std::string::npos && end > start) {
      std::string elemStr = typeStr.substr(start, end - start);

      // Remove any shape information (e.g., "10x20xf64" -> "f64")
      size_t lastX = elemStr.rfind('x');
      if (lastX != std::string::npos) {
        elemStr = elemStr.substr(lastX + 1);
      }

      return convertType(elemStr);
    }
  }

  return builder.getF32Type();
}

//===----------------------------------------------------------------------===//
// Location Management
//===----------------------------------------------------------------------===//

mlir::Location MLIRCodeGenContext::getLocation(int line, int col) {
  return mlir::FileLineColLoc::get(&mlirContext, sourceFileName, line, col);
}

//===----------------------------------------------------------------------===//
// AST Lowering - Main Entry Point
//===----------------------------------------------------------------------===//

mlir::ModuleOp MLIRCodeGenContext::lowerAST(BlockAST* programBlock) {
  if (!programBlock) {
    llvm::errs() << "Error: Null program block\n";
    return nullptr;
  }

  // Set insertion point to module body
  builder.setInsertionPointToEnd(module.getBody());

  // Lower all statements in the program block
  // Note: BlockAST::statements is a vector of raw pointers
  for (auto* stmt : programBlock->statements) {
    // Use getKind() to identify statement type
    if (stmt->getKind() == ASTKind::FunctionDecl) {
      FunctionAST* func = static_cast<FunctionAST*>(stmt);
      if (!lowerFunction(func)) {
        llvm::errs() << "Error: Failed to lower function\n";
        return nullptr;
      }
    } else {
      // Handle other top-level statements
      if (failed(lowerStatement(stmt))) {
        llvm::errs() << "Error: Failed to lower statement\n";
        return nullptr;
      }
    }
  }

  // Verify the module
  if (failed(mlir::verify(module))) {
    llvm::errs() << "Error: Module verification failed\n";
    module.dump();
    return nullptr;
  }

  return module;
}

//===----------------------------------------------------------------------===//
// Expression Lowering
//===----------------------------------------------------------------------===//

mlir::Value MLIRCodeGenContext::lowerExpression(ExprAST* expr) {
  if (!expr) {
    LOG_ERROR("Null expression encountered in lowerExpression");
    return nullptr;
  }

  // Log expression type for debugging
  static const char* kindNames[] = {
    "NumberExpr", "VariableExpr", "AssignmentExpr", "BinaryExpr", "UnaryExpr",
    "CallExpr", "ArrayCreateExpr", "ArrayAccessExpr", "ArrayStoreExpr",
    "MatMulExpr", "CastExpr", "TensorGetExpr", "TensorSetExpr"
  };
  int kindIndex = static_cast<int>(expr->getKind());
  LOG_DEBUG("Lowering expression: ",
            (kindIndex >= 0 && kindIndex < 13 ? kindNames[kindIndex] : "Unknown"),
            " (kind=", kindIndex, ")");

  // Use getKind() for type identification without RTTI
  switch (expr->getKind()) {
    case ASTKind::NumberExpr:
      return lowerLiteral(static_cast<NumberExprAST*>(expr));

    case ASTKind::VariableExpr:
      return lowerVariable(static_cast<VariableExprAST*>(expr));

    case ASTKind::AssignmentExpr:
      return lowerAssignment(static_cast<AssignmentExprAST*>(expr));

    case ASTKind::BinaryExpr:
      return lowerBinaryOp(static_cast<BinaryExprAST*>(expr));

    case ASTKind::UnaryExpr:
      return lowerUnaryOp(static_cast<UnaryExprAST*>(expr));

    case ASTKind::CallExpr:
      return lowerCall(static_cast<CallExprAST*>(expr));

    case ASTKind::ArrayCreateExpr:
      return lowerArrayCreate(static_cast<ArrayCreateExprAST*>(expr));

    case ASTKind::ArrayAccessExpr:
      return lowerArrayAccess(static_cast<ArrayAccessExprAST*>(expr));

    case ASTKind::ArrayStoreExpr:
      return lowerArrayStore(static_cast<ArrayStoreExprAST*>(expr));

    case ASTKind::MatMulExpr:
      return lowerMatMul(static_cast<MatMulExprAST*>(expr));

    case ASTKind::CastExpr:
      return lowerCast(static_cast<CastExprAST*>(expr));

    default:
      llvm::errs() << "Error: Unsupported expression kind " << static_cast<int>(expr->getKind())
                   << " in MLIR lowering\n";
      return nullptr;
  }
}

mlir::Value MLIRCodeGenContext::lowerLiteral(NumberExprAST* literal) {
  auto loc = getUnknownLocation();

  // Determine type based on the literal value using accessor
  double val = literal->getValue();
  mlir::Type type = builder.getF32Type(); // Default float type (matches existing compiler)
  mlir::Attribute value;

  // Check if it's an integer or floating-point
  if (val == std::floor(val) && literal->isIntegerLiteral()) {
    // Integer value
    type = builder.getI64Type();
    value = builder.getI64IntegerAttr(static_cast<int64_t>(val));
  } else {
    // Floating-point value - use f32 to match existing compiler
    type = builder.getF32Type();
    value = builder.getF32FloatAttr(static_cast<float>(val));
  }

  // Create simp.constant operation
  return builder.create<mlir::simp::ConstantOp>(loc, type, value);
}

mlir::Value MLIRCodeGenContext::lowerVariable(VariableExprAST* varExpr) {
  return lookupVariable(varExpr->getName());
}

mlir::Value MLIRCodeGenContext::lowerAssignment(AssignmentExprAST* assignment) {
  // In SSA form, assignment just creates a new SSA value
  // and updates the symbol table

  // Lower the RHS expression
  mlir::Value value = lowerExpression(assignment->getRHS());
  if (!value) {
    llvm::errs() << "Error: Failed to lower assignment RHS\n";
    return nullptr;
  }

  // Get the variable name from LHS
  const std::string& varName = assignment->getLHS()->getName();

  // Update the symbol table with the new value
  // In SSA, this creates a new binding for the variable name
  declareVariable(varName, value);

  // Return the assigned value
  return value;
}

mlir::Value MLIRCodeGenContext::lowerBinaryOp(BinaryExprAST* binOp) {
  auto loc = getUnknownLocation();

  // Lower left and right operands using accessors
  mlir::Value lhs = lowerExpression(binOp->getLeft());
  mlir::Value rhs = lowerExpression(binOp->getRight());

  if (!lhs || !rhs) {
    llvm::errs() << "Error: Failed to lower binary operation operands\n";
    return nullptr;
  }

  // Check if operands are tensors - use tensor-specific operations
  auto lhsTensorType = lhs.getType().dyn_cast<mlir::simp::SimpTensorType>();
  auto rhsTensorType = rhs.getType().dyn_cast<mlir::simp::SimpTensorType>();

  if (lhsTensorType && rhsTensorType) {
    // Both operands are tensors - use tensor element-wise operations
    switch (binOp->getOp()) {
      case OpAdd:
        return builder.create<mlir::simp::TensorAddOp>(loc, lhsTensorType, lhs, rhs);
      case OpSub:
        return builder.create<mlir::simp::TensorSubOp>(loc, lhsTensorType, lhs, rhs);
      case OpMul:
        return builder.create<mlir::simp::TensorMulOp>(loc, lhsTensorType, lhs, rhs);
      case OpDiv:
        return builder.create<mlir::simp::TensorDivOp>(loc, lhsTensorType, lhs, rhs);
      default:
        llvm::errs() << "Error: Unsupported binary operation on tensors\n";
        return nullptr;
    }
  }

  // Apply C++ style type promotion for arithmetic operations on scalars
  auto promoted = applyUsualArithmeticConversions(lhs, rhs, builder, loc);
  lhs = promoted.first;
  rhs = promoted.second;

  // Result type is the promoted common type
  mlir::Type resultType = lhs.getType();

  // Create the appropriate operation based on the operator
  // Note: MLIR operations need result type as first argument
  switch (binOp->getOp()) {
    case OpAdd:
      return builder.create<mlir::simp::AddOp>(loc, resultType, lhs, rhs);
    case OpSub:
      return builder.create<mlir::simp::SubOp>(loc, resultType, lhs, rhs);
    case OpMul:
      return builder.create<mlir::simp::MulOp>(loc, resultType, lhs, rhs);
    case OpDiv:
      return builder.create<mlir::simp::DivOp>(loc, resultType, lhs, rhs);

    // Comparison operations - use arith dialect
    // Result type is always i1 (boolean) for comparisons
    // Need to check operand type: use CmpIOp for integers, CmpFOp for floats
    case OpLT:
      if (lhs.getType().isa<mlir::IntegerType>()) {
        return builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::slt, lhs, rhs);
      } else {
        return builder.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OLT, lhs, rhs);
      }
    case OpGT:
      if (lhs.getType().isa<mlir::IntegerType>()) {
        return builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::sgt, lhs, rhs);
      } else {
        return builder.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OGT, lhs, rhs);
      }
    case OpLE:
      if (lhs.getType().isa<mlir::IntegerType>()) {
        return builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::sle, lhs, rhs);
      } else {
        return builder.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OLE, lhs, rhs);
      }
    case OpGE:
      if (lhs.getType().isa<mlir::IntegerType>()) {
        return builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::sge, lhs, rhs);
      } else {
        return builder.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OGE, lhs, rhs);
      }
    case OpEQ:
      if (lhs.getType().isa<mlir::IntegerType>()) {
        return builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::eq, lhs, rhs);
      } else {
        return builder.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OEQ, lhs, rhs);
      }
    case OpNE:
      if (lhs.getType().isa<mlir::IntegerType>()) {
        return builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::ne, lhs, rhs);
      } else {
        return builder.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::ONE, lhs, rhs);
      }

    // Bitwise operations - only valid for integer types
    case OpAnd:
      if (!lhs.getType().isa<mlir::IntegerType>()) {
        llvm::errs() << "Error: Bitwise AND (&) can only be applied to integer types\n";
        return nullptr;
      }
      return builder.create<mlir::arith::AndIOp>(loc, lhs, rhs);

    case OpOr:
      if (!lhs.getType().isa<mlir::IntegerType>()) {
        llvm::errs() << "Error: Bitwise OR (|) can only be applied to integer types\n";
        return nullptr;
      }
      return builder.create<mlir::arith::OrIOp>(loc, lhs, rhs);

    case OpXor:
      if (!lhs.getType().isa<mlir::IntegerType>()) {
        llvm::errs() << "Error: Bitwise XOR (^) can only be applied to integer types\n";
        return nullptr;
      }
      return builder.create<mlir::arith::XOrIOp>(loc, lhs, rhs);

    case OpLShift:
      if (!lhs.getType().isa<mlir::IntegerType>()) {
        llvm::errs() << "Error: Left shift (<<) can only be applied to integer types\n";
        return nullptr;
      }
      return builder.create<mlir::arith::ShLIOp>(loc, lhs, rhs);

    case OpRShift:
      if (!lhs.getType().isa<mlir::IntegerType>()) {
        llvm::errs() << "Error: Right shift (>>) can only be applied to integer types\n";
        return nullptr;
      }
      // Use arithmetic right shift (sign-extending for signed integers)
      return builder.create<mlir::arith::ShRSIOp>(loc, lhs, rhs);

    case OpMod:
      if (lhs.getType().isa<mlir::IntegerType>()) {
        return builder.create<mlir::arith::RemSIOp>(loc, lhs, rhs);
      } else {
        return builder.create<mlir::arith::RemFOp>(loc, lhs, rhs);
      }

    default:
      llvm::errs() << "Error: Unsupported binary operator: " << binOp->getOp() << "\n";
      return nullptr;
  }
}

mlir::Value MLIRCodeGenContext::lowerUnaryOp(UnaryExprAST* unaryOp) {
  auto loc = getUnknownLocation();

  // Lower the operand
  mlir::Value operand = lowerExpression(unaryOp->getOperand());
  if (!operand) {
    llvm::errs() << "Error: Failed to lower unary operand\n";
    return nullptr;
  }

  // Get result type (same as operand type for negation)
  mlir::Type resultType = operand.getType();

  // Handle the unary operation
  switch (unaryOp->getOp()) {
    case OpNeg:
      return builder.create<mlir::simp::NegOp>(loc, resultType, operand);

    default:
      llvm::errs() << "Error: Unsupported unary operator: " << unaryOp->getOp() << "\n";
      return nullptr;
  }
}

mlir::Value MLIRCodeGenContext::lowerCast(CastExprAST* castExpr) {
  auto loc = getUnknownLocation();

  // Lower the expression being cast
  mlir::Value value = lowerExpression(castExpr->getExpr());
  if (!value) {
    llvm::errs() << "Error: Failed to lower cast expression\n";
    return nullptr;
  }

  // Get the target type
  mlir::Type targetType = getMLIRType(castExpr->getTargetType());
  mlir::Type sourceType = value.getType();

  // If types are already the same, no cast needed
  if (sourceType == targetType) {
    return value;
  }

  // Handle conversions
  bool sourceIsInt = sourceType.isInteger(64);
  bool targetIsInt = targetType.isInteger(64);
  bool sourceIsFloat = sourceType.isa<mlir::Float32Type>() || sourceType.isa<mlir::Float64Type>();
  bool targetIsFloat = targetType.isa<mlir::Float32Type>() || targetType.isa<mlir::Float64Type>();

  // int -> float
  if (sourceIsInt && targetIsFloat) {
    return builder.create<mlir::arith::SIToFPOp>(loc, targetType, value);
  }

  // float -> int
  if (sourceIsFloat && targetIsInt) {
    return builder.create<mlir::arith::FPToSIOp>(loc, targetType, value);
  }

  // float -> float (different precision)
  if (sourceIsFloat && targetIsFloat) {
    if (sourceType.getIntOrFloatBitWidth() < targetType.getIntOrFloatBitWidth()) {
      // Extend
      return builder.create<mlir::arith::ExtFOp>(loc, targetType, value);
    } else {
      // Truncate
      return builder.create<mlir::arith::TruncFOp>(loc, targetType, value);
    }
  }

  // int -> int (different width)
  if (sourceIsInt && targetIsInt) {
    if (sourceType.getIntOrFloatBitWidth() < targetType.getIntOrFloatBitWidth()) {
      // Extend (signed)
      return builder.create<mlir::arith::ExtSIOp>(loc, targetType, value);
    } else {
      // Truncate
      return builder.create<mlir::arith::TruncIOp>(loc, targetType, value);
    }
  }

  llvm::errs() << "Error: Unsupported cast from " << sourceType << " to " << targetType << "\n";
  return nullptr;
}

mlir::Value MLIRCodeGenContext::lowerArrayCreate(ArrayCreateExprAST* arrayCreate) {
  auto loc = getUnknownLocation();

  const auto& expressions = arrayCreate->getDimensions();
  if (expressions.empty()) {
    llvm::errs() << "Error: Array must have at least one dimension or initializer\n";
    return nullptr;
  }

  mlir::Type elemType = getMLIRType(arrayCreate->getElementType());
  mlir::Type arrayType = mlir::simp::ArrayType::get(&mlirContext, elemType);

  // Check if this is an initializer list
  if (arrayCreate->isInitializer()) {
    // Create array with size = number of initializer values
    mlir::Value size = builder.create<mlir::simp::ConstantOp>(
        loc, builder.getI64Type(),
        builder.getI64IntegerAttr(expressions.size()));

    mlir::Value array = builder.create<mlir::simp::ArrayCreateOp>(loc, arrayType, size);

    // Initialize each element
    for (size_t i = 0; i < expressions.size(); ++i) {
      mlir::Value value = lowerExpression(expressions[i].get());
      if (!value) {
        llvm::errs() << "Error: Failed to lower initializer expression\n";
        return nullptr;
      }

      mlir::Value index = builder.create<mlir::simp::ConstantOp>(
          loc, builder.getI64Type(), builder.getI64IntegerAttr(i));

      array = builder.create<mlir::simp::ArraySetOp>(loc, arrayType, array, index, value);
    }

    return array;
  } else {
    // Dimension specification: lower all dimensions
    mlir::SmallVector<mlir::Value> dimValues;
    for (const auto& dim : expressions) {
      mlir::Value dimValue = lowerExpression(dim.get());
      if (!dimValue) {
        llvm::errs() << "Error: Failed to lower array dimension\n";
        return nullptr;
      }
      dimValues.push_back(dimValue);
    }

    // Calculate total size as product of all dimensions
    mlir::Value totalSize = dimValues[0];
    for (size_t i = 1; i < dimValues.size(); ++i) {
      totalSize = builder.create<mlir::arith::MulIOp>(loc, totalSize, dimValues[i]);
    }

    // Create simp.array_create operation with total flattened size
    return builder.create<mlir::simp::ArrayCreateOp>(loc, arrayType, totalSize);
  }
}

mlir::Value MLIRCodeGenContext::lowerArrayAccess(ArrayAccessExprAST* arrayAccess,
                                                  mlir::Value newValue) {
  auto loc = getUnknownLocation();

  // Lower the array expression
  mlir::Value array = lowerExpression(arrayAccess->getArray());
  if (!array) {
    llvm::errs() << "Error: Failed to lower array expression\n";
    return nullptr;
  }

  // Lower all index expressions
  const auto& indices = arrayAccess->getIndices();
  if (indices.empty()) {
    llvm::errs() << "Error: Array/tensor access requires at least one index\n";
    return nullptr;
  }

  mlir::SmallVector<mlir::Value> indexValues;
  for (const auto& idx : indices) {
    mlir::Value indexValue = lowerExpression(idx.get());
    if (!indexValue) {
      llvm::errs() << "Error: Failed to lower index\n";
      return nullptr;
    }
    indexValues.push_back(indexValue);
  }

  // Check if this is a tensor type (static multi-dimensional)
  auto tensorType = array.getType().dyn_cast<mlir::simp::SimpTensorType>();
  if (tensorType) {
    // For tensors, use multi-dimensional indexing directly
    if (newValue) {
      // tensor_set: returns updated tensor
      return builder.create<mlir::simp::TensorSetOp>(loc, tensorType, array, indexValues, newValue);
    } else {
      // tensor_get: returns element
      mlir::Type elemType = tensorType.getElementType();
      return builder.create<mlir::simp::TensorGetOp>(loc, elemType, array, indexValues);
    }
  }

  // Compute flattened index for multi-dimensional arrays
  mlir::Value flatIndex;

  if (indexValues.size() == 1) {
    // Simple 1D case
    flatIndex = indexValues[0];
  } else {
    // Multi-dimensional: compute flattened index using row-major layout
    // flat_index = i0 * (D1*D2*...*Dn) + i1 * (D2*D3*...*Dn) + ... + in

    // Get the array variable name (if this is a variable reference)
    std::string arrayVarName;
    if (arrayAccess->getArray()->getKind() == ASTKind::VariableExpr) {
      VariableExprAST* varExpr = static_cast<VariableExprAST*>(arrayAccess->getArray());
      arrayVarName = varExpr->getName();
    }

    // Look up stored dimensions for this array by variable name
    auto dimIt = arrayDimensions.find(arrayVarName);
    if (dimIt == arrayDimensions.end() || arrayVarName.empty()) {
      llvm::errs() << "Error: Multi-dimensional array access but dimensions not tracked\n";
      llvm::errs() << "       Variable name: " << (arrayVarName.empty() ? "<unknown>" : arrayVarName) << "\n";
      llvm::errs() << "       This array may not have been created with explicit dimensions\n";
      return nullptr;
    }

    const auto& dims = dimIt->second;
    if (indexValues.size() != dims.size()) {
      llvm::errs() << "Error: Index count (" << indexValues.size()
                   << ") doesn't match array dimensions (" << dims.size() << ")\n";
      return nullptr;
    }

    // Compute strides: stride[i] = product(dims[i+1] ... dims[n-1])
    // For [D0, D1, D2], strides are [D1*D2, D2, 1]
    mlir::SmallVector<mlir::Value> strides = computeRowMajorStrides(dims, builder, loc);

    // Compute flat_index = sum(indices[i] * strides[i])
    // Cast all values to index type for arithmetic
    mlir::Value idx0 = castToIndex(indexValues[0], builder, loc);
    flatIndex = builder.create<mlir::arith::MulIOp>(loc, idx0, strides[0]);
    for (size_t i = 1; i < indexValues.size(); ++i) {
      mlir::Value idx = castToIndex(indexValues[i], builder, loc);
      mlir::Value term = builder.create<mlir::arith::MulIOp>(loc, idx, strides[i]);
      flatIndex = builder.create<mlir::arith::AddIOp>(loc, flatIndex, term);
    }
  }

  // Ensure final index is i64 for array operations
  mlir::Value index = flatIndex;
  if (!index.getType().isa<mlir::IntegerType>() ||
      index.getType().cast<mlir::IntegerType>().getWidth() != 64) {
    index = builder.create<mlir::arith::IndexCastOp>(loc, builder.getI64Type(), flatIndex);
  }

  // If newValue is provided, this is an array_set
  if (newValue) {
    // Cast value to match array element type if needed
    auto arrayType = array.getType().dyn_cast<mlir::simp::ArrayType>();
    if (arrayType) {
      mlir::Type elemType = arrayType.getElementType();
      mlir::Type valueType = newValue.getType();

      if (elemType != valueType) {
        // Handle float type mismatches
        if (elemType.isa<mlir::FloatType>() && valueType.isa<mlir::FloatType>()) {
          auto expectedFloat = elemType.cast<mlir::FloatType>();
          auto actualFloat = valueType.cast<mlir::FloatType>();

          if (expectedFloat.getWidth() > actualFloat.getWidth()) {
            // Extend: f32 -> f64, f16 -> f32, etc.
            newValue = builder.create<mlir::arith::ExtFOp>(loc, elemType, newValue);
          } else {
            // Truncate: f64 -> f32, f32 -> f16, etc.
            newValue = builder.create<mlir::arith::TruncFOp>(loc, elemType, newValue);
          }
        } else if (elemType.isa<mlir::IntegerType>() && valueType.isa<mlir::IntegerType>()) {
          // Handle integer conversions
          auto expectedInt = elemType.cast<mlir::IntegerType>();
          auto actualInt = valueType.cast<mlir::IntegerType>();

          if (expectedInt.getWidth() > actualInt.getWidth()) {
            newValue = builder.create<mlir::arith::ExtSIOp>(loc, elemType, newValue);
          } else if (expectedInt.getWidth() < actualInt.getWidth()) {
            newValue = builder.create<mlir::arith::TruncIOp>(loc, elemType, newValue);
          }
        } else if (elemType.isa<mlir::FloatType>() && valueType.isa<mlir::IntegerType>()) {
          // Int to float conversion (C++ style promotion)
          newValue = builder.create<mlir::arith::SIToFPOp>(loc, elemType, newValue);
        } else if (elemType.isa<mlir::IntegerType>() && valueType.isa<mlir::FloatType>()) {
          // Float to int conversion (truncation)
          newValue = builder.create<mlir::arith::FPToSIOp>(loc, elemType, newValue);
        }
      }
    }

    return builder.create<mlir::simp::ArraySetOp>(loc, array.getType(),
                                                   array, index, newValue);
  } else {
    // Otherwise, it's an array_get
    // Get element type from array type
    auto arrayType = array.getType().dyn_cast<mlir::simp::ArrayType>();
    if (!arrayType) {
      llvm::errs() << "Error: Array access on non-array type\n";
      return nullptr;
    }

    mlir::Type elemType = arrayType.getElementType();
    return builder.create<mlir::simp::ArrayGetOp>(loc, elemType, array, index);
  }
}

mlir::Value MLIRCodeGenContext::lowerArrayStore(ArrayStoreExprAST* arrayStore) {
  auto loc = getUnknownLocation();

  // Check if we're storing to a variable (need to update symbol table)
  std::string varName;
  if (arrayStore->getArray()->getKind() == ASTKind::VariableExpr) {
    auto* varExpr = static_cast<VariableExprAST*>(arrayStore->getArray());
    varName = varExpr->getName();
  }

  // Lower the array expression
  mlir::Value array = lowerExpression(arrayStore->getArray());
  if (!array) {
    llvm::errs() << "Error: Failed to lower array expression in store\n";
    return nullptr;
  }

  // Lower all index expressions
  const auto& indices = arrayStore->getIndices();
  if (indices.empty()) {
    llvm::errs() << "Error: Array store requires at least one index\n";
    return nullptr;
  }

  mlir::SmallVector<mlir::Value> indexValues;
  for (const auto& idx : indices) {
    mlir::Value indexValue = lowerExpression(idx.get());
    if (!indexValue) {
      llvm::errs() << "Error: Failed to lower array index in store\n";
      return nullptr;
    }
    indexValues.push_back(indexValue);
  }

  // Lower the value to store
  mlir::Value value = lowerExpression(arrayStore->getValue());
  if (!value) {
    llvm::errs() << "Error: Failed to lower store value\n";
    return nullptr;
  }

  // Check if this is a tensor type
  auto tensorType = array.getType().dyn_cast<mlir::simp::SimpTensorType>();
  if (tensorType) {
    // For tensors, use tensor_set with multi-dimensional indices
    mlir::Value result = builder.create<mlir::simp::TensorSetOp>(loc, tensorType, array, indexValues, value);

    // Update symbol table if this is a variable assignment
    if (!varName.empty()) {
      declareVariable(varName, result);
    }

    return result;
  }

  // Compute flattened index for multi-dimensional arrays
  mlir::Value index;
  if (indexValues.size() == 1) {
    index = indexValues[0];
  } else {
    // Multi-dimensional: flatten using row-major stride computation
    // Look up dimensions by variable name
    auto dimIt = arrayDimensions.find(varName);
    if (dimIt == arrayDimensions.end()) {
      llvm::errs() << "Error: Multi-dimensional array dimensions not tracked for " << varName << "\n";
      return nullptr;
    }

    const auto& dims = dimIt->second;
    if (dims.size() != indexValues.size()) {
      llvm::errs() << "Error: Dimension mismatch in array store\n";
      return nullptr;
    }

    // Compute row-major strides
    mlir::SmallVector<mlir::Value> strides = computeRowMajorStrides(dims, builder, loc);

    // Flatten: index = i0*stride[0] + i1*stride[1] + ... + in*stride[n]
    mlir::Value idx0 = castToIndex(indexValues[0], builder, loc);
    index = builder.create<mlir::arith::MulIOp>(loc, idx0, strides[0]);

    for (size_t i = 1; i < indexValues.size(); ++i) {
      mlir::Value idx = castToIndex(indexValues[i], builder, loc);
      mlir::Value term = builder.create<mlir::arith::MulIOp>(loc, idx, strides[i]);
      index = builder.create<mlir::arith::AddIOp>(loc, index, term);
    }
  }

  // Cast to i64 for array operations
  if (!index.getType().isa<mlir::IntegerType>() ||
      index.getType().cast<mlir::IntegerType>().getWidth() != 64) {
    index = builder.create<mlir::arith::IndexCastOp>(loc, builder.getI64Type(), index);
  }

  // Cast value to match array element type if needed (value already lowered above)
  auto arrayType = array.getType().dyn_cast<mlir::simp::ArrayType>();
  if (arrayType) {
    mlir::Type elemType = arrayType.getElementType();
    mlir::Type valueType = value.getType();

    if (elemType != valueType) {
      // Handle float type mismatches
      if (elemType.isa<mlir::FloatType>() && valueType.isa<mlir::FloatType>()) {
        auto expectedFloat = elemType.cast<mlir::FloatType>();
        auto actualFloat = valueType.cast<mlir::FloatType>();

        if (expectedFloat.getWidth() > actualFloat.getWidth()) {
          // Extend: f32 -> f64, f16 -> f32, etc.
          value = builder.create<mlir::arith::ExtFOp>(loc, elemType, value);
        } else {
          // Truncate: f64 -> f32, f32 -> f16, etc.
          value = builder.create<mlir::arith::TruncFOp>(loc, elemType, value);
        }
      } else if (elemType.isa<mlir::IntegerType>() && valueType.isa<mlir::IntegerType>()) {
        // Handle integer conversions
        auto expectedInt = elemType.cast<mlir::IntegerType>();
        auto actualInt = valueType.cast<mlir::IntegerType>();

        if (expectedInt.getWidth() > actualInt.getWidth()) {
          value = builder.create<mlir::arith::ExtSIOp>(loc, elemType, value);
        } else if (expectedInt.getWidth() < actualInt.getWidth()) {
          value = builder.create<mlir::arith::TruncIOp>(loc, elemType, value);
        }
      } else if (elemType.isa<mlir::FloatType>() && valueType.isa<mlir::IntegerType>()) {
        // Int to float conversion (C++ style promotion)
        value = builder.create<mlir::arith::SIToFPOp>(loc, elemType, value);
      } else if (elemType.isa<mlir::IntegerType>() && valueType.isa<mlir::FloatType>()) {
        // Float to int conversion (truncation)
        value = builder.create<mlir::arith::FPToSIOp>(loc, elemType, value);
      }
    }
  }

  // Create simp.array_set operation (returns new array in SSA form)
  mlir::Value newArray = builder.create<mlir::simp::ArraySetOp>(
      loc, array.getType(), array, index, value);

  // Update the symbol table if storing to a variable
  if (!varName.empty()) {
    declareVariable(varName, newArray);
  }

  return newArray;
}

mlir::Value MLIRCodeGenContext::lowerMatMul(MatMulExprAST* matmul) {
  auto loc = getUnknownLocation();

  // Lower the left-hand side matrix (A: MxK)
  mlir::Value lhs = lowerExpression(matmul->getLHS());
  if (!lhs) {
    llvm::errs() << "Error: Failed to lower matmul LHS\n";
    return nullptr;
  }

  // Lower the right-hand side matrix (B: KxN)
  mlir::Value rhs = lowerExpression(matmul->getRHS());
  if (!rhs) {
    llvm::errs() << "Error: Failed to lower matmul RHS\n";
    return nullptr;
  }

  // Lower the dimension arguments: m, k, n
  mlir::Value m = lowerExpression(matmul->getM());
  if (!m) {
    llvm::errs() << "Error: Failed to lower matmul dimension m\n";
    return nullptr;
  }

  mlir::Value k = lowerExpression(matmul->getK());
  if (!k) {
    llvm::errs() << "Error: Failed to lower matmul dimension k\n";
    return nullptr;
  }

  mlir::Value n = lowerExpression(matmul->getN());
  if (!n) {
    llvm::errs() << "Error: Failed to lower matmul dimension n\n";
    return nullptr;
  }

  // Lower the output buffer (pre-allocated by caller)
  mlir::Value output = lowerExpression(matmul->getOutput());
  if (!output) {
    llvm::errs() << "Error: Failed to lower matmul output buffer\n";
    return nullptr;
  }

  // Lower the offset arguments
  mlir::Value lhs_offset = lowerExpression(matmul->getLHSOffset());
  if (!lhs_offset) {
    llvm::errs() << "Error: Failed to lower matmul lhs_offset\n";
    return nullptr;
  }

  mlir::Value rhs_offset = lowerExpression(matmul->getRHSOffset());
  if (!rhs_offset) {
    llvm::errs() << "Error: Failed to lower matmul rhs_offset\n";
    return nullptr;
  }

  mlir::Value output_offset = lowerExpression(matmul->getOutputOffset());
  if (!output_offset) {
    llvm::errs() << "Error: Failed to lower matmul output_offset\n";
    return nullptr;
  }

  // Get the array type
  auto arrayType = output.getType().dyn_cast<mlir::simp::ArrayType>();
  if (!arrayType) {
    llvm::errs() << "Error: MatMul output is not an array type\n";
    return nullptr;
  }

  // Create the simp.matmul operation with provided offsets
  // The output buffer is pre-allocated - matmul writes in-place
  return builder.create<mlir::simp::MatMulOp>(
      loc, arrayType, lhs, rhs, output, m, k, n, lhs_offset, rhs_offset, output_offset);
}

mlir::Value MLIRCodeGenContext::lowerCall(CallExprAST* call) {
  auto loc = getUnknownLocation();

  // Get the callee name
  const std::string& calleeName = call->getCallee();

  // Lower all argument expressions
  llvm::SmallVector<mlir::Value, 4> args;
  for (auto* argExpr : call->getArguments()) {
    mlir::Value arg = lowerExpression(argExpr);
    if (!arg) {
      llvm::errs() << "Error: Failed to lower call argument\n";
      return nullptr;
    }
    args.push_back(arg);
  }

  // Handle builtin functions: conv2d
  if (calleeName == "conv2d") {
    // conv2d(input, weights, bias, output, batch, in_h, in_w, in_c, out_c, k_h, k_w, stride_h, stride_w, pad_h, pad_w)
    if (args.size() != 15) {
      llvm::errs() << "Error: conv2d requires 15 arguments (got " << args.size() << ")\n";
      return nullptr;
    }

    mlir::Value input = args[0];
    mlir::Value weights = args[1];
    mlir::Value bias = args[2];
    mlir::Value output = args[3];
    mlir::Value batch = args[4];
    mlir::Value in_h = args[5];
    mlir::Value in_w = args[6];
    mlir::Value in_c = args[7];
    mlir::Value out_c = args[8];
    mlir::Value kernel_h = args[9];
    mlir::Value kernel_w = args[10];
    mlir::Value stride_h = args[11];
    mlir::Value stride_w = args[12];
    mlir::Value pad_h = args[13];
    mlir::Value pad_w = args[14];

    // Get the output array type to determine return type
    mlir::Type outputArrayType = output.getType();

    // Create the simp.conv2d operation
    auto conv2dOp = builder.create<simp::Conv2DOp>(
        loc,
        outputArrayType,  // Result type (same as output buffer)
        input, weights, bias, output,
        batch, in_h, in_w, in_c,
        out_c, kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w
    );

    return conv2dOp.getResult();
  }

  // Handle builtin: matmul(lhs, rhs, output, m, k, n, lhs_offset, rhs_offset, output_offset)
  if (calleeName == "matmul") {
    if (args.size() != 9) {
      llvm::errs() << "Error: matmul requires 9 arguments (got " << args.size() << ")\n";
      return nullptr;
    }

    mlir::Value lhs = args[0];
    mlir::Value rhs = args[1];
    mlir::Value output = args[2];
    mlir::Value m = args[3];
    mlir::Value k = args[4];
    mlir::Value n = args[5];
    mlir::Value lhs_offset = args[6];
    mlir::Value rhs_offset = args[7];
    mlir::Value output_offset = args[8];

    mlir::Type outputArrayType = output.getType();

    auto matmulOp = builder.create<simp::MatMulOp>(
        loc, outputArrayType,
        lhs, rhs, output, m, k, n, lhs_offset, rhs_offset, output_offset
    );

    return matmulOp.getResult();
  }

  // Handle builtin: rmsnorm(input, weight, output, size, epsilon, weight_offset)
  if (calleeName == "rmsnorm") {
    if (args.size() != 6) {
      llvm::errs() << "Error: rmsnorm requires 6 arguments (got " << args.size() << ")\n";
      return nullptr;
    }

    mlir::Value input = args[0];
    mlir::Value weight = args[1];
    mlir::Value output = args[2];
    mlir::Value size = args[3];
    mlir::Value epsilon = args[4];
    mlir::Value weight_offset = args[5];

    mlir::Type outputArrayType = output.getType();

    auto rmsnormOp = builder.create<simp::RMSNormOp>(
        loc, outputArrayType,
        input, weight, output, size, epsilon, weight_offset
    );

    return rmsnormOp.getResult();
  }

  // Handle builtin: softmax(input, output, size, input_offset, output_offset)
  if (calleeName == "softmax") {
    if (args.size() != 5) {
      llvm::errs() << "Error: softmax requires 5 arguments (got " << args.size() << ")\n";
      return nullptr;
    }

    mlir::Value input = args[0];
    mlir::Value output = args[1];
    mlir::Value size = args[2];
    mlir::Value input_offset = args[3];
    mlir::Value output_offset = args[4];

    mlir::Type outputArrayType = output.getType();

    auto softmaxOp = builder.create<simp::SoftmaxOp>(
        loc, outputArrayType,
        input, output, size, input_offset, output_offset
    );

    return softmaxOp.getResult();
  }

  // Handle builtin: silu(input, output, size)
  if (calleeName == "silu") {
    if (args.size() != 3) {
      llvm::errs() << "Error: silu requires 3 arguments (got " << args.size() << ")\n";
      return nullptr;
    }

    mlir::Value input = args[0];
    mlir::Value output = args[1];
    mlir::Value size = args[2];

    mlir::Type outputArrayType = output.getType();

    auto siluOp = builder.create<simp::SiLUOp>(
        loc, outputArrayType,
        input, output, size
    );

    return siluOp.getResult();
  }

  // Handle builtin: dequant_w4(qweights, scales, zeros, idx, group_size)
  if (calleeName == "dequant_w4") {
    if (args.size() != 5) {
      llvm::errs() << "Error: dequant_w4 requires 5 arguments (got " << args.size() << ")\n";
      return nullptr;
    }

    mlir::Value qweights = args[0];
    mlir::Value scales = args[1];
    mlir::Value zeros = args[2];
    mlir::Value idx = args[3];
    mlir::Value group_size = args[4];

    // Result type is f32
    mlir::Type f32Type = builder.getF32Type();

    auto dequantOp = builder.create<simp::DequantW4Op>(
        loc, f32Type,
        qweights, scales, zeros, idx, group_size
    );

    return dequantOp.getResult();
  }

  // Handle builtin: matmul_quant(qweights, scales, zeros, input, output, rows, cols, group_size, offset)
  if (calleeName == "matmul_quant") {
    if (args.size() != 9) {
      llvm::errs() << "Error: matmul_quant requires 9 arguments (got " << args.size() << ")\n";
      return nullptr;
    }

    mlir::Value qweights = args[0];
    mlir::Value scales = args[1];
    mlir::Value zeros = args[2];
    mlir::Value input = args[3];
    mlir::Value output = args[4];
    mlir::Value rows = args[5];
    mlir::Value cols = args[6];
    mlir::Value group_size = args[7];
    mlir::Value offset = args[8];

    mlir::Type outputArrayType = output.getType();

    auto matmulQuantOp = builder.create<simp::MatMulQuantOp>(
        loc, outputArrayType,
        qweights, scales, zeros, input, output, rows, cols, group_size, offset
    );

    return matmulQuantOp.getResult();
  }

  // Handle builtin math functions: sqrt, exp
  if (calleeName == "sqrt") {
    if (args.size() != 1) {
      llvm::errs() << "Error: sqrt requires 1 argument (got " << args.size() << ")\n";
      return nullptr;
    }
    return builder.create<math::SqrtOp>(loc, args[0]);
  }

  if (calleeName == "exp") {
    if (args.size() != 1) {
      llvm::errs() << "Error: exp requires 1 argument (got " << args.size() << ")\n";
      return nullptr;
    }
    return builder.create<math::ExpOp>(loc, args[0]);
  }

  if (calleeName == "log") {
    if (args.size() != 1) {
      llvm::errs() << "Error: log requires 1 argument (got " << args.size() << ")\n";
      return nullptr;
    }
    return builder.create<math::LogOp>(loc, args[0]);
  }

  if (calleeName == "pow") {
    if (args.size() != 2) {
      llvm::errs() << "Error: pow requires 2 arguments (got " << args.size() << ")\n";
      return nullptr;
    }
    return builder.create<math::PowFOp>(loc, args[0], args[1]);
  }

  if (calleeName == "cos") {
    if (args.size() != 1) {
      llvm::errs() << "Error: cos requires 1 argument (got " << args.size() << ")\n";
      return nullptr;
    }
    return builder.create<math::CosOp>(loc, args[0]);
  }

  if (calleeName == "sin") {
    if (args.size() != 1) {
      llvm::errs() << "Error: sin requires 1 argument (got " << args.size() << ")\n";
      return nullptr;
    }
    return builder.create<math::SinOp>(loc, args[0]);
  }

  // Handle tensor reduction builtins
  if (calleeName == "tensor_sum") {
    if (args.size() < 1 || args.size() > 2) {
      llvm::errs() << "Error: tensor_sum requires 1-2 arguments (tensor, [axis]), got " << args.size() << "\n";
      return nullptr;
    }
    mlir::Value tensor = args[0];
    auto tensorType = tensor.getType().dyn_cast<mlir::simp::SimpTensorType>();
    if (!tensorType) {
      llvm::errs() << "Error: tensor_sum requires a tensor argument\n";
      return nullptr;
    }

    mlir::Value axisValue = nullptr;
    mlir::Type resultType;

    if (args.size() == 2) {
      axisValue = args[1];

      // Extract axis value to compute result shape
      int64_t axis = -1;
      if (auto simpConstOp = axisValue.getDefiningOp<mlir::simp::ConstantOp>()) {
        if (auto intAttr = simpConstOp.value().dyn_cast<mlir::IntegerAttr>()) {
          axis = intAttr.getInt();
        }
      } else if (auto arithConstOp = axisValue.getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto intAttr = arithConstOp.getValue().dyn_cast<mlir::IntegerAttr>()) {
          axis = intAttr.getInt();
        }
      }

      if (axis != -1) {
        auto shape = tensorType.getShape();
        int64_t rank = shape.size();
        if (axis < 0) axis += rank;

        // Compute result shape (remove axis dimension)
        llvm::SmallVector<int64_t, 4> resultShape;
        for (int64_t i = 0; i < rank; i++) {
          if (i != axis) {
            resultShape.push_back(shape[i]);
          }
        }
        resultType = mlir::simp::SimpTensorType::get(builder.getContext(), resultShape, tensorType.getElementType());
      } else {
        // Couldn't extract axis, use input type as placeholder
        resultType = tensorType;
      }
    } else {
      // No axis: result is scalar
      resultType = tensorType.getElementType();
    }

    return builder.create<mlir::simp::TensorSumOp>(loc, resultType, tensor, axisValue);
  }

  // Helper lambda to compute result type for axis reductions
  auto computeAxisReductionType = [&](mlir::simp::SimpTensorType tensorType, mlir::Value axisValue, mlir::Type elemType) -> mlir::Type {
    if (!axisValue) return elemType;  // No axis: scalar result

    // Extract axis to compute result shape
    int64_t axis = -1;
    if (auto simpConstOp = axisValue.getDefiningOp<mlir::simp::ConstantOp>()) {
      if (auto intAttr = simpConstOp.value().dyn_cast<mlir::IntegerAttr>()) {
        axis = intAttr.getInt();
      }
    } else if (auto arithConstOp = axisValue.getDefiningOp<mlir::arith::ConstantOp>()) {
      if (auto intAttr = arithConstOp.getValue().dyn_cast<mlir::IntegerAttr>()) {
        axis = intAttr.getInt();
      }
    }

    if (axis != -1) {
      auto shape = tensorType.getShape();
      int64_t rank = shape.size();
      if (axis < 0) axis += rank;

      llvm::SmallVector<int64_t, 4> resultShape;
      for (int64_t i = 0; i < rank; i++) {
        if (i != axis) {
          resultShape.push_back(shape[i]);
        }
      }
      return mlir::simp::SimpTensorType::get(builder.getContext(), resultShape, elemType);
    }
    return tensorType;  // Fallback
  };

  if (calleeName == "tensor_mean") {
    if (args.size() < 1 || args.size() > 2) {
      llvm::errs() << "Error: tensor_mean requires 1-2 arguments (tensor, [axis]), got " << args.size() << "\n";
      return nullptr;
    }
    mlir::Value tensor = args[0];
    auto tensorType = tensor.getType().dyn_cast<mlir::simp::SimpTensorType>();
    if (!tensorType) {
      llvm::errs() << "Error: tensor_mean requires a tensor argument\n";
      return nullptr;
    }

    mlir::Value axisValue = (args.size() == 2) ? args[1] : nullptr;
    mlir::Type resultType = computeAxisReductionType(tensorType, axisValue, tensorType.getElementType());
    return builder.create<mlir::simp::TensorMeanOp>(loc, resultType, tensor, axisValue);
  }

  if (calleeName == "tensor_max") {
    if (args.size() < 1 || args.size() > 2) {
      llvm::errs() << "Error: tensor_max requires 1-2 arguments (tensor, [axis]), got " << args.size() << "\n";
      return nullptr;
    }
    mlir::Value tensor = args[0];
    auto tensorType = tensor.getType().dyn_cast<mlir::simp::SimpTensorType>();
    if (!tensorType) {
      llvm::errs() << "Error: tensor_max requires a tensor argument\n";
      return nullptr;
    }

    mlir::Value axisValue = (args.size() == 2) ? args[1] : nullptr;
    mlir::Type resultType = computeAxisReductionType(tensorType, axisValue, tensorType.getElementType());
    return builder.create<mlir::simp::TensorMaxOp>(loc, resultType, tensor, axisValue);
  }

  if (calleeName == "tensor_min") {
    if (args.size() < 1 || args.size() > 2) {
      llvm::errs() << "Error: tensor_min requires 1-2 arguments (tensor, [axis]), got " << args.size() << "\n";
      return nullptr;
    }
    mlir::Value tensor = args[0];
    auto tensorType = tensor.getType().dyn_cast<mlir::simp::SimpTensorType>();
    if (!tensorType) {
      llvm::errs() << "Error: tensor_min requires a tensor argument\n";
      return nullptr;
    }

    mlir::Value axisValue = (args.size() == 2) ? args[1] : nullptr;
    mlir::Type resultType = computeAxisReductionType(tensorType, axisValue, tensorType.getElementType());
    return builder.create<mlir::simp::TensorMinOp>(loc, resultType, tensor, axisValue);
  }

  if (calleeName == "tensor_argmax") {
    if (args.size() < 1 || args.size() > 2) {
      llvm::errs() << "Error: tensor_argmax requires 1-2 arguments (tensor, [axis]), got " << args.size() << "\n";
      return nullptr;
    }
    mlir::Value tensor = args[0];
    auto tensorType = tensor.getType().dyn_cast<mlir::simp::SimpTensorType>();
    if (!tensorType) {
      llvm::errs() << "Error: tensor_argmax requires a tensor argument\n";
      return nullptr;
    }

    mlir::Value axisValue = (args.size() == 2) ? args[1] : nullptr;
    mlir::Type resultType = computeAxisReductionType(tensorType, axisValue, builder.getI64Type());
    return builder.create<mlir::simp::TensorArgmaxOp>(loc, resultType, tensor, axisValue);
  }

  // tensor_reshape: tensor_reshape(tensor, dim0, dim1, ...)
  if (calleeName == "tensor_reshape") {
    if (args.size() < 2) {
      llvm::errs() << "Error: tensor_reshape requires at least 2 arguments (tensor, new_dims...), got " << args.size() << "\n";
      return nullptr;
    }

    mlir::Value tensor = args[0];
    auto tensorType = tensor.getType().dyn_cast<mlir::simp::SimpTensorType>();
    if (!tensorType) {
      llvm::errs() << "Error: tensor_reshape requires a tensor argument\n";
      return nullptr;
    }

    // Extract new shape from arguments
    llvm::SmallVector<int64_t, 4> newShape;
    for (size_t i = 1; i < args.size(); i++) {
      if (auto constOp = args[i].getDefiningOp<mlir::simp::ConstantOp>()) {
        if (auto intAttr = constOp.value().dyn_cast<mlir::IntegerAttr>()) {
          newShape.push_back(intAttr.getInt());
        }
      } else if (auto arithConstOp = args[i].getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto intAttr = arithConstOp.getValue().dyn_cast<mlir::IntegerAttr>()) {
          newShape.push_back(intAttr.getInt());
        }
      }
    }

    if (newShape.empty()) {
      llvm::errs() << "Error: tensor_reshape requires constant dimension arguments\n";
      return nullptr;
    }

    mlir::Type resultType = mlir::simp::SimpTensorType::get(builder.getContext(), newShape, tensorType.getElementType());
    llvm::SmallVector<mlir::Value, 4> shapeArgs(args.begin() + 1, args.end());
    return builder.create<mlir::simp::TensorReshapeOp>(loc, resultType, tensor, shapeArgs);
  }

  // tensor_transpose: tensor_transpose(tensor) or tensor_transpose(tensor, perm0, perm1, ...)
  if (calleeName == "tensor_transpose") {
    if (args.size() < 1) {
      llvm::errs() << "Error: tensor_transpose requires at least 1 argument (tensor, [perm...]), got " << args.size() << "\n";
      return nullptr;
    }

    mlir::Value tensor = args[0];
    auto tensorType = tensor.getType().dyn_cast<mlir::simp::SimpTensorType>();
    if (!tensorType) {
      llvm::errs() << "Error: tensor_transpose requires a tensor argument\n";
      return nullptr;
    }

    auto shape = tensorType.getShape();
    int64_t rank = shape.size();

    // Compute result shape
    llvm::SmallVector<int64_t, 4> resultShape;
    if (args.size() == 1 && rank == 2) {
      // Default 2D transpose: swap dims
      resultShape = {shape[1], shape[0]};
    } else if (args.size() > 1) {
      // Extract permutation from args
      llvm::SmallVector<int64_t, 4> perm;
      for (size_t i = 1; i < args.size(); i++) {
        if (auto constOp = args[i].getDefiningOp<mlir::simp::ConstantOp>()) {
          if (auto intAttr = constOp.value().dyn_cast<mlir::IntegerAttr>()) {
            perm.push_back(intAttr.getInt());
          }
        } else if (auto arithConstOp = args[i].getDefiningOp<mlir::arith::ConstantOp>()) {
          if (auto intAttr = arithConstOp.getValue().dyn_cast<mlir::IntegerAttr>()) {
            perm.push_back(intAttr.getInt());
          }
        }
      }

      if (perm.size() != rank) {
        llvm::errs() << "Error: transpose permutation size must match tensor rank\n";
        return nullptr;
      }

      // Compute result shape from permutation
      for (auto p : perm) {
        resultShape.push_back(shape[p]);
      }
    } else {
      llvm::errs() << "Error: tensor_transpose requires permutation for rank > 2\n";
      return nullptr;
    }

    mlir::Type resultType = mlir::simp::SimpTensorType::get(builder.getContext(), resultShape, tensorType.getElementType());
    llvm::SmallVector<mlir::Value, 4> permArgs(args.begin() + 1, args.end());
    return builder.create<mlir::simp::TensorTransposeOp>(loc, resultType, tensor, permArgs);
  }

  // tensor_slice: tensor_slice(tensor, start0, end0, start1, end1, ...)
  if (calleeName == "tensor_slice") {
    if (args.size() < 1) {
      llvm::errs() << "Error: tensor_slice requires arguments (tensor, indices...), got " << args.size() << "\n";
      return nullptr;
    }

    mlir::Value tensor = args[0];
    auto tensorType = tensor.getType().dyn_cast<mlir::simp::SimpTensorType>();
    if (!tensorType) {
      llvm::errs() << "Error: tensor_slice requires a tensor argument\n";
      return nullptr;
    }

    auto shape = tensorType.getShape();
    int64_t rank = shape.size();

    if (args.size() != 1 + 2 * rank) {
      llvm::errs() << "Error: tensor_slice requires 2 indices per dimension (start, end), got " << (args.size() - 1) << " for rank " << rank << "\n";
      return nullptr;
    }

    // Extract start/end pairs and compute result shape
    llvm::SmallVector<int64_t, 4> resultShape;
    for (int64_t i = 0; i < rank; i++) {
      int64_t start = -1, end = -1;

      if (auto constOp = args[1 + i * 2].getDefiningOp<mlir::simp::ConstantOp>()) {
        if (auto intAttr = constOp.value().dyn_cast<mlir::IntegerAttr>()) {
          start = intAttr.getInt();
        }
      } else if (auto arithConstOp = args[1 + i * 2].getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto intAttr = arithConstOp.getValue().dyn_cast<mlir::IntegerAttr>()) {
          start = intAttr.getInt();
        }
      }

      if (auto constOp = args[1 + i * 2 + 1].getDefiningOp<mlir::simp::ConstantOp>()) {
        if (auto intAttr = constOp.value().dyn_cast<mlir::IntegerAttr>()) {
          end = intAttr.getInt();
        }
      } else if (auto arithConstOp = args[1 + i * 2 + 1].getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto intAttr = arithConstOp.getValue().dyn_cast<mlir::IntegerAttr>()) {
          end = intAttr.getInt();
        }
      }

      if (start == -1 || end == -1) {
        llvm::errs() << "Error: tensor_slice requires constant start/end indices\n";
        return nullptr;
      }

      resultShape.push_back(end - start);
    }

    mlir::Type resultType = mlir::simp::SimpTensorType::get(builder.getContext(), resultShape, tensorType.getElementType());
    llvm::SmallVector<mlir::Value, 4> indexArgs(args.begin() + 1, args.end());
    return builder.create<mlir::simp::TensorSliceOp>(loc, resultType, tensor, indexArgs);
  }

  // tensor_gather: tensor_gather(source, indices, [axis])
  if (calleeName == "tensor_gather") {
    if (args.size() < 2 || args.size() > 3) {
      llvm::errs() << "Error: tensor_gather requires 2-3 arguments (source, indices, [axis]), got " << args.size() << "\n";
      return nullptr;
    }

    mlir::Value source = args[0];
    mlir::Value indices = args[1];
    auto sourceType = source.getType().dyn_cast<mlir::simp::SimpTensorType>();
    auto indicesType = indices.getType().dyn_cast<mlir::simp::SimpTensorType>();

    if (!sourceType) {
      llvm::errs() << "Error: tensor_gather requires tensor source argument\n";
      return nullptr;
    }
    if (!indicesType) {
      llvm::errs() << "Error: tensor_gather requires tensor indices argument\n";
      return nullptr;
    }

    // Validate indices tensor is 1D and i64
    if (indicesType.getShape().size() != 1) {
      llvm::errs() << "Error: tensor_gather indices must be 1D tensor, got rank " << indicesType.getShape().size() << "\n";
      return nullptr;
    }

    // Get axis (default 0)
    int64_t axis = 0;
    mlir::Value axisVal = nullptr;
    if (args.size() == 3) {
      if (auto constOp = args[2].getDefiningOp<mlir::simp::ConstantOp>()) {
        if (auto intAttr = constOp.value().dyn_cast<mlir::IntegerAttr>()) {
          axis = intAttr.getInt();
        }
      } else if (auto arithConstOp = args[2].getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto intAttr = arithConstOp.getValue().dyn_cast<mlir::IntegerAttr>()) {
          axis = intAttr.getInt();
        }
      }
      axisVal = args[2];
    }

    auto sourceShape = sourceType.getShape();
    int64_t rank = sourceShape.size();
    int64_t numIndices = indicesType.getShape()[0];

    // Handle negative axis
    if (axis < 0) {
      axis += rank;
    }
    if (axis < 0 || axis >= rank) {
      llvm::errs() << "Error: tensor_gather axis " << axis << " out of bounds for rank " << rank << "\n";
      return nullptr;
    }

    // Compute result shape: replace axis dimension with numIndices
    llvm::SmallVector<int64_t, 4> resultShape;
    for (int64_t i = 0; i < rank; i++) {
      if (i == axis) {
        resultShape.push_back(numIndices);
      } else {
        resultShape.push_back(sourceShape[i]);
      }
    }

    mlir::Type resultType = mlir::simp::SimpTensorType::get(builder.getContext(), resultShape, sourceType.getElementType());

    if (axisVal) {
      return builder.create<mlir::simp::TensorGatherOp>(loc, resultType, source, indices, axisVal);
    } else {
      return builder.create<mlir::simp::TensorGatherOp>(loc, resultType, source, indices, nullptr);
    }
  }

  // tensor_scatter: tensor_scatter(dst, indices, values, [axis])
  if (calleeName == "tensor_scatter") {
    if (args.size() < 3 || args.size() > 4) {
      llvm::errs() << "Error: tensor_scatter requires 3-4 arguments (dst, indices, values, [axis]), got " << args.size() << "\n";
      return nullptr;
    }

    mlir::Value dst = args[0];
    mlir::Value indices = args[1];
    mlir::Value values = args[2];
    auto dstType = dst.getType().dyn_cast<mlir::simp::SimpTensorType>();
    auto indicesType = indices.getType().dyn_cast<mlir::simp::SimpTensorType>();
    auto valuesType = values.getType().dyn_cast<mlir::simp::SimpTensorType>();

    if (!dstType) {
      llvm::errs() << "Error: tensor_scatter requires tensor dst argument\n";
      return nullptr;
    }
    if (!indicesType) {
      llvm::errs() << "Error: tensor_scatter requires tensor indices argument\n";
      return nullptr;
    }
    if (!valuesType) {
      llvm::errs() << "Error: tensor_scatter requires tensor values argument\n";
      return nullptr;
    }

    // Validate indices tensor is 1D
    if (indicesType.getShape().size() != 1) {
      llvm::errs() << "Error: tensor_scatter indices must be 1D tensor, got rank " << indicesType.getShape().size() << "\n";
      return nullptr;
    }

    // Get axis (default 0)
    int64_t axis = 0;
    mlir::Value axisVal = nullptr;
    if (args.size() == 4) {
      if (auto constOp = args[3].getDefiningOp<mlir::simp::ConstantOp>()) {
        if (auto intAttr = constOp.value().dyn_cast<mlir::IntegerAttr>()) {
          axis = intAttr.getInt();
        }
      } else if (auto arithConstOp = args[3].getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto intAttr = arithConstOp.getValue().dyn_cast<mlir::IntegerAttr>()) {
          axis = intAttr.getInt();
        }
      }
      axisVal = args[3];
    }

    auto dstShape = dstType.getShape();
    int64_t rank = dstShape.size();
    int64_t numIndices = indicesType.getShape()[0];

    // Handle negative axis
    if (axis < 0) {
      axis += rank;
    }
    if (axis < 0 || axis >= rank) {
      llvm::errs() << "Error: tensor_scatter axis " << axis << " out of bounds for rank " << rank << "\n";
      return nullptr;
    }

    // Validate values shape matches dst shape with axis replaced by numIndices
    auto valuesShape = valuesType.getShape();
    if (valuesShape.size() != rank) {
      llvm::errs() << "Error: tensor_scatter values rank must match dst rank\n";
      return nullptr;
    }
    for (int64_t i = 0; i < rank; i++) {
      if (i == axis) {
        if (valuesShape[i] != numIndices) {
          llvm::errs() << "Error: tensor_scatter values axis dimension must match indices length\n";
          return nullptr;
        }
      } else {
        if (valuesShape[i] != dstShape[i]) {
          llvm::errs() << "Error: tensor_scatter values shape must match dst shape (except axis)\n";
          return nullptr;
        }
      }
    }

    // Result type is same as dst
    mlir::Type resultType = dstType;

    if (axisVal) {
      return builder.create<mlir::simp::TensorScatterOp>(loc, resultType, dst, indices, values, axisVal);
    } else {
      return builder.create<mlir::simp::TensorScatterOp>(loc, resultType, dst, indices, values, nullptr);
    }
  }

  // tensor_from_array is handled specially in variable declarations
  // See lowerVariableDeclaration for the implementation
  if (calleeName == "tensor_from_array") {
    llvm::errs() << "Error: tensor_from_array must be used in a tensor variable declaration\n";
    llvm::errs() << "  Example: f32<32000, 768> embedding = tensor_from_array(array);\n";
    return nullptr;
  }

  // tensor_matmul: tensor_matmul(lhs, rhs)
  if (calleeName == "tensor_matmul") {
    if (args.size() != 2) {
      llvm::errs() << "Error: tensor_matmul requires 2 arguments (lhs, rhs), got " << args.size() << "\n";
      return nullptr;
    }

    mlir::Value lhs = args[0];
    mlir::Value rhs = args[1];
    auto lhsType = lhs.getType().dyn_cast<mlir::simp::SimpTensorType>();
    auto rhsType = rhs.getType().dyn_cast<mlir::simp::SimpTensorType>();

    if (!lhsType) {
      llvm::errs() << "Error: tensor_matmul requires tensor lhs argument\n";
      return nullptr;
    }
    if (!rhsType) {
      llvm::errs() << "Error: tensor_matmul requires tensor rhs argument\n";
      return nullptr;
    }

    auto lhsShape = lhsType.getShape();
    auto rhsShape = rhsType.getShape();
    int64_t lhsRank = lhsShape.size();
    int64_t rhsRank = rhsShape.size();

    // Compute result shape based on input dimensions
    llvm::SmallVector<int64_t, 4> resultShape;

    if (lhsRank == 2 && rhsRank == 2) {
      // 2D matmul: (M, K) × (K, N) → (M, N)
      int64_t M = lhsShape[0];
      int64_t K_lhs = lhsShape[1];
      int64_t K_rhs = rhsShape[0];
      int64_t N = rhsShape[1];

      if (K_lhs != K_rhs) {
        llvm::errs() << "Error: tensor_matmul dimension mismatch: lhs K=" << K_lhs << ", rhs K=" << K_rhs << "\n";
        return nullptr;
      }

      resultShape = {M, N};

    } else if (lhsRank == 3 && rhsRank == 3) {
      // Batched 3D matmul: (B, M, K) × (B, K, N) → (B, M, N)
      int64_t B_lhs = lhsShape[0];
      int64_t B_rhs = rhsShape[0];
      int64_t M = lhsShape[1];
      int64_t K_lhs = lhsShape[2];
      int64_t K_rhs = rhsShape[1];
      int64_t N = rhsShape[2];

      if (B_lhs != B_rhs) {
        llvm::errs() << "Error: tensor_matmul batch size mismatch\n";
        return nullptr;
      }
      if (K_lhs != K_rhs) {
        llvm::errs() << "Error: tensor_matmul dimension mismatch\n";
        return nullptr;
      }

      resultShape = {B_lhs, M, N};

    } else if (lhsRank == 4 && rhsRank == 2) {
      // 4D NHWC matmul: (N, H, W, C_in) × (C_out, C_in) → (N, H, W, C_out)
      int64_t N = lhsShape[0];
      int64_t H = lhsShape[1];
      int64_t W = lhsShape[2];
      int64_t C_in_lhs = lhsShape[3];
      int64_t C_out = rhsShape[0];
      int64_t C_in_rhs = rhsShape[1];

      if (C_in_lhs != C_in_rhs) {
        llvm::errs() << "Error: tensor_matmul channel mismatch\n";
        return nullptr;
      }

      resultShape = {N, H, W, C_out};

    } else {
      llvm::errs() << "Error: tensor_matmul unsupported dimensions: lhs rank=" << lhsRank
                   << ", rhs rank=" << rhsRank << "\n";
      return nullptr;
    }

    // For i8/i16 inputs, promote result type to i32 to prevent overflow
    mlir::Type elemType = lhsType.getElementType();
    if (elemType.isInteger(8) || elemType.isInteger(16)) {
      elemType = builder.getI32Type();
    }

    mlir::Type resultType = mlir::simp::SimpTensorType::get(
        builder.getContext(), resultShape, elemType);

    return builder.create<mlir::simp::TensorMatMulOp>(loc, resultType, lhs, rhs, /*layout=*/nullptr);
  }

  // tensor_dot: tensor_dot(lhs, rhs)
  if (calleeName == "tensor_dot") {
    if (args.size() != 2) {
      llvm::errs() << "Error: tensor_dot requires 2 arguments (lhs, rhs), got " << args.size() << "\n";
      return nullptr;
    }

    mlir::Value lhs = args[0];
    mlir::Value rhs = args[1];
    auto lhsType = lhs.getType().dyn_cast<mlir::simp::SimpTensorType>();
    auto rhsType = rhs.getType().dyn_cast<mlir::simp::SimpTensorType>();

    if (!lhsType) {
      llvm::errs() << "Error: tensor_dot requires tensor lhs argument\n";
      return nullptr;
    }
    if (!rhsType) {
      llvm::errs() << "Error: tensor_dot requires tensor rhs argument\n";
      return nullptr;
    }

    auto lhsShape = lhsType.getShape();
    auto rhsShape = rhsType.getShape();

    if (lhsShape.size() != 1 || rhsShape.size() != 1) {
      llvm::errs() << "Error: tensor_dot requires 1D tensors\n";
      return nullptr;
    }

    if (lhsShape[0] != rhsShape[0]) {
      llvm::errs() << "Error: tensor_dot dimension mismatch: " << lhsShape[0] << " vs " << rhsShape[0] << "\n";
      return nullptr;
    }

    // Result is a scalar of the same element type
    mlir::Type resultType = lhsType.getElementType();

    return builder.create<mlir::simp::TensorDotOp>(loc, resultType, lhs, rhs);
  }

  // Look up user-defined functions in the module
  mlir::FuncOp callee = module.lookupSymbol<mlir::FuncOp>(calleeName);
  if (!callee) {
    llvm::errs() << "Error: Undefined function '" << calleeName << "'\n";
    return nullptr;
  }

  // Create the call operation (Standard dialect in MLIR 14)
  // CallOp takes: location, callee symbol, arguments
  auto callOp = builder.create<mlir::CallOp>(
      loc,
      callee,
      args
  );

  // Return the first result (functions return a single value)
  if (callOp.getNumResults() == 0) {
    llvm::errs() << "Error: Function '" << calleeName << "' has no return value\n";
    return nullptr;
  }

  return callOp.getResult(0);
}

//===----------------------------------------------------------------------===//
// Statement Lowering
//===----------------------------------------------------------------------===//

mlir::LogicalResult MLIRCodeGenContext::lowerStatement(StmtAST* stmt) {
  if (!stmt) {
    return mlir::failure();
  }

  // Use getKind() for type identification without RTTI
  switch (stmt->getKind()) {
    case ASTKind::VariableDecl:
      return lowerDeclaration(static_cast<VariableDeclarationAST*>(stmt));

    case ASTKind::ReturnStmt:
      return lowerReturn(static_cast<ReturnAST*>(stmt));

    case ASTKind::ExpressionStmt:
      return lowerExpressionStmt(static_cast<ExpressionStmtAST*>(stmt));

    case ASTKind::IfStmt:
      return lowerIf(static_cast<IfAST*>(stmt));

    case ASTKind::WhileStmt:
      return lowerWhile(static_cast<WhileAST*>(stmt));

    case ASTKind::FunctionDecl:
      // Functions are handled at top level, not as statements
      llvm::errs() << "Warning: Function declarations should be lowered at module level\n";
      return mlir::success();

    case ASTKind::BlockStmt:
    case ASTKind::IncludeStmt:
      // Not yet implemented
      llvm::errs() << "Warning: Statement kind not yet implemented in MLIR lowering\n";
      return mlir::success();

    default:
      llvm::errs() << "Error: Unknown statement kind in MLIR lowering\n";
      return mlir::failure();
  }
}

mlir::LogicalResult MLIRCodeGenContext::lowerDeclaration(VariableDeclarationAST* decl) {
  LOG_DEBUG("Lowering variable declaration: ", decl->getName(),
            ", staticType=", (decl->isStaticallyTyped() ? "yes" : "no"),
            ", isTensor=", (decl->isStaticallyTyped() && decl->getStaticType()->isTensor() ? "yes" : "no"));

  // Check if this is an array creation - we need to track dimensions
  ExprAST* initExpr = decl->getAssignmentExpr();
  ArrayCreateExprAST* arrayCreate = nullptr;
  if (initExpr && initExpr->getKind() == ASTKind::ArrayCreateExpr) {
    arrayCreate = static_cast<ArrayCreateExprAST*>(initExpr);
  }

  mlir::Value initValue;

  // Handle uninitialized tensor declarations (auto-create tensor)
  if (!initExpr && decl->isStaticallyTyped() && decl->getStaticType()->isTensor()) {
    auto loc = getUnknownLocation();
    auto tensorType = getMLIRType(const_cast<TypeInfo*>(decl->getStaticType()));
    initValue = builder.create<mlir::simp::TensorCreateOp>(loc, tensorType);
  }
  // Handle tensor declarations with initializer lists (e.g., f32<2,3> a = {1.0, 2.0, ...})
  else if (arrayCreate && decl->isStaticallyTyped() && decl->getStaticType()->isTensor()) {
    auto loc = getUnknownLocation();
    // Get the full tensor type from the declaration
    auto tensorType = getMLIRType(const_cast<TypeInfo*>(decl->getStaticType()));
    auto simpTensorType = tensorType.dyn_cast<mlir::simp::SimpTensorType>();

    if (!simpTensorType) {
      llvm::errs() << "Error: Expected SimpTensorType for tensor declaration\n";
      return mlir::failure();
    }

    // Create a tensor with the proper type
    initValue = builder.create<mlir::simp::TensorCreateOp>(loc, tensorType);

    // Now populate it with values from the initializer list
    auto& values = arrayCreate->getDimensions();  // dimensionExprs contains the initializer values
    auto shape = simpTensorType.getShape();

    // Compute total number of elements
    int64_t totalElements = 1;
    for (int64_t dim : shape) {
      totalElements *= dim;
    }

    if (values.size() != static_cast<size_t>(totalElements)) {
      llvm::errs() << "Error: Initializer list size (" << values.size()
                   << ") doesn't match tensor size (" << totalElements << ")\n";
      return mlir::failure();
    }

    // Directly create tensor with the proper shape (not via array conversion)
    initValue = builder.create<mlir::simp::TensorCreateOp>(loc, tensorType);

    // Initialize tensor elements with values from initializer list
    // Store values using multi-dimensional indices (row-major order)
    for (size_t flatIdx = 0; flatIdx < values.size(); ++flatIdx) {
      // Compute multi-dimensional indices from flat index
      // For shape [D0, D1, D2], index i maps to:
      // i0 = i / (D1*D2), i1 = (i % (D1*D2)) / D2, i2 = i % D2
      llvm::SmallVector<mlir::Value, 4> indices;
      size_t remainingIndex = flatIdx;
      size_t stride = totalElements;

      for (size_t dim = 0; dim < shape.size(); ++dim) {
        stride /= shape[dim];
        size_t dimIndex = remainingIndex / stride;
        remainingIndex %= stride;

        // Create index value as i64 (TensorSetOp expects i64 indices)
        mlir::Value idxValue = builder.create<mlir::arith::ConstantIntOp>(
            loc, dimIndex, builder.getI64Type());
        indices.push_back(idxValue);
      }

      // Lower the initializer value expression
      mlir::Value valueToStore = lowerExpression(values[flatIdx].get());
      if (!valueToStore) {
        llvm::errs() << "Error: Failed to lower initializer value at index " << flatIdx << "\n";
        return mlir::failure();
      }

      // Store value into tensor using TensorSetOp (functional, returns new tensor)
      initValue = builder.create<mlir::simp::TensorSetOp>(loc, tensorType, initValue, indices, valueToStore);
    }
  }
  // Handle tensor_from_array conversion
  else if (initExpr && initExpr->getKind() == ASTKind::CallExpr &&
           decl->isStaticallyTyped() && decl->getStaticType()->isTensor()) {
    auto* funcCall = static_cast<CallExprAST*>(initExpr);
    if (funcCall->getCallee() == "tensor_from_array") {
      LOG_DEBUG("Processing tensor_from_array for variable: ", decl->getName());

      auto loc = getUnknownLocation();
      auto tensorType = getMLIRType(const_cast<TypeInfo*>(decl->getStaticType()));

      // Get arguments: tensor_from_array(array) or tensor_from_array(array, offset)
      size_t numArgs = funcCall->getArguments().size();
      LOG_DEBUG("tensor_from_array has ", numArgs, " arguments");
      if (numArgs != 1 && numArgs != 2) {
        llvm::errs() << "Error: tensor_from_array requires 1 or 2 arguments (array, [offset])\n";
        return mlir::failure();
      }

      mlir::Value arrayArg = lowerExpression(funcCall->getArguments()[0]);
      if (!arrayArg) {
        llvm::errs() << "Error: Failed to lower tensor_from_array array argument\n";
        return mlir::failure();
      }

      // Get offset (default to 0 if not provided)
      mlir::Value offsetArg;
      if (numArgs == 2) {
        offsetArg = lowerExpression(funcCall->getArguments()[1]);
        if (!offsetArg) {
          llvm::errs() << "Error: Failed to lower tensor_from_array offset argument\n";
          return mlir::failure();
        }
      } else {
        // Default offset = 0
        offsetArg = builder.create<mlir::arith::ConstantIntOp>(loc, 0, builder.getI64Type());
      }

      // Create tensor_from_array operation with explicit result type
      initValue = builder.create<mlir::simp::TensorFromArrayOp>(loc, tensorType, arrayArg, offsetArg);
    } else {
      // Regular function call
      initValue = lowerExpression(decl->getAssignmentExpr());
      if (!initValue) {
        return mlir::failure();
      }
    }
  }
  // Lower the initialization expression if provided
  else if (initExpr) {
    LOG_DEBUG("Lowering init expression for: ", decl->getName());
    initValue = lowerExpression(initExpr);
    if (!initValue) {
      LOG_ERROR("Failed to lower init expression for: ", decl->getName());
      return mlir::failure();
    }
  }
  // Handle uninitialized arrays (allocate memory)
  else if (decl->isStaticallyTyped() && decl->getStaticType()->isArray()) {
    LOG_DEBUG("Creating uninitialized array for: ", decl->getName());
    auto loc = getUnknownLocation();
    auto arrayType = getMLIRType(const_cast<TypeInfo*>(decl->getStaticType()));

    // Cast to ArrayTypeInfo to get size
    auto* arrayTypeInfo = static_cast<const ArrayTypeInfo*>(decl->getStaticType());
    int arraySize = arrayTypeInfo->size;

    if (arraySize <= 0) {
      LOG_ERROR("Array ", decl->getName(), " has invalid size: ", arraySize);
      return mlir::failure();
    }

    // Create size constant
    mlir::Value sizeValue = builder.create<mlir::simp::ConstantOp>(
        loc, builder.getI64Type(), builder.getI64IntegerAttr(arraySize));

    // For arrays, create an ArrayCreateOp with the declared size
    initValue = builder.create<mlir::simp::ArrayCreateOp>(loc, arrayType, sizeValue);
  }
  else {
    LOG_ERROR("No initialization for variable: ", decl->getName());
    return mlir::failure();
  }

  // Declare the variable in the symbol table
  declareVariable(decl->getName(), initValue);

  // If this was an array creation with multiple dimensions, store the dimensions
  if (arrayCreate && arrayCreate->getDimensions().size() > 1) {
    llvm::SmallVector<mlir::Value, 4> dimValues;
    for (const auto& dim : arrayCreate->getDimensions()) {
      mlir::Value dimValue = lowerExpression(dim.get());
      if (dimValue) {
        dimValues.push_back(dimValue);
      }
    }
    if (dimValues.size() > 1) {
      arrayDimensions[decl->getName()] = dimValues;
    }
  }

  // Also track dimensions for tensor types (from tensor operations)
  auto tensorType = initValue.getType().dyn_cast<mlir::simp::SimpTensorType>();
  if (tensorType) {
    auto shape = tensorType.getShape();
    if (shape.size() > 1) {
      llvm::SmallVector<mlir::Value, 4> dimValues;
      auto loc = getUnknownLocation();
      for (int64_t dim : shape) {
        dimValues.push_back(builder.create<mlir::arith::ConstantIndexOp>(loc, dim));
      }
      arrayDimensions[decl->getName()] = dimValues;
    }
  }

  return mlir::success();
}

mlir::LogicalResult MLIRCodeGenContext::lowerReturn(ReturnAST* ret) {
  auto loc = getUnknownLocation();

  // Lower the return value expression
  mlir::Value returnValue = lowerExpression(ret->getExpression());
  if (!returnValue) {
    return mlir::failure();
  }

  // Check if we need to cast to match function return type
  if (currentFunction) {
    mlir::Type expectedType = currentFunction.getType().getResult(0);
    mlir::Type actualType = returnValue.getType();

    if (expectedType != actualType) {
      // Need to cast - handle float type mismatches (f32 <-> f64, f16, bf16)
      if (expectedType.isa<mlir::FloatType>() && actualType.isa<mlir::FloatType>()) {
        // Use fpext (extend) or fptrunc (truncate) based on bitwidths
        auto expectedFloat = expectedType.cast<mlir::FloatType>();
        auto actualFloat = actualType.cast<mlir::FloatType>();

        if (expectedFloat.getWidth() > actualFloat.getWidth()) {
          // Extend: f32 -> f64, f16 -> f32, etc.
          returnValue = builder.create<mlir::arith::ExtFOp>(loc, expectedType, returnValue);
        } else {
          // Truncate: f64 -> f32, f32 -> f16, etc.
          returnValue = builder.create<mlir::arith::TruncFOp>(loc, expectedType, returnValue);
        }
      } else if (expectedType.isa<mlir::IntegerType>() && actualType.isa<mlir::IntegerType>()) {
        // Integer conversions
        auto expectedInt = expectedType.cast<mlir::IntegerType>();
        auto actualInt = actualType.cast<mlir::IntegerType>();

        if (expectedInt.getWidth() > actualInt.getWidth()) {
          // Sign extend for now (could check signedness)
          returnValue = builder.create<mlir::arith::ExtSIOp>(loc, expectedType, returnValue);
        } else if (expectedInt.getWidth() < actualInt.getWidth()) {
          // Truncate
          returnValue = builder.create<mlir::arith::TruncIOp>(loc, expectedType, returnValue);
        }
      }
      // TODO: Handle other type mismatches (int <-> float, etc.)
    }
  }

  // Create return operation (Standard dialect in MLIR 14)
  builder.create<mlir::ReturnOp>(loc, returnValue);

  return mlir::success();
}

mlir::LogicalResult MLIRCodeGenContext::lowerExpressionStmt(ExpressionStmtAST* exprStmt) {
  // Just lower the expression (side effects will be captured)
  mlir::Value result = lowerExpression(exprStmt->getExpression());
  return result ? mlir::success() : mlir::failure();
}

mlir::LogicalResult MLIRCodeGenContext::lowerIf(IfAST* ifStmt) {
  auto loc = getUnknownLocation();

  // Lower the condition expression
  mlir::Value condition = lowerExpression(ifStmt->getCondition());
  if (!condition) {
    llvm::errs() << "Error: Failed to lower if condition\n";
    return mlir::failure();
  }

  // The condition must be i1 type for scf.if
  if (!condition.getType().isInteger(1)) {
    // For now, we'll just warn about this limitation
    llvm::errs() << "Warning: Non-boolean condition in if statement\n";
  }

  // Get the set of variables that exist before the if statement
  auto existingVars = getCurrentVariableNames();

  // Track which variables are modified in the if/else blocks
  // Only track modifications to pre-existing variables (not new locals)
  auto thenModified = trackModifiedVariables(ifStmt->getThenBlock(), existingVars);
  auto elseModified = trackModifiedVariables(ifStmt->getElseBlock(), existingVars);

  // Find the union of variables modified in either branch
  std::set<std::string> allModified;
  allModified.insert(thenModified.begin(), thenModified.end());
  allModified.insert(elseModified.begin(), elseModified.end());

  // Collect initial values of modified variables (for else branch defaults)
  std::vector<mlir::Value> initialValues = collectVariableValues(allModified);

  // Determine result types for the scf.if operation
  llvm::SmallVector<mlir::Type, 4> resultTypes;
  for (const auto& value : initialValues) {
    resultTypes.push_back(value.getType());
  }

  // Create the scf.if operation with result types
  auto ifOp = builder.create<mlir::scf::IfOp>(
      loc, resultTypes, condition,
      /*withElseRegion=*/true);  // Always create else region when we have results

  // Lower the then block
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

    pushScope();
    BlockAST* thenBlock = ifStmt->getThenBlock();
    if (thenBlock) {
      for (auto* stmt : thenBlock->statements) {
        if (failed(lowerStatement(stmt))) {
          popScope();
          return mlir::failure();
        }
      }
    }

    // Collect final values of modified variables for yielding
    std::vector<mlir::Value> thenValues;
    for (const auto& varName : allModified) {
      mlir::Value value = lookupVariable(varName);
      if (!value) {
        // Variable wasn't modified in then branch, use initial value
        auto it = std::find(allModified.begin(), allModified.end(), varName);
        size_t idx = std::distance(allModified.begin(), it);
        value = initialValues[idx];
      }
      thenValues.push_back(value);
    }

    popScope();

    // Terminate with scf.yield, passing the modified values
    builder.create<mlir::scf::YieldOp>(loc, thenValues);
  }

  // Lower the else block
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());

    if (ifStmt->getElseBlock()) {
      pushScope();
      BlockAST* elseBlock = ifStmt->getElseBlock();
      for (auto* stmt : elseBlock->statements) {
        if (failed(lowerStatement(stmt))) {
          popScope();
          return mlir::failure();
        }
      }

      // Collect final values of modified variables for yielding
      std::vector<mlir::Value> elseValues;
      for (const auto& varName : allModified) {
        mlir::Value value = lookupVariable(varName);
        if (!value) {
          // Variable wasn't modified in else branch, use initial value
          auto it = std::find(allModified.begin(), allModified.end(), varName);
          size_t idx = std::distance(allModified.begin(), it);
          value = initialValues[idx];
        }
        elseValues.push_back(value);
      }

      popScope();

      // Terminate with scf.yield, passing the modified values
      builder.create<mlir::scf::YieldOp>(loc, elseValues);
    } else {
      // No else block - create implicit else that yields initial values
      std::vector<mlir::Value> elseValues(initialValues);
      builder.create<mlir::scf::YieldOp>(loc, elseValues);
    }
  }

  // Update symbol table with the results from scf.if
  if (!allModified.empty()) {
    // Extract modified variable values
    std::vector<mlir::Value> modifiedResults;
    for (size_t i = 0; i < allModified.size(); ++i) {
      modifiedResults.push_back(ifOp.getResult(i));
    }
    updateSymbolTableWithResults(allModified, modifiedResults);
  }

  return mlir::success();
}

mlir::LogicalResult MLIRCodeGenContext::lowerWhile(WhileAST* whileLoop) {
  auto loc = getUnknownLocation();

  // Get the set of variables that exist before the while loop
  auto existingVars = getCurrentVariableNames();

  // Track which variables are modified in the loop body
  // Only track modifications to pre-existing variables (not new loop locals)
  auto modifiedVars = trackModifiedVariables(whileLoop->getBody(), existingVars);

  // Collect initial values of loop-carried variables
  std::vector<mlir::Value> initialValues = collectVariableValues(modifiedVars);

  // PROPER TYPE INFERENCE: Analyze loop body to determine final types after promotions
  // We need to simulate the loop body to see what types variables will have
  std::map<std::string, mlir::Type> inferredTypes;

  // Start with current types
  size_t idx = 0;
  for (const auto& varName : modifiedVars) {
    inferredTypes[varName] = initialValues[idx].getType();
    idx++;
  }

  // Helper lambda to recursively analyze statements for type promotions
  std::function<void(StmtAST*)> analyzeStatement = [&](StmtAST* stmt) {
    if (!stmt) return;

    if (stmt->getKind() == ASTKind::ExpressionStmt) {
      auto* exprStmt = static_cast<ExpressionStmtAST*>(stmt);
      ExprAST* expr = exprStmt->getExpression();

      // Check for assignments that might promote types
      if (expr && expr->getKind() == ASTKind::AssignmentExpr) {
        auto* assign = static_cast<AssignmentExprAST*>(expr);
        VariableExprAST* lhs = assign->getLHS();
        if (!lhs) return;

        std::string varName = lhs->getName();

        // If this is a modified variable, infer its promoted type
        if (modifiedVars.find(varName) != modifiedVars.end()) {
          // Detect patterns like: sum = sum + array[i] where array[i] might be wider type
          ExprAST* rhs = assign->getRHS();
          if (rhs && rhs->getKind() == ASTKind::BinaryExpr) {
            auto* binOp = static_cast<BinaryExprAST*>(rhs);
            // Check if one operand is array access (might be f64)
            bool hasArrayAccess = false;
            if (binOp->getLeft() && binOp->getLeft()->getKind() == ASTKind::ArrayAccessExpr) hasArrayAccess = true;
            if (binOp->getRight() && binOp->getRight()->getKind() == ASTKind::ArrayAccessExpr) hasArrayAccess = true;

            if (hasArrayAccess) {
              // Array access involved - assume widest float type
              mlir::Type currentType = inferredTypes[varName];
              if (currentType.isa<mlir::FloatType>()) {
                inferredTypes[varName] = builder.getF64Type();
              }
            }
          }
        }
      }
    } else if (stmt->getKind() == ASTKind::WhileStmt) {
      // RECURSIVE CASE: Analyze nested while loop body
      auto* nestedWhile = static_cast<WhileAST*>(stmt);
      BlockAST* nestedBody = nestedWhile->getBody();
      if (nestedBody) {
        for (auto* nestedStmt : nestedBody->statements) {
          analyzeStatement(nestedStmt);  // Recurse into nested loop
        }
      }
    }
  };

  // Simulate loop body type evolution by recursively analyzing all statements
  BlockAST* body = whileLoop->getBody();
  if (body) {
    for (auto* stmt : body->statements) {
      analyzeStatement(stmt);
    }
  }

  // Build final iter_args types and promoted initial values
  llvm::SmallVector<mlir::Type, 4> iterTypes;
  std::vector<mlir::Value> promotedInitialValues;

  idx = 0;
  for (const auto& varName : modifiedVars) {
    mlir::Type originalType = initialValues[idx].getType();
    mlir::Type inferredType = inferredTypes[varName];

    // Use the wider of original and inferred types
    mlir::Type useType = originalType;
    if (inferredType != originalType) {
      int origPrec = getTypePrecedence(originalType);
      int inferPrec = getTypePrecedence(inferredType);
      if (inferPrec > origPrec) {
        useType = inferredType;
      }
    }

    iterTypes.push_back(useType);

    // Promote initial value if needed
    mlir::Value promotedValue = initialValues[idx];
    if (useType != originalType) {
      promotedValue = promoteType(initialValues[idx], useType, builder, loc);
    }
    promotedInitialValues.push_back(promotedValue);

    idx++;
  }

  // Create the scf.while operation with iter_args using inferred types
  auto whileOp = builder.create<mlir::scf::WhileOp>(loc, iterTypes, promotedInitialValues);

  // Build the "before" region (condition check)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Block* beforeBlock = builder.createBlock(&whileOp.getBefore());

    // Add block arguments for loop-carried values
    for (auto type : iterTypes) {
      beforeBlock->addArgument(type, loc);
    }

    builder.setInsertionPointToStart(beforeBlock);

    // Update symbol table with block arguments
    pushScope();
    auto varIt = modifiedVars.begin();
    for (size_t i = 0; i < beforeBlock->getNumArguments(); ++i, ++varIt) {
      declareVariable(*varIt, beforeBlock->getArgument(i));
    }

    // Lower the condition
    mlir::Value condition = lowerExpression(whileLoop->getCondition());
    if (!condition) {
      llvm::errs() << "Error: Failed to lower while condition\n";
      popScope();
      return mlir::failure();
    }

    // The condition must be i1 type
    if (!condition.getType().isInteger(1)) {
      llvm::errs() << "Warning: Non-boolean condition in while loop\n";
    }

    // Collect current values for passing to loop body
    std::vector<mlir::Value> conditionValues;
    for (const auto& varName : modifiedVars) {
      mlir::Value value = lookupVariable(varName);
      if (!value) {
        // Use the block argument if not found
        auto it = std::find(modifiedVars.begin(), modifiedVars.end(), varName);
        size_t idx = std::distance(modifiedVars.begin(), it);
        value = beforeBlock->getArgument(idx);
      }
      conditionValues.push_back(value);
    }

    popScope();

    // Terminate with scf.condition, passing values to the body
    builder.create<mlir::scf::ConditionOp>(loc, condition, conditionValues);
  }

  // Build the "after" region (loop body)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Block* afterBlock = builder.createBlock(&whileOp.getAfter());

    // Add block arguments for loop-carried values
    for (auto type : iterTypes) {
      afterBlock->addArgument(type, loc);
    }

    builder.setInsertionPointToStart(afterBlock);

    pushScope();

    // Update symbol table with block arguments
    auto varIt = modifiedVars.begin();
    for (size_t i = 0; i < afterBlock->getNumArguments(); ++i, ++varIt) {
      declareVariable(*varIt, afterBlock->getArgument(i));
    }

    // Lower the loop body
    BlockAST* body = whileLoop->getBody();
    if (body) {
      for (auto* stmt : body->statements) {
        if (failed(lowerStatement(stmt))) {
          popScope();
          return mlir::failure();
        }
      }
    }

    // Collect updated values for the next iteration
    std::vector<mlir::Value> nextIterValues;
    for (const auto& varName : modifiedVars) {
      mlir::Value value = lookupVariable(varName);
      if (!value) {
        // This shouldn't happen if tracking is correct
        auto it = std::find(modifiedVars.begin(), modifiedVars.end(), varName);
        size_t varIdx = std::distance(modifiedVars.begin(), it);
        value = afterBlock->getArgument(varIdx);
      }
      nextIterValues.push_back(value);
    }

    popScope();

    // Terminate with scf.yield, passing updated values back to condition
    // Types should match now due to proper type inference above
    builder.create<mlir::scf::YieldOp>(loc, nextIterValues);
  }

  // Update symbol table with the final results from the while loop
  if (!modifiedVars.empty()) {
    updateSymbolTableWithResults(modifiedVars, whileOp.getResults());
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Control Flow Helpers
//===----------------------------------------------------------------------===//

std::set<std::string> MLIRCodeGenContext::getCurrentVariableNames() const {
  std::set<std::string> varNames;

  // Collect all variable names from all scopes
  for (const auto& scope : symbolTable) {
    for (const auto& entry : scope) {
      varNames.insert(entry.first);
    }
  }

  return varNames;
}

std::set<std::string> MLIRCodeGenContext::trackModifiedVariables(
    BlockAST* block, const std::set<std::string>& existingVars) {
  std::set<std::string> modifiedVars;

  if (!block) return modifiedVars;

  // Scan through all statements to find assignments to existing variables
  for (auto* stmt : block->statements) {
    if (!stmt) continue;

    switch (stmt->getKind()) {
      case ASTKind::VariableDecl: {
        // Variable declarations: check if re-declaring an existing variable
        auto* decl = static_cast<VariableDeclarationAST*>(stmt);
        const std::string& varName = decl->getName();

        // Only track if this variable existed before this block
        if (existingVars.find(varName) != existingVars.end()) {
          modifiedVars.insert(varName);
        }
        // Otherwise it's a new local variable, not loop-carried
        break;
      }

      case ASTKind::ExpressionStmt: {
        auto* exprStmt = static_cast<ExpressionStmtAST*>(stmt);
        if (!exprStmt->getExpression()) break;

        // Check for regular assignment: x = value
        if (exprStmt->getExpression()->getKind() == ASTKind::AssignmentExpr) {
          auto* assign = static_cast<AssignmentExprAST*>(exprStmt->getExpression());
          // Extract variable name from LHS (assuming it's a VariableExprAST)
          if (assign->getLHS() &&
              assign->getLHS()->getKind() == ASTKind::VariableExpr) {
            auto* varExpr = static_cast<VariableExprAST*>(assign->getLHS());
            const std::string& varName = varExpr->getName();

            // Only track if this variable existed before this block
            if (existingVars.find(varName) != existingVars.end()) {
              modifiedVars.insert(varName);
            }
          }
        }
        // Check for array element assignment: A[i] = value
        // In Simp dialect, this is functional: A = array_set(A, i, value)
        else if (exprStmt->getExpression()->getKind() == ASTKind::ArrayStoreExpr) {
          auto* arrayStore = static_cast<ArrayStoreExprAST*>(exprStmt->getExpression());
          // Extract variable name from the array being modified
          if (arrayStore->getArray() &&
              arrayStore->getArray()->getKind() == ASTKind::VariableExpr) {
            auto* varExpr = static_cast<VariableExprAST*>(arrayStore->getArray());
            const std::string& varName = varExpr->getName();

            // Only track if this variable existed before this block
            if (existingVars.find(varName) != existingVars.end()) {
              modifiedVars.insert(varName);
            }
          }
        }
        break;
      }

      case ASTKind::IfStmt: {
        // Recursively track modified vars in nested if
        auto* ifStmt = static_cast<IfAST*>(stmt);
        auto thenVars = trackModifiedVariables(ifStmt->getThenBlock(), existingVars);
        auto elseVars = trackModifiedVariables(ifStmt->getElseBlock(), existingVars);
        modifiedVars.insert(thenVars.begin(), thenVars.end());
        modifiedVars.insert(elseVars.begin(), elseVars.end());
        break;
      }

      case ASTKind::WhileStmt: {
        // Recursively track modified vars in nested while
        auto* whileStmt = static_cast<WhileAST*>(stmt);
        auto bodyVars = trackModifiedVariables(whileStmt->getBody(), existingVars);
        modifiedVars.insert(bodyVars.begin(), bodyVars.end());
        break;
      }

      default:
        // Other statement types don't modify variables
        break;
    }
  }

  return modifiedVars;
}

std::vector<mlir::Value> MLIRCodeGenContext::collectVariableValues(
    const std::set<std::string>& varNames) {
  std::vector<mlir::Value> values;

  for (const auto& name : varNames) {
    mlir::Value value = lookupVariable(name);
    if (value) {
      values.push_back(value);
    } else {
      // This shouldn't happen if tracking is correct
      llvm::errs() << "Warning: Variable '" << name
                   << "' not found when collecting values\n";
    }
  }

  return values;
}

void MLIRCodeGenContext::updateSymbolTableWithResults(
    const std::set<std::string>& varNames,
    mlir::ValueRange results) {

  // Ensure we have the right number of results
  if (varNames.size() != results.size()) {
    llvm::errs() << "Error: Mismatch between variable count ("
                 << varNames.size() << ") and result count ("
                 << results.size() << ")\n";
    return;
  }

  // Update symbol table with new SSA values
  auto varIt = varNames.begin();
  for (size_t i = 0; i < results.size(); ++i, ++varIt) {
    declareVariable(*varIt, results[i]);
  }
}

//===----------------------------------------------------------------------===//
// Function Lowering
//===----------------------------------------------------------------------===//

mlir::FuncOp MLIRCodeGenContext::lowerFunction(FunctionAST* funcAst) {
  auto loc = getUnknownLocation();

  // Save the current insertion point (should be at module level)
  auto savedInsertionPoint = builder.saveInsertionPoint();

  // Build function type
  llvm::SmallVector<mlir::Type, 4> argTypes;
  for (auto* arg : funcAst->getArguments()) {
    // Get type from VariableDeclarationAST
    mlir::Type argType = builder.getF32Type(); // Default (matches existing compiler)
    if (arg->isStaticallyTyped()) {
      argType = getMLIRType(const_cast<TypeInfo*>(arg->getStaticType()));
    }
    argTypes.push_back(argType);
  }

  // Get return type
  mlir::Type returnType = builder.getF32Type(); // Default (matches existing compiler)
  if (funcAst->hasStaticReturnType()) {
    returnType = getMLIRType(const_cast<TypeInfo*>(funcAst->getReturnType()));
  }

  auto funcType = builder.getFunctionType(argTypes, returnType);

  // Create the function (Standard dialect in MLIR 14)
  // Note: builder.create() automatically inserts at the current insertion point
  auto func = builder.create<mlir::FuncOp>(loc, funcAst->getName(), funcType);

  // Create entry block
  auto* entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Set as current function
  setCurrentFunction(func);

  // Push new scope for function body
  pushScope();

  // Declare function arguments
  const auto& args = funcAst->getArguments();
  for (size_t i = 0; i < args.size(); ++i) {
    declareVariable(args[i]->getName(), entryBlock->getArgument(i));
  }

  // Lower function body
  BlockAST* body = funcAst->getBody();
  if (body) {
    for (auto* stmt : body->statements) {
      if (failed(lowerStatement(stmt))) {
        func.erase();
        popScope();
        builder.restoreInsertionPoint(savedInsertionPoint);
        return nullptr;
      }
    }
  }

  // Pop function scope
  popScope();

  // Restore insertion point to module level for next function
  builder.restoreInsertionPoint(savedInsertionPoint);

  return func;
}

mlir::FuncOp MLIRCodeGenContext::getOrCreateFunction(const std::string& name,
                                                     mlir::FunctionType funcType) {
  // Look for existing function in module
  if (auto func = module.lookupSymbol<mlir::FuncOp>(name)) {
    return func;
  }

  // Create new function declaration (Standard dialect in MLIR 14)
  // Note: builder.create() automatically inserts at the current insertion point
  auto loc = getUnknownLocation();
  auto func = builder.create<mlir::FuncOp>(loc, name, funcType);

  return func;
}
