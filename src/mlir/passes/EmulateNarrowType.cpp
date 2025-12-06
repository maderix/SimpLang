//===- EmulateNarrowType.cpp - Emulate sub-byte integer types -------------===//
//
// Part of the SimpLang Project
//
// This pass emulates sub-byte integer types (i4, i2, etc.) by packing them
// into i8 containers. Required for LLVM lowering since LLVM doesn't support
// sub-byte addressable memory.
//
// Strategy:
// - memref<NxMxi4> → memref<Nx(M/2)xi8> (2 values per byte)
// - Load i4: load i8, extract nibble (low or high based on index parity)
// - Store i4: load i8, modify nibble, store i8
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "emulate-narrow-type"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Type Converter: i4 memref → i8 packed memref
//===----------------------------------------------------------------------===//

class NarrowTypeConverter : public TypeConverter {
public:
  NarrowTypeConverter() {
    addConversion([](Type type) { return type; });

    // Convert memref with sub-byte element types
    addConversion([](MemRefType memrefType) -> Type {
      Type elemType = memrefType.getElementType();

      // Only handle i4 for now (can extend to i2, i1 later)
      if (auto intType = elemType.dyn_cast<IntegerType>()) {
        unsigned width = intType.getWidth();
        if (width == 4) {
          // Pack 2 i4 values per i8 byte
          // Last dimension gets halved (rounded up for odd sizes)
          auto shape = memrefType.getShape();
          SmallVector<int64_t> newShape(shape.begin(), shape.end());

          // Halve the last dimension (pack along innermost)
          if (!newShape.empty()) {
            newShape.back() = (newShape.back() + 1) / 2;
          }

          auto i8Type = IntegerType::get(memrefType.getContext(), 8);
          return MemRefType::get(newShape, i8Type, memrefType.getLayout(),
                                  memrefType.getMemorySpace());
        }
      }
      return memrefType;
    });
  }
};

//===----------------------------------------------------------------------===//
// Lowering Patterns
//===----------------------------------------------------------------------===//

/// Convert memref.alloc of i4 to i8 packed allocation
struct AllocOpLowering : public OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::AllocOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto oldType = op.getType();
    auto elemType = oldType.getElementType();

    // Only convert i4 memrefs
    auto intType = elemType.dyn_cast<IntegerType>();
    if (!intType || intType.getWidth() != 4)
      return failure();

    // Create packed i8 memref type
    auto shape = oldType.getShape();
    SmallVector<int64_t> newShape(shape.begin(), shape.end());
    if (!newShape.empty()) {
      newShape.back() = (newShape.back() + 1) / 2;
    }

    auto i8Type = rewriter.getIntegerType(8);
    auto newType = MemRefType::get(newShape, i8Type, oldType.getLayout(),
                                    oldType.getMemorySpace());

    rewriter.replaceOpWithNewOp<memref::AllocOp>(op, newType);
    return success();
  }
};

/// Convert memref.load of i4 to extract from packed i8
struct LoadOpLowering : public OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::LoadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto memrefType = op.getMemRef().getType().cast<MemRefType>();
    auto elemType = memrefType.getElementType();

    // Only convert i4 loads
    auto intType = elemType.dyn_cast<IntegerType>();
    if (!intType || intType.getWidth() != 4)
      return failure();

    Location loc = op.getLoc();
    Value memref = adaptor.memref();
    auto indices = adaptor.indices();
    size_t numIndices = indices.size();

    if (numIndices == 0)
      return failure();

    // Compute byte index and nibble position
    // byteIdx = indices.back() / 2
    // isHighNibble = indices.back() % 2
    Value lastIdx = indices[numIndices - 1];

    auto i8Type = rewriter.getIntegerType(8);
    auto i32Type = rewriter.getI32Type();
    auto indexType = rewriter.getIndexType();

    // Convert to index arithmetic
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    Value byteIdx = rewriter.create<arith::DivUIOp>(loc, lastIdx, c2);
    Value nibblePos = rewriter.create<arith::RemUIOp>(loc, lastIdx, c2);

    // Build new indices with byte index
    SmallVector<Value> newIndices;
    for (size_t i = 0; i < numIndices - 1; ++i) {
      newIndices.push_back(indices[i]);
    }
    newIndices.push_back(byteIdx);

    // Load packed byte
    Value packedByte = rewriter.create<memref::LoadOp>(loc, memref, newIndices);

    // Extract nibble based on position
    // if even: low nibble (& 0x0F)
    // if odd: high nibble (>> 4)
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value isLowNibble = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, nibblePos, zero);

    Value c4_i8 = rewriter.create<arith::ConstantOp>(
        loc, i8Type, rewriter.getI8IntegerAttr(4));
    Value c15_i8 = rewriter.create<arith::ConstantOp>(
        loc, i8Type, rewriter.getI8IntegerAttr(15));

    // Low nibble: byte & 0x0F
    Value lowNibble = rewriter.create<arith::AndIOp>(loc, packedByte, c15_i8);

    // High nibble: (byte >> 4) & 0x0F
    Value shifted = rewriter.create<arith::ShRUIOp>(loc, packedByte, c4_i8);
    Value highNibble = rewriter.create<arith::AndIOp>(loc, shifted, c15_i8);

    // Select based on nibble position
    Value result8 = rewriter.create<SelectOp>(loc, isLowNibble, lowNibble, highNibble);

    // Sign-extend from i4 to i8 (for signed values)
    // Check if bit 3 is set, if so extend sign
    Value c8_i8 = rewriter.create<arith::ConstantOp>(
        loc, i8Type, rewriter.getI8IntegerAttr(8));
    Value hasSign = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::uge, result8, c8_i8);
    Value signExtend = rewriter.create<arith::ConstantOp>(
        loc, i8Type, rewriter.getI8IntegerAttr(0xF0));
    Value extended = rewriter.create<arith::OrIOp>(loc, result8, signExtend);
    Value signedResult = rewriter.create<SelectOp>(loc, hasSign, extended, result8);

    // Truncate to i4
    auto i4Type = rewriter.getIntegerType(4);
    Value result = rewriter.create<arith::TruncIOp>(loc, i4Type, signedResult);

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Convert memref.store of i4 to read-modify-write of packed i8
struct StoreOpLowering : public OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::StoreOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto memrefType = op.getMemRef().getType().cast<MemRefType>();
    auto elemType = memrefType.getElementType();

    // Only convert i4 stores
    auto intType = elemType.dyn_cast<IntegerType>();
    if (!intType || intType.getWidth() != 4)
      return failure();

    Location loc = op.getLoc();
    Value memref = adaptor.memref();
    Value value = adaptor.value();
    auto indices = adaptor.indices();
    size_t numIndices = indices.size();

    if (numIndices == 0)
      return failure();

    Value lastIdx = indices[numIndices - 1];

    auto i8Type = rewriter.getIntegerType(8);

    // Compute byte index and nibble position
    Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    Value byteIdx = rewriter.create<arith::DivUIOp>(loc, lastIdx, c2);
    Value nibblePos = rewriter.create<arith::RemUIOp>(loc, lastIdx, c2);

    // Build new indices
    SmallVector<Value> newIndices;
    for (size_t i = 0; i < numIndices - 1; ++i) {
      newIndices.push_back(indices[i]);
    }
    newIndices.push_back(byteIdx);

    // Load current packed byte
    Value packedByte = rewriter.create<memref::LoadOp>(loc, memref, newIndices);

    // Extend value to i8 and mask to 4 bits
    Value value8 = rewriter.create<arith::ExtUIOp>(loc, i8Type, value);
    Value c15_i8 = rewriter.create<arith::ConstantOp>(
        loc, i8Type, rewriter.getI8IntegerAttr(15));
    value8 = rewriter.create<arith::AndIOp>(loc, value8, c15_i8);

    // Determine if low or high nibble
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value isLowNibble = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, nibblePos, zero);

    // For low nibble: clear low 4 bits, OR with value
    Value c240_i8 = rewriter.create<arith::ConstantOp>(
        loc, i8Type, rewriter.getI8IntegerAttr(0xF0));
    Value clearedLow = rewriter.create<arith::AndIOp>(loc, packedByte, c240_i8);
    Value newByteLow = rewriter.create<arith::OrIOp>(loc, clearedLow, value8);

    // For high nibble: clear high 4 bits, OR with (value << 4)
    Value c4_i8 = rewriter.create<arith::ConstantOp>(
        loc, i8Type, rewriter.getI8IntegerAttr(4));
    Value shiftedValue = rewriter.create<arith::ShLIOp>(loc, value8, c4_i8);
    Value clearedHigh = rewriter.create<arith::AndIOp>(loc, packedByte, c15_i8);
    Value newByteHigh = rewriter.create<arith::OrIOp>(loc, clearedHigh, shiftedValue);

    // Select based on nibble position
    Value newByte = rewriter.create<SelectOp>(loc, isLowNibble, newByteLow, newByteHigh);

    // Store modified byte
    rewriter.create<memref::StoreOp>(loc, newByte, memref, newIndices);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct EmulateNarrowTypePass
    : public PassWrapper<EmulateNarrowTypePass, OperationPass<ModuleOp>> {

  StringRef getArgument() const override { return "emulate-narrow-type"; }
  StringRef getDescription() const override {
    return "Emulate sub-byte integer types (i4, i2) by packing into i8";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
    registry.insert<arith::ArithmeticDialect>();
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // Check if we have any i4 types to convert
    bool hasNarrowTypes = false;
    module.walk([&](Operation *op) {
      for (auto type : op->getResultTypes()) {
        if (auto memrefType = type.dyn_cast<MemRefType>()) {
          if (auto intType = memrefType.getElementType().dyn_cast<IntegerType>()) {
            if (intType.getWidth() < 8) {
              hasNarrowTypes = true;
              return WalkResult::interrupt();
            }
          }
        }
      }
      return WalkResult::advance();
    });

    if (!hasNarrowTypes) {
      LLVM_DEBUG(llvm::dbgs() << "No narrow types found, skipping pass\n");
      return;
    }

    LLVM_DEBUG(llvm::dbgs() << "Found narrow types, applying emulation\n");

    // Set up conversion target
    ConversionTarget target(*ctx);
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<arith::ArithmeticDialect>();
    target.addLegalDialect<scf::SCFDialect>();

    // Mark i4 memref operations as illegal
    target.addDynamicallyLegalOp<memref::AllocOp>([](memref::AllocOp op) {
      auto elemType = op.getType().getElementType();
      if (auto intType = elemType.dyn_cast<IntegerType>())
        return intType.getWidth() >= 8;
      return true;
    });

    target.addDynamicallyLegalOp<memref::LoadOp>([](memref::LoadOp op) {
      auto memrefType = op.getMemRef().getType().cast<MemRefType>();
      auto elemType = memrefType.getElementType();
      if (auto intType = elemType.dyn_cast<IntegerType>())
        return intType.getWidth() >= 8;
      return true;
    });

    target.addDynamicallyLegalOp<memref::StoreOp>([](memref::StoreOp op) {
      auto memrefType = op.getMemRef().getType().cast<MemRefType>();
      auto elemType = memrefType.getElementType();
      if (auto intType = elemType.dyn_cast<IntegerType>())
        return intType.getWidth() >= 8;
      return true;
    });

    // Set up type converter and patterns
    NarrowTypeConverter typeConverter;
    RewritePatternSet patterns(ctx);

    patterns.add<AllocOpLowering, LoadOpLowering, StoreOpLowering>(
        typeConverter, ctx);

    // Apply conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

namespace mlir {
namespace simp {

std::unique_ptr<Pass> createEmulateNarrowTypePass() {
  return std::make_unique<EmulateNarrowTypePass>();
}

} // namespace simp
} // namespace mlir
