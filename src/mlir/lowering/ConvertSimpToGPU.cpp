//===- ConvertSimpToGPU.cpp - Simp to GPU dialect lowering ----------------===//
//
// This file implements the lowering of Simp tensor operations to GPU dialect.
// Key transformations:
// - tensor_create -> gpu.alloc (device memory)
// - tensor_matmul -> gpu.launch with GEMM kernel OR cuBLAS call
// - Element-wise ops -> gpu.launch with parallel kernels
//
//===----------------------------------------------------------------------===//

#ifdef USE_CUDA

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/gpu/GPUPasses.h"

#include <iostream>

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Check if a memref type is suitable for GPU execution
bool isGPUCompatibleMemRef(MemRefType type) {
    // GPU works best with contiguous memory and standard element types
    if (!type.hasStaticShape()) return false;

    Type elemType = type.getElementType();
    return elemType.isF32() || elemType.isF16() || elemType.isInteger(32) ||
           elemType.isInteger(16) || elemType.isInteger(8);
}

/// Calculate total number of elements in a memref
int64_t getNumElements(MemRefType type) {
    int64_t numElems = 1;
    for (int64_t dim : type.getShape()) {
        numElems *= dim;
    }
    return numElems;
}

/// Get block and grid dimensions for a 1D kernel launch
std::pair<int64_t, int64_t> get1DLaunchConfig(int64_t numElements) {
    const int64_t blockSize = 256;  // Standard block size
    int64_t gridSize = (numElements + blockSize - 1) / blockSize;
    return {gridSize, blockSize};
}

/// Get block and grid dimensions for a 2D kernel launch (for matmul)
std::tuple<int64_t, int64_t, int64_t, int64_t> get2DLaunchConfig(int64_t M, int64_t N) {
    const int64_t tileM = 16;  // Thread block tile size
    const int64_t tileN = 16;
    int64_t gridM = (M + tileM - 1) / tileM;
    int64_t gridN = (N + tileN - 1) / tileN;
    return {gridM, gridN, tileM, tileN};
}

//===----------------------------------------------------------------------===//
// GPU Memory Allocation Pattern
// Converts memref.alloc to gpu.alloc for device memory
//===----------------------------------------------------------------------===//

struct AllocToGPUAllocPattern : public OpRewritePattern<memref::AllocOp> {
    using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(memref::AllocOp op,
                                  PatternRewriter& rewriter) const override {
        MemRefType memRefType = op.getType();

        if (!isGPUCompatibleMemRef(memRefType)) {
            return failure();
        }

        // Create gpu.alloc operation
        auto gpuAlloc = rewriter.create<gpu::AllocOp>(
            op.getLoc(),
            memRefType,
            /*asyncToken=*/Type(),
            /*asyncDependencies=*/ValueRange(),
            /*dynamicSizes=*/ValueRange(),
            /*symbolOperands=*/ValueRange()
        );

        rewriter.replaceOp(op, gpuAlloc.getResult(0));
        return success();
    }
};

//===----------------------------------------------------------------------===//
// GPU Memory Deallocation Pattern
//===----------------------------------------------------------------------===//

struct DeallocToGPUDeallocPattern : public OpRewritePattern<memref::DeallocOp> {
    using OpRewritePattern<memref::DeallocOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(memref::DeallocOp op,
                                  PatternRewriter& rewriter) const override {
        Value memref = op.memref();

        // Check if this memref was allocated with gpu.alloc
        if (auto gpuAlloc = memref.getDefiningOp<gpu::AllocOp>()) {
            rewriter.create<gpu::DeallocOp>(
                op.getLoc(),
                /*asyncToken=*/Type(),
                /*asyncDependencies=*/ValueRange(),
                memref
            );
            rewriter.eraseOp(op);
            return success();
        }

        return failure();
    }
};

//===----------------------------------------------------------------------===//
// GPU Memory Copy Patterns (Host <-> Device)
//===----------------------------------------------------------------------===//

struct CopyToGPUMemcpyPattern : public OpRewritePattern<memref::CopyOp> {
    using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(memref::CopyOp op,
                                  PatternRewriter& rewriter) const override {
        Value src = op.source();
        Value dst = op.target();

        // Check if either source or dest is GPU memory
        bool srcIsGPU = src.getDefiningOp<gpu::AllocOp>() != nullptr;
        bool dstIsGPU = dst.getDefiningOp<gpu::AllocOp>() != nullptr;

        if (!srcIsGPU && !dstIsGPU) {
            return failure();  // Neither is GPU memory
        }

        // Create gpu.memcpy operation
        rewriter.create<gpu::MemcpyOp>(
            op.getLoc(),
            /*asyncToken=*/Type(),
            /*asyncDependencies=*/ValueRange(),
            dst, src
        );

        rewriter.eraseOp(op);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Linalg.generic to GPU Launch Pattern
// Converts element-wise operations to GPU kernels
//===----------------------------------------------------------------------===//

struct LinalgGenericToGPUPattern : public OpRewritePattern<linalg::GenericOp> {
    using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::GenericOp op,
                                  PatternRewriter& rewriter) const override {
        // Only handle element-wise operations (parallel iterators only)
        for (auto iterType : op.iterator_types()) {
            if (iterType.cast<StringAttr>().getValue() != "parallel") {
                return failure();  // Not purely parallel
            }
        }

        // Get output memref to determine launch dimensions
        if (op.getNumOutputs() == 0) return failure();

        Value output = op.outputs()[0];
        auto memRefType = output.getType().dyn_cast<MemRefType>();
        if (!memRefType || !isGPUCompatibleMemRef(memRefType)) {
            return failure();
        }

        Location loc = op.getLoc();
        int64_t numElements = getNumElements(memRefType);
        auto [gridSize, blockSize] = get1DLaunchConfig(numElements);

        // Create constants for launch dimensions
        Value gridSizeVal = rewriter.create<arith::ConstantIndexOp>(loc, gridSize);
        Value blockSizeVal = rewriter.create<arith::ConstantIndexOp>(loc, blockSize);
        Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

        // Create GPU launch operation
        auto launchOp = rewriter.create<gpu::LaunchOp>(
            loc,
            gridSizeVal, one, one,      // Grid dimensions
            blockSizeVal, one, one      // Block dimensions
        );

        // Build the kernel body
        rewriter.setInsertionPointToStart(&launchOp.body().front());

        // Calculate global thread ID
        Value blockIdx = launchOp.getBlockIds().x;
        Value threadIdx = launchOp.getThreadIds().x;
        Value blockDim = launchOp.getBlockSize().x;

        Value globalIdx = rewriter.create<arith::AddIOp>(
            loc,
            rewriter.create<arith::MulIOp>(loc, blockIdx, blockDim),
            threadIdx
        );

        // Bounds check
        Value numElemsVal = rewriter.create<arith::ConstantIndexOp>(loc, numElements);
        Value inBounds = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ult, globalIdx, numElemsVal
        );

        // Create conditional execution
        auto ifOp = rewriter.create<scf::IfOp>(loc, inBounds, /*withElse=*/false);
        rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());

        // Clone the operation body into the GPU kernel
        // This is a simplified version - real implementation would need to
        // convert the linalg region to work with the thread index

        // For now, just add a terminator
        rewriter.setInsertionPointToEnd(&launchOp.body().front());
        rewriter.create<gpu::TerminatorOp>(loc);

        // TODO: Properly lower the linalg body to indexed GPU operations
        // This requires converting the linalg affine maps to thread indexing

        return success();
    }
};

//===----------------------------------------------------------------------===//
// Linalg.matmul to cuBLAS Call Pattern
// For large matrices, use cuBLAS SGEMM instead of custom kernels
//===----------------------------------------------------------------------===//

struct MatmulToCublasPattern : public OpRewritePattern<linalg::MatmulOp> {
    using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                  PatternRewriter& rewriter) const override {
        Location loc = op.getLoc();

        // Get operand types
        Value A = op.inputs()[0];
        Value B = op.inputs()[1];
        Value C = op.outputs()[0];

        auto aType = A.getType().dyn_cast<MemRefType>();
        auto bType = B.getType().dyn_cast<MemRefType>();
        auto cType = C.getType().dyn_cast<MemRefType>();

        if (!aType || !bType || !cType) return failure();
        if (!aType.getElementType().isF32()) return failure();  // cuBLAS SGEMM

        // Get dimensions: A[M,K], B[K,N], C[M,N]
        if (aType.getRank() != 2 || bType.getRank() != 2 || cType.getRank() != 2) {
            return failure();
        }

        int64_t M = aType.getShape()[0];
        int64_t K = aType.getShape()[1];
        int64_t N = bType.getShape()[1];

        // For small matrices, let MLIR generate GPU kernels
        // For large matrices (M,N,K >= 128), use cuBLAS
        if (M < 128 && N < 128 && K < 128) {
            return failure();  // Let other patterns handle small matrices
        }

        // Create call to cuBLAS SGEMM
        // This requires the external function to be declared in the module
        ModuleOp module = op->getParentOfType<ModuleOp>();

        // Look for or create cublasSgemm function declaration
        auto sgemm = module.lookupSymbol<FuncOp>("simp_cublas_sgemm");
        if (!sgemm) {
            // Create function declaration
            // void simp_cublas_sgemm(bool transA, bool transB, int M, int N, int K,
            //                        float alpha, float* A, int lda, float* B, int ldb,
            //                        float beta, float* C, int ldc)
            auto funcType = rewriter.getFunctionType(
                {rewriter.getI1Type(), rewriter.getI1Type(),
                 rewriter.getI32Type(), rewriter.getI32Type(), rewriter.getI32Type(),
                 rewriter.getF32Type(),
                 MemRefType::get({-1, -1}, rewriter.getF32Type()),
                 rewriter.getI32Type(),
                 MemRefType::get({-1, -1}, rewriter.getF32Type()),
                 rewriter.getI32Type(),
                 rewriter.getF32Type(),
                 MemRefType::get({-1, -1}, rewriter.getF32Type()),
                 rewriter.getI32Type()},
                {}
            );

            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(module.getBody());
            sgemm = rewriter.create<FuncOp>(loc, "simp_cublas_sgemm", funcType);
            sgemm.setPrivate();
        }

        // Create constants
        Value falseVal = rewriter.create<arith::ConstantIntOp>(loc, 0, 1);
        Value mVal = rewriter.create<arith::ConstantIntOp>(loc, M, 32);
        Value nVal = rewriter.create<arith::ConstantIntOp>(loc, N, 32);
        Value kVal = rewriter.create<arith::ConstantIntOp>(loc, K, 32);
        Value alpha = rewriter.create<arith::ConstantFloatOp>(
            loc, APFloat(1.0f), rewriter.getF32Type());
        Value beta = rewriter.create<arith::ConstantFloatOp>(
            loc, APFloat(0.0f), rewriter.getF32Type());
        Value ldaVal = rewriter.create<arith::ConstantIntOp>(loc, K, 32);
        Value ldbVal = rewriter.create<arith::ConstantIntOp>(loc, N, 32);
        Value ldcVal = rewriter.create<arith::ConstantIntOp>(loc, N, 32);

        // Cast static memrefs to dynamic memrefs for the external function call
        auto dynamicMemRefType = MemRefType::get({-1, -1}, rewriter.getF32Type());
        Value aCast = rewriter.create<memref::CastOp>(loc, dynamicMemRefType, A);
        Value bCast = rewriter.create<memref::CastOp>(loc, dynamicMemRefType, B);
        Value cCast = rewriter.create<memref::CastOp>(loc, dynamicMemRefType, C);

        // Call cuBLAS
        rewriter.create<CallOp>(loc, sgemm,
            ValueRange{falseVal, falseVal, mVal, nVal, kVal,
                       alpha, aCast, ldaVal, bCast, ldbVal, beta, cCast, ldcVal});

        rewriter.eraseOp(op);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// ConvertSimpToGPU Pass
//===----------------------------------------------------------------------===//

struct ConvertSimpToGPUPass
    : public PassWrapper<ConvertSimpToGPUPass, OperationPass<ModuleOp>> {

    void getDependentDialects(DialectRegistry& registry) const override {
        registry.insert<gpu::GPUDialect>();
        registry.insert<NVVM::NVVMDialect>();
        registry.insert<memref::MemRefDialect>();
        registry.insert<arith::ArithmeticDialect>();
        registry.insert<scf::SCFDialect>();
    }

    StringRef getArgument() const override { return "convert-simp-to-gpu"; }
    StringRef getDescription() const override {
        return "Convert Simp tensor operations to GPU dialect";
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        MLIRContext* ctx = &getContext();

        RewritePatternSet patterns(ctx);

        // Add conversion patterns
        // NOTE: Memory patterns disabled for now - gpu.alloc at module level doesn't translate
        // to LLVM IR. Need proper CUDA runtime integration for memory management.
        // patterns.add<AllocToGPUAllocPattern>(ctx);
        // patterns.add<DeallocToGPUDeallocPattern>(ctx);
        // patterns.add<CopyToGPUMemcpyPattern>(ctx);

        // MatMul pattern for cuBLAS calls (large matrices)
        patterns.add<MatmulToCublasPattern>(ctx);
        // patterns.add<LinalgGenericToGPUPattern>(ctx);  // TODO: Complete implementation

        if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
            signalPassFailure();
            return;
        }

        llvm::outs() << "[GPU] ConvertSimpToGPU pass completed\n";
    }
};

} // anonymous namespace

namespace mlir {
namespace simp {

std::unique_ptr<Pass> createConvertSimpToGPUPass() {
    return std::make_unique<ConvertSimpToGPUPass>();
}

void registerGPUPasses() {
    PassRegistration<ConvertSimpToGPUPass>();
}

} // namespace simp
} // namespace mlir

#endif // USE_CUDA
