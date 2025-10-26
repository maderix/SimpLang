// PromoteLargeAllocaToHeap.cpp
// Custom LLVM pass to replace large stack allocations with heap allocations
//
// Problem: MLIR's vectorization with tiling creates large alloca instructions
// that exceed stack limits (e.g., [64 x <64 x float>] = 16KB per buffer).
// This causes stack overflow and segfaults.
//
// Solution: Replace alloca > threshold with malloc/free for heap allocation.

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Constants.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

namespace {

struct PromoteLargeAllocaToHeap : public FunctionPass {
  static char ID;

  // Threshold: promote allocas larger than 8KB to heap
  // 32x32 float = 4KB (stays on stack)
  // 64x64 float = 16KB (promoted to heap)
  static constexpr uint64_t HEAP_THRESHOLD_BYTES = 4 * 1024;  // 4KB threshold (catches 64x64 tile buffers = 16KB)

  PromoteLargeAllocaToHeap() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    bool Modified = false;
    Module *M = F.getParent();
    const DataLayout &DL = M->getDataLayout();

    SmallVector<AllocaInst*, 16> LargeAllocas;

    // Collect large allocas
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (AllocaInst *AI = dyn_cast<AllocaInst>(&I)) {
          Type *AllocatedType = AI->getAllocatedType();
          uint64_t AllocSize = DL.getTypeAllocSize(AllocatedType);

          if (AllocSize > 8000) {  // Only debug large ones
            errs() << "Checking large alloca: " << *AI << "\n";
            errs() << "  Allocated type size: " << AllocSize << "\n";
            errs() << "  Is array allocation: " << AI->isArrayAllocation() << "\n";
          }

          // Handle array allocations
          if (AI->isArrayAllocation()) {
            Value *ArraySize = AI->getArraySize();
            errs() << "  Array allocation with size: " << *ArraySize << "\n";

            if (ConstantInt *CI = dyn_cast<ConstantInt>(ArraySize)) {
              AllocSize *= CI->getZExtValue();
              errs() << "  Direct ConstantInt: " << CI->getZExtValue() << "\n";
            } else {
              // Try to evaluate constant expression
              errs() << "  Not a ConstantInt, trying to evaluate...\n";
              if (ConstantExpr *CE = dyn_cast<ConstantExpr>(ArraySize)) {
                errs() << "  Is ConstantExpr\n";
                if (Constant *C = ConstantFoldConstant(CE, DL)) {
                  if (ConstantInt *CI = dyn_cast<ConstantInt>(C)) {
                    AllocSize *= CI->getZExtValue();
                    errs() << "  Evaluated constant expr to: " << CI->getZExtValue() << "\n";
                  } else {
                    // Dynamic array size - conservatively promote
                    AllocSize = HEAP_THRESHOLD_BYTES + 1;
                    errs() << "  Could not fold constant expr, promoting conservatively\n";
                  }
                } else {
                  AllocSize = HEAP_THRESHOLD_BYTES + 1;
                  errs() << "  Constant folding failed, promoting conservatively\n";
                }
              } else {
                // Dynamic array size - conservatively promote
                AllocSize = HEAP_THRESHOLD_BYTES + 1;
                errs() << "  Dynamic array size, promoting conservatively\n";
              }
            }
          }

          if (AllocSize > HEAP_THRESHOLD_BYTES) {
            LargeAllocas.push_back(AI);
            errs() << "Found large alloca: " << AllocSize << " bytes in function "
                   << F.getName() << "\n";
          }
        }
      }
    }

    // Replace large allocas with malloc/free
    for (AllocaInst *AI : LargeAllocas) {
      errs() << "Replacing alloca: " << *AI << "\n";

      IRBuilder<> Builder(AI);

      // Calculate total allocation size
      Type *AllocatedType = AI->getAllocatedType();
      uint64_t TypeSize = DL.getTypeAllocSize(AllocatedType);
      Value *ArraySize = AI->getArraySize();

      // FIX MLIR BUG: Check if ArraySize is ptrtoint(gep(null, 1)) pattern
      // This evaluates to sizeof(TYPE), but MLIR incorrectly uses it as array count!
      // Correct: alloca TYPE, 1
      // Wrong: alloca TYPE, sizeof(TYPE)  <- allocates sizeof(TYPE) copies!
      uint64_t ArrayCount = 1;
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(ArraySize)) {
        if (Constant *C = ConstantFoldConstant(CE, DL)) {
          if (ConstantInt *CI = dyn_cast<ConstantInt>(C)) {
            ArrayCount = CI->getZExtValue();
            // If array count == sizeof(type), it's the MLIR bug! Fix to 1.
            if (ArrayCount == TypeSize) {
              errs() << "  MLIR BUG DETECTED: ArraySize=" << ArrayCount
                     << " == TypeSize=" << TypeSize << ", fixing to 1\n";
              ArrayCount = 1;
            }
          }
        }
      } else if (ConstantInt *CI = dyn_cast<ConstantInt>(ArraySize)) {
        ArrayCount = CI->getZExtValue();
        if (ArrayCount == TypeSize) {
          errs() << "  MLIR BUG DETECTED: ArraySize=" << ArrayCount
                 << " == TypeSize=" << TypeSize << ", fixing to 1\n";
          ArrayCount = 1;
        }
      }

      Value *TotalSize = Builder.getInt64(TypeSize * ArrayCount);
      errs() << "  Total size to allocate: " << (TypeSize * ArrayCount) << " bytes\n";

      // Insert malloc call
      FunctionCallee MallocFunc = M->getOrInsertFunction(
          "malloc",
          PointerType::get(Builder.getInt8Ty(), 0),
          Builder.getInt64Ty()
      );

      CallInst *MallocCall = Builder.CreateCall(MallocFunc, {TotalSize});
      MallocCall->setName(AI->getName() + ".heap");

      errs() << "  Created malloc call\n";

      // Bitcast malloc result to correct type
      Value *BitCast = Builder.CreateBitCast(
          MallocCall,
          AI->getType(),
          AI->getName() + ".cast"
      );

      errs() << "  Created bitcast\n";

      // Replace alloca uses with malloc
      AI->replaceAllUsesWith(BitCast);
      errs() << "  Replaced all uses\n";

      // Insert free calls before all returns
      FunctionCallee FreeFunc = M->getOrInsertFunction(
          "free",
          Builder.getVoidTy(),
          PointerType::get(Builder.getInt8Ty(), 0)
      );

      for (BasicBlock &BB : F) {
        if (ReturnInst *RI = dyn_cast<ReturnInst>(BB.getTerminator())) {
          Builder.SetInsertPoint(RI);
          Builder.CreateCall(FreeFunc, {MallocCall});
        }
      }

      // Remove the alloca
      AI->eraseFromParent();
      Modified = true;
    }

    return Modified;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    // This pass modifies the CFG
  }
};

} // anonymous namespace

char PromoteLargeAllocaToHeap::ID = 0;

// Registration for opt tool
static RegisterPass<PromoteLargeAllocaToHeap> X(
    "promote-large-alloca-to-heap",
    "Promote large stack allocations to heap allocations",
    false, false);

// Export the pass for use in the pipeline
namespace llvm {
FunctionPass *createPromoteLargeAllocaToHeapPass() {
  return new PromoteLargeAllocaToHeap();
}
} // namespace llvm
