//===- VNNIPass.cpp - LLVM Pass to emit vpdpbusd for INT8 dot products ----===//
//
// This pass identifies reduction loops with i8*i8->i32 pattern and replaces
// them with AVX512-VNNI vpdpbusd instructions.
//
// FUTURE OPTIMIZATION (I=4 Tiling):
// C++ benchmark shows I=4 tiling achieves 397 GIOP/s vs 217 GIOP/s baseline.
// To implement:
//   1. Detect outer I loop (grandparent of K loop)
//   2. Modify I loop to step by 4
//   3. Generate 4 accumulators, load B once, load 4 A rows
//   4. Store 4 results per iteration
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsX86.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

namespace {

//===----------------------------------------------------------------------===//
// VNNICandidate - Information about a detected VNNI-compatible loop
//===----------------------------------------------------------------------===//
struct VNNICandidate {
  // Core loop components
  PHINode *AccPhi;              // Accumulator PHI (i32) - may be null for load-store pattern
  PHINode *IndPhi;              // Induction variable PHI (i64)
  BinaryOperator *Add;          // acc = acc + product
  BinaryOperator *Mul;          // product = a * b

  // Memory access pattern for A and B
  Instruction *ExtA;            // sext/zext from i8
  Instruction *ExtB;            // sext/zext from i8
  LoadInst *LoadA;
  LoadInst *LoadB;
  GetElementPtrInst *GEPA;
  GetElementPtrInst *GEPB;

  // Memory access pattern for C (load-store pattern, when AccPhi is null)
  LoadInst *LoadC;              // Load from C for accumulation
  StoreInst *StoreC;            // Store to C after accumulation
  GetElementPtrInst *GEPC;      // GEP for C array

  // Loop properties
  int64_t TripCount;
  int64_t RowStrideA;           // Stride between A rows (for future I-tiling)
  bool BothSigned;              // true if both inputs are signed (need bias correction)
  bool IsLoadStorePattern;      // true if using load-store instead of PHI
};

//===----------------------------------------------------------------------===//
// VNNIPass - Main pass implementation
//===----------------------------------------------------------------------===//
struct VNNIPass : public FunctionPass {
  static char ID;
  VNNIPass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    bool Changed = false;
    auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

    // Collect all innermost loops (K loops in matmul)
    SmallVector<Loop*, 8> InnermostLoops;
    for (Loop *L : LI) {
      collectInnermostLoops(L, InnermostLoops);
    }

    // Transform each qualifying loop
    for (Loop *L : InnermostLoops) {
      VNNICandidate Candidate;
      if (detectVNNIPattern(L, Candidate)) {
        errs() << "VNNI: Detected pattern, trip=" << Candidate.TripCount
               << ", stride=" << Candidate.RowStrideA << "\n";

        // Step 1: Detect parent loops
        // K loop -> J loop (parent) -> I loop (grandparent)
        if (Loop *JLoop = L->getParentLoop()) {
          errs() << "VNNI: Found J loop (parent of K)\n";
          analyzeParentLoop(JLoop);
          if (Loop *ILoop = JLoop->getParentLoop()) {
            errs() << "VNNI: Found I loop (grandparent of K) - THIS IS THE TARGET\n";
            analyzeParentLoop(ILoop);
          } else {
            errs() << "VNNI: No I loop (grandparent) found\n";
          }
        } else {
          errs() << "VNNI: No parent loop found\n";
        }

        if (transformToVNNI(F, L, Candidate, LI)) {
          Changed = true;
        }
      }
    }
    return Changed;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
  }

private:
  //===--------------------------------------------------------------------===//
  // Loop Collection
  //===--------------------------------------------------------------------===//
  void collectInnermostLoops(Loop *L, SmallVectorImpl<Loop*> &Result) {
    if (L->getSubLoops().empty()) {
      Result.push_back(L);
    } else {
      for (Loop *SubL : L->getSubLoops()) {
        collectInnermostLoops(SubL, Result);
      }
    }
  }

  //===--------------------------------------------------------------------===//
  // Parent Loop Analysis and Modification (for I=4 tiling)
  //===--------------------------------------------------------------------===//

  // Returns the parent loop's induction PHI if suitable for I=4 tiling
  PHINode* getParentLoopIndPhi(Loop *ParentLoop) {
    BasicBlock *Header = ParentLoop->getHeader();
    if (!Header) return nullptr;

    for (PHINode &Phi : Header->phis()) {
      if (Phi.getType()->isIntegerTy(64)) {
        return &Phi;
      }
    }
    return nullptr;
  }

  // Modify parent loop to step by 4 instead of 1
  bool modifyParentLoopStep(Loop *ParentLoop, int NewStep) {
    PHINode *IndPhi = getParentLoopIndPhi(ParentLoop);
    if (!IndPhi) {
      errs() << "VNNI:   - No i64 induction PHI found\n";
      return false;
    }

    errs() << "VNNI:   - Induction PHI: " << *IndPhi << "\n";

    // Find and modify the increment
    for (unsigned i = 0; i < IndPhi->getNumIncomingValues(); i++) {
      if (ParentLoop->contains(IndPhi->getIncomingBlock(i))) {
        Value *Inc = IndPhi->getIncomingValue(i);

        if (auto *Add = dyn_cast<BinaryOperator>(Inc)) {
          if (Add->getOpcode() == Instruction::Add) {
            // Check which operand is the constant step
            if (auto *CI = dyn_cast<ConstantInt>(Add->getOperand(1))) {
              if (CI->getSExtValue() == 1) {
                errs() << "VNNI:   - Changing step from 1 to " << NewStep << "\n";
                Add->setOperand(1, ConstantInt::get(CI->getType(), NewStep));
                return true;
              }
            } else if (auto *CI = dyn_cast<ConstantInt>(Add->getOperand(0))) {
              if (CI->getSExtValue() == 1) {
                errs() << "VNNI:   - Changing step from 1 to " << NewStep << "\n";
                Add->setOperand(0, ConstantInt::get(CI->getType(), NewStep));
                return true;
              }
            }
          }
        }
      }
    }

    errs() << "VNNI:   - Could not find step=1 increment\n";
    return false;
  }

  void analyzeParentLoop(Loop *L) {
    BasicBlock *Header = L->getHeader();
    if (!Header) {
      errs() << "VNNI:   - No header\n";
      return;
    }

    PHINode *IndPhi = getParentLoopIndPhi(L);
    if (!IndPhi) {
      errs() << "VNNI:   - No i64 induction PHI found\n";
      return;
    }

    errs() << "VNNI:   - Induction PHI: " << *IndPhi << "\n";

    // Find the increment (should be +1 currently)
    for (unsigned i = 0; i < IndPhi->getNumIncomingValues(); i++) {
      if (L->contains(IndPhi->getIncomingBlock(i))) {
        Value *Inc = IndPhi->getIncomingValue(i);
        errs() << "VNNI:   - Increment value: " << *Inc << "\n";

        if (auto *Add = dyn_cast<BinaryOperator>(Inc)) {
          if (Add->getOpcode() == Instruction::Add) {
            if (auto *CI = dyn_cast<ConstantInt>(Add->getOperand(1))) {
              errs() << "VNNI:   - Step size: " << CI->getSExtValue() << "\n";
            } else if (auto *CI = dyn_cast<ConstantInt>(Add->getOperand(0))) {
              errs() << "VNNI:   - Step size: " << CI->getSExtValue() << "\n";
            }
          }
        }
      }
    }

    // Find trip count
    if (auto *Br = dyn_cast<BranchInst>(Header->getTerminator())) {
      if (Br->isConditional()) {
        if (auto *Cmp = dyn_cast<ICmpInst>(Br->getCondition())) {
          errs() << "VNNI:   - Condition: " << *Cmp << "\n";
        }
      }
    }
  }

  //===--------------------------------------------------------------------===//
  // Pattern Detection
  //===--------------------------------------------------------------------===//
  bool detectVNNIPattern(Loop *L, VNNICandidate &C) {
    BasicBlock *Header = L->getHeader();
    if (!Header || !L->getLoopLatch()) return false;

    // Find accumulator PHI (i32) and induction PHI (i64)
    if (!findLoopPHIs(L, Header, C)) return false;

    // Find the multiply-accumulate pattern
    if (!findMulAccPattern(L, C)) return false;

    // Find memory access pattern (GEPs and loads)
    if (!findMemoryPattern(C)) return false;

    // Extract trip count (must be multiple of 64 for VNNI)
    if (!extractTripCount(L, Header, C)) return false;

    // Extract row stride for future I-tiling optimization
    C.RowStrideA = extractRowStride(C);

    // Check if signed×signed (needs bias correction)
    C.BothSigned = isa<SExtInst>(C.ExtA) && isa<SExtInst>(C.ExtB);

    return true;
  }

  bool findLoopPHIs(Loop *L, BasicBlock *Header, VNNICandidate &C) {
    C.AccPhi = nullptr;
    C.IndPhi = nullptr;
    C.LoadC = nullptr;
    C.StoreC = nullptr;
    C.GEPC = nullptr;
    C.IsLoadStorePattern = false;

    for (PHINode &Phi : Header->phis()) {
      if (Phi.getType()->isIntegerTy(32)) {
        // Look for accumulator pattern: phi feeds into add
        for (unsigned i = 0; i < Phi.getNumIncomingValues(); i++) {
          if (L->contains(Phi.getIncomingBlock(i))) {
            if (auto *Add = dyn_cast<BinaryOperator>(Phi.getIncomingValue(i))) {
              if (Add->getOpcode() == Instruction::Add) {
                C.AccPhi = &Phi;
                C.Add = Add;
                break;
              }
            }
          }
        }
      } else if (Phi.getType()->isIntegerTy(64)) {
        C.IndPhi = &Phi;
      }
    }

    // If we found PHI pattern, we're done
    if (C.AccPhi != nullptr) {
      return true;
    }

    // Otherwise, look for load-store pattern: load C, add, store C
    // Pattern: store(add(load(C), mul(...)), C)
    for (BasicBlock *BB : L->blocks()) {
      for (Instruction &I : *BB) {
        if (auto *SI = dyn_cast<StoreInst>(&I)) {
          // Check if stored value is an add
          if (auto *Add = dyn_cast<BinaryOperator>(SI->getValueOperand())) {
            if (Add->getOpcode() == Instruction::Add) {
              // Check if one operand is a load from same location
              Value *Op0 = Add->getOperand(0);
              Value *Op1 = Add->getOperand(1);
              LoadInst *LI = dyn_cast<LoadInst>(Op0);
              if (!LI) LI = dyn_cast<LoadInst>(Op1);

              if (LI && LI->getType()->isIntegerTy(32)) {
                // Verify load and store access same base (C array)
                auto *LoadGEP = dyn_cast<GetElementPtrInst>(LI->getPointerOperand());
                auto *StoreGEP = dyn_cast<GetElementPtrInst>(SI->getPointerOperand());

                if (LoadGEP && StoreGEP) {
                  // Found load-store pattern!
                  C.Add = Add;
                  C.LoadC = LI;
                  C.StoreC = SI;
                  C.GEPC = StoreGEP;
                  C.IsLoadStorePattern = true;
                  errs() << "VNNI: Found load-store pattern (no PHI accumulator)\n";
                  return true;
                }
              }
            }
          }
        }
      }
    }

    return false;
  }

  bool findMulAccPattern(Loop *L, VNNICandidate &C) {
    Value *AddOp0 = C.Add->getOperand(0);
    Value *AddOp1 = C.Add->getOperand(1);

    Value *MulOrExt = nullptr;
    if (C.IsLoadStorePattern) {
      // For load-store pattern: add(load(C), mul(...))
      // The mul is the operand that isn't the load
      MulOrExt = (AddOp0 == C.LoadC) ? AddOp1 :
                 (AddOp1 == C.LoadC) ? AddOp0 : nullptr;
    } else {
      // For PHI pattern: add(phi, mul(...))
      MulOrExt = (AddOp0 == C.AccPhi) ? AddOp1 :
                 (AddOp1 == C.AccPhi) ? AddOp0 : nullptr;
    }
    if (!MulOrExt) return false;

    // Pattern: sext(i16 mul) -> i32, where mul is from i8 operands
    if (auto *OuterSExt = dyn_cast<SExtInst>(MulOrExt)) {
      if (OuterSExt->getSrcTy()->isIntegerTy(16)) {
        if (auto *I16Mul = dyn_cast<BinaryOperator>(OuterSExt->getOperand(0))) {
          if (I16Mul->getOpcode() == Instruction::Mul) {
            C.Mul = I16Mul;
          }
        }
      }
    }

    // Pattern: direct i32 mul from i8 operands
    if (!C.Mul) {
      C.Mul = dyn_cast<BinaryOperator>(MulOrExt);
      if (!C.Mul || C.Mul->getOpcode() != Instruction::Mul) return false;
    }

    // Find sext/zext from i8
    auto checkExt = [](Value *V) -> Instruction* {
      if (auto *S = dyn_cast<SExtInst>(V)) {
        if (S->getSrcTy()->isIntegerTy(8)) return S;
      }
      if (auto *Z = dyn_cast<ZExtInst>(V)) {
        if (Z->getSrcTy()->isIntegerTy(8)) return Z;
      }
      return nullptr;
    };

    C.ExtA = checkExt(C.Mul->getOperand(0));
    C.ExtB = checkExt(C.Mul->getOperand(1));

    return C.ExtA && C.ExtB;
  }

  bool findMemoryPattern(VNNICandidate &C) {
    C.LoadA = dyn_cast<LoadInst>(C.ExtA->getOperand(0));
    C.LoadB = dyn_cast<LoadInst>(C.ExtB->getOperand(0));
    if (!C.LoadA || !C.LoadB) return false;

    C.GEPA = dyn_cast<GetElementPtrInst>(C.LoadA->getPointerOperand());
    C.GEPB = dyn_cast<GetElementPtrInst>(C.LoadB->getPointerOperand());

    return C.GEPA && C.GEPB;
  }

  bool extractTripCount(Loop *L, BasicBlock *Header, VNNICandidate &C) {
    auto *HeaderBr = dyn_cast<BranchInst>(Header->getTerminator());
    if (!HeaderBr || !HeaderBr->isConditional()) return false;

    auto *Cmp = dyn_cast<ICmpInst>(HeaderBr->getCondition());
    if (!Cmp) return false;

    ConstantInt *TripConst = dyn_cast<ConstantInt>(Cmp->getOperand(1));
    if (!TripConst) TripConst = dyn_cast<ConstantInt>(Cmp->getOperand(0));
    if (!TripConst) return false;

    C.TripCount = TripConst->getSExtValue();

    // Must be multiple of 4 for VNNI (vpdpbusd processes 4 bytes at a time)
    if (C.TripCount % 4 != 0) {
      errs() << "VNNI: Trip count " << C.TripCount << " not multiple of 4\n";
      return false;
    }

    return true;
  }

  int64_t extractRowStride(VNNICandidate &C) {
    // Pattern: index = i * stride + k
    Value *Idx = C.GEPA->getOperand(C.GEPA->getNumOperands() - 1);

    if (auto *Add = dyn_cast<BinaryOperator>(Idx)) {
      if (Add->getOpcode() == Instruction::Add) {
        // Find the mul part (row * stride)
        for (Value *Op : {Add->getOperand(0), Add->getOperand(1)}) {
          if (auto *Mul = dyn_cast<BinaryOperator>(Op)) {
            if (Mul->getOpcode() == Instruction::Mul) {
              // Extract stride constant
              if (auto *CI = dyn_cast<ConstantInt>(Mul->getOperand(1)))
                return CI->getSExtValue();
              if (auto *CI = dyn_cast<ConstantInt>(Mul->getOperand(0)))
                return CI->getSExtValue();
            }
          }
        }
      }
    }
    return 0;
  }

  //===--------------------------------------------------------------------===//
  // Value Tracing (for finding loop-invariant base pointers)
  //===--------------------------------------------------------------------===//
  Value* traceToBase(Value *V, Function &F, Loop *L, LoopInfo &LI, IRBuilder<> &Builder) {
    // Trace through PHI chains to find function-level base
    auto tracePhis = [&](Value *V) -> Value* {
      SmallPtrSet<Value*, 8> Visited;
      while (auto *Phi = dyn_cast<PHINode>(V)) {
        if (Visited.count(V)) break;
        Visited.insert(V);

        if (Phi->getNumIncomingValues() == 1) {
          V = Phi->getIncomingValue(0);
          continue;
        }

        for (unsigned i = 0; i < Phi->getNumIncomingValues(); i++) {
          BasicBlock *BB = Phi->getIncomingBlock(i);
          if (&F.getEntryBlock() == BB || !LI.getLoopFor(BB)) {
            V = Phi->getIncomingValue(i);
            break;
          }
        }
        break;
      }
      return V;
    };

    // Trace insertvalue chain to find the base pointer
    // For malloc: insertvalue chain ends with bitcast of malloc result
    auto traceInsertValue = [&](Value *V, ArrayRef<unsigned> Indices) -> Value* {
      SmallPtrSet<Value*, 16> Visited;
      while (V && !Visited.count(V)) {
        Visited.insert(V);

        if (auto *IV = dyn_cast<InsertValueInst>(V)) {
          // Check if this insertvalue sets the index we want
          if (IV->getIndices() == Indices) {
            return IV->getInsertedValueOperand();
          }
          // Otherwise trace the aggregate operand
          V = IV->getAggregateOperand();
        } else if (isa<UndefValue>(V)) {
          return nullptr;
        } else {
          break;
        }
      }
      return nullptr;
    };

    // Handle MLIR memref descriptors (extractvalue from struct)
    if (auto *EV = dyn_cast<ExtractValueInst>(V)) {
      Value *Agg = EV->getAggregateOperand();

      // First, trace through PHIs to find loop-invariant aggregate
      Value *TracedAgg = tracePhis(Agg);

      // If the aggregate is from an insertvalue chain (malloc case),
      // trace it to find the actual base pointer
      if (auto *IV = dyn_cast<InsertValueInst>(TracedAgg)) {
        Value *BasePtr = traceInsertValue(IV, EV->getIndices());
        if (BasePtr) {
          errs() << "VNNI: Traced insertvalue chain to base: " << *BasePtr << "\n";
          return BasePtr;
        }
      }

      // Original path: create extractvalue from traced aggregate
      if (TracedAgg != Agg) {
        return Builder.CreateExtractValue(TracedAgg, EV->getIndices(), "base.ptr");
      }
    }

    // Trace through all enclosing loops
    Loop *CurLoop = L;
    while (CurLoop) {
      if (auto *Inst = dyn_cast<Instruction>(V)) {
        if (CurLoop->contains(Inst->getParent())) {
          V = tracePhis(V);
        }
      }
      CurLoop = CurLoop->getParentLoop();
    }

    return V;
  }

  Value* computeRowOffset(Value *Idx, VNNICandidate &C, IRBuilder<> &Builder, Loop *L) {
    if (auto *Add = dyn_cast<BinaryOperator>(Idx)) {
      if (Add->getOpcode() == Instruction::Add) {
        Value *Op0 = Add->getOperand(0);
        Value *Op1 = Add->getOperand(1);
        Value *RowPart = (Op0 == C.IndPhi) ? Op1 : Op0;

        if (auto *Mul = dyn_cast<BinaryOperator>(RowPart)) {
          if (Mul->getOpcode() == Instruction::Mul) {
            Value *OuterIdx = isa<ConstantInt>(Mul->getOperand(1)) ?
                              Mul->getOperand(0) : Mul->getOperand(1);
            Value *Stride = isa<ConstantInt>(Mul->getOperand(1)) ?
                            Mul->getOperand(1) : Mul->getOperand(0);

            // Trace through LCSSA PHIs
            if (auto *Phi = dyn_cast<PHINode>(OuterIdx)) {
              if (L->contains(Phi->getParent()) && Phi->getNumIncomingValues() == 1) {
                OuterIdx = Phi->getIncomingValue(0);
              }
            }
            return Builder.CreateMul(OuterIdx, Stride, "row.off");
          }
        }
        return RowPart;
      }
    }
    return Idx;
  }

  //===--------------------------------------------------------------------===//
  // Loop Transformation
  //===--------------------------------------------------------------------===//
  bool transformToVNNI(Function &F, Loop *L, VNNICandidate &C, LoopInfo &LI) {
    Module *M = F.getParent();
    LLVMContext &Ctx = M->getContext();

    BasicBlock *Preheader = L->getLoopPreheader();
    BasicBlock *Header = L->getHeader();
    BasicBlock *Exit = L->getExitBlock();

    if (!Preheader || !Exit) return false;

    // Get vpdpbusd intrinsic
    Function *VPDPBUSD = Intrinsic::getDeclaration(M, Intrinsic::x86_avx512_vpdpbusd_512);
    if (!VPDPBUSD) return false;

    // Types
    Type *I8Ty = Type::getInt8Ty(Ctx);
    Type *I32Ty = Type::getInt32Ty(Ctx);
    Type *I64Ty = Type::getInt64Ty(Ctx);
    auto *V16I32Ty = FixedVectorType::get(I32Ty, 16);
    Type *V16I32PtrTy = PointerType::get(V16I32Ty, 0);

    // For small K (< 64), we need to handle partial vectors
    // Calculate how many bytes to load per iteration
    int64_t BytesPerIter = std::min((int64_t)64, C.TripCount);
    // Number of valid i32 elements (each holds 4 bytes for vpdpbusd)
    int NumValidElements = BytesPerIter / 4;

    IRBuilder<> Builder(Ctx);
    Builder.SetInsertPoint(Preheader->getTerminator());

    // Trace base pointers
    Value *BaseA = traceToBase(C.GEPA->getPointerOperand(), F, L, LI, Builder);
    Value *BaseB = traceToBase(C.GEPB->getPointerOperand(), F, L, LI, Builder);

    // Compute row offsets
    Value *IdxA = C.GEPA->getOperand(C.GEPA->getNumOperands() - 1);
    Value *IdxB = C.GEPB->getOperand(C.GEPB->getNumOperands() - 1);
    Value *RowOffA = computeRowOffset(IdxA, C, Builder, L);
    Value *RowOffB = computeRowOffset(IdxB, C, Builder, L);

    // Create VNNI loop blocks
    BasicBlock *VecHeader = BasicBlock::Create(Ctx, "vnni.hdr", &F);
    BasicBlock *VecBody = BasicBlock::Create(Ctx, "vnni.body", &F);
    BasicBlock *VecExit = BasicBlock::Create(Ctx, "vnni.exit", &F);

    Value *ZeroVec = ConstantVector::getSplat(ElementCount::getFixed(16),
                                               ConstantInt::get(I32Ty, 0));
    Value *VecEnd = ConstantInt::get(I64Ty, C.TripCount);

    // === I=4 Tiling: Compute row offsets for 4 A rows ===
    Value *Stride = ConstantInt::get(I64Ty, C.RowStrideA);
    Value *RowOffA0 = RowOffA;  // i*stride
    Value *RowOffA1 = Builder.CreateAdd(RowOffA, Stride);  // (i+1)*stride
    Value *RowOffA2 = Builder.CreateAdd(RowOffA1, Stride); // (i+2)*stride
    Value *RowOffA3 = Builder.CreateAdd(RowOffA2, Stride); // (i+3)*stride

    // === VNNI Header: PHIs and condition ===
    Builder.SetInsertPoint(VecHeader);
    PHINode *VecIdx = Builder.CreatePHI(I64Ty, 2, "k");

    // 4 accumulators for I=4 tiling
    PHINode *VecAcc0 = Builder.CreatePHI(V16I32Ty, 2, "acc0");
    PHINode *VecAcc1 = Builder.CreatePHI(V16I32Ty, 2, "acc1");
    PHINode *VecAcc2 = Builder.CreatePHI(V16I32Ty, 2, "acc2");
    PHINode *VecAcc3 = Builder.CreatePHI(V16I32Ty, 2, "acc3");
    PHINode *BiasAcc = C.BothSigned ? Builder.CreatePHI(V16I32Ty, 2, "bias") : nullptr;

    Value *Cond = Builder.CreateICmpSLT(VecIdx, VecEnd);
    Builder.CreateCondBr(Cond, VecBody, VecExit);

    // === VNNI Body: load B once, load 4 A rows, compute 4 dot products ===
    Builder.SetInsertPoint(VecBody);

    // For K < 64, we pad to 64 by loading available data + zeros
    // This works because 0 * x = 0 in dot product
    // We always load 64 bytes but use masked load for safety when K < 64

    // Helper: create padded load - loads K bytes and pads rest with zeros
    auto loadPadded = [&](Value *BasePtr, Value *Offset) -> Value* {
      Value *Ptr = Builder.CreateGEP(I8Ty, BasePtr, Offset);

      if (C.TripCount >= 64) {
        // Full 64-byte load, no padding needed
        return Builder.CreateAlignedLoad(V16I32Ty,
                 Builder.CreateBitCast(Ptr, V16I32PtrTy), Align(1));
      }

      // K < 64: Load K bytes into smaller vector, then extend with zeros
      // NumValidElements = K / 4 (number of i32 elements)
      auto *SmallVecTy = FixedVectorType::get(I32Ty, NumValidElements);
      Type *SmallVecPtrTy = PointerType::get(SmallVecTy, 0);
      Value *SmallVec = Builder.CreateAlignedLoad(SmallVecTy,
                          Builder.CreateBitCast(Ptr, SmallVecPtrTy), Align(1));

      // Extend to 16 elements: [v0, v1, ..., vN, 0, 0, ..., 0]
      // First pad SmallVec to 16 elements with undef, then shuffle with zeros
      Value *SmallZero = ConstantVector::getSplat(
                           ElementCount::getFixed(NumValidElements),
                           ConstantInt::get(I32Ty, 0));

      // Shuffle: indices 0..NumValidElements-1 from SmallVec, rest from SmallZero (as 0)
      // Since both operands are same size, we can shuffle properly
      SmallVector<int, 16> PadMask;
      for (int i = 0; i < NumValidElements * 2; i++) {
        PadMask.push_back(i < NumValidElements ? i : NumValidElements); // second half = zeros
      }
      Value *PaddedSmall = Builder.CreateShuffleVector(SmallVec, SmallZero, PadMask);

      // Now extend PaddedSmall (2*NumValidElements) to 16 elements
      // If NumValidElements=8 (K=32), PaddedSmall is 16 elements, we're done
      if (NumValidElements * 2 == 16) {
        return PaddedSmall;
      }

      // Otherwise need another shuffle to get to 16
      Value *FullZero = ConstantVector::getSplat(
                          ElementCount::getFixed(NumValidElements * 2),
                          ConstantInt::get(I32Ty, 0));
      SmallVector<int, 16> ExtendMask;
      for (int i = 0; i < 16; i++) {
        ExtendMask.push_back(i < NumValidElements * 2 ? i : NumValidElements * 2);
      }
      return Builder.CreateShuffleVector(PaddedSmall, FullZero, ExtendMask);
    };

    // Load B vector ONCE (shared across all 4 A rows)
    Value *VecB = loadPadded(BaseB, Builder.CreateAdd(RowOffB, VecIdx));

    // Load 4 A rows
    Value *VecA0 = loadPadded(BaseA, Builder.CreateAdd(RowOffA0, VecIdx));
    Value *VecA1 = loadPadded(BaseA, Builder.CreateAdd(RowOffA1, VecIdx));
    Value *VecA2 = loadPadded(BaseA, Builder.CreateAdd(RowOffA2, VecIdx));
    Value *VecA3 = loadPadded(BaseA, Builder.CreateAdd(RowOffA3, VecIdx));

    // Signed conversion and bias accumulation
    Value *NewBiasAcc = nullptr;
    if (C.BothSigned) {
      Value *SignFlip = ConstantVector::getSplat(ElementCount::getFixed(16),
                          ConstantInt::get(I32Ty, 0x80808080));
      VecA0 = Builder.CreateXor(VecA0, SignFlip);
      VecA1 = Builder.CreateXor(VecA1, SignFlip);
      VecA2 = Builder.CreateXor(VecA2, SignFlip);
      VecA3 = Builder.CreateXor(VecA3, SignFlip);

      Value *Ones = ConstantVector::getSplat(ElementCount::getFixed(16),
                      ConstantInt::get(I32Ty, 0x01010101));
      NewBiasAcc = Builder.CreateCall(VPDPBUSD, {BiasAcc, Ones, VecB});
    }

    // vpdpbusd: 4 dot products, reusing B
    Value *NewAcc0 = Builder.CreateCall(VPDPBUSD, {VecAcc0, VecA0, VecB});
    Value *NewAcc1 = Builder.CreateCall(VPDPBUSD, {VecAcc1, VecA1, VecB});
    Value *NewAcc2 = Builder.CreateCall(VPDPBUSD, {VecAcc2, VecA2, VecB});
    Value *NewAcc3 = Builder.CreateCall(VPDPBUSD, {VecAcc3, VecA3, VecB});

    // Step by 64 bytes or TripCount if smaller (for K < 64 cases)
    int64_t StepSize = std::min((int64_t)64, C.TripCount);
    Value *NextIdx = Builder.CreateAdd(VecIdx, ConstantInt::get(I64Ty, StepSize));
    Builder.CreateBr(VecHeader);

    // Complete PHIs
    VecIdx->addIncoming(ConstantInt::get(I64Ty, 0), Preheader);
    VecIdx->addIncoming(NextIdx, VecBody);
    VecAcc0->addIncoming(ZeroVec, Preheader);
    VecAcc0->addIncoming(NewAcc0, VecBody);
    VecAcc1->addIncoming(ZeroVec, Preheader);
    VecAcc1->addIncoming(NewAcc1, VecBody);
    VecAcc2->addIncoming(ZeroVec, Preheader);
    VecAcc2->addIncoming(NewAcc2, VecBody);
    VecAcc3->addIncoming(ZeroVec, Preheader);
    VecAcc3->addIncoming(NewAcc3, VecBody);
    if (C.BothSigned) {
      BiasAcc->addIncoming(ZeroVec, Preheader);
      BiasAcc->addIncoming(NewBiasAcc, VecBody);
    }

    // === VNNI Exit: horizontal reduction ===
    Builder.SetInsertPoint(VecExit);

    auto hreduce = [&](Value *Vec) -> Value* {
      for (int W = 8; W >= 1; W /= 2) {
        SmallVector<int, 16> Mask;
        for (int i = 0; i < 16; i++) Mask.push_back((i + W) % 16);
        Vec = Builder.CreateAdd(Vec, Builder.CreateShuffleVector(Vec, Vec, Mask));
      }
      return Builder.CreateExtractElement(Vec, (uint64_t)0);
    };

    // Reduce all 4 accumulators
    Value *Result0 = hreduce(VecAcc0);
    Value *Result1 = hreduce(VecAcc1);
    Value *Result2 = hreduce(VecAcc2);
    Value *Result3 = hreduce(VecAcc3);

    if (C.BothSigned) {
      Value *BiasSum = hreduce(BiasAcc);
      Value *Correction = Builder.CreateMul(BiasSum, ConstantInt::get(I32Ty, 128));
      Result0 = Builder.CreateSub(Result0, Correction);
      Result1 = Builder.CreateSub(Result1, Correction);
      Result2 = Builder.CreateSub(Result2, Correction);
      Result3 = Builder.CreateSub(Result3, Correction);
    }

    // === I=4 Tiling: Store all 4 results ===
    // Find the store instruction that uses the exit PHI
    // K loop -> J loop (parent) -> I loop (grandparent)
    Loop *JLoop = L->getParentLoop();
    Loop *ILoop = JLoop ? JLoop->getParentLoop() : nullptr;
    StoreInst *OrigStore = nullptr;
    GetElementPtrInst *StoreGEP = nullptr;

    // Find store in exit block or parent loop
    for (PHINode &PN : Exit->phis()) {
      for (User *U : PN.users()) {
        if (auto *SI = dyn_cast<StoreInst>(U)) {
          OrigStore = SI;
          StoreGEP = dyn_cast<GetElementPtrInst>(SI->getPointerOperand());
          break;
        }
      }
      if (OrigStore) break;
    }

    Builder.CreateBr(Exit);

    // === Rewire control flow ===
    if (auto *Br = dyn_cast<BranchInst>(Preheader->getTerminator())) {
      Br->setSuccessor(0, VecHeader);
    }

    // Update exit PHIs - Result0 goes to original PHI
    for (PHINode &PN : Exit->phis()) {
      for (unsigned i = 0; i < PN.getNumIncomingValues(); i++) {
        if (PN.getIncomingBlock(i) == Header) {
          PN.setIncomingBlock(i, VecExit);
          if (PN.getIncomingValue(i) == C.AccPhi) {
            PN.setIncomingValue(i, Result0);
          }
        }
      }
    }

    // === Generate stores for Result1, Result2, Result3 ===
    if (OrigStore && StoreGEP && ILoop) {
      // Insert stores right after the original store
      Builder.SetInsertPoint(OrigStore->getNextNode());

      Value *BaseC = StoreGEP->getPointerOperand();
      Value *OrigIdx = StoreGEP->getOperand(StoreGEP->getNumOperands() - 1);

      // C is stored as C[i * stride + j], so consecutive rows are stride apart
      // Use the same stride as matrix A (RowStrideA = N)
      Value *CStride = ConstantInt::get(I64Ty, C.RowStrideA);

      // Store Result1 at index + stride (next row)
      Value *Idx1 = Builder.CreateAdd(OrigIdx, CStride);
      Value *Ptr1 = Builder.CreateGEP(I32Ty, BaseC, Idx1);
      Builder.CreateStore(Result1, Ptr1);

      // Store Result2 at index + 2*stride
      Value *Idx2 = Builder.CreateAdd(Idx1, CStride);
      Value *Ptr2 = Builder.CreateGEP(I32Ty, BaseC, Idx2);
      Builder.CreateStore(Result2, Ptr2);

      // Store Result3 at index + 3*stride
      Value *Idx3 = Builder.CreateAdd(Idx2, CStride);
      Value *Ptr3 = Builder.CreateGEP(I32Ty, BaseC, Idx3);
      Builder.CreateStore(Result3, Ptr3);

      errs() << "VNNI: Added 3 extra stores for I=4 tiling (stride=" << C.RowStrideA << ")\n";

      // Modify I loop (grandparent) to step by 4 - NOT J loop!
      if (modifyParentLoopStep(ILoop, 4)) {
        errs() << "VNNI: Modified I loop (grandparent) step to 4\n";
      }
    } else {
      errs() << "VNNI: Could not find store pattern or I loop for I=4 tiling\n";
      if (!ILoop) errs() << "VNNI:   - No I loop (grandparent) found\n";
    }

    // === Delete old loop ===
    deleteLoop(L, LI, C);

    errs() << "VNNI: Transformed loop (trip=" << C.TripCount << ")\n";
    return true;
  }

  void deleteLoop(Loop *L, LoopInfo &LI, VNNICandidate &C) {
    SmallVector<BasicBlock*, 4> Blocks(L->blocks().begin(), L->blocks().end());

    // Remove from LoopInfo
    if (Loop *Parent = L->getParentLoop()) {
      Parent->removeChildLoop(std::find(Parent->begin(), Parent->end(), L));
    } else {
      LI.removeLoop(std::find(LI.begin(), LI.end(), L));
    }

    // Replace external uses
    for (Use &U : C.AccPhi->uses()) {
      if (auto *I = dyn_cast<Instruction>(U.getUser())) {
        if (!L->contains(I->getParent())) {
          // Already handled by exit PHI update
        }
      }
    }

    // Replace internal uses with undef and delete
    for (BasicBlock *BB : Blocks) {
      for (Instruction &I : *BB) {
        if (!I.use_empty()) {
          I.replaceAllUsesWith(UndefValue::get(I.getType()));
        }
      }
    }
    for (BasicBlock *BB : Blocks) {
      BB->getTerminator()->eraseFromParent();
    }
    for (BasicBlock *BB : Blocks) {
      while (!BB->empty()) BB->back().eraseFromParent();
    }
    for (BasicBlock *BB : Blocks) {
      BB->eraseFromParent();
    }
  }
};

} // anonymous namespace

char VNNIPass::ID = 0;
static RegisterPass<VNNIPass> X("vnni", "VNNI vpdpbusd optimization pass", false, false);

namespace llvm {
  Pass *createVNNIPass() { return new VNNIPass(); }
}
