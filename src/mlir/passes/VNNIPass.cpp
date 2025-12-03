//===- VNNIPass.cpp - LLVM Pass to emit vpdpbusd for INT8 dot products ----===//
//
// This pass identifies reduction loops with u8*i8->i32 pattern and replaces
// them with AVX512-VNNI vpdpbusd instructions.
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

struct VNNICandidate {
  PHINode *AccPhi;
  PHINode *IndPhi;
  BinaryOperator *Add;
  BinaryOperator *Mul;        // The i16 or i32 mul
  Instruction *ExtA;          // sext/zext from i8
  Instruction *ExtB;          // sext/zext from i8
  LoadInst *LoadA;
  LoadInst *LoadB;
  GetElementPtrInst *GEPA;
  GetElementPtrInst *GEPB;
  int64_t TripCount;
  bool ViaI16;                // true if pattern goes i8->i16->mul->i32
};

struct VNNIPass : public FunctionPass {
  static char ID;
  VNNIPass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    bool Changed = false;
    auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

    SmallVector<Loop*, 8> InnermostLoops;
    for (Loop *L : LI) {
      collectInnermostLoops(L, InnermostLoops);
    }

    for (Loop *L : InnermostLoops) {
      VNNICandidate Candidate;
      if (findVNNIPattern(L, Candidate)) {
        errs() << "VNNI Pass: Found VNNI pattern in loop!\n";
        if (transformLoop(F, L, Candidate, LI)) {
          Changed = true;
        }
      }
    }
    return Changed;
  }

  void collectInnermostLoops(Loop *L, SmallVectorImpl<Loop*> &Result) {
    if (L->getSubLoops().empty()) {
      Result.push_back(L);
    } else {
      for (Loop *SubL : L->getSubLoops()) {
        collectInnermostLoops(SubL, Result);
      }
    }
  }

  bool findVNNIPattern(Loop *L, VNNICandidate &Candidate) {
    BasicBlock *Header = L->getHeader();
    BasicBlock *Latch = L->getLoopLatch();
    if (!Header || !Latch) return false;

    PHINode *AccPhi = nullptr;
    PHINode *IndPhi = nullptr;

    for (PHINode &Phi : Header->phis()) {
      if (Phi.getType()->isIntegerTy(32)) {
        for (unsigned i = 0; i < Phi.getNumIncomingValues(); i++) {
          if (L->contains(Phi.getIncomingBlock(i))) {
            if (auto *Add = dyn_cast<BinaryOperator>(Phi.getIncomingValue(i))) {
              if (Add->getOpcode() == Instruction::Add) {
                AccPhi = &Phi;
                Candidate.Add = Add;
                break;
              }
            }
          }
        }
      } else if (Phi.getType()->isIntegerTy(64)) {
        IndPhi = &Phi;
      }
    }

    if (!AccPhi) return false;
    Candidate.AccPhi = AccPhi;
    Candidate.IndPhi = IndPhi;

    Value *AddOp0 = Candidate.Add->getOperand(0);
    Value *AddOp1 = Candidate.Add->getOperand(1);
    Value *MulOrExt = nullptr;

    if (AddOp0 == AccPhi) MulOrExt = AddOp1;
    else if (AddOp1 == AccPhi) MulOrExt = AddOp0;
    if (!MulOrExt) return false;

    // Pattern 1: Direct i32 mul (i8->i32 extensions)
    // Pattern 2: i8->i16->mul i16->sext i32 (what MLIR generates)
    BinaryOperator *Mul = nullptr;
    Instruction *ExtA = nullptr;
    Instruction *ExtB = nullptr;
    Candidate.ViaI16 = false;

    // Check for Pattern 2: sext i16 to i32 wrapping a mul i16
    if (auto *OuterSExt = dyn_cast<SExtInst>(MulOrExt)) {
      if (OuterSExt->getSrcTy()->isIntegerTy(16) && OuterSExt->getDestTy()->isIntegerTy(32)) {
        if (auto *I16Mul = dyn_cast<BinaryOperator>(OuterSExt->getOperand(0))) {
          if (I16Mul->getOpcode() == Instruction::Mul && I16Mul->getType()->isIntegerTy(16)) {
            Mul = I16Mul;
            Candidate.ViaI16 = true;
            errs() << "VNNI Pass: Found i8->i16->mul->i32 pattern\n";
          }
        }
      }
    }

    // Check for Pattern 1: direct i32 mul
    if (!Mul) {
      Mul = dyn_cast<BinaryOperator>(MulOrExt);
      if (!Mul || Mul->getOpcode() != Instruction::Mul) return false;
      if (!Mul->getType()->isIntegerTy(32)) return false;
    }

    Candidate.Mul = Mul;
    Value *MulOp0 = Mul->getOperand(0);
    Value *MulOp1 = Mul->getOperand(1);

    // Look for sext/zext from i8 to i16 or i32
    auto checkExt = [&](Value *V) -> Instruction* {
      if (auto *S = dyn_cast<SExtInst>(V)) {
        if (S->getSrcTy()->isIntegerTy(8)) return S;
      }
      if (auto *Z = dyn_cast<ZExtInst>(V)) {
        if (Z->getSrcTy()->isIntegerTy(8)) return Z;
      }
      return nullptr;
    };

    ExtA = checkExt(MulOp0);
    ExtB = checkExt(MulOp1);

    if (!ExtA || !ExtB) {
      errs() << "VNNI Pass: Extensions not found from i8\n";
      return false;
    }

    Candidate.ExtA = ExtA;
    Candidate.ExtB = ExtB;

    Candidate.LoadA = dyn_cast<LoadInst>(ExtA->getOperand(0));
    Candidate.LoadB = dyn_cast<LoadInst>(ExtB->getOperand(0));
    if (!Candidate.LoadA || !Candidate.LoadB) return false;

    Candidate.GEPA = dyn_cast<GetElementPtrInst>(Candidate.LoadA->getPointerOperand());
    Candidate.GEPB = dyn_cast<GetElementPtrInst>(Candidate.LoadB->getPointerOperand());
    if (!Candidate.GEPA || !Candidate.GEPB) return false;

    BranchInst *HeaderBr = dyn_cast<BranchInst>(Header->getTerminator());
    if (!HeaderBr || !HeaderBr->isConditional()) return false;

    ICmpInst *Cmp = dyn_cast<ICmpInst>(HeaderBr->getCondition());
    if (!Cmp) return false;

    ConstantInt *TripCountConst = nullptr;
    if (auto *CI = dyn_cast<ConstantInt>(Cmp->getOperand(1))) {
      TripCountConst = CI;
    } else if (auto *CI = dyn_cast<ConstantInt>(Cmp->getOperand(0))) {
      TripCountConst = CI;
    }

    if (!TripCountConst) return false;

    int64_t TC = TripCountConst->getSExtValue();
    if (TC % 64 != 0) {
      errs() << "VNNI Pass: Trip count " << TC << " not multiple of 64\n";
      return false;
    }

    Candidate.TripCount = TC;
    errs() << "VNNI Pass: Pattern matched! Trip count = " << TC << "\n";
    return true;
  }

  bool transformLoop(Function &F, Loop *L, VNNICandidate &Candidate, LoopInfo &LI) {
    Module *M = F.getParent();
    LLVMContext &Ctx = M->getContext();

    BasicBlock *Preheader = L->getLoopPreheader();
    BasicBlock *Header = L->getHeader();
    BasicBlock *Exit = L->getExitBlock();

    if (!Preheader || !Exit) {
      errs() << "VNNI Pass: Missing preheader or exit\n";
      return false;
    }

    Function *VPDPBUSDFn = Intrinsic::getDeclaration(M, Intrinsic::x86_avx512_vpdpbusd_512);
    if (!VPDPBUSDFn) {
      errs() << "VNNI Pass: vpdpbusd intrinsic not available\n";
      return false;
    }

    Type *I8Ty = Type::getInt8Ty(Ctx);
    Type *I32Ty = Type::getInt32Ty(Ctx);
    Type *I64Ty = Type::getInt64Ty(Ctx);
    auto *V16I32Ty = FixedVectorType::get(I32Ty, 16);
    Type *V16I32PtrTy = PointerType::get(V16I32Ty, 0);

    errs() << "VNNI Pass: Transforming loop with trip count " << Candidate.TripCount << "\n";

    // Extract row offset from the GEP index
    // The GEP index is typically (row_offset + k) where k is the loop induction variable
    GetElementPtrInst *GEPA = Candidate.GEPA;
    GetElementPtrInst *GEPB = Candidate.GEPB;
    Value *IdxA = GEPA->getOperand(GEPA->getNumOperands() - 1);
    Value *IdxB = GEPB->getOperand(GEPB->getNumOperands() - 1);

    Value *RowOffsetA = nullptr;
    Value *RowOffsetB = nullptr;

    // Find the row offset (the non-induction-variable part of the index)
    if (auto *AddA = dyn_cast<BinaryOperator>(IdxA)) {
      if (AddA->getOpcode() == Instruction::Add) {
        Value *Op0 = AddA->getOperand(0);
        Value *Op1 = AddA->getOperand(1);
        // Check which operand is the induction variable
        if (Op0 == Candidate.IndPhi || (isa<PHINode>(Op0) && L->contains(cast<PHINode>(Op0)->getParent()))) {
          RowOffsetA = Op1;
        } else {
          RowOffsetA = Op0;
        }
      }
    }

    if (auto *AddB = dyn_cast<BinaryOperator>(IdxB)) {
      if (AddB->getOpcode() == Instruction::Add) {
        Value *Op0 = AddB->getOperand(0);
        Value *Op1 = AddB->getOperand(1);
        if (Op0 == Candidate.IndPhi || (isa<PHINode>(Op0) && L->contains(cast<PHINode>(Op0)->getParent()))) {
          RowOffsetB = Op1;
        } else {
          RowOffsetB = Op0;
        }
      }
    }

    if (!RowOffsetA || !RowOffsetB) {
      errs() << "VNNI Pass: Could not extract row offsets\n";
      return false;
    }

    Value *BaseA = GEPA->getPointerOperand();
    Value *BaseB = GEPB->getPointerOperand();

    IRBuilder<> Builder(Ctx);
    Builder.SetInsertPoint(Preheader->getTerminator());

    // Helper to trace through PHI chains to find the ultimate source value
    std::function<Value*(Value*)> tracePhiChain = [&](Value *V) -> Value* {
      SmallPtrSet<Value*, 8> Visited;
      while (auto *Phi = dyn_cast<PHINode>(V)) {
        if (Visited.count(V)) break;  // Avoid infinite loops
        Visited.insert(V);

        // For single-incoming PHIs (LCSSA style), just follow through
        if (Phi->getNumIncomingValues() == 1) {
          V = Phi->getIncomingValue(0);
          continue;
        }

        // For multi-incoming PHIs, look for value from entry/preheader
        bool found = false;
        for (unsigned i = 0; i < Phi->getNumIncomingValues(); i++) {
          BasicBlock *InBB = Phi->getIncomingBlock(i);
          // If incoming from entry block or outside all loops
          if (&F.getEntryBlock() == InBB || !LI.getLoopFor(InBB)) {
            V = Phi->getIncomingValue(i);
            found = true;
            break;
          }
        }
        if (!found) break;
      }
      return V;
    };

    // Helper to trace back through PHIs to find a value that dominates preheader
    // For MLIR memref descriptors, we need to extract the raw pointer
    std::function<Value*(Value*, Loop*)> traceToLoopInvariant = [&](Value *V, Loop *L) -> Value* {
      // For extractvalue - trace the aggregate and rebuild
      if (auto *EV = dyn_cast<ExtractValueInst>(V)) {
        Value *Agg = EV->getAggregateOperand();
        Value *TracedAgg = tracePhiChain(Agg);

        errs() << "VNNI Pass: ExtractValue aggregate traced: " << *Agg << " -> " << *TracedAgg << "\n";

        if (TracedAgg != Agg) {
          // Create new extractvalue in preheader
          return Builder.CreateExtractValue(TracedAgg, EV->getIndices(), "traced.base");
        }
        return V;
      }

      // Already loop invariant
      if (auto *Inst = dyn_cast<Instruction>(V)) {
        if (!L->contains(Inst->getParent())) return V;
      } else {
        return V;  // Constants, arguments, etc.
      }

      // Trace through PHIs
      Value *Traced = tracePhiChain(V);
      if (Traced != V) return Traced;

      return V;
    };

    // Helper to trace through all enclosing loops
    auto traceToFunctionLevel = [&](Value *V) -> Value* {
      Loop *CurLoop = L;
      Value *Result = V;
      while (CurLoop) {
        Result = traceToLoopInvariant(Result, CurLoop);
        CurLoop = CurLoop->getParentLoop();
      }
      return Result;
    };

    errs() << "VNNI Pass: Original BaseA: " << *BaseA << "\n";
    errs() << "VNNI Pass: Original BaseB: " << *BaseB << "\n";

    // Trace base pointers back to function-level values
    BaseA = traceToFunctionLevel(BaseA);
    BaseB = traceToFunctionLevel(BaseB);

    errs() << "VNNI Pass: Traced BaseA: " << *BaseA << "\n";
    errs() << "VNNI Pass: Traced BaseB: " << *BaseB << "\n";

    // For row offsets - we need the outer loop indices multiplied by stride
    // These come from the GEP index computation: row * stride + k
    // The row offset is "row * stride" part
    // IMPORTANT: Don't trace outer loop indices to entry - keep them as PHIs!
    auto traceRowOffset = [&](Value *V) -> Value* {
      // If it's an add, one operand is row*stride, other is k (induction var)
      if (auto *Add = dyn_cast<BinaryOperator>(V)) {
        if (Add->getOpcode() == Instruction::Add) {
          Value *Op0 = Add->getOperand(0);
          Value *Op1 = Add->getOperand(1);

          // Find which operand is NOT the innermost induction variable
          bool Op0IsInd = (Op0 == Candidate.IndPhi);

          Value *RowPart = Op0IsInd ? Op1 : Op0;

          // Trace the row part - it's typically a mul from outer loop
          if (auto *Mul = dyn_cast<BinaryOperator>(RowPart)) {
            if (Mul->getOpcode() == Instruction::Mul) {
              // Get operands - one is stride (constant), other is outer loop index
              Value *MulOp0 = Mul->getOperand(0);
              Value *MulOp1 = Mul->getOperand(1);

              Value *OuterIdx = isa<ConstantInt>(MulOp1) ? MulOp0 : MulOp1;
              Value *Stride = isa<ConstantInt>(MulOp1) ? MulOp1 : MulOp0;

              // DON'T trace outer loop index - we need its current value!
              // The mul instruction itself is in the loop body, we need to
              // recreate it in preheader using the outer loop's PHI value
              // that's available at preheader entry

              // Check if OuterIdx is defined in our innermost loop
              if (auto *OuterInst = dyn_cast<Instruction>(OuterIdx)) {
                if (L->contains(OuterInst->getParent())) {
                  // It's inside our loop - trace through LCSSA PHIs only
                  if (auto *Phi = dyn_cast<PHINode>(OuterIdx)) {
                    // Single-entry LCSSA phi - get incoming value
                    if (Phi->getNumIncomingValues() == 1) {
                      OuterIdx = Phi->getIncomingValue(0);
                    }
                  }
                }
              }

              return Builder.CreateMul(OuterIdx, Stride, "row.offset");
            }
          }
          // If RowPart is already a value available at preheader, use it
          return RowPart;
        }
      }
      return V;
    };

    RowOffsetA = traceRowOffset(IdxA);
    RowOffsetB = traceRowOffset(IdxB);

    errs() << "VNNI Pass: RowOffsetA = " << *RowOffsetA << "\n";
    errs() << "VNNI Pass: RowOffsetB = " << *RowOffsetB << "\n";

    // Create new vectorized loop blocks
    BasicBlock *VecHeader = BasicBlock::Create(Ctx, "vnni.header", &F);
    BasicBlock *VecBody = BasicBlock::Create(Ctx, "vnni.body", &F);
    BasicBlock *VecExit = BasicBlock::Create(Ctx, "vnni.exit", &F);

    // Check if both extensions are signed (need bias correction)
    bool BothSigned = isa<SExtInst>(Candidate.ExtA) && isa<SExtInst>(Candidate.ExtB);

    Value *ZeroVec = ConstantVector::getSplat(ElementCount::getFixed(16), ConstantInt::get(I32Ty, 0));
    Value *VecEnd = ConstantInt::get(I64Ty, Candidate.TripCount);

    // VecHeader: PHIs and loop condition
    Builder.SetInsertPoint(VecHeader);
    PHINode *VecIdx = Builder.CreatePHI(I64Ty, 2, "vnni.idx");
    PHINode *VecAcc = Builder.CreatePHI(V16I32Ty, 2, "vnni.acc");

    // For signed×signed: accumulate sum(B) in the SAME loop (no extra memory reads!)
    PHINode *BiasSumAcc = nullptr;
    if (BothSigned) {
      BiasSumAcc = Builder.CreatePHI(V16I32Ty, 2, "bias.acc");
    }

    Value *VecCond = Builder.CreateICmpSLT(VecIdx, VecEnd, "vnni.cond");
    Builder.CreateCondBr(VecCond, VecBody, VecExit);

    // VecBody: compute new index, load, vpdpbusd, increment
    Builder.SetInsertPoint(VecBody);

    // Compute address: base + row_offset + vec_idx
    Value *IdxANew = Builder.CreateAdd(RowOffsetA, VecIdx, "idxA");
    Value *IdxBNew = Builder.CreateAdd(RowOffsetB, VecIdx, "idxB");
    Value *PtrA = Builder.CreateGEP(I8Ty, BaseA, IdxANew, "ptrA");
    Value *PtrB = Builder.CreateGEP(I8Ty, BaseB, IdxBNew, "ptrB");

    Value *VecPtrA = Builder.CreateBitCast(PtrA, V16I32PtrTy);
    Value *VecPtrB = Builder.CreateBitCast(PtrB, V16I32PtrTy);
    Value *VecA = Builder.CreateAlignedLoad(V16I32Ty, VecPtrA, Align(1), "vecA");
    Value *VecB = Builder.CreateAlignedLoad(V16I32Ty, VecPtrB, Align(1), "vecB");

    // For signed×signed: XOR A with 0x80808080 to convert to unsigned
    // Also accumulate sum(B) using vpdpbusd with ones vector (reuses VecB we already loaded!)
    Value *NewBiasSumAcc = nullptr;
    if (BothSigned) {
      Value *SignFlip = ConstantVector::getSplat(ElementCount::getFixed(16),
          ConstantInt::get(I32Ty, 0x80808080));
      VecA = Builder.CreateXor(VecA, SignFlip, "vecA.unsigned");

      // Accumulate sum(B) - reuses VecB, no extra load!
      Value *OnesVec = ConstantVector::getSplat(ElementCount::getFixed(16),
          ConstantInt::get(I32Ty, 0x01010101));
      NewBiasSumAcc = Builder.CreateCall(VPDPBUSDFn, {BiasSumAcc, OnesVec, VecB}, "bias.sum");
    }

    Value *NewAcc = Builder.CreateCall(VPDPBUSDFn, {VecAcc, VecA, VecB}, "vpdpbusd");

    Value *NextIdx = Builder.CreateAdd(VecIdx, ConstantInt::get(I64Ty, 64), "next.idx");
    Builder.CreateBr(VecHeader);

    // Complete PHI nodes
    VecIdx->addIncoming(ConstantInt::get(I64Ty, 0), Preheader);
    VecIdx->addIncoming(NextIdx, VecBody);
    VecAcc->addIncoming(ZeroVec, Preheader);
    VecAcc->addIncoming(NewAcc, VecBody);
    if (BothSigned) {
      BiasSumAcc->addIncoming(ZeroVec, Preheader);
      BiasSumAcc->addIncoming(NewBiasSumAcc, VecBody);
    }

    // VecExit: horizontal reduction
    Builder.SetInsertPoint(VecExit);

    Value *Sum = VecAcc;
    for (int Width = 8; Width >= 1; Width /= 2) {
      SmallVector<int, 16> Mask;
      for (int i = 0; i < 16; i++) Mask.push_back((i + Width) % 16);
      Value *Shuffled = Builder.CreateShuffleVector(Sum, Sum, Mask);
      Sum = Builder.CreateAdd(Sum, Shuffled);
    }
    Value *ScalarResult = Builder.CreateExtractElement(Sum, (uint64_t)0, "vnni.result");

    // Apply bias correction: result = vpdpbusd_result - 128 * sum(B)
    if (BothSigned) {
      // Horizontal reduce the bias sum
      Value *BiasSum = BiasSumAcc;
      for (int Width = 8; Width >= 1; Width /= 2) {
        SmallVector<int, 16> Mask;
        for (int i = 0; i < 16; i++) Mask.push_back((i + Width) % 16);
        Value *Shuffled = Builder.CreateShuffleVector(BiasSum, BiasSum, Mask);
        BiasSum = Builder.CreateAdd(BiasSum, Shuffled);
      }
      Value *TotalBiasSum = Builder.CreateExtractElement(BiasSum, (uint64_t)0, "total.bias");
      Value *Correction = Builder.CreateMul(TotalBiasSum, ConstantInt::get(I32Ty, 128), "correction");
      ScalarResult = Builder.CreateSub(ScalarResult, Correction, "corrected.result");
    }

    Builder.CreateBr(Exit);

    // Redirect preheader to our vector loop
    BranchInst *PreheaderBr = dyn_cast<BranchInst>(Preheader->getTerminator());
    if (PreheaderBr) {
      PreheaderBr->setSuccessor(0, VecHeader);
    }

    // Update Exit PHIs
    for (PHINode &PN : Exit->phis()) {
      for (unsigned i = 0; i < PN.getNumIncomingValues(); i++) {
        if (PN.getIncomingBlock(i) == Header) {
          PN.setIncomingBlock(i, VecExit);
          if (PN.getIncomingValue(i) == Candidate.AccPhi) {
            PN.setIncomingValue(i, ScalarResult);
          }
        }
      }
    }

    // Delete the old loop
    SmallVector<BasicBlock*, 4> ToDelete;
    for (BasicBlock *BB : L->blocks()) {
      ToDelete.push_back(BB);
    }

    // Remove loop from LoopInfo
    if (Loop *Parent = L->getParentLoop()) {
      Parent->removeChildLoop(std::find(Parent->begin(), Parent->end(), L));
    } else {
      LI.removeLoop(std::find(LI.begin(), LI.end(), L));
    }

    // Replace external uses of AccPhi with ScalarResult
    SmallVector<Use*, 8> UsesToReplace;
    for (Use &U : Candidate.AccPhi->uses()) {
      Instruction *UserI = cast<Instruction>(U.getUser());
      if (!L->contains(UserI->getParent())) {
        UsesToReplace.push_back(&U);
      }
    }
    for (Use *U : UsesToReplace) {
      U->set(ScalarResult);
    }

    // Replace all remaining uses with undef
    for (BasicBlock *BB : ToDelete) {
      for (Instruction &I : *BB) {
        if (!I.use_empty()) {
          I.replaceAllUsesWith(UndefValue::get(I.getType()));
        }
      }
    }

    // Delete terminators
    for (BasicBlock *BB : ToDelete) {
      BB->getTerminator()->eraseFromParent();
    }

    // Delete instructions
    for (BasicBlock *BB : ToDelete) {
      while (!BB->empty()) {
        BB->back().eraseFromParent();
      }
    }

    // Delete blocks
    for (BasicBlock *BB : ToDelete) {
      BB->eraseFromParent();
    }

    errs() << "VNNI Pass: Loop transformation complete\n";
    return true;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
  }
};

} // anonymous namespace

char VNNIPass::ID = 0;
static RegisterPass<VNNIPass> X("vnni", "VNNI vpdpbusd optimization pass", false, false);

namespace llvm {
  Pass *createVNNIPass() { return new VNNIPass(); }
}
