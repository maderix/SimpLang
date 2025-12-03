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
  BinaryOperator *Mul;
  ZExtInst *ZExt;
  SExtInst *SExt;
  LoadInst *LoadA;
  LoadInst *LoadB;
  GetElementPtrInst *GEPA;
  GetElementPtrInst *GEPB;
  int64_t TripCount;
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
    BinaryOperator *Mul = nullptr;

    if (AddOp0 == AccPhi) Mul = dyn_cast<BinaryOperator>(AddOp1);
    else if (AddOp1 == AccPhi) Mul = dyn_cast<BinaryOperator>(AddOp0);

    if (!Mul || Mul->getOpcode() != Instruction::Mul) return false;
    Candidate.Mul = Mul;

    ZExtInst *ZExt = nullptr;
    SExtInst *SExt = nullptr;
    Value *MulOp0 = Mul->getOperand(0);
    Value *MulOp1 = Mul->getOperand(1);

    if (auto *Z = dyn_cast<ZExtInst>(MulOp0)) {
      ZExt = Z;
      SExt = dyn_cast<SExtInst>(MulOp1);
    } else if (auto *Z = dyn_cast<ZExtInst>(MulOp1)) {
      ZExt = Z;
      SExt = dyn_cast<SExtInst>(MulOp0);
    }

    if (!ZExt || !SExt) return false;
    if (!ZExt->getSrcTy()->isIntegerTy(8) || !SExt->getSrcTy()->isIntegerTy(8)) return false;

    Candidate.ZExt = ZExt;
    Candidate.SExt = SExt;

    Candidate.LoadA = dyn_cast<LoadInst>(ZExt->getOperand(0));
    Candidate.LoadB = dyn_cast<LoadInst>(SExt->getOperand(0));
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

    // Clone RowOffsetA to preheader if it's defined inside the loop
    if (auto *InstA = dyn_cast<Instruction>(RowOffsetA)) {
      if (L->contains(InstA->getParent())) {
        Builder.SetInsertPoint(Preheader->getTerminator());
        // Clone the instruction and its operand chain
        if (auto *MulA = dyn_cast<BinaryOperator>(InstA)) {
          // It's a mul, clone it: %315 = mul i64 %305, 64
          Value *NewRowA = Builder.CreateMul(MulA->getOperand(0), MulA->getOperand(1), "row.offset.A");
          RowOffsetA = NewRowA;
        }
      }
    }

    // Clone RowOffsetB to preheader if it's defined inside the loop
    if (auto *InstB = dyn_cast<Instruction>(RowOffsetB)) {
      if (L->contains(InstB->getParent())) {
        Builder.SetInsertPoint(Preheader->getTerminator());
        if (auto *MulB = dyn_cast<BinaryOperator>(InstB)) {
          Value *NewRowB = Builder.CreateMul(MulB->getOperand(0), MulB->getOperand(1), "row.offset.B");
          RowOffsetB = NewRowB;
        }
      }
    }

    // Create new vectorized loop blocks
    BasicBlock *VecHeader = BasicBlock::Create(Ctx, "vnni.header", &F);
    BasicBlock *VecBody = BasicBlock::Create(Ctx, "vnni.body", &F);
    BasicBlock *VecExit = BasicBlock::Create(Ctx, "vnni.exit", &F);

    // VecHeader: PHIs and loop condition
    Builder.SetInsertPoint(VecHeader);
    PHINode *VecIdx = Builder.CreatePHI(I64Ty, 2, "vnni.idx");
    PHINode *VecAcc = Builder.CreatePHI(V16I32Ty, 2, "vnni.acc");

    Value *ZeroVec = ConstantVector::getSplat(ElementCount::getFixed(16), ConstantInt::get(I32Ty, 0));
    Value *VecEnd = ConstantInt::get(I64Ty, Candidate.TripCount);
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

    Value *NewAcc = Builder.CreateCall(VPDPBUSDFn, {VecAcc, VecA, VecB}, "vpdpbusd");

    Value *NextIdx = Builder.CreateAdd(VecIdx, ConstantInt::get(I64Ty, 64), "next.idx");
    Builder.CreateBr(VecHeader);

    // Complete PHI nodes
    VecIdx->addIncoming(ConstantInt::get(I64Ty, 0), Preheader);
    VecIdx->addIncoming(NextIdx, VecBody);
    VecAcc->addIncoming(ZeroVec, Preheader);
    VecAcc->addIncoming(NewAcc, VecBody);

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
