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
// GDB Debugging Helpers - Set breakpoints on these functions
//===----------------------------------------------------------------------===//

// Call this to trigger a GDB breakpoint - set: break vnni_debug_break
extern "C" void vnni_debug_break(const char* msg, int line) {
  errs() << "\n=== VNNI DEBUG BREAK [" << msg << "] at line " << line << " ===\n";
  // GDB: break vnni_debug_break
  // Then: print msg, print line
}

// Debug macro - use VNNI_BREAK("message") to add breakpoints
#define VNNI_BREAK(msg) vnni_debug_break(msg, __LINE__)
#define VNNI_LOG(msg) errs() << "VNNI[" << __LINE__ << "]: " << msg << "\n"

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
  int64_t RowStrideB;           // Stride between B rows (for non-square matrix detection)
  bool BothSigned;              // true if both inputs are signed (need bias correction)
  bool IsLoadStorePattern;      // true if using load-store instead of PHI
  bool NeedsTranspose;          // true if B needs transpose for contiguous access

  // Debug dump - call from GDB: call C.dump()
  void dump() const {
    errs() << "\n=== VNNICandidate Dump ===\n";
    errs() << "  AccPhi: " << (AccPhi ? "yes" : "NULL") << "\n";
    if (AccPhi) errs() << "    " << *AccPhi << "\n";
    errs() << "  IndPhi: " << (IndPhi ? "yes" : "NULL") << "\n";
    if (IndPhi) errs() << "    " << *IndPhi << "\n";
    errs() << "  Add: " << (Add ? "yes" : "NULL") << "\n";
    if (Add) errs() << "    " << *Add << "\n";
    errs() << "  Mul: " << (Mul ? "yes" : "NULL") << "\n";
    if (Mul) errs() << "    " << *Mul << "\n";
    errs() << "  ExtA: " << (ExtA ? "yes" : "NULL") << "\n";
    if (ExtA) errs() << "    " << *ExtA << "\n";
    errs() << "  ExtB: " << (ExtB ? "yes" : "NULL") << "\n";
    if (ExtB) errs() << "    " << *ExtB << "\n";
    errs() << "  LoadA: " << (LoadA ? "yes" : "NULL") << "\n";
    if (LoadA) errs() << "    " << *LoadA << "\n";
    errs() << "  LoadB: " << (LoadB ? "yes" : "NULL") << "\n";
    if (LoadB) errs() << "    " << *LoadB << "\n";
    errs() << "  GEPA: " << (GEPA ? "yes" : "NULL") << "\n";
    if (GEPA) errs() << "    " << *GEPA << "\n";
    errs() << "  GEPB: " << (GEPB ? "yes" : "NULL") << "\n";
    if (GEPB) errs() << "    " << *GEPB << "\n";
    errs() << "  TripCount: " << TripCount << "\n";
    errs() << "  RowStrideA: " << RowStrideA << "\n";
    errs() << "  RowStrideB: " << RowStrideB << "\n";
    errs() << "  BothSigned: " << BothSigned << "\n";
    errs() << "  IsLoadStorePattern: " << IsLoadStorePattern << "\n";
    errs() << "  NeedsTranspose: " << NeedsTranspose << "\n";
    errs() << "=========================\n\n";
  }
};

//===----------------------------------------------------------------------===//
// TiledMatmulLoops - Clean structure to hold all 6 loop levels
//===----------------------------------------------------------------------===//
struct TiledMatmulLoops {
  // Outer tile loops (step = TILE_SIZE, typically 16)
  Loop *IOuterLoop = nullptr;   // depth 1: for i_outer in 0..M step TILE
  Loop *JOuterLoop = nullptr;   // depth 2: for j_outer in 0..N step TILE
  Loop *KOuterLoop = nullptr;   // depth 3: for k_outer in 0..K step TILE

  // Inner tile loops (step = 1)
  Loop *IILoop = nullptr;       // depth 4: for ii in 0..TILE
  Loop *JJLoop = nullptr;       // depth 5: for jj in 0..TILE
  Loop *KKLoop = nullptr;       // depth 6: for kk in 0..TILE (innermost, VNNI target)

  // Induction variable PHI nodes for each loop
  PHINode *i_outer = nullptr;
  PHINode *j_outer = nullptr;
  PHINode *k_outer = nullptr;
  PHINode *ii = nullptr;
  PHINode *jj = nullptr;
  PHINode *kk = nullptr;

  // Matrix dimensions (full, not tile)
  int64_t M = 0;  // A rows, C rows
  int64_t N = 0;  // B cols, C cols
  int64_t K = 0;  // A cols, B rows (reduction dimension)
  int64_t TileSize = 16;

  bool isValid() const {
    return IOuterLoop && JOuterLoop && KOuterLoop &&
           IILoop && JJLoop && KKLoop &&
           i_outer && j_outer && k_outer &&
           ii && jj && kk;
  }

  void dump() const {
    errs() << "\n=== TiledMatmulLoops Dump ===\n";
    errs() << "  Loops: "
           << (IOuterLoop ? "I " : "- ")
           << (JOuterLoop ? "J " : "- ")
           << (KOuterLoop ? "K " : "- ")
           << (IILoop ? "ii " : "- ")
           << (JJLoop ? "jj " : "- ")
           << (KKLoop ? "kk" : "-") << "\n";
    errs() << "  PHIs:  "
           << (i_outer ? "i " : "- ")
           << (j_outer ? "j " : "- ")
           << (k_outer ? "k " : "- ")
           << (ii ? "ii " : "- ")
           << (jj ? "jj " : "- ")
           << (kk ? "kk" : "-") << "\n";
    errs() << "  Dims: M=" << M << " N=" << N << " K=" << K
           << " Tile=" << TileSize << "\n";
    errs() << "  Valid: " << (isValid() ? "YES" : "NO") << "\n";
    errs() << "=============================\n\n";
  }
};

//===----------------------------------------------------------------------===//
// MatmulOperands - Base pointers and strides for A, B, C matrices
//===----------------------------------------------------------------------===//
struct MatmulOperands {
  Value *BaseA = nullptr;       // Base pointer for A matrix
  Value *BaseB = nullptr;       // Base pointer for B matrix
  Value *BaseC = nullptr;       // Base pointer for C matrix
  Value *BaseB_T = nullptr;     // Transposed B (generated by pass)

  int64_t StrideA = 0;  // Row stride for A (= K for row-major A[M,K])
  int64_t StrideB = 0;  // Row stride for B (= N for row-major B[K,N])
  int64_t StrideC = 0;  // Row stride for C (= N for row-major C[M,N])

  bool BothSigned = false;      // Both A and B are signed int8
  bool NeedsTranspose = true;   // B needs transpose (almost always true)

  // Original instructions for deletion
  StoreInst *OrigStore = nullptr;

  bool isValid() const {
    return BaseA && BaseB && BaseC && StrideA > 0 && StrideC > 0;
  }

  void dump() const {
    errs() << "\n=== MatmulOperands Dump ===\n";
    errs() << "  BaseA: " << (BaseA ? "set" : "NULL") << "\n";
    errs() << "  BaseB: " << (BaseB ? "set" : "NULL") << "\n";
    errs() << "  BaseC: " << (BaseC ? "set" : "NULL") << "\n";
    errs() << "  BaseB_T: " << (BaseB_T ? "set" : "NULL") << "\n";
    errs() << "  StrideA: " << StrideA << "\n";
    errs() << "  StrideB: " << StrideB << "\n";
    errs() << "  StrideC: " << StrideC << "\n";
    errs() << "  BothSigned: " << BothSigned << "\n";
    errs() << "  Valid: " << (isValid() ? "YES" : "NO") << "\n";
    errs() << "===========================\n\n";
  }
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

    VNNI_LOG("=== Starting VNNI pass on function: " << F.getName() << " ===");
    VNNI_BREAK("runOnFunction entry");

    // Collect all innermost loops (K loops in matmul)
    SmallVector<Loop*, 8> InnermostLoops;
    for (Loop *L : LI) {
      collectInnermostLoops(L, InnermostLoops);
    }
    VNNI_LOG("Found " << InnermostLoops.size() << " innermost loops");

    // Transform each qualifying loop
    for (Loop *L : InnermostLoops) {
      VNNI_LOG("--- Analyzing loop: " << L->getHeader()->getName() << " ---");
      VNNICandidate Candidate;
      if (detectVNNIPattern(L, Candidate)) {
        VNNI_BREAK("Pattern detected");
        Candidate.dump();
        errs() << "VNNI: Detected pattern, trip=" << Candidate.TripCount
               << ", stride=" << Candidate.RowStrideA << "\n";

        // === NEW: Test the clean loop and operand analysis ===
        TiledMatmulLoops LoopNest;
        MatmulOperands Operands;
        bool newAnalysisOK = false;

        if (analyzeLoopNest(L, LoopNest)) {
          errs() << "VNNI-NEW: Clean loop analysis PASSED - all 6 levels found!\n";

          if (extractMatmulOperands(Candidate, LoopNest, Operands)) {
            errs() << "VNNI-NEW: Operand extraction PASSED!\n";
            errs() << "VNNI-NEW: Ready for clean codegen with:\n";
            errs() << "  - i_outer PHI: " << *LoopNest.i_outer << "\n";
            errs() << "  - j_outer PHI: " << *LoopNest.j_outer << "\n";
            errs() << "  - k_outer PHI: " << *LoopNest.k_outer << "\n";
            errs() << "  - K=" << LoopNest.K << " N=" << LoopNest.N << "\n";
            newAnalysisOK = true;

            // Generate B transpose
            if (generateBTranspose(F, LoopNest, Operands)) {
              errs() << "VNNI-NEW: B transpose generated, BaseB_T = " << *Operands.BaseB_T << "\n";

              // Generate VNNI loop with correct indices
              if (generateVNNILoop(F, LoopNest, Operands, Candidate)) {
                errs() << "VNNI-NEW: === CLEAN CODEGEN COMPLETE ===\n";
                Changed = true;
              } else {
                errs() << "VNNI-NEW: VNNI codegen skipped (K too small or error)\n";
              }
              continue;  // Skip old buggy transformToVNNI
            }
          } else {
            errs() << "VNNI-NEW: Operand extraction FAILED\n";
          }
        } else {
          errs() << "VNNI-NEW: Clean loop analysis FAILED - skipping (not 6-level tiled)\n";
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

  //===--------------------------------------------------------------------===//
  // NEW: Clean Loop Nest Analysis (populates TiledMatmulLoops)
  //===--------------------------------------------------------------------===//

  // Analyze 6-level tiled matmul loop nest starting from innermost
  // Returns true if valid 6-level structure found
  bool analyzeLoopNest(Loop *Innermost, TiledMatmulLoops &Loops) {
    errs() << "VNNI-NEW: Analyzing loop nest from innermost...\n";

    // Innermost is KK loop (depth 6)
    Loops.KKLoop = Innermost;
    Loops.kk = getParentLoopIndPhi(Innermost);
    errs() << "  L5 (kk): depth=" << Innermost->getLoopDepth()
           << " phi=" << (Loops.kk ? "found" : "MISSING") << "\n";

    // Walk up to find all parent loops
    Loop *L = Innermost;

    // JJ loop (depth 5) - parent of KK
    L = L->getParentLoop();
    if (!L) { errs() << "  FAIL: No parent for KK\n"; return false; }
    Loops.JJLoop = L;
    Loops.jj = getParentLoopIndPhi(L);
    errs() << "  L4 (jj): depth=" << L->getLoopDepth()
           << " phi=" << (Loops.jj ? "found" : "MISSING") << "\n";

    // II loop (depth 4) - grandparent of KK
    L = L->getParentLoop();
    if (!L) { errs() << "  FAIL: No grandparent for KK\n"; return false; }
    Loops.IILoop = L;
    Loops.ii = getParentLoopIndPhi(L);
    errs() << "  L3 (ii): depth=" << L->getLoopDepth()
           << " phi=" << (Loops.ii ? "found" : "MISSING") << "\n";

    // K_outer loop (depth 3)
    L = L->getParentLoop();
    if (!L) { errs() << "  FAIL: No K_outer loop\n"; return false; }
    Loops.KOuterLoop = L;
    Loops.k_outer = getParentLoopIndPhi(L);
    errs() << "  L2 (k_outer): depth=" << L->getLoopDepth()
           << " phi=" << (Loops.k_outer ? "found" : "MISSING") << "\n";

    // J_outer loop (depth 2)
    L = L->getParentLoop();
    if (!L) { errs() << "  FAIL: No J_outer loop\n"; return false; }
    Loops.JOuterLoop = L;
    Loops.j_outer = getParentLoopIndPhi(L);
    errs() << "  L1 (j_outer): depth=" << L->getLoopDepth()
           << " phi=" << (Loops.j_outer ? "found" : "MISSING") << "\n";

    // I_outer loop (depth 1)
    L = L->getParentLoop();
    if (!L) { errs() << "  FAIL: No I_outer loop\n"; return false; }
    Loops.IOuterLoop = L;
    Loops.i_outer = getParentLoopIndPhi(L);
    errs() << "  L0 (i_outer): depth=" << L->getLoopDepth()
           << " phi=" << (Loops.i_outer ? "found" : "MISSING") << "\n";

    // Extract tile size from innermost loop trip count
    if (auto *BI = dyn_cast<BranchInst>(Loops.KKLoop->getHeader()->getTerminator())) {
      if (BI->isConditional()) {
        if (auto *Cmp = dyn_cast<ICmpInst>(BI->getCondition())) {
          if (auto *CI = dyn_cast<ConstantInt>(Cmp->getOperand(1))) {
            Loops.TileSize = CI->getSExtValue();
            errs() << "  TileSize: " << Loops.TileSize << "\n";
          }
        }
      }
    }

    // Validate and dump
    bool valid = Loops.isValid();
    errs() << "VNNI-NEW: Loop nest analysis " << (valid ? "SUCCESS" : "FAILED") << "\n";
    if (valid) {
      Loops.dump();
    }
    return valid;
  }

  //===--------------------------------------------------------------------===//
  // NEW: Extract matmul operands (base pointers and strides)
  //===--------------------------------------------------------------------===//

  // Trace through GEPs and PHIs to find base pointer
  Value* traceBasePointer(Value *Ptr, Loop *OutermostLoop) {
    Value *V = Ptr;
    int depth = 0;
    const int maxDepth = 20;

    while (depth++ < maxDepth) {
      // If it's a GEP, get the pointer operand
      if (auto *GEP = dyn_cast<GetElementPtrInst>(V)) {
        V = GEP->getPointerOperand();
        continue;
      }
      // If it's a PHI in a loop header, trace to preheader value
      if (auto *Phi = dyn_cast<PHINode>(V)) {
        // Find the incoming value from outside the loop
        for (unsigned i = 0; i < Phi->getNumIncomingValues(); i++) {
          BasicBlock *InBB = Phi->getIncomingBlock(i);
          if (!OutermostLoop->contains(InBB)) {
            V = Phi->getIncomingValue(i);
            break;
          }
        }
        continue;
      }
      // If it's an extractvalue (MLIR memref descriptor), trace it
      if (auto *EV = dyn_cast<ExtractValueInst>(V)) {
        // For MLIR memrefs, the base pointer is in field 0 or 1
        if (auto *IV = dyn_cast<InsertValueInst>(EV->getAggregateOperand())) {
          // Trace to find the allocated pointer
          while (IV) {
            if (IV->getIndices() == EV->getIndices()) {
              V = IV->getInsertedValueOperand();
              break;
            }
            if (auto *NextIV = dyn_cast<InsertValueInst>(IV->getAggregateOperand())) {
              IV = NextIV;
            } else {
              break;
            }
          }
        }
        continue;
      }
      // Stop at allocations or arguments
      if (isa<AllocaInst>(V) || isa<CallInst>(V) || isa<Argument>(V)) {
        break;
      }
      break;
    }
    return V;
  }

  // Extract stride from GEP index pattern: idx = row * stride + col
  int64_t extractStrideFromIndex(Value *Idx) {
    if (auto *Add = dyn_cast<BinaryOperator>(Idx)) {
      if (Add->getOpcode() == Instruction::Add) {
        for (Value *Op : {Add->getOperand(0), Add->getOperand(1)}) {
          if (auto *Mul = dyn_cast<BinaryOperator>(Op)) {
            if (Mul->getOpcode() == Instruction::Mul) {
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

  // Extract matmul operands from detected pattern
  bool extractMatmulOperands(VNNICandidate &C, TiledMatmulLoops &Loops,
                              MatmulOperands &Ops) {
    errs() << "VNNI-NEW: Extracting matmul operands...\n";

    // Get base pointers by tracing through GEPs
    if (!C.GEPA || !C.GEPB) {
      errs() << "  FAIL: Missing GEPs for A or B\n";
      return false;
    }

    // Trace to base pointers
    Ops.BaseA = traceBasePointer(C.GEPA->getPointerOperand(), Loops.IOuterLoop);
    Ops.BaseB = traceBasePointer(C.GEPB->getPointerOperand(), Loops.IOuterLoop);

    errs() << "  BaseA: " << *Ops.BaseA << "\n";
    errs() << "  BaseB: " << *Ops.BaseB << "\n";

    // Get C base pointer
    if (C.IsLoadStorePattern && C.GEPC) {
      Ops.BaseC = traceBasePointer(C.GEPC->getPointerOperand(), Loops.IOuterLoop);
      errs() << "  BaseC: " << *Ops.BaseC << "\n";
      Ops.OrigStore = C.StoreC;
    } else {
      errs() << "  FAIL: No C GEP found\n";
      return false;
    }

    // Extract strides from GEP indices
    Value *IdxA = C.GEPA->getOperand(C.GEPA->getNumOperands() - 1);
    Value *IdxB = C.GEPB->getOperand(C.GEPB->getNumOperands() - 1);
    Value *IdxC = C.GEPC->getOperand(C.GEPC->getNumOperands() - 1);

    Ops.StrideA = extractStrideFromIndex(IdxA);
    Ops.StrideB = extractStrideFromIndex(IdxB);
    Ops.StrideC = extractStrideFromIndex(IdxC);

    errs() << "  StrideA: " << Ops.StrideA << "\n";
    errs() << "  StrideB: " << Ops.StrideB << "\n";
    errs() << "  StrideC: " << Ops.StrideC << "\n";

    // Check if both inputs are signed (sext from i8)
    Ops.BothSigned = C.BothSigned;
    errs() << "  BothSigned: " << (Ops.BothSigned ? "yes" : "no") << "\n";

    // Set matrix dimensions from strides
    // For A[M,K]: StrideA = K
    // For B[K,N]: StrideB = N
    // For C[M,N]: StrideC = N
    Loops.K = Ops.StrideA;
    Loops.N = Ops.StrideC;
    // M is harder to determine, leave as 0 for now

    bool valid = Ops.isValid();
    errs() << "VNNI-NEW: Operand extraction " << (valid ? "SUCCESS" : "FAILED") << "\n";
    if (valid) {
      Ops.dump();
    }
    return valid;
  }

  //===--------------------------------------------------------------------===//
  // NEW: Generate B transpose at outermost loop preheader
  //===--------------------------------------------------------------------===//

  // Generate B_T[j,k] = B[k,j] transpose
  // B is K x N (row-major), B_T is N x K (row-major)
  // After transpose: B_T[j*K + k] is contiguous along k
  bool generateBTranspose(Function &F, TiledMatmulLoops &Loops,
                          MatmulOperands &Ops) {
    errs() << "VNNI-NEW: Generating B transpose...\n";

    LLVMContext &Ctx = F.getContext();
    Module *M = F.getParent();

    // Use aligned_alloc for 64-byte alignment (required for AVX-512 loads)
    Type *I64Ty = Type::getInt64Ty(Ctx);
    FunctionCallee AlignedAllocFn = M->getOrInsertFunction(
        "aligned_alloc", PointerType::get(Ctx, 0), I64Ty, I64Ty);

    BasicBlock *Preheader = Loops.IOuterLoop->getLoopPreheader();
    if (!Preheader) {
      errs() << "  FAIL: No preheader for outermost loop\n";
      return false;
    }

    IRBuilder<> Builder(Preheader->getTerminator());
    Type *I8Ty = Type::getInt8Ty(Ctx);

    int64_t K = Loops.K;
    int64_t N = Loops.N;
    int64_t AllocSize = ((N * K + 63) / 64) * 64;  // Round up to 64-byte multiple
    errs() << "  Transpose size: " << N << " x " << K << " = " << AllocSize << " bytes (aligned)\n";

    Value *Align = ConstantInt::get(I64Ty, 64);
    Value *Size = ConstantInt::get(I64Ty, AllocSize);
    Ops.BaseB_T = Builder.CreateCall(AlignedAllocFn, {Align, Size}, "B_T");
    errs() << "  Allocated B_T: " << *Ops.BaseB_T << "\n";

    // Create transpose loop blocks
    BasicBlock *JHeader = BasicBlock::Create(Ctx, "transpose.j.hdr", &F);
    BasicBlock *KHeader = BasicBlock::Create(Ctx, "transpose.k.hdr", &F);
    BasicBlock *KBody = BasicBlock::Create(Ctx, "transpose.k.body", &F);
    BasicBlock *KExit = BasicBlock::Create(Ctx, "transpose.k.exit", &F);
    BasicBlock *JExit = BasicBlock::Create(Ctx, "transpose.j.exit", &F);

    // Insert blocks before the preheader's successor (the loop header)
    BasicBlock *LoopEntry = Preheader->getTerminator()->getSuccessor(0);

    // Redirect preheader to transpose J header
    Preheader->getTerminator()->setSuccessor(0, JHeader);

    // J loop header: for j in 0..N
    Builder.SetInsertPoint(JHeader);
    PHINode *TJ = Builder.CreatePHI(I64Ty, 2, "tj");
    TJ->addIncoming(ConstantInt::get(I64Ty, 0), Preheader);
    Value *JCond = Builder.CreateICmpSLT(TJ, ConstantInt::get(I64Ty, N));
    Builder.CreateCondBr(JCond, KHeader, JExit);

    // K loop header: for k in 0..K
    Builder.SetInsertPoint(KHeader);
    PHINode *TK = Builder.CreatePHI(I64Ty, 2, "tk");
    TK->addIncoming(ConstantInt::get(I64Ty, 0), JHeader);
    Value *KCond = Builder.CreateICmpSLT(TK, ConstantInt::get(I64Ty, K));
    Builder.CreateCondBr(KCond, KBody, KExit);

    // K loop body: B_T[j*K + k] = B[k*N + j]
    Builder.SetInsertPoint(KBody);

    // Source index: B[k*N + j] (B is K x N, row-major)
    Value *SrcIdx = Builder.CreateAdd(
        Builder.CreateMul(TK, ConstantInt::get(I64Ty, N)),
        TJ, "src.idx");
    Value *SrcPtr = Builder.CreateGEP(I8Ty, Ops.BaseB, SrcIdx, "src.ptr");
    Value *Val = Builder.CreateLoad(I8Ty, SrcPtr, "b.val");

    // Dest index: B_T[j*K + k] (B_T is N x K, row-major, contiguous along k)
    Value *DstIdx = Builder.CreateAdd(
        Builder.CreateMul(TJ, ConstantInt::get(I64Ty, K)),
        TK, "dst.idx");
    Value *DstPtr = Builder.CreateGEP(I8Ty, Ops.BaseB_T, DstIdx, "dst.ptr");
    Builder.CreateStore(Val, DstPtr);

    // K loop increment
    Value *TKNext = Builder.CreateAdd(TK, ConstantInt::get(I64Ty, 1));
    TK->addIncoming(TKNext, KBody);
    Builder.CreateBr(KHeader);

    // K loop exit -> J increment
    Builder.SetInsertPoint(KExit);
    Value *TJNext = Builder.CreateAdd(TJ, ConstantInt::get(I64Ty, 1));
    TJ->addIncoming(TJNext, KExit);
    Builder.CreateBr(JHeader);

    // J loop exit -> continue to original loop
    Builder.SetInsertPoint(JExit);
    Builder.CreateBr(LoopEntry);

    // Fix PHIs in LoopEntry (I_outer header) - they expect Preheader,
    // but now JExit is the predecessor
    for (PHINode &Phi : LoopEntry->phis()) {
      int Idx = Phi.getBasicBlockIndex(Preheader);
      if (Idx >= 0) {
        Phi.setIncomingBlock(Idx, JExit);
      }
    }

    errs() << "VNNI-NEW: B transpose generation SUCCESS\n";
    return true;
  }

  //===--------------------------------------------------------------------===//
  // NEW: Generate VNNI loop with CORRECT index calculations
  //===--------------------------------------------------------------------===//

  // Generate VNNI loops with OWN ii/jj loops (replaces k_outer and its children)
  // At k_outer preheader, only i_outer and j_outer are valid
  // We create our own ii/jj loops, then VNNI k loop inside
  bool generateVNNILoop(Function &F, TiledMatmulLoops &Loops,
                        MatmulOperands &Ops, VNNICandidate &C) {
    errs() << "VNNI-NEW: Generating VNNI with own ii/jj/k loops...\n";

    LLVMContext &Ctx = F.getContext();
    Module *M = F.getParent();

    Type *I8Ty = Type::getInt8Ty(Ctx);
    Type *I32Ty = Type::getInt32Ty(Ctx);
    Type *I64Ty = Type::getInt64Ty(Ctx);
    auto *V16I32Ty = FixedVectorType::get(I32Ty, 16);
    auto *V16I32PtrTy = PointerType::get(V16I32Ty, 0);

    Function *VPDPBUSD = Intrinsic::getOrInsertDeclaration(
        M, Intrinsic::x86_avx512_vpdpbusd_512);
    if (!VPDPBUSD) {
      errs() << "  FAIL: Could not get vpdpbusd intrinsic\n";
      return false;
    }

    BasicBlock *KOuterPreheader = Loops.KOuterLoop->getLoopPreheader();
    BasicBlock *KOuterExit = Loops.KOuterLoop->getExitBlock();
    if (!KOuterPreheader || !KOuterExit) {
      errs() << "  FAIL: k_outer loop missing preheader or exit\n";
      return false;
    }

    int64_t K = Loops.K;
    int64_t N = Loops.N;
    int64_t TileSize = Loops.TileSize;
    const int64_t VNNI_STEP = 64;
    const int64_t I_STEP = 4;

    if (K < VNNI_STEP) {
      errs() << "  SKIP: K=" << K << " < " << VNNI_STEP << "\n";
      return false;
    }

    errs() << "  Creating ii/jj/k loops: ii=0.." << TileSize << " step " << I_STEP
           << ", jj=0.." << TileSize << ", k=0.." << K << " step " << VNNI_STEP << "\n";

    Value *KVal = ConstantInt::get(I64Ty, K);
    Value *NVal = ConstantInt::get(I64Ty, N);
    Value *TileSizeVal = ConstantInt::get(I64Ty, TileSize);
    Value *ZeroI64 = ConstantInt::get(I64Ty, 0);
    Value *ZeroVec = ConstantVector::getSplat(ElementCount::getFixed(16), ConstantInt::get(I32Ty, 0));

    // Create all basic blocks
    BasicBlock *IIHeader = BasicBlock::Create(Ctx, "ii.hdr", &F);
    BasicBlock *JJHeader = BasicBlock::Create(Ctx, "jj.hdr", &F);
    BasicBlock *VNNIPre = BasicBlock::Create(Ctx, "vnni.pre", &F);
    BasicBlock *VNNIHeader = BasicBlock::Create(Ctx, "vnni.hdr", &F);
    BasicBlock *VNNIBody = BasicBlock::Create(Ctx, "vnni.body", &F);
    BasicBlock *VNNIExit = BasicBlock::Create(Ctx, "vnni.exit", &F);
    BasicBlock *JJLatch = BasicBlock::Create(Ctx, "jj.latch", &F);
    BasicBlock *IILatch = BasicBlock::Create(Ctx, "ii.latch", &F);

    // Remove k_outer PHI incoming from preheader BEFORE redirect
    BasicBlock *KOuterHeader = Loops.KOuterLoop->getHeader();
    for (PHINode &Phi : KOuterHeader->phis()) {
      int Idx = Phi.getBasicBlockIndex(KOuterPreheader);
      if (Idx >= 0) Phi.removeIncomingValue(Idx, false);
    }

    // Redirect k_outer preheader to our II header
    KOuterPreheader->getTerminator()->setSuccessor(0, IIHeader);

    IRBuilder<> Builder(Ctx);

    // === II HEADER: ii loop (0..TileSize step 4) ===
    Builder.SetInsertPoint(IIHeader);
    PHINode *IIPhi = Builder.CreatePHI(I64Ty, 2, "ii");
    IIPhi->addIncoming(ZeroI64, KOuterPreheader);
    Value *IICond = Builder.CreateICmpSLT(IIPhi, TileSizeVal, "ii.cond");
    Builder.CreateCondBr(IICond, JJHeader, KOuterExit);

    // === JJ HEADER: jj loop (0..TileSize) ===
    Builder.SetInsertPoint(JJHeader);
    PHINode *JJPhi = Builder.CreatePHI(I64Ty, 2, "jj");
    JJPhi->addIncoming(ZeroI64, IIHeader);
    Value *JJCond = Builder.CreateICmpSLT(JJPhi, TileSizeVal, "jj.cond");
    Builder.CreateCondBr(JJCond, VNNIPre, IILatch);

    // === VNNI PREHEADER: compute indices with valid ii/jj ===
    Builder.SetInsertPoint(VNNIPre);

    // i_full = i_outer + ii (both valid here!)
    Value *i_full = Builder.CreateAdd(Loops.i_outer, IIPhi, "i.full");
    Value *j_full = Builder.CreateAdd(Loops.j_outer, JJPhi, "j.full");

    // A row offsets for 4 consecutive rows
    Value *RowOffA0 = Builder.CreateMul(i_full, KVal, "row.a0");
    Value *RowOffA1 = Builder.CreateAdd(RowOffA0, KVal, "row.a1");
    Value *RowOffA2 = Builder.CreateAdd(RowOffA1, KVal, "row.a2");
    Value *RowOffA3 = Builder.CreateAdd(RowOffA2, KVal, "row.a3");

    // B_T row offset
    Value *RowOffB = Builder.CreateMul(j_full, KVal, "row.b");

    // C indices for 4 rows
    Value *IdxC0 = Builder.CreateAdd(Builder.CreateMul(i_full, NVal), j_full, "idx.c0");
    Value *IdxC1 = Builder.CreateAdd(IdxC0, NVal, "idx.c1");
    Value *IdxC2 = Builder.CreateAdd(IdxC1, NVal, "idx.c2");
    Value *IdxC3 = Builder.CreateAdd(IdxC2, NVal, "idx.c3");

    Builder.CreateBr(VNNIHeader);

    // === VNNI HEADER: k loop PHIs ===
    Builder.SetInsertPoint(VNNIHeader);
    PHINode *VK = Builder.CreatePHI(I64Ty, 2, "vk");
    VK->addIncoming(ZeroI64, VNNIPre);

    PHINode *Acc0 = Builder.CreatePHI(V16I32Ty, 2, "acc0");
    PHINode *Acc1 = Builder.CreatePHI(V16I32Ty, 2, "acc1");
    PHINode *Acc2 = Builder.CreatePHI(V16I32Ty, 2, "acc2");
    PHINode *Acc3 = Builder.CreatePHI(V16I32Ty, 2, "acc3");
    Acc0->addIncoming(ZeroVec, VNNIPre);
    Acc1->addIncoming(ZeroVec, VNNIPre);
    Acc2->addIncoming(ZeroVec, VNNIPre);
    Acc3->addIncoming(ZeroVec, VNNIPre);

    PHINode *Bias = nullptr;
    if (Ops.BothSigned) {
      Bias = Builder.CreatePHI(V16I32Ty, 2, "bias");
      Bias->addIncoming(ZeroVec, VNNIPre);
    }

    Value *VKCond = Builder.CreateICmpSLT(VK, KVal, "vk.cond");
    Builder.CreateCondBr(VKCond, VNNIBody, VNNIExit);

    // === VNNI BODY: load B once, 4 A rows, 4 vpdpbusd ===
    Builder.SetInsertPoint(VNNIBody);

    Value *IdxB = Builder.CreateAdd(RowOffB, VK, "idx.b");
    Value *PtrB = Builder.CreateGEP(I8Ty, Ops.BaseB_T, IdxB, "ptr.b");
    Value *VecB = Builder.CreateLoad(V16I32Ty, Builder.CreateBitCast(PtrB, V16I32PtrTy), "vec.b");

    Value *IdxA0 = Builder.CreateAdd(RowOffA0, VK, "idx.a0");
    Value *IdxA1 = Builder.CreateAdd(RowOffA1, VK, "idx.a1");
    Value *IdxA2 = Builder.CreateAdd(RowOffA2, VK, "idx.a2");
    Value *IdxA3 = Builder.CreateAdd(RowOffA3, VK, "idx.a3");

    Value *PtrA0 = Builder.CreateGEP(I8Ty, Ops.BaseA, IdxA0, "ptr.a0");
    Value *PtrA1 = Builder.CreateGEP(I8Ty, Ops.BaseA, IdxA1, "ptr.a1");
    Value *PtrA2 = Builder.CreateGEP(I8Ty, Ops.BaseA, IdxA2, "ptr.a2");
    Value *PtrA3 = Builder.CreateGEP(I8Ty, Ops.BaseA, IdxA3, "ptr.a3");

    Value *VecA0 = Builder.CreateLoad(V16I32Ty, Builder.CreateBitCast(PtrA0, V16I32PtrTy), "vec.a0");
    Value *VecA1 = Builder.CreateLoad(V16I32Ty, Builder.CreateBitCast(PtrA1, V16I32PtrTy), "vec.a1");
    Value *VecA2 = Builder.CreateLoad(V16I32Ty, Builder.CreateBitCast(PtrA2, V16I32PtrTy), "vec.a2");
    Value *VecA3 = Builder.CreateLoad(V16I32Ty, Builder.CreateBitCast(PtrA3, V16I32PtrTy), "vec.a3");

    Value *NewBias = Bias;
    if (Ops.BothSigned) {
      Value *SignFlip = ConstantVector::getSplat(ElementCount::getFixed(16),
                          ConstantInt::get(I32Ty, 0x80808080));
      VecA0 = Builder.CreateXor(VecA0, SignFlip);
      VecA1 = Builder.CreateXor(VecA1, SignFlip);
      VecA2 = Builder.CreateXor(VecA2, SignFlip);
      VecA3 = Builder.CreateXor(VecA3, SignFlip);

      Value *Ones = ConstantVector::getSplat(ElementCount::getFixed(16),
                      ConstantInt::get(I32Ty, 0x01010101));
      NewBias = Builder.CreateCall(VPDPBUSD, {Bias, Ones, VecB});
    }

    Value *NewAcc0 = Builder.CreateCall(VPDPBUSD, {Acc0, VecA0, VecB});
    Value *NewAcc1 = Builder.CreateCall(VPDPBUSD, {Acc1, VecA1, VecB});
    Value *NewAcc2 = Builder.CreateCall(VPDPBUSD, {Acc2, VecA2, VecB});
    Value *NewAcc3 = Builder.CreateCall(VPDPBUSD, {Acc3, VecA3, VecB});

    Value *VKNext = Builder.CreateAdd(VK, ConstantInt::get(I64Ty, VNNI_STEP), "vk.next");
    VK->addIncoming(VKNext, VNNIBody);
    Acc0->addIncoming(NewAcc0, VNNIBody);
    Acc1->addIncoming(NewAcc1, VNNIBody);
    Acc2->addIncoming(NewAcc2, VNNIBody);
    Acc3->addIncoming(NewAcc3, VNNIBody);
    if (Bias) Bias->addIncoming(NewBias, VNNIBody);

    Builder.CreateBr(VNNIHeader);

    // === VNNI EXIT: hreduce and store 4 results ===
    Builder.SetInsertPoint(VNNIExit);

    auto hreduce = [&](Value *Vec) -> Value* {
      for (int W = 8; W >= 1; W /= 2) {
        SmallVector<int, 16> Mask;
        for (int i = 0; i < 16; i++) Mask.push_back((i + W) % 16);
        Vec = Builder.CreateAdd(Vec, Builder.CreateShuffleVector(Vec, Vec, Mask));
      }
      return Builder.CreateExtractElement(Vec, (uint64_t)0);
    };

    Value *Sum0 = hreduce(Acc0);
    Value *Sum1 = hreduce(Acc1);
    Value *Sum2 = hreduce(Acc2);
    Value *Sum3 = hreduce(Acc3);

    if (Ops.BothSigned && Bias) {
      Value *BiasSum = hreduce(Bias);
      Value *Corr = Builder.CreateMul(BiasSum, ConstantInt::get(I32Ty, 128));
      Sum0 = Builder.CreateSub(Sum0, Corr);
      Sum1 = Builder.CreateSub(Sum1, Corr);
      Sum2 = Builder.CreateSub(Sum2, Corr);
      Sum3 = Builder.CreateSub(Sum3, Corr);
    }

    Value *PtrC0 = Builder.CreateGEP(I32Ty, Ops.BaseC, IdxC0);
    Value *PtrC1 = Builder.CreateGEP(I32Ty, Ops.BaseC, IdxC1);
    Value *PtrC2 = Builder.CreateGEP(I32Ty, Ops.BaseC, IdxC2);
    Value *PtrC3 = Builder.CreateGEP(I32Ty, Ops.BaseC, IdxC3);

    Builder.CreateStore(Sum0, PtrC0);
    Builder.CreateStore(Sum1, PtrC1);
    Builder.CreateStore(Sum2, PtrC2);
    Builder.CreateStore(Sum3, PtrC3);

    Builder.CreateBr(JJLatch);

    // === JJ LATCH: jj++ ===
    Builder.SetInsertPoint(JJLatch);
    Value *JJNext = Builder.CreateAdd(JJPhi, ConstantInt::get(I64Ty, 1), "jj.next");
    JJPhi->addIncoming(JJNext, JJLatch);
    Builder.CreateBr(JJHeader);

    // === II LATCH: ii += 4 ===
    Builder.SetInsertPoint(IILatch);
    Value *IINext = Builder.CreateAdd(IIPhi, ConstantInt::get(I64Ty, I_STEP), "ii.next");
    IIPhi->addIncoming(IINext, IILatch);
    Builder.CreateBr(IIHeader);

    // Fix PHI nodes in k_outer exit (redirect from old k_outer to our IIHeader)
    for (PHINode &Phi : KOuterExit->phis()) {
      for (unsigned i = 0; i < Phi.getNumIncomingValues(); i++) {
        if (Loops.KOuterLoop->contains(Phi.getIncomingBlock(i))) {
          Phi.setIncomingBlock(i, IIHeader);
        }
      }
    }

    errs() << "VNNI-NEW: SUCCESS - created ii/jj/k loops\n";
    return true;
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
    VNNI_LOG("detectVNNIPattern: starting");
    BasicBlock *Header = L->getHeader();
    if (!Header || !L->getLoopLatch()) {
      VNNI_LOG("detectVNNIPattern: FAIL - no header or latch");
      return false;
    }

    // Find accumulator PHI (i32) and induction PHI (i64)
    VNNI_LOG("detectVNNIPattern: calling findLoopPHIs");
    if (!findLoopPHIs(L, Header, C)) {
      VNNI_LOG("detectVNNIPattern: FAIL - findLoopPHIs failed");
      return false;
    }
    VNNI_LOG("detectVNNIPattern: findLoopPHIs OK - AccPhi=" << (C.AccPhi ? "yes" : "no")
             << ", IndPhi=" << (C.IndPhi ? "yes" : "no")
             << ", IsLoadStore=" << C.IsLoadStorePattern);

    // Find the multiply-accumulate pattern
    VNNI_LOG("detectVNNIPattern: calling findMulAccPattern");
    if (!findMulAccPattern(L, C)) {
      VNNI_LOG("detectVNNIPattern: FAIL - findMulAccPattern failed");
      return false;
    }
    VNNI_LOG("detectVNNIPattern: findMulAccPattern OK - Mul=" << (C.Mul ? "yes" : "no"));

    // Find memory access pattern (GEPs and loads)
    VNNI_LOG("detectVNNIPattern: calling findMemoryPattern");
    if (!findMemoryPattern(C)) {
      VNNI_LOG("detectVNNIPattern: FAIL - findMemoryPattern failed");
      return false;
    }
    VNNI_LOG("detectVNNIPattern: findMemoryPattern OK");

    // Extract trip count (must be multiple of 64 for VNNI)
    VNNI_LOG("detectVNNIPattern: calling extractTripCount");
    if (!extractTripCount(L, Header, C)) {
      VNNI_LOG("detectVNNIPattern: FAIL - extractTripCount failed");
      return false;
    }
    VNNI_LOG("detectVNNIPattern: extractTripCount OK - trip=" << C.TripCount);

    // Extract row stride for future I-tiling optimization
    C.RowStrideA = extractRowStride(C);
    C.RowStrideB = extractStrideFromGEP(C.GEPB);
    VNNI_LOG("detectVNNIPattern: strideA=" << C.RowStrideA << ", strideB=" << C.RowStrideB);

    // B access pattern analysis for VNNI:
    // For matmul C[i,j] = sum_k A[i,k] * B[k,j]:
    //   - A[i,k] = A[i*strideA + k] -> contiguous along k (GOOD)
    //   - B[k,j] = B[k*strideB + j] -> strided by strideB along k (BAD for SIMD)
    //
    // VNNI needs contiguous 64-byte loads. B is ALWAYS strided along k.
    // Options:
    //   1. Transpose B to B_T[j,k] = B[k,j] so B_T[j*K+k] is contiguous
    //   2. Use gather loads (slower)
    //   3. Check if B is already transposed in input (strideB == 1)
    //
    // Current: We only transpose if strideB > TripCount (indicates strided access)
    // For tiled loops, TripCount is tile size (16), strideB is full matrix stride
    // So transpose is ALWAYS triggered, which is correct for VNNI
    // Enable transpose when B access is strided (stride > tile size means column access)
    C.NeedsTranspose = (C.RowStrideB != 0 && C.RowStrideB > C.TripCount);
    if (C.NeedsTranspose) {
      VNNI_LOG("detectVNNIPattern: B access is strided (stride=" << C.RowStrideB
               << " > tile=" << C.TripCount << ") - generating transpose");
    } else {
      VNNI_LOG("detectVNNIPattern: B access is contiguous - no transpose needed");
    }

    // Check if signed×signed (needs bias correction)
    C.BothSigned = isa<SExtInst>(C.ExtA) && isa<SExtInst>(C.ExtB);
    VNNI_LOG("detectVNNIPattern: BothSigned=" << C.BothSigned);

    VNNI_BREAK("detectVNNIPattern SUCCESS");
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
    C.Mul = nullptr;
    C.ExtA = nullptr;
    C.ExtB = nullptr;

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

  // Extract stride from a GEP instruction (for C matrix which may have different stride than A)
  int64_t extractStrideFromGEP(GetElementPtrInst *GEP) {
    if (!GEP) return 0;

    // Pattern: index = i * stride + j
    Value *Idx = GEP->getOperand(GEP->getNumOperands() - 1);

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

    // Trace through GEP chain to find base pointer
    while (auto *GEP = dyn_cast<GetElementPtrInst>(V)) {
      V = GEP->getPointerOperand();
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

    // Final GEP trace
    while (auto *GEP = dyn_cast<GetElementPtrInst>(V)) {
      V = GEP->getPointerOperand();
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
    if (!VPDPBUSD) {
      errs() << "VNNI: ERROR - Could not get vpdpbusd intrinsic!\n";
      return false;
    }
    errs() << "VNNI: Got vpdpbusd intrinsic: " << VPDPBUSD->getName() << "\n";

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
    errs() << "VNNI: BaseA = " << *BaseA << "\n";
    errs() << "VNNI: BaseB = " << *BaseB << "\n";
    if (isa<UndefValue>(BaseA) || isa<UndefValue>(BaseB)) {
      errs() << "VNNI: ERROR - Base pointer is undef!\n";
      return false;
    }

    // Compute row offsets
    Value *IdxA = C.GEPA->getOperand(C.GEPA->getNumOperands() - 1);
    Value *IdxB = C.GEPB->getOperand(C.GEPB->getNumOperands() - 1);
    Value *RowOffA = computeRowOffset(IdxA, C, Builder, L);
    Value *RowOffB = computeRowOffset(IdxB, C, Builder, L);

    // === Full matrix B transpose (Option 1) ===
    // For matmul A[M,K] × B[K,N], B is accessed as B[k,j] = B[k*N + j]
    // This is strided along k (stride = N), bad for VNNI which needs contiguous loads.
    // Solution: Transpose entire B matrix ONCE before all tile loops.
    // B_T[j,k] = B[k,j], so B_T[j*K + k] is contiguous along k.
    //
    // CRITICAL: Insert transpose at OUTERMOST matmul loop preheader, not inner K loop!
    //
    Value *BaseBT = BaseB;  // Will point to transposed B if needed
    int64_t K_full = C.RowStrideA;  // Full K dimension (not tile)
    int64_t N_full = C.RowStrideB;  // Full N dimension (not tile)

    if (C.NeedsTranspose && K_full > 0 && N_full > 0) {
      errs() << "VNNI: Generating FULL B transpose (K_full=" << K_full
             << ", N_full=" << N_full << ")\n";

      // Find the OUTERMOST loop (keep going up until no more parents)
      Loop *OuterLoop = L;
      while (OuterLoop->getParentLoop()) {
        OuterLoop = OuterLoop->getParentLoop();
      }
      BasicBlock *OuterPreheader = OuterLoop->getLoopPreheader();
      if (!OuterPreheader) {
        errs() << "VNNI: ERROR - No outermost preheader for transpose!\n";
        return false;
      }
      errs() << "VNNI: Inserting transpose at outermost preheader: "
             << OuterPreheader->getName() << "\n";

      // Save current insert point, switch to outer preheader
      auto SavedIP = Builder.saveIP();
      Builder.SetInsertPoint(OuterPreheader->getTerminator());

      // Allocate transposed buffer: N_full * K_full bytes
      Function *MallocFn = M->getFunction("malloc");
      if (!MallocFn) {
        FunctionType *MallocTy = FunctionType::get(
            PointerType::get(I8Ty, 0), {I64Ty}, false);
        MallocFn = Function::Create(MallocTy, Function::ExternalLinkage, "malloc", M);
      }
      Value *TransposeSize = ConstantInt::get(I64Ty, N_full * K_full);
      BaseBT = Builder.CreateCall(MallocFn, {TransposeSize}, "B_T");

      // Generate transpose loops at OUTER preheader
      // for j in 0..N_full: for k in 0..K_full: B_T[j*K_full+k] = B[k*N_full+j]
      BasicBlock *TransposeJHdr = BasicBlock::Create(Ctx, "transpose.j.hdr", &F, OuterPreheader->getNextNode());
      BasicBlock *TransposeKHdr = BasicBlock::Create(Ctx, "transpose.k.hdr", &F, TransposeJHdr->getNextNode());
      BasicBlock *TransposeKBody = BasicBlock::Create(Ctx, "transpose.k.body", &F, TransposeKHdr->getNextNode());
      BasicBlock *TransposeKExit = BasicBlock::Create(Ctx, "transpose.k.exit", &F, TransposeKBody->getNextNode());
      BasicBlock *TransposeJExit = BasicBlock::Create(Ctx, "transpose.j.exit", &F, TransposeKExit->getNextNode());

      // Get the original destination of outer preheader (the outer loop header)
      BasicBlock *OuterLoopHeader = OuterLoop->getHeader();

      // Redirect outer preheader to transpose loops
      Instruction *OuterTerm = OuterPreheader->getTerminator();
      Builder.SetInsertPoint(OuterTerm);
      Builder.CreateBr(TransposeJHdr);
      OuterTerm->eraseFromParent();

      // J loop header: for j in 0..N_full
      Builder.SetInsertPoint(TransposeJHdr);
      PHINode *JIdx = Builder.CreatePHI(I64Ty, 2, "tj");
      JIdx->addIncoming(ConstantInt::get(I64Ty, 0), OuterPreheader);
      Value *JCond = Builder.CreateICmpSLT(JIdx, ConstantInt::get(I64Ty, N_full));
      Builder.CreateCondBr(JCond, TransposeKHdr, TransposeJExit);

      // K loop header: for k in 0..K_full
      Builder.SetInsertPoint(TransposeKHdr);
      PHINode *KIdx = Builder.CreatePHI(I64Ty, 2, "tk");
      KIdx->addIncoming(ConstantInt::get(I64Ty, 0), TransposeJHdr);
      Value *KCond = Builder.CreateICmpSLT(KIdx, ConstantInt::get(I64Ty, K_full));
      Builder.CreateCondBr(KCond, TransposeKBody, TransposeKExit);

      // K loop body: B_T[j*K_full+k] = B[k*N_full+j]
      Builder.SetInsertPoint(TransposeKBody);
      Value *SrcIdx = Builder.CreateAdd(
          Builder.CreateMul(KIdx, ConstantInt::get(I64Ty, N_full)), JIdx);
      Value *DstIdx = Builder.CreateAdd(
          Builder.CreateMul(JIdx, ConstantInt::get(I64Ty, K_full)), KIdx);
      Value *SrcPtr = Builder.CreateGEP(I8Ty, BaseB, SrcIdx);
      Value *DstPtr = Builder.CreateGEP(I8Ty, BaseBT, DstIdx);
      Value *Val = Builder.CreateLoad(I8Ty, SrcPtr);
      Builder.CreateStore(Val, DstPtr);
      Value *KNext = Builder.CreateAdd(KIdx, ConstantInt::get(I64Ty, 1));
      KIdx->addIncoming(KNext, TransposeKBody);
      Builder.CreateBr(TransposeKHdr);

      // K loop exit -> J increment
      Builder.SetInsertPoint(TransposeKExit);
      Value *JNext = Builder.CreateAdd(JIdx, ConstantInt::get(I64Ty, 1));
      JIdx->addIncoming(JNext, TransposeKExit);
      Builder.CreateBr(TransposeJHdr);

      // J loop exit -> continue to original outer loop header
      Builder.SetInsertPoint(TransposeJExit);
      Builder.CreateBr(OuterLoopHeader);

      // Update PHIs in outer loop header to come from TransposeJExit instead of OuterPreheader
      for (PHINode &Phi : OuterLoopHeader->phis()) {
        for (unsigned i = 0; i < Phi.getNumIncomingValues(); i++) {
          if (Phi.getIncomingBlock(i) == OuterPreheader) {
            Phi.setIncomingBlock(i, TransposeJExit);
          }
        }
      }

      // Restore insert point to inner K loop preheader for VNNI generation
      Builder.restoreIP(SavedIP);

      // Update RowOffB for transposed layout:
      // Original: B[k*N + j] with row offset = j*stride (but we load strided)
      // Transposed: B_T[j*K + k] - now row j of B_T gives column j of B, contiguous!
      //
      // For 6-level tiled matmul: i, j, k, ii, jj, kk
      // Full j index = j_base + jj where:
      //   - jj is the inner J tile index (0..16) from L->getParentLoop()
      //   - j_base is the outer J loop index (0,16,32...)
      //
      // RowOffB = (j_base + jj) * K_full
      //
      Loop *JJLoop = L->getParentLoop();  // Inner J tile loop (jj)
      PHINode *JJPhi = JJLoop ? getParentLoopIndPhi(JJLoop) : nullptr;

      // Find outer J loop: go up 4 levels from kk (kk -> jj -> ii -> k -> j)
      Loop *JLoop = JJLoop;
      for (int i = 0; i < 3 && JLoop; i++) {
        JLoop = JLoop->getParentLoop();
      }
      PHINode *JBasePhi = JLoop ? getParentLoopIndPhi(JLoop) : nullptr;

      if (JJPhi) {
        Value *JFull = JJPhi;
        if (JBasePhi) {
          // j_full = j_base + jj
          JFull = Builder.CreateAdd(JBasePhi, JJPhi, "j.full");
          errs() << "VNNI: Found outer J loop, using j_base + jj\n";
        } else {
          errs() << "VNNI: Warning - no outer J loop, using jj only\n";
        }
        RowOffB = Builder.CreateMul(JFull, ConstantInt::get(I64Ty, K_full), "row.off.transposed");
        errs() << "VNNI: Transposed RowOffB = j_full * " << K_full << "\n";
      } else {
        errs() << "VNNI: ERROR - Could not get JJ PHI for transpose offset!\n";
        return false;
      }

      errs() << "VNNI: Full transpose generated (" << N_full << "x" << K_full << " bytes)\n";
    }

    // === Find outer K loop base for correct offset calculation ===
    // For 6-level tiled matmul: i, j, k, ii, jj, kk
    // We need k_base from the outer K loop (3 levels up from L)
    // kk (L) -> jj -> ii -> k
    Loop *KLoop = L;
    for (int i = 0; i < 3 && KLoop; i++) {
      KLoop = KLoop->getParentLoop();
    }
    PHINode *KBasePhi = KLoop ? getParentLoopIndPhi(KLoop) : nullptr;

    if (KBasePhi) {
      errs() << "VNNI: Found outer K loop (depth " << KLoop->getLoopDepth()
             << ") base PHI: " << *KBasePhi << "\n";
    } else {
      errs() << "VNNI: Warning - No outer K loop found (assuming non-tiled or 3-level loop)\n";
    }

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

    // Compute full K offset: k_base + VecIdx
    // For tiled loops: k_full = outer_k + inner_kk
    // VecIdx is the VNNI loop iterator (replaces kk), KBasePhi is outer k
    Value *KOffset = VecIdx;
    if (KBasePhi) {
      KOffset = Builder.CreateAdd(KBasePhi, VecIdx, "k.full");
      errs() << "VNNI: Using k_full = k_base + VecIdx for memory offsets\n";
    }

    // Load B vector ONCE (shared across all 4 A rows)
    // Use BaseBT (transposed B) if transpose enabled, otherwise BaseB
    Value *VecB = loadPadded(BaseBT, Builder.CreateAdd(RowOffB, KOffset));

    // Load 4 A rows - also need k_full for correct A offset
    Value *VecA0 = loadPadded(BaseA, Builder.CreateAdd(RowOffA0, KOffset));
    Value *VecA1 = loadPadded(BaseA, Builder.CreateAdd(RowOffA1, KOffset));
    Value *VecA2 = loadPadded(BaseA, Builder.CreateAdd(RowOffA2, KOffset));
    Value *VecA3 = loadPadded(BaseA, Builder.CreateAdd(RowOffA3, KOffset));

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
    BranchInst *LatchBr = Builder.CreateBr(VecHeader);

    // Add loop metadata to disable unrolling (prevents register spilling)
    // LLVM O3 over-unrolls this loop causing 40x more vpdpbusd instructions
    // The metadata must be on the latch (back-edge) branch
    // Using multiple hints to ensure LLVM respects no-unroll
    LLVMContext &LoopCtx = F.getContext();
    MDNode *LoopID = MDNode::get(LoopCtx, {
        MDNode::get(LoopCtx, {}),  // self-reference placeholder
        MDNode::get(LoopCtx, {
            MDString::get(LoopCtx, "llvm.loop.unroll.disable")
        }),
        MDNode::get(LoopCtx, {
            MDString::get(LoopCtx, "llvm.loop.unroll.count"),
            ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(LoopCtx), 1))
        }),
        MDNode::get(LoopCtx, {
            MDString::get(LoopCtx, "llvm.loop.unroll.full"),
            ConstantAsMetadata::get(ConstantInt::get(Type::getInt1Ty(LoopCtx), false))
        })
    });
    LoopID->replaceOperandWith(0, LoopID);  // fix self-reference
    LatchBr->setMetadata("llvm.loop", LoopID);

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

    // For load-store pattern, we already have the store in C.StoreC
    if (C.IsLoadStorePattern && C.StoreC) {
      OrigStore = C.StoreC;
      StoreGEP = C.GEPC;
      errs() << "VNNI: Using load-store pattern store instruction\n";
    } else {
      // Find store in exit block or parent loop (PHI pattern)
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
    }

    Builder.CreateBr(Exit);

    // === Rewire control flow ===
    // Preheader may be TransposeJExit which has no terminator yet
    if (Instruction *Term = Preheader->getTerminator()) {
      if (auto *Br = dyn_cast<BranchInst>(Term)) {
        Br->setSuccessor(0, VecHeader);
      }
    } else {
      // No terminator - create branch to VNNI loop
      IRBuilder<> PreBuilder(Preheader);
      PreBuilder.CreateBr(VecHeader);
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

    // === Generate stores for all 4 results (I=4 tiling) ===
    if (OrigStore && StoreGEP && ILoop && JLoop) {
      // For load-store pattern, the original store is in K loop which will be deleted
      // We need to insert stores in VecExit block instead
      Builder.SetInsertPoint(VecExit->getTerminator());

      // Trace BaseC through GEP chain to get actual array base pointer
      Value *BaseC = StoreGEP->getPointerOperand();
      while (auto *GEP = dyn_cast<GetElementPtrInst>(BaseC)) {
        BaseC = GEP->getPointerOperand();
      }

      // Get I and J loop induction variables - these remain valid after K loop deletion
      // Unlike OrigIdx which comes from the K loop and becomes invalid
      PHINode *IPhi = getParentLoopIndPhi(ILoop);
      PHINode *JPhi = getParentLoopIndPhi(JLoop);

      if (!IPhi || !JPhi) {
        errs() << "VNNI: Could not find I or J induction PHIs for store generation\n";
        if (!IPhi) errs() << "VNNI:   - IPhi is null\n";
        if (!JPhi) errs() << "VNNI:   - JPhi is null\n";
      } else {
        errs() << "VNNI: Using I/J induction PHIs for store index computation\n";
        errs() << "VNNI:   IPhi: " << *IPhi << "\n";
        errs() << "VNNI:   JPhi: " << *JPhi << "\n";
      }

      // C is stored as C[i * strideC + j], so consecutive rows are strideC apart
      // For matmul A[M,K] × B[K,N] → C[M,N]:
      //   strideA = K (e.g., 1024 for Attn)
      //   strideC = N (e.g., 64 for output)
      // IMPORTANT: Must extract strideC from the store GEP, not use strideA!
      int64_t StrideC = extractStrideFromGEP(StoreGEP);
      if (StrideC == 0) {
        // Fallback: for square matrices, strideA = strideC
        StrideC = C.RowStrideA;
        errs() << "VNNI: Warning - could not extract C stride, using A stride as fallback\n";
      }
      Value *CStride = ConstantInt::get(I64Ty, StrideC);

      // Compute index from I and J: C[i][j] = C[i * strideC + j]
      Value *IxStride = Builder.CreateMul(IPhi, CStride, "i_x_stride");
      Value *Idx0 = Builder.CreateAdd(IxStride, JPhi, "idx_i0_j");

      // Store Result0 at C[i][j]
      Value *Ptr0 = Builder.CreateGEP(I32Ty, BaseC, Idx0, "ptr_c_i0_j");
      Builder.CreateStore(Result0, Ptr0);

      // Store Result1 at C[i+1][j] = idx0 + strideC
      Value *Idx1 = Builder.CreateAdd(Idx0, CStride, "idx_i1_j");
      Value *Ptr1 = Builder.CreateGEP(I32Ty, BaseC, Idx1, "ptr_c_i1_j");
      Builder.CreateStore(Result1, Ptr1);

      // Store Result2 at C[i+2][j] = idx1 + strideC
      Value *Idx2 = Builder.CreateAdd(Idx1, CStride, "idx_i2_j");
      Value *Ptr2 = Builder.CreateGEP(I32Ty, BaseC, Idx2, "ptr_c_i2_j");
      Builder.CreateStore(Result2, Ptr2);

      // Store Result3 at C[i+3][j] = idx2 + strideC
      Value *Idx3 = Builder.CreateAdd(Idx2, CStride, "idx_i3_j");
      Value *Ptr3 = Builder.CreateGEP(I32Ty, BaseC, Idx3, "ptr_c_i3_j");
      Builder.CreateStore(Result3, Ptr3);

      errs() << "VNNI: Added 4 stores for I=4 tiling (strideC=" << StrideC << ", strideA=" << C.RowStrideA << ")\n";

      // Modify I loop (grandparent) to step by 4 - NOT J loop!
      if (modifyParentLoopStep(ILoop, 4)) {
        errs() << "VNNI: Modified I loop (grandparent) step to 4\n";
      }
    } else {
      errs() << "VNNI: Could not find store pattern or loops for I=4 tiling\n";
      if (!OrigStore) errs() << "VNNI:   - No OrigStore found\n";
      if (!StoreGEP) errs() << "VNNI:   - No StoreGEP found\n";
      if (!ILoop) errs() << "VNNI:   - No I loop (grandparent) found\n";
      if (!JLoop) errs() << "VNNI:   - No J loop (parent) found\n";
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

    // Replace external uses (only for PHI pattern, not load-store)
    if (C.AccPhi) {
      for (Use &U : C.AccPhi->uses()) {
        if (auto *I = dyn_cast<Instruction>(U.getUser())) {
          if (!L->contains(I->getParent())) {
            // Already handled by exit PHI update
          }
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
