# VNNI Pass Rewrite Plan

## Problem Summary

The current VNNIPass.cpp (1300 lines) has critical bugs in index calculation:
- A load: uses `ii*K` instead of `(i_outer+ii)*K`
- C store: uses `ii*N + jj` instead of `(i_outer+ii)*N + (j_outer+jj)`
- B_T load: correct (uses `j_full*K + k_full`)

Root cause: The pass doesn't properly track and use all 6 loop indices.

## Target Loop Structure

```
for i_outer in 0..M step TILE:        # L0 (depth 1)
  for j_outer in 0..N step TILE:      # L1 (depth 2)
    for k_outer in 0..K step TILE:    # L2 (depth 3)
      for ii in 0..TILE:              # L3 (depth 4)
        for jj in 0..TILE:            # L4 (depth 5)
          for kk in 0..TILE:          # L5 (depth 6) <- VNNI target
            C[i_outer+ii, j_outer+jj] += A[i_outer+ii, k_outer+kk] * B[k_outer+kk, j_outer+jj]
```

Memory layout (row-major):
- `A[i,k] = A[i*K + k]` - contiguous along k ✓
- `B[k,j] = B[k*N + j]` - strided along k, needs transpose
- `C[i,j] = C[i*N + j]`

## Rewrite Architecture

### Phase 1: Data Structures (~50 lines)

```cpp
// Clean structure to hold all loop info
struct TiledMatmulLoops {
  // Outer tile loops (step = TILE_SIZE)
  Loop *IOuterLoop = nullptr;   // depth 1
  Loop *JOuterLoop = nullptr;   // depth 2
  Loop *KOuterLoop = nullptr;   // depth 3

  // Inner tile loops (step = 1, or 4 for I with tiling)
  Loop *IILoop = nullptr;       // depth 4
  Loop *JJLoop = nullptr;       // depth 5
  Loop *KKLoop = nullptr;       // depth 6 (innermost, VNNI target)

  // Induction variables (PHI nodes)
  PHINode *i_outer = nullptr;
  PHINode *j_outer = nullptr;
  PHINode *k_outer = nullptr;
  PHINode *ii = nullptr;
  PHINode *jj = nullptr;
  PHINode *kk = nullptr;

  // Dimensions
  int64_t M, N, K;
  int64_t TileSize = 16;

  bool isValid() const {
    return IOuterLoop && JOuterLoop && KOuterLoop &&
           IILoop && JJLoop && KKLoop &&
           i_outer && j_outer && k_outer &&
           ii && jj && kk;
  }
};

struct MatmulOperands {
  Value *BaseA = nullptr;      // Base pointer for A
  Value *BaseB = nullptr;      // Base pointer for B
  Value *BaseC = nullptr;      // Base pointer for C
  Value *BaseB_T = nullptr;    // Transposed B (generated)

  int64_t StrideA;  // K (row stride for A)
  int64_t StrideB;  // N (row stride for B)
  int64_t StrideC;  // N (row stride for C)

  bool BothSigned = false;
};
```

### Phase 2: Loop Analysis (~100 lines)

```cpp
// Analyze loop nest starting from innermost
bool analyzeLoopNest(Loop *Innermost, TiledMatmulLoops &Loops) {
  // Innermost is KK loop
  Loops.KKLoop = Innermost;
  Loops.kk = getCanonicalInductionPHI(Innermost);

  // Walk up the loop tree
  Loop *L = Innermost;

  // JJ loop (parent of KK)
  L = L->getParentLoop();
  if (!L) return false;
  Loops.JJLoop = L;
  Loops.jj = getCanonicalInductionPHI(L);

  // II loop (grandparent of KK)
  L = L->getParentLoop();
  if (!L) return false;
  Loops.IILoop = L;
  Loops.ii = getCanonicalInductionPHI(L);

  // K_outer loop
  L = L->getParentLoop();
  if (!L) return false;
  Loops.KOuterLoop = L;
  Loops.k_outer = getCanonicalInductionPHI(L);

  // J_outer loop
  L = L->getParentLoop();
  if (!L) return false;
  Loops.JOuterLoop = L;
  Loops.j_outer = getCanonicalInductionPHI(L);

  // I_outer loop
  L = L->getParentLoop();
  if (!L) return false;
  Loops.IOuterLoop = L;
  Loops.i_outer = getCanonicalInductionPHI(L);

  // Extract dimensions from loop bounds
  Loops.TileSize = extractTripCount(Loops.KKLoop);
  Loops.K = extractUpperBound(Loops.KOuterLoop) + Loops.TileSize;
  // ... similar for M, N

  return Loops.isValid();
}
```

### Phase 3: Pattern Detection (~100 lines)

```cpp
bool detectMatmulPattern(Loop *KKLoop, MatmulOperands &Ops) {
  // Find the multiply-accumulate pattern in KK loop body:
  // C[...] += A[...] * B[...]

  // 1. Find store instruction
  StoreInst *Store = findStoreInLoop(KKLoop);
  if (!Store) return false;

  // 2. Trace back to find: add(load_c, mul(load_a, load_b))
  // ... pattern matching code

  // 3. Extract base pointers
  Ops.BaseA = traceToBasePointer(LoadA);
  Ops.BaseB = traceToBasePointer(LoadB);
  Ops.BaseC = traceToBasePointer(Store);

  // 4. Extract strides from GEP patterns
  Ops.StrideA = extractStride(GEPA);
  Ops.StrideB = extractStride(GEPB);
  Ops.StrideC = extractStride(GEPC);

  // 5. Check if both inputs are signed
  Ops.BothSigned = isSigned(ExtA) && isSigned(ExtB);

  return true;
}
```

### Phase 4: B Transpose Generation (~80 lines)

```cpp
void generateBTranspose(TiledMatmulLoops &Loops, MatmulOperands &Ops,
                        IRBuilder<> &Builder) {
  // Insert at outermost loop preheader
  BasicBlock *Preheader = Loops.IOuterLoop->getLoopPreheader();
  Builder.SetInsertPoint(Preheader->getTerminator());

  // Allocate B_T: N x K bytes
  Value *Size = Builder.getInt64(Loops.N * Loops.K);
  Ops.BaseB_T = Builder.CreateCall(MallocFn, {Size}, "B_T");

  // Generate transpose loops:
  // for j in 0..N:
  //   for k in 0..K:
  //     B_T[j*K + k] = B[k*N + j]

  // ... loop generation code
}
```

### Phase 5: VNNI Code Generation (~200 lines)

```cpp
void generateVNNILoop(TiledMatmulLoops &Loops, MatmulOperands &Ops,
                      IRBuilder<> &Builder) {
  // Insert point: KK loop preheader
  BasicBlock *KKPreheader = Loops.KKLoop->getLoopPreheader();

  // === Compute full indices (THE KEY FIX) ===
  // i_full = i_outer + ii
  Value *i_full = Builder.CreateAdd(Loops.i_outer, Loops.ii, "i.full");
  // j_full = j_outer + jj
  Value *j_full = Builder.CreateAdd(Loops.j_outer, Loops.jj, "j.full");
  // k_base = k_outer (added to VNNI loop index later)
  Value *k_base = Loops.k_outer;

  // === Compute row offsets ===
  // A: row = i_full, stride = K
  Value *RowOffA = Builder.CreateMul(i_full, Builder.getInt64(Ops.StrideA), "row.off.a");
  // B_T: row = j_full, stride = K
  Value *RowOffB = Builder.CreateMul(j_full, Builder.getInt64(Loops.K), "row.off.b");
  // C: row = i_full, stride = N, col = j_full
  Value *RowOffC = Builder.CreateMul(i_full, Builder.getInt64(Ops.StrideC), "row.off.c");
  Value *IdxC = Builder.CreateAdd(RowOffC, j_full, "idx.c");

  // === Create VNNI loop structure ===
  BasicBlock *VNNIHeader = BasicBlock::Create(Ctx, "vnni.hdr", &F);
  BasicBlock *VNNIBody = BasicBlock::Create(Ctx, "vnni.body", &F);
  BasicBlock *VNNIExit = BasicBlock::Create(Ctx, "vnni.exit", &F);

  // PHIs: k (0..TileSize step 64), accumulator
  PHINode *K = Builder.CreatePHI(I64Ty, 2, "k");
  PHINode *Acc = Builder.CreatePHI(V16I32Ty, 2, "acc");

  // === VNNI Body ===
  Builder.SetInsertPoint(VNNIBody);

  // k_full = k_base + k
  Value *k_full = Builder.CreateAdd(k_base, K, "k.full");

  // Load B_T[j_full * K + k_full] - 64 bytes contiguous
  Value *IdxB = Builder.CreateAdd(RowOffB, k_full);
  Value *PtrB = Builder.CreateGEP(I8Ty, Ops.BaseB_T, IdxB);
  Value *VecB = Builder.CreateLoad(V16I32Ty, PtrB);

  // Load A[i_full * K + k_full] - 64 bytes contiguous
  Value *IdxA = Builder.CreateAdd(RowOffA, k_full);
  Value *PtrA = Builder.CreateGEP(I8Ty, Ops.BaseA, IdxA);
  Value *VecA = Builder.CreateLoad(V16I32Ty, PtrA);

  // Handle signed: XOR with 0x80808080
  if (Ops.BothSigned) {
    VecA = Builder.CreateXor(VecA, SignFlipMask);
    // ... bias accumulation
  }

  // vpdpbusd(acc, a, b)
  Value *NewAcc = Builder.CreateCall(VPDPBUSD, {Acc, VecA, VecB});

  // Loop increment
  Value *NextK = Builder.CreateAdd(K, Builder.getInt64(64));

  // === VNNI Exit: horizontal sum and store ===
  Builder.SetInsertPoint(VNNIExit);
  Value *Sum = horizontalSum(Acc);  // Reduce 16xi32 to scalar

  // Store to C[i_full * N + j_full]
  Value *PtrC = Builder.CreateGEP(I32Ty, Ops.BaseC, IdxC);
  Builder.CreateStore(Sum, PtrC);
}
```

### Phase 6: Optional I=4 Tiling (~100 lines)

Process 4 rows of A simultaneously for better register utilization:
- Load B once, load 4 A rows
- 4 accumulators, 4 vpdpbusd calls
- 4 stores at the end

### Phase 7: Cleanup (~50 lines)

- Delete original KK loop
- Update loop info
- Free B_T at function exit

## File Structure

```
src/mlir/passes/VNNIPass.cpp  (~600 lines total, down from 1300)
├── Data structures (50 lines)
│   ├── TiledMatmulLoops
│   └── MatmulOperands
├── Analysis (150 lines)
│   ├── analyzeLoopNest()
│   ├── detectMatmulPattern()
│   └── helper functions
├── Code Generation (350 lines)
│   ├── generateBTranspose()
│   ├── generateVNNILoop()
│   ├── generateI4Tiling() [optional]
│   └── horizontalSum()
└── Pass infrastructure (50 lines)
    ├── runOnFunction()
    └── cleanup
```

## Implementation Order

1. **Phase 1**: Create new data structures, keep old code working
2. **Phase 2**: Implement loop analysis, verify it finds all 6 levels
3. **Phase 3**: Implement pattern detection, verify operands found
4. **Phase 4**: Implement B transpose (can test independently)
5. **Phase 5**: Implement basic VNNI codegen (no I=4 tiling)
6. **Phase 6**: Test correctness with small matrices
7. **Phase 7**: Add I=4 tiling optimization
8. **Phase 8**: Performance testing

## Key Invariants to Maintain

1. **All 6 loop indices must be tracked**: i_outer, j_outer, k_outer, ii, jj, kk
2. **Full indices computed before use**:
   - `i_full = i_outer + ii` for A row and C row
   - `j_full = j_outer + jj` for B_T row and C col
   - `k_full = k_outer + k` for A col and B_T col
3. **B transpose layout**: `B_T[j,k] = B[k,j]`, stored as `B_T[j*K + k]`
4. **Signed handling**: XOR with 0x80808080, accumulate bias correction

## Testing Strategy

1. **Unit test**: 4x4 matrix, verify each element
2. **Tile boundary**: 16x16, 32x32 (single tile, multiple tiles)
3. **Non-tile-aligned**: 17x17, 33x33
4. **Large matrices**: 256x256, 1024x1024
5. **Compare with**: scalar reference implementation
