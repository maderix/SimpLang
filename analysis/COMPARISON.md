# VNNI Pass Comparison: LLVM 14 vs LLVM 21

## File Sizes
- LLVM 14: 1047 lines (41KB) - **Full implementation**
- LLVM 21: 388 lines (13KB) - **Pattern detection + hints only**

## Key Differences

### 1. LLVM 14: Actual VNNI Intrinsic Emission
```cpp
Function *VPDPBUSD = Intrinsic::getDeclaration(M, Intrinsic::x86_avx512_vpdpbusd_512);
Value *NewAcc0 = Builder.CreateCall(VPDPBUSD, {VecAcc0, VecA0, VecB});
```

### 2. LLVM 14: I=4 Tiling (4 rows processed per iteration)
- Creates 4 accumulators (acc0-acc3)
- Loads B once, loads 4 A rows
- Computes 4 dot products per iteration
- Stores 4 results (Result0-Result3)
- Modifies I loop step from 1 to 4

### 3. LLVM 14: Full Loop Transformation
- Detects K/J/I loop hierarchy
- Creates new VNNI loop blocks (vnni.hdr, vnni.body, vnni.exit)
- Deletes original scalar loop
- Rewires control flow

### 4. LLVM 14: Signed×Signed Handling
```cpp
if (C.BothSigned) {
    Value *SignFlip = ConstantVector::getSplat(ElementCount::getFixed(16),
                        ConstantInt::get(I32Ty, 0x80808080));
    VecA0 = Builder.CreateXor(VecA0, SignFlip);  // Convert signed to unsigned
    // ... accumulate bias for correction
    Result0 = Builder.CreateSub(Result0, Correction);  // Apply bias correction
}
```

### 5. LLVM 14: Horizontal Reduction
```cpp
auto hreduce = [&](Value *Vec) -> Value* {
    for (int W = 8; W >= 1; W /= 2) {
        SmallVector<int, 16> Mask;
        for (int i = 0; i < 16; i++) Mask.push_back((i + W) % 16);
        Vec = Builder.CreateAdd(Vec, Builder.CreateShuffleVector(Vec, Vec, Mask));
    }
    return Builder.CreateExtractElement(Vec, (uint64_t)0);
};
```

### 6. LLVM 21: Only Metadata Hints
```cpp
MDNode *UnrollMD = MDNode::get(Ctx, {
    MDString::get(Ctx, "llvm.loop.unroll.count"),
    ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx), 4))
});
MDNode *VectorizeMD = MDNode::get(Ctx, {
    MDString::get(Ctx, "llvm.loop.vectorize.enable"),
    ConstantAsMetadata::get(ConstantInt::get(Type::getInt1Ty(Ctx), 1))
});
```

## Performance Impact
| Size      | LLVM 14 (vs VNNI) | LLVM 21 (vs VNNI) |
|-----------|-------------------|-------------------|
| 2048×2048 | **99.2%**         | 5.8%              |
| 1024×1024 | **91.7%**         | 6.6%              |
| 512×512   | **78.8%**         | 6.2%              |
| 256×256   | **61.2%**         | 7.7%              |

## What Needs Porting to LLVM 21
1. `transformToVNNI()` - Full loop transformation with VNNI intrinsics
2. `deleteLoop()` - Old loop removal
3. `modifyParentLoopStep()` - I loop step modification (1→4)
4. `hreduce` - Horizontal vector reduction
5. Signed bias correction logic
6. API changes for LLVM 21 (Triple.str(), CreateLoad/CreateStore with types, etc.)
