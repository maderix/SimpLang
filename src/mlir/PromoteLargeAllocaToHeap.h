#ifndef PROMOTE_LARGE_ALLOCA_TO_HEAP_H
#define PROMOTE_LARGE_ALLOCA_TO_HEAP_H

namespace llvm {
class FunctionPass;

/// Create a pass that promotes large stack allocations (alloca) to heap
/// allocations (malloc/free). This prevents stack overflow when MLIR's
/// tiling creates large temporary buffers.
///
/// Allocations larger than 8KB are promoted to heap.
FunctionPass *createPromoteLargeAllocaToHeapPass();

} // namespace llvm

#endif // PROMOTE_LARGE_ALLOCA_TO_HEAP_H
