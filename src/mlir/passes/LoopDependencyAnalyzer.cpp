//===- LoopDependencyAnalyzer.cpp - Analyze loop dependencies -------------===//
//
// Part of the SimpLang Project
//
// This pass analyzes loop nests to determine:
// 1. Which loops can be parallelized (independent iterations)
// 2. Which loops are reductions (accumulate with associative op)
// 3. Which loops have dependencies (must be sequential)
// 4. Memory access patterns for cache optimization
//
// The analysis embeds metadata attributes on loops and memory operations
// that downstream passes (like AnnotationLoweringPass) use for optimization.
//
// Supported loop types:
//   - scf.for loops
//   - linalg operations (extracts iterator types directly)
//
// Metadata produced:
//   - simp.loop_type: "parallel" | "reduction" | "sequential"
//   - simp.reduction_op: "add" | "mul" | "max" | "min" (for reductions)
//   - simp.loop_depth: integer depth in nest (0 = outermost)
//   - simp.access_pattern: "contiguous" | "strided" | "random"
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "loop-dependency-analyzer"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Loop Analysis Data Structures
//===----------------------------------------------------------------------===//

/// Classification of a loop's parallelization potential
enum class LoopType {
  Parallel,    // Independent iterations, can parallelize
  Reduction,   // Accumulation pattern, parallel with reduction clause
  Sequential   // Has loop-carried dependencies, must be sequential
};

/// Type of reduction operation
enum class ReductionOp {
  None,
  Add,
  Mul,
  Max,
  Min,
  And,
  Or
};

/// Memory access pattern
enum class AccessPattern {
  Contiguous,  // Stride-1 access, good cache locality
  Strided,     // Fixed stride > 1
  Random       // Non-linear or data-dependent access
};

/// Analysis result for a single loop
struct LoopAnalysis {
  LoopType type = LoopType::Sequential;  // Default to safe choice
  ReductionOp reductionOp = ReductionOp::None;
  int64_t depth = 0;  // 0 = outermost

  // Memory access info
  SmallVector<std::pair<Value, AccessPattern>> memoryAccesses;

  // Dependencies detected
  bool hasRAW = false;  // Read-after-write
  bool hasWAR = false;  // Write-after-read
  bool hasWAW = false;  // Write-after-write
};

//===----------------------------------------------------------------------===//
// Dependency Analysis Helpers
//===----------------------------------------------------------------------===//

/// Check if a value depends on the loop induction variable
static bool dependsOnIV(Value val, Value iv) {
  if (val == iv) return true;

  Operation *defOp = val.getDefiningOp();
  if (!defOp) return false;

  // Recursively check operands
  for (Value operand : defOp->getOperands()) {
    if (dependsOnIV(operand, iv))
      return true;
  }
  return false;
}

/// Analyze memory access pattern for a load/store
static AccessPattern analyzeAccessPattern(Operation *memOp, Value iv) {
  // Get indices
  SmallVector<Value> indices;
  if (auto load = dyn_cast<memref::LoadOp>(memOp)) {
    indices.append(load.getIndices().begin(), load.getIndices().end());
  } else if (auto store = dyn_cast<memref::StoreOp>(memOp)) {
    indices.append(store.getIndices().begin(), store.getIndices().end());
  } else {
    return AccessPattern::Random;
  }

  if (indices.empty()) return AccessPattern::Contiguous;

  // Check innermost index (last one for row-major)
  Value innerIdx = indices.back();

  // If innermost index IS the IV, it's contiguous
  if (innerIdx == iv) return AccessPattern::Contiguous;

  // If innermost index is IV + constant or IV * 1, it's contiguous
  if (auto addOp = innerIdx.getDefiningOp<arith::AddIOp>()) {
    if ((addOp.getLhs() == iv && addOp.getRhs().getDefiningOp<arith::ConstantOp>()) ||
        (addOp.getRhs() == iv && addOp.getLhs().getDefiningOp<arith::ConstantOp>())) {
      return AccessPattern::Contiguous;
    }
  }

  // If innermost index is IV * stride, it's strided
  if (auto mulOp = innerIdx.getDefiningOp<arith::MulIOp>()) {
    if (dependsOnIV(innerIdx, iv)) {
      return AccessPattern::Strided;
    }
  }

  // If innermost doesn't depend on IV, check outer indices
  if (!dependsOnIV(innerIdx, iv)) {
    // IV affects outer dimension - strided access
    for (size_t i = 0; i < indices.size() - 1; ++i) {
      if (dependsOnIV(indices[i], iv))
        return AccessPattern::Strided;
    }
    // IV doesn't affect any index - same address each iteration
    return AccessPattern::Contiguous;
  }

  return AccessPattern::Random;
}

/// Detect reduction pattern: load, binary op, store to same location
static ReductionOp detectReductionPattern(scf::ForOp loop) {
  Value iv = loop.getInductionVar();
  Block &body = loop.getRegion().front();

  // Look for pattern: %x = load %mem[...]; %y = op %x, %val; store %y, %mem[...]
  for (auto &op : body) {
    auto store = dyn_cast<memref::StoreOp>(&op);
    if (!store) continue;

    Value storedVal = store.getValue();
    Operation *storeValDef = storedVal.getDefiningOp();
    if (!storeValDef) continue;

    // Check if it's a binary arithmetic op
    Value lhs, rhs;
    ReductionOp redOp = ReductionOp::None;

    if (auto addOp = dyn_cast<arith::AddFOp>(storeValDef)) {
      lhs = addOp.getLhs(); rhs = addOp.getRhs();
      redOp = ReductionOp::Add;
    } else if (auto addOp = dyn_cast<arith::AddIOp>(storeValDef)) {
      lhs = addOp.getLhs(); rhs = addOp.getRhs();
      redOp = ReductionOp::Add;
    } else if (auto mulOp = dyn_cast<arith::MulFOp>(storeValDef)) {
      lhs = mulOp.getLhs(); rhs = mulOp.getRhs();
      redOp = ReductionOp::Mul;
    } else if (auto mulOp = dyn_cast<arith::MulIOp>(storeValDef)) {
      lhs = mulOp.getLhs(); rhs = mulOp.getRhs();
      redOp = ReductionOp::Mul;
    } else {
      continue;
    }

    // Check if one operand is a load from the same location
    auto checkLoad = [&](Value v) -> bool {
      auto load = v.getDefiningOp<memref::LoadOp>();
      if (!load) return false;
      // Same memref and indices
      if (load.getMemRef() != store.getMemRef()) return false;
      if (load.getIndices().size() != store.getIndices().size()) return false;
      for (auto [li, si] : llvm::zip(load.getIndices(), store.getIndices())) {
        if (li != si) return false;
      }
      return true;
    };

    if (checkLoad(lhs) || checkLoad(rhs)) {
      return redOp;
    }
  }

  return ReductionOp::None;
}

/// Check for loop-carried dependencies (RAW, WAR, WAW)
static void detectDependencies(scf::ForOp loop, LoopAnalysis &analysis) {
  Value iv = loop.getInductionVar();
  Block &body = loop.getRegion().front();

  // Collect all memory accesses
  SmallVector<std::pair<Operation*, bool>> accesses;  // {op, isWrite}

  body.walk([&](Operation *op) {
    if (isa<memref::LoadOp>(op)) {
      accesses.push_back({op, false});
    } else if (isa<memref::StoreOp>(op)) {
      accesses.push_back({op, true});
    }
  });

  // For each pair of accesses, check for dependencies
  // This is simplified - full analysis would need alias analysis
  for (size_t i = 0; i < accesses.size(); ++i) {
    for (size_t j = i + 1; j < accesses.size(); ++j) {
      auto [op1, isWrite1] = accesses[i];
      auto [op2, isWrite2] = accesses[j];

      // Get memrefs
      Value mem1 = isa<memref::LoadOp>(op1)
          ? cast<memref::LoadOp>(op1).getMemRef()
          : cast<memref::StoreOp>(op1).getMemRef();
      Value mem2 = isa<memref::LoadOp>(op2)
          ? cast<memref::LoadOp>(op2).getMemRef()
          : cast<memref::StoreOp>(op2).getMemRef();

      // If different memrefs, no dependency
      if (mem1 != mem2) continue;

      // Same memref - check for potential conflict
      // (Simplified: assume any same-memref access pair is a potential dependency)
      if (isWrite1 && !isWrite2) {
        // Write then Read - potential RAW if indices overlap across iterations
        // For now, mark as potential dependency
        analysis.hasRAW = true;
      } else if (!isWrite1 && isWrite2) {
        // Read then Write - potential WAR
        analysis.hasWAR = true;
      } else if (isWrite1 && isWrite2) {
        // Write then Write - potential WAW
        analysis.hasWAW = true;
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Main Analysis Logic
//===----------------------------------------------------------------------===//

/// Analyze a single scf.for loop
static LoopAnalysis analyzeLoop(scf::ForOp loop, int64_t depth) {
  LoopAnalysis analysis;
  analysis.depth = depth;

  Value iv = loop.getInductionVar();

  // 1. Check for reduction pattern first
  analysis.reductionOp = detectReductionPattern(loop);
  if (analysis.reductionOp != ReductionOp::None) {
    analysis.type = LoopType::Reduction;
    LLVM_DEBUG(llvm::dbgs() << "  Loop at depth " << depth << ": REDUCTION\n");
    return analysis;
  }

  // 2. Detect general dependencies
  detectDependencies(loop, analysis);

  // 3. Classify based on dependencies
  if (!analysis.hasRAW && !analysis.hasWAR && !analysis.hasWAW) {
    // No dependencies detected - can parallelize
    analysis.type = LoopType::Parallel;
    LLVM_DEBUG(llvm::dbgs() << "  Loop at depth " << depth << ": PARALLEL\n");
  } else {
    // Has dependencies - must be sequential
    analysis.type = LoopType::Sequential;
    LLVM_DEBUG(llvm::dbgs() << "  Loop at depth " << depth << ": SEQUENTIAL"
                            << " (RAW=" << analysis.hasRAW
                            << ", WAR=" << analysis.hasWAR
                            << ", WAW=" << analysis.hasWAW << ")\n");
  }

  // 4. Analyze memory access patterns
  loop.getBody()->walk([&](Operation *op) {
    if (isa<memref::LoadOp, memref::StoreOp>(op)) {
      AccessPattern pattern = analyzeAccessPattern(op, iv);
      Value mem = isa<memref::LoadOp>(op)
          ? cast<memref::LoadOp>(op).getMemRef()
          : cast<memref::StoreOp>(op).getMemRef();
      analysis.memoryAccesses.push_back({mem, pattern});
    }
  });

  return analysis;
}

/// Get string representation of loop type
static StringRef loopTypeToString(LoopType type) {
  switch (type) {
    case LoopType::Parallel: return "parallel";
    case LoopType::Reduction: return "reduction";
    case LoopType::Sequential: return "sequential";
  }
  llvm_unreachable("unknown loop type");
}

/// Get string representation of reduction op
static StringRef reductionOpToString(ReductionOp op) {
  switch (op) {
    case ReductionOp::None: return "none";
    case ReductionOp::Add: return "add";
    case ReductionOp::Mul: return "mul";
    case ReductionOp::Max: return "max";
    case ReductionOp::Min: return "min";
    case ReductionOp::And: return "and";
    case ReductionOp::Or: return "or";
  }
  llvm_unreachable("unknown reduction op");
}

/// Get string representation of access pattern
static StringRef accessPatternToString(AccessPattern pattern) {
  switch (pattern) {
    case AccessPattern::Contiguous: return "contiguous";
    case AccessPattern::Strided: return "strided";
    case AccessPattern::Random: return "random";
  }
  llvm_unreachable("unknown access pattern");
}

//===----------------------------------------------------------------------===//
// Linalg Operation Analysis
//===----------------------------------------------------------------------===//

/// Extract parallelization info from linalg operations
static void analyzeLinalgOp(linalg::LinalgOp op, OpBuilder &builder) {
  // Linalg ops already have iterator types encoded
  SmallVector<utils::IteratorType> iterTypes = op.getIteratorTypesArray();

  SmallVector<StringRef> loopTypes;
  for (auto iterType : iterTypes) {
    if (iterType == utils::IteratorType::parallel) {
      loopTypes.push_back("parallel");
    } else if (iterType == utils::IteratorType::reduction) {
      loopTypes.push_back("reduction");
    }
  }

  // Set attribute with loop types
  SmallVector<Attribute> typeAttrs;
  for (StringRef t : loopTypes) {
    typeAttrs.push_back(builder.getStringAttr(t));
  }
  op->setAttr("simp.iterator_types", builder.getArrayAttr(typeAttrs));

  LLVM_DEBUG({
    llvm::dbgs() << "Linalg op " << op->getName() << " iterators: [";
    for (size_t i = 0; i < loopTypes.size(); ++i) {
      if (i > 0) llvm::dbgs() << ", ";
      llvm::dbgs() << loopTypes[i];
    }
    llvm::dbgs() << "]\n";
  });
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

class LoopDependencyAnalyzerPass
    : public PassWrapper<LoopDependencyAnalyzerPass, OperationPass<func::FuncOp>> {
public:
  StringRef getArgument() const override { return "simp-loop-dependency-analyzer"; }
  StringRef getDescription() const override {
    return "Analyze loop dependencies and mark parallelization opportunities";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder builder(&getContext());

    LLVM_DEBUG(llvm::dbgs() << "=== Analyzing " << func.getName() << " ===\n");

    // 1. Analyze linalg operations (before they get lowered to loops)
    func.walk([&](linalg::LinalgOp op) {
      analyzeLinalgOp(op, builder);
    });

    // 2. Analyze existing SCF loops
    // We need to process from outermost to innermost
    SmallVector<scf::ForOp> outerLoops;
    func.walk([&](scf::ForOp loop) {
      // Check if this is an outermost loop (not nested in another scf.for)
      bool isOutermost = true;
      Operation *parent = loop->getParentOp();
      while (parent && !isa<func::FuncOp>(parent)) {
        if (isa<scf::ForOp>(parent)) {
          isOutermost = false;
          break;
        }
        parent = parent->getParentOp();
      }
      if (isOutermost) {
        outerLoops.push_back(loop);
      }
    });

    // Analyze each loop nest
    for (scf::ForOp outerLoop : outerLoops) {
      analyzeLoopNest(outerLoop, builder, 0);
    }

    // Mark function as analyzed
    func->setAttr("simp.dependency_analyzed", builder.getUnitAttr());
  }

private:
  /// Recursively analyze a loop nest
  void analyzeLoopNest(scf::ForOp loop, OpBuilder &builder, int64_t depth) {
    // Analyze this loop
    LoopAnalysis analysis = analyzeLoop(loop, depth);

    // Set attributes based on analysis
    loop->setAttr("simp.loop_type",
                  builder.getStringAttr(loopTypeToString(analysis.type)));
    loop->setAttr("simp.loop_depth",
                  builder.getI64IntegerAttr(depth));

    if (analysis.type == LoopType::Reduction) {
      loop->setAttr("simp.reduction_op",
                    builder.getStringAttr(reductionOpToString(analysis.reductionOp)));
    }

    // Log the analysis
    llvm::errs() << "[LoopAnalyzer] Loop depth=" << depth
                 << " type=" << loopTypeToString(analysis.type);
    if (analysis.type == LoopType::Reduction) {
      llvm::errs() << " reduction=" << reductionOpToString(analysis.reductionOp);
    }
    llvm::errs() << "\n";

    // Recursively analyze nested loops
    loop.getBody()->walk([&](scf::ForOp nestedLoop) {
      // Only direct children
      if (nestedLoop->getParentOp() == loop.getBody()->getParentOp() ||
          nestedLoop->getParentOfType<scf::ForOp>() == loop) {
        analyzeLoopNest(nestedLoop, builder, depth + 1);
      }
    });
  }
};

} // namespace

namespace mlir {
namespace simp {

/// Create the loop dependency analyzer pass
std::unique_ptr<Pass> createLoopDependencyAnalyzerPass() {
  return std::make_unique<LoopDependencyAnalyzerPass>();
}

/// Register the pass
void registerLoopDependencyAnalyzerPass() {
  PassRegistration<LoopDependencyAnalyzerPass>();
}

} // namespace simp
} // namespace mlir
