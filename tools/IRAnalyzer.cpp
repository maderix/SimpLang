/**
 * IRAnalyzer - LLVM-based hierarchical IR structure analyzer
 *
 * Uses LLVM's LoopInfo, ScalarEvolution, and other analyses to provide
 * accurate loop hierarchy, bounds, and key operation summaries.
 *
 * Enhanced features:
 *   - Memory access pattern analysis (shows which loop indices are used)
 *   - Missing index detection (warns when loop var not used in memory ops)
 *   - GEP decomposition (shows base + offset breakdown)
 *
 * Build:
 *   clang++ -O2 IRAnalyzer.cpp -o ir_analyzer \
 *     $(llvm-config --cxxflags --ldflags --libs core analysis passes support irreader) \
 *     -lLLVM
 *
 * Usage:
 *   ./ir_analyzer input.ll [function_name]
 */

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Passes/PassBuilder.h"
#include <map>
#include <set>

using namespace llvm;

cl::opt<std::string> InputFilename(cl::Positional, cl::desc("<input.ll>"), cl::Required);
cl::opt<std::string> FunctionName(cl::Positional, cl::desc("[function_name]"), cl::Optional);
cl::opt<bool> Verbose("v", cl::desc("Verbose mode - show memory access details"));

// ANSI colors for output
namespace Color {
  const char* Reset = "\033[0m";
  const char* Bold = "\033[1m";
  const char* Red = "\033[31m";
  const char* Green = "\033[32m";
  const char* Yellow = "\033[33m";
  const char* Blue = "\033[34m";
  const char* Magenta = "\033[35m";
  const char* Cyan = "\033[36m";
  const char* BoldRed = "\033[1;31m";
  const char* BoldYellow = "\033[1;33m";
}

struct BlockStats {
  int loads = 0;
  int stores = 0;
  int phis = 0;
  int adds = 0;
  int muls = 0;
  int exts = 0;
  int calls = 0;
  std::vector<std::string> callNames;
  std::vector<std::string> phiTypes;
  std::vector<std::string> phiNames;
};

// Track which loop indices are used in memory accesses
struct LoopIndexUsage {
  PHINode *IndVar;
  std::string Name;
  int Depth;
  bool UsedInLoads = false;
  bool UsedInStores = false;
};

// Check if a value depends on a specific PHI node
bool dependsOn(Value *V, PHINode *Phi, std::set<Value*> &Visited) {
  if (!V || Visited.count(V)) return false;
  Visited.insert(V);

  if (V == Phi) return true;

  if (auto *I = dyn_cast<Instruction>(V)) {
    for (Use &U : I->operands()) {
      if (dependsOn(U.get(), Phi, Visited)) return true;
    }
  }
  return false;
}

bool dependsOn(Value *V, PHINode *Phi) {
  std::set<Value*> Visited;
  return dependsOn(V, Phi, Visited);
}

// Get a short description of a GEP's index computation
std::string describeIndex(Value *V, std::map<PHINode*, std::string> &PhiNames) {
  std::string result;
  raw_string_ostream OS(result);

  if (auto *Phi = dyn_cast<PHINode>(V)) {
    if (PhiNames.count(Phi)) {
      OS << PhiNames[Phi];
      return result;
    }
  }

  if (auto *CI = dyn_cast<ConstantInt>(V)) {
    OS << CI->getSExtValue();
    return result;
  }

  if (auto *BO = dyn_cast<BinaryOperator>(V)) {
    std::string op0 = describeIndex(BO->getOperand(0), PhiNames);
    std::string op1 = describeIndex(BO->getOperand(1), PhiNames);

    switch (BO->getOpcode()) {
      case Instruction::Add: OS << "(" << op0 << " + " << op1 << ")"; break;
      case Instruction::Mul: OS << op0 << "*" << op1; break;
      case Instruction::Sub: OS << "(" << op0 << " - " << op1 << ")"; break;
      default: OS << "?"; break;
    }
    return result;
  }

  if (V->hasName()) {
    OS << "%" << V->getName().str();
  } else {
    OS << "?";
  }
  return result;
}

BlockStats analyzeBlock(BasicBlock &BB) {
  BlockStats stats;
  for (Instruction &I : BB) {
    if (auto *Phi = dyn_cast<PHINode>(&I)) {
      stats.phis++;
      stats.phiTypes.push_back(I.getType()->isIntegerTy(64) ? "i64" :
                               I.getType()->isIntegerTy(32) ? "i32" :
                               I.getType()->isVectorTy() ? "vec" : "?");
      if (Phi->hasName()) {
        stats.phiNames.push_back(Phi->getName().str());
      } else {
        stats.phiNames.push_back("");
      }
    } else if (isa<LoadInst>(I)) {
      stats.loads++;
    } else if (isa<StoreInst>(I)) {
      stats.stores++;
    } else if (isa<BinaryOperator>(I)) {
      auto *BO = cast<BinaryOperator>(&I);
      if (BO->getOpcode() == Instruction::Add) stats.adds++;
      else if (BO->getOpcode() == Instruction::Mul) stats.muls++;
    } else if (isa<SExtInst>(I) || isa<ZExtInst>(I)) {
      stats.exts++;
    } else if (auto *CI = dyn_cast<CallInst>(&I)) {
      stats.calls++;
      if (Function *F = CI->getCalledFunction()) {
        stats.callNames.push_back(F->getName().str());
      }
    }
  }
  return stats;
}

std::string getLoopBounds(Loop *L, ScalarEvolution &SE) {
  std::string result;
  raw_string_ostream OS(result);

  if (auto *TC = SE.getBackedgeTakenCount(L)) {
    if (auto *Const = dyn_cast<SCEVConstant>(TC)) {
      OS << "trip=" << Const->getValue()->getSExtValue() + 1;
    } else {
      OS << "trip=?";
    }
  }

  // Try to get step
  if (PHINode *IndVar = L->getInductionVariable(SE)) {
    if (auto *AR = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(IndVar))) {
      if (auto *Step = dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE))) {
        int64_t step = Step->getValue()->getSExtValue();
        if (step != 1) {
          OS << " step=" << step;
        }
      }
    }
  }

  return result;
}

// Analyze memory accesses in a loop and check which indices are used
void analyzeMemoryAccesses(Loop *L, ScalarEvolution &SE,
                           std::vector<LoopIndexUsage> &AllIndices,
                           std::map<PHINode*, std::string> &PhiNames,
                           int depth) {
  std::string indent(depth * 2, ' ');

  // Collect all loads and stores in this loop (not subloops)
  std::set<BasicBlock*> subLoopBlocks;
  for (Loop *SubL : L->getSubLoops()) {
    for (BasicBlock *BB : SubL->blocks()) {
      subLoopBlocks.insert(BB);
    }
  }

  std::vector<LoadInst*> loads;
  std::vector<StoreInst*> stores;

  for (BasicBlock *BB : L->blocks()) {
    if (subLoopBlocks.count(BB)) continue;
    for (Instruction &I : *BB) {
      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        loads.push_back(LI);
      } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
        stores.push_back(SI);
      }
    }
  }

  if (Verbose && (loads.size() > 0 || stores.size() > 0)) {
    outs() << indent << "  " << Color::Blue << "Memory Access Analysis:" << Color::Reset << "\n";

    for (LoadInst *LI : loads) {
      Value *Ptr = LI->getPointerOperand();
      if (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
        std::string idx = describeIndex(GEP->getOperand(GEP->getNumOperands() - 1), PhiNames);
        outs() << indent << "    LOAD[" << idx << "]\n";

        // Check which loop indices this depends on
        for (auto &Usage : AllIndices) {
          if (dependsOn(GEP, Usage.IndVar)) {
            Usage.UsedInLoads = true;
          }
        }
      }
    }

    for (StoreInst *SI : stores) {
      Value *Ptr = SI->getPointerOperand();
      if (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
        std::string idx = describeIndex(GEP->getOperand(GEP->getNumOperands() - 1), PhiNames);
        outs() << indent << "    STORE[" << idx << "]\n";

        // Check which loop indices this depends on
        for (auto &Usage : AllIndices) {
          if (dependsOn(GEP, Usage.IndVar)) {
            Usage.UsedInStores = true;
          }
        }
      }
    }
  } else {
    // Still track usage even in non-verbose mode
    for (LoadInst *LI : loads) {
      Value *Ptr = LI->getPointerOperand();
      for (auto &Usage : AllIndices) {
        if (dependsOn(Ptr, Usage.IndVar)) {
          Usage.UsedInLoads = true;
        }
      }
    }
    for (StoreInst *SI : stores) {
      Value *Ptr = SI->getPointerOperand();
      for (auto &Usage : AllIndices) {
        if (dependsOn(Ptr, Usage.IndVar)) {
          Usage.UsedInStores = true;
        }
      }
    }
  }
}

void printLoopHierarchy(Loop *L, ScalarEvolution &SE,
                        std::vector<LoopIndexUsage> &AllIndices,
                        std::map<PHINode*, std::string> &PhiNames,
                        int depth = 0) {
  std::string indent(depth * 2, ' ');
  BasicBlock *Header = L->getHeader();

  // Loop header info
  outs() << indent << Color::Cyan << "[LOOP] " << Color::Reset
         << Header->getName();

  std::string bounds = getLoopBounds(L, SE);
  if (!bounds.empty()) {
    outs() << Color::Yellow << " [" << bounds << "]" << Color::Reset;
  }
  outs() << "\n";

  // Track ALL i64 PHIs in header as potential loop indices
  // (not just canonical induction variables which LLVM often can't detect)
  int phiIdx = 0;
  for (PHINode &Phi : Header->phis()) {
    if (!Phi.getType()->isIntegerTy(64)) continue;

    std::string varName;
    if (Phi.hasName()) {
      varName = Phi.getName().str();
    } else {
      varName = "L" + std::to_string(depth) + "_" + std::to_string(phiIdx);
    }
    PhiNames[&Phi] = varName;
    phiIdx++;

    // Show this PHI
    outs() << indent << "  " << Color::Blue << "Idx: " << varName << Color::Reset;

    // Show SCEV if available
    const SCEV *S = SE.getSCEV(&Phi);
    if (auto *AR = dyn_cast<SCEVAddRecExpr>(S)) {
      if (auto *Start = dyn_cast<SCEVConstant>(AR->getStart())) {
        outs() << " [" << Start->getValue()->getSExtValue() << "..";
        if (auto *Step = dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE))) {
          int64_t step = Step->getValue()->getSExtValue();
          if (step != 1) {
            outs() << " step=" << step;
          }
        }
        outs() << "]";
      }
    }
    outs() << "\n";

    // Track this index
    LoopIndexUsage usage;
    usage.IndVar = &Phi;
    usage.Name = varName;
    usage.Depth = depth;
    AllIndices.push_back(usage);
  }

  // Analyze blocks in loop
  std::set<BasicBlock*> loopBlocks(L->block_begin(), L->block_end());
  std::set<BasicBlock*> subLoopBlocks;
  for (Loop *SubL : L->getSubLoops()) {
    for (BasicBlock *BB : SubL->blocks()) {
      subLoopBlocks.insert(BB);
    }
  }

  // Print non-subloop blocks
  for (BasicBlock *BB : L->blocks()) {
    if (subLoopBlocks.count(BB)) continue;  // Skip subloop blocks

    BlockStats stats = analyzeBlock(*BB);
    std::string blockIndent = indent + "  ";

    // Only print interesting blocks
    if (stats.loads + stats.stores + stats.calls > 0 ||
        (BB == Header && stats.phis > 0)) {

      if (BB == Header) {
        // Show PHI details
        if (stats.phis > 0 && Verbose) {
          outs() << blockIndent << "  PHI: ";
          for (size_t i = 0; i < stats.phiTypes.size(); i++) {
            if (i > 0) outs() << ", ";
            if (!stats.phiNames[i].empty()) {
              outs() << stats.phiNames[i] << ":";
            }
            outs() << stats.phiTypes[i];
          }
          outs() << "\n";
        }
      } else {
        outs() << blockIndent << BB->getName() << ":\n";
      }

      std::string detailIndent = blockIndent + "  ";
      if (stats.loads > 0)
        outs() << detailIndent << Color::Green << "LOAD: " << stats.loads << "x" << Color::Reset << "\n";
      if (stats.stores > 0)
        outs() << detailIndent << Color::Red << "STORE: " << stats.stores << "x" << Color::Reset << "\n";
      if (stats.calls > 0) {
        for (const auto &name : stats.callNames) {
          if (name.find("vpdpbusd") != std::string::npos) {
            outs() << detailIndent << Color::Magenta << "VNNI: vpdpbusd" << Color::Reset << "\n";
          } else if (name.find("malloc") != std::string::npos) {
            outs() << detailIndent << "CALL: malloc\n";
          } else if (name.find("free") != std::string::npos) {
            outs() << detailIndent << "CALL: free\n";
          } else {
            outs() << detailIndent << "CALL: " << name << "\n";
          }
        }
      }
      if (stats.muls + stats.adds > 0) {
        outs() << detailIndent << "ARITH: mul:" << stats.muls << " add:" << stats.adds;
        if (stats.exts > 0) outs() << " ext:" << stats.exts;
        outs() << "\n";
      }
    }
  }

  // Analyze memory accesses for this loop level
  analyzeMemoryAccesses(L, SE, AllIndices, PhiNames, depth);

  // Print subloops
  for (Loop *SubL : L->getSubLoops()) {
    printLoopHierarchy(SubL, SE, AllIndices, PhiNames, depth + 1);
  }
}

void analyzeFunction(Function &F) {
  outs() << Color::Bold << "\n========================================\n"
         << "FUNCTION: " << F.getName() << "\n"
         << "========================================" << Color::Reset << "\n";

  // Set up analysis managers
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  PassBuilder PB;
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  // Get analyses
  DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);
  ScalarEvolution &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);

  // Find pre-loop blocks (init phase)
  std::set<BasicBlock*> inLoop;
  for (Loop *L : LI) {
    for (BasicBlock *BB : L->blocks()) {
      inLoop.insert(BB);
    }
  }

  outs() << "\n" << Color::Yellow << "=== PRE-LOOP PHASE ===" << Color::Reset << "\n";
  for (BasicBlock &BB : F) {
    if (inLoop.count(&BB)) continue;

    BlockStats stats = analyzeBlock(BB);
    if (stats.loads + stats.stores + stats.calls > 0) {
      outs() << "  " << BB.getName() << ":\n";
      if (stats.calls > 0) {
        for (const auto &name : stats.callNames) {
          outs() << "    CALL: " << name << "\n";
        }
      }
      if (stats.loads > 0) outs() << "    LOAD: " << stats.loads << "x\n";
      if (stats.stores > 0) outs() << "    STORE: " << stats.stores << "x\n";
    }
  }

  outs() << "\n" << Color::Yellow << "=== LOOP HIERARCHY ===" << Color::Reset << "\n";

  std::vector<LoopIndexUsage> AllIndices;
  std::map<PHINode*, std::string> PhiNames;

  for (Loop *L : LI) {
    AllIndices.clear();
    PhiNames.clear();
    printLoopHierarchy(L, SE, AllIndices, PhiNames);

    // Check for unused indices - potential bugs!
    bool hasWarnings = false;
    for (auto &Usage : AllIndices) {
      if (!Usage.UsedInLoads && !Usage.UsedInStores) {
        if (!hasWarnings) {
          outs() << "\n" << Color::BoldYellow << "  ⚠ INDEX USAGE WARNINGS:" << Color::Reset << "\n";
          hasWarnings = true;
        }
        outs() << "    " << Color::BoldRed << "• " << Usage.Name
               << " (depth " << Usage.Depth << ") NOT USED in memory accesses!"
               << Color::Reset << "\n";
      }
    }
  }

  // Summary
  outs() << "\n" << Color::Yellow << "=== SUMMARY ===" << Color::Reset << "\n";
  outs() << "  Top-level loops: " << std::distance(LI.begin(), LI.end()) << "\n";

  int maxDepth = 0;
  std::function<void(Loop*, int)> countDepth = [&](Loop *L, int d) {
    maxDepth = std::max(maxDepth, d);
    for (Loop *Sub : L->getSubLoops()) countDepth(Sub, d + 1);
  };
  for (Loop *L : LI) countDepth(L, 1);
  outs() << "  Max loop depth: " << maxDepth << "\n";
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "LLVM IR Hierarchical Analyzer\n");

  LLVMContext Context;
  SMDiagnostic Err;

  std::unique_ptr<Module> M = parseIRFile(InputFilename, Err, Context);
  if (!M) {
    Err.print(argv[0], errs());
    return 1;
  }

  outs() << "Analyzing: " << InputFilename << "\n";
  outs() << "Module: " << M->getName() << " (" << M->size() << " functions)\n";
  if (Verbose) {
    outs() << Color::Cyan << "(Verbose mode enabled)" << Color::Reset << "\n";
  }

  if (!FunctionName.empty()) {
    Function *F = M->getFunction(FunctionName);
    if (!F) {
      errs() << "Function '" << FunctionName << "' not found.\n";
      errs() << "Available functions:\n";
      for (Function &Fn : *M) {
        if (!Fn.isDeclaration()) {
          errs() << "  - " << Fn.getName() << "\n";
        }
      }
      return 1;
    }
    analyzeFunction(*F);
  } else {
    for (Function &F : *M) {
      if (!F.isDeclaration() && F.size() > 3) {  // Skip trivial functions
        analyzeFunction(F);
      }
    }
  }

  return 0;
}
