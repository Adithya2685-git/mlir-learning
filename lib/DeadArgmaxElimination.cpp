//===----------------------------------------------------------------------===//
// DeadArgmaxElimination.cpp
//
// Runs liveness analysis, then eliminates dead linalg.generic results.
// GPT-2 has 12 linalg.generic ops where result #1 (softmax argmax) is dead.
// Replaces each with a single-result version yielding only the live result.
//
// Run: tutorial-opt --eliminate-dead-argmax gpt2_linalg.mlir
//===----------------------------------------------------------------------===//

#include "lib/DeadArgmaxElimination.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/LivenessAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::dataflow;

namespace mlir {
namespace tutorial {

void DeadArgmaxEliminationPass::runOnOperation() {
  ModuleOp module = getOperation();
  auto func = *module.getOps<func::FuncOp>().begin();

  // ---- Step 1: Run dataflow liveness --------------------------------------
  SymbolTableCollection symbolTable;
  DataFlowSolver solver;
  solver.load<DeadCodeAnalysis>();
  solver.load<LivenessAnalysis>(symbolTable);

  if (failed(solver.initializeAndRun(func))) {
    llvm::errs() << "Dataflow solver failed\n";
    signalPassFailure();
    return;
  }

  // ---- Step 2: Find generics with dead results ----------------------------
  struct ToFix {
    linalg::GenericOp op;
    int deadResultIdx; // which result index is dead
  };
  SmallVector<ToFix> fixList;

  func.walk([&](linalg::GenericOp generic) {
    if (generic.getNumResults() < 2)
      return;

    for (auto [idx, result] : llvm::enumerate(generic.getResults())) {
      auto *lattice = solver.lookupState<Liveness>(result);
      bool dead = (!lattice || !lattice->isLive);
      if (dead && !result.use_empty()) {
        llvm::errs() << "WARNING: dead result has uses, skipping\n";
        continue;
      }
      if (dead)
        fixList.push_back({generic, (int)idx});
    }
  });

  llvm::outs() << "Found " << fixList.size()
               << " dead results in linalg.generic ops\n";

  if (fixList.empty()) {
    llvm::outs() << "Nothing to eliminate.\n";
    return;
  }

  // ---- Step 3: Eliminate dead results -------------------------------------
  for (auto &fix : fixList) {
    linalg::GenericOp oldOp = fix.op;
    int deadIdx = fix.deadResultIdx;
    unsigned numResults = oldOp.getNumResults();

    // If removing this result would leave zero results, skip
    if (numResults <= 1) {
      llvm::errs() << "WARNING: skipping single-result op, deadIdx="
                   << deadIdx << "\n";
      continue;
    }

    // Build new op: same inputs, same outputs minus dead one
    SmallVector<Value> newInputs(oldOp.getInputs());
    SmallVector<Value> newOutputs;
    SmallVector<Type> newResultTypes;
    SmallVector<AffineMap> newIndexingMaps;

    for (unsigned i = 0; i < numResults; i++) {
      if ((int)i != deadIdx) {
        newOutputs.push_back(oldOp.getOutputs()[i]);
        newResultTypes.push_back(oldOp.getResult(i).getType());
      }
    }

    // Build indexing maps: input maps stay, output maps skip dead result
    SmallVector<AffineMap> allMaps = oldOp.getIndexingMapsArray();
    unsigned numInputs = oldOp.getNumDpsInputs();
    // Input maps: first numInputs entries
    for (unsigned i = 0; i < numInputs; i++)
      newIndexingMaps.push_back(allMaps[i]);
    // Output maps: entries after numInputs, skip deadIdx
    for (unsigned i = 0; i < numResults; i++) {
      if ((int)i != deadIdx)
        newIndexingMaps.push_back(allMaps[numInputs + i]);
    }

    SmallVector<utils::IteratorType> newIteratorTypes;
    for (auto it : oldOp.getIteratorTypesArray())
      newIteratorTypes.push_back(it);

    // Create new op
    OpBuilder builder(oldOp);
    auto newOp = builder.create<linalg::GenericOp>(
        oldOp.getLoc(), newResultTypes, newInputs, newOutputs, newIndexingMaps,
        newIteratorTypes);

    // Move the region body from old op to new op
    newOp.getRegion().takeBody(oldOp.getRegion());
    Block &block = newOp.getRegion().front();

    // FIRST: remove the dead yield operand and get the value that was yielded
    auto yieldOp = cast<linalg::YieldOp>(block.getTerminator());
    Value deadYieldValue = yieldOp.getOperand(deadIdx);
    SmallVector<Value> newYieldOperands;
    for (unsigned i = 0; i < yieldOp.getNumOperands(); i++) {
      if ((int)i != deadIdx)
        newYieldOperands.push_back(yieldOp.getOperand(i));
    }
    yieldOp->setOperands(newYieldOperands);

    // Remove ops inside the region that ONLY contribute to the dead yield
    // Walk backward from the dead yield value, collect dead ops
    SmallVector<Operation *> deadInnerOps;
    SmallVector<Value> worklist = {deadYieldValue};
    SmallPtrSet<Value, 8> deadValues;
    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      deadValues.insert(v);
      // If this value is a block argument, stop (we'll erase the arg later)
      if (isa<BlockArgument>(v))
        continue;
      // If defined by an op, check if all its uses are dead
      Operation *defOp = v.getDefiningOp();
      if (!defOp)
        continue;
      bool allUsesDead = true;
      for (auto &use : defOp->getUses()) {
        if (!deadValues.contains(use.get()))
          allUsesDead = false;
      }
      if (allUsesDead) {
        deadInnerOps.push_back(defOp);
        for (Value operand : defOp->getOperands())
          worklist.push_back(operand);
      }
    }
    // Erase dead inner ops in insertion order (consumers before producers,
    // which is reverse of def-before-use — this is correct for erasure)
    for (Operation *op : deadInnerOps)
      op->erase();

    // THEN: erase the dead block argument (now has no uses)
    block.eraseArgument(numInputs + deadIdx);

    // Replace uses of live old results with new results
    for (unsigned i = 0, newIdx = 0; i < numResults; i++) {
      if ((int)i != deadIdx) {
        oldOp.getResult(i).replaceAllUsesWith(newOp.getResult(newIdx));
        newIdx++;
      }
    }

    // Erase old op
    oldOp.erase();
  }

  llvm::outs() << "Eliminated " << fixList.size() << " dead result(s)\n";
}

} // namespace tutorial
} // namespace mlir
