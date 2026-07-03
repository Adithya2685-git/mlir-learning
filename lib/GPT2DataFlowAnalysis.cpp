//===----------------------------------------------------------------------===//
// GPT2DataFlowAnalysis.cpp — Sparse forward dataflow on GPT-2
//
// Run: tutorial-opt --gpt2-dataflow gpt2_linalg.mlir > /dev/null
//===----------------------------------------------------------------------===//

#include "lib/GPT2DataFlowAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/LivenessAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::dataflow;

namespace mlir {
namespace tutorial {

void GPT2DataFlowAnalysisPass::runOnOperation() {
  ModuleOp module = getOperation();
  auto func = *module.getOps<func::FuncOp>().begin();

  // ---- Run solver ----------------------------------------------------------
  SymbolTableCollection symbolTable;
  DataFlowSolver solver;
  solver.load<DeadCodeAnalysis>();
  solver.load<SparseConstantPropagation>();
  solver.load<IntegerRangeAnalysis>();
  solver.load<LivenessAnalysis>(symbolTable);

  if (failed(solver.initializeAndRun(func))) {
    llvm::errs() << "Dataflow solver failed\n";
    signalPassFailure();
    return;
  }

  llvm::outs() << "==================================================\n";
  llvm::outs() << "DEAD VALUE TRACKING — verifying 12 dead values\n";
  llvm::outs() << "==================================================\n\n";

  // ---- Track each dead value individually ---------------------------------
  struct DeadInfo {
    std::string opName;
    std::string loc;
    int resultNum;
    int totalUses;
    bool hasNoUses;
    std::string valueStr;
  };
  SmallVector<DeadInfo> deadValues;

  func.walk([&](Operation *op) {
    for (auto [idx, result] : llvm::enumerate(op->getResults())) {
      auto *lattice = solver.lookupState<Liveness>(result);
      bool isDead = (!lattice || !lattice->isLive);

      if (!isDead)
        continue;

      DeadInfo info;
      info.opName = op->getName().getStringRef().str();
      info.resultNum = idx;
      info.totalUses = 0;
      info.hasNoUses = result.use_empty();

      // Count uses
      for (auto &use : result.getUses()) {
        info.totalUses++;
        info.valueStr = "user: " + use.getOwner()->getName().getStringRef().str();
      }

      // Location
      std::string locStr;
      llvm::raw_string_ostream rso(locStr);
      op->getLoc().print(rso);
      info.loc = rso.str();

      deadValues.push_back(info);
    }
  });

  // ---- Print each dead value with full context -----------------------------
  llvm::outs() << "Found " << deadValues.size() << " dead values:\n\n";

  for (int i = 0; i < (int)deadValues.size(); i++) {
    auto &d = deadValues[i];
    llvm::outs() << "[" << i << "] " << d.opName << " result #" << d.resultNum << "\n";
    llvm::outs() << "    Location: " << d.loc << "\n";
    llvm::outs() << "    Uses:     " << d.totalUses << "\n";
    if (d.hasNoUses)
      llvm::outs() << "    STATUS:   DEAD — zero uses. Value is computed and discarded.\n";
    else
      llvm::outs() << "    STATUS:   Used by " << d.totalUses
                   << " consumer(s) but lattice says dead. " << d.valueStr << "\n";
    llvm::outs() << "\n";
  }

  // ---- Now trace each dead value: walk its parents to find root cause ------
  llvm::outs() << "==================================================\n";
  llvm::outs() << "ROOT CAUSE ANALYSIS — why each value is dead\n";
  llvm::outs() << "==================================================\n\n";

  for (int i = 0; i < (int)deadValues.size(); i++) {
    llvm::outs() << "[" << i << "] Tracing from dead value...\n";

    // Find the specific dead value's operation + result
    func.walk([&](Operation *op) {
      if (i >= (int)deadValues.size()) return;

      // Walk uses forward to see if they lead to a live result
      for (auto [idx, result] : llvm::enumerate(op->getResults())) {
        auto *lat = solver.lookupState<Liveness>(result);
        if (lat && lat->isLive) continue;
        if (!lat || !lat->isLive) {
          // This matches a dead value
          // Show the op and its immediate users
          llvm::outs() << "    Producer: " << op->getName().getStringRef()
                       << " result #" << idx << "\n";
          op->getLoc().print(llvm::outs());
          llvm::outs() << "\n";

          if (result.use_empty()) {
            llvm::outs() << "    Zero uses — dead on arrival.\n";
          } else {
            llvm::outs() << "    Consumers:\n";
            for (auto &use : result.getUses()) {
              Operation *user = use.getOwner();
              llvm::outs() << "      → " << user->getName().getStringRef()
                           << " (operand #" << use.getOperandNumber() << ")";
              user->getLoc().print(llvm::outs());
              llvm::outs() << "\n";

              // Check if consumer itself is live
              bool consumerLive = false;
              for (Value r : user->getResults()) {
                auto *clat = solver.lookupState<Liveness>(r);
                if (clat && clat->isLive) consumerLive = true;
              }
              auto *clat = solver.lookupState<Liveness>(
                  user->getResult(0));
              llvm::outs() << "        consumer lattice: "
                           << (clat ? (clat->isLive ? "LIVE" : "DEAD")
                                    : "NO LATTICE")
                           << "\n";
              if (!consumerLive && user->getNumResults() > 0) {
                llvm::outs() << "        => consumer also dead — dead chain.\n";
              }
            }
          }
          llvm::outs() << "\n";
        }
      }
    });
  }
}

} // namespace tutorial
} // namespace mlir
