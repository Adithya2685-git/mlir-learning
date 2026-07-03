//===----------------------------------------------------------------------===//
// ElementwiseFusion.cpp
//
// Fuses element-wise linalg.generic ops into their consumers.
// Uses MLIR's own fuseElementwiseOps with a control function that:
//   - Only fuses non-reduction producers (all parallel loops)
//   - Skips producers with multiple consumers (avoids compute duplication)
//   - Operates top-down (producers before consumers)
//
// Prerequisite: run --linalg-generalize-named-ops first to convert matmuls
// and batch_matmuls to generics so they can participate in fusion.
//
// Run: tutorial-opt --fuse-elementwise gpt2_clean.mlir
//===----------------------------------------------------------------------===//

#include "lib/ElementwiseFusion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::linalg;

namespace mlir {
namespace tutorial {

void ElementwiseFusionPass::runOnOperation() {
  ModuleOp module = getOperation();

  // ---- Step 1: Generalize named ops so matmuls can participate -----------
  // Convert linalg.matmul and linalg.batch_matmul to linalg.generic
  // so that element-wise fusion can fuse producers into them.

  llvm::outs() << "Generalizing named ops (matmul/batch_matmul → generic)...\n";
  {
    RewritePatternSet generalizePatterns(&getContext());
    linalg::populateLinalgNamedOpsGeneralizationPatterns(generalizePatterns);
    if (failed(applyPatternsGreedily(module, std::move(generalizePatterns)))) {
      llvm::errs() << "Generalization failed\n";
      signalPassFailure();
      return;
    }
  }

  // Count matmuls/batch_matmuls that were generalized
  int remainingNamed = 0;
  module.walk([&](linalg::MatmulOp) { remainingNamed++; });
  module.walk([&](linalg::BatchMatmulOp) { remainingNamed++; });
  llvm::outs() << "  Named ops remaining: " << remainingNamed
               << " (should be 0)\n\n";

  // ---- Step 2: Fuse element-wise ops into their consumers ---------------
  // Control function: only fuse if producer has all-parallel iterators
  // and producer result has exactly one use (avoid duplicating compute).
  llvm::outs() << "Fusing element-wise generics into consumers...\n";

  int fusedCount = 0;
  auto controlFn = [&](OpOperand *fusedOperand) -> bool {
    // Get the producer generic
    auto producer =
        fusedOperand->get().getDefiningOp<linalg::GenericOp>();
    if (!producer)
      return false;

    // Only fuse all-parallel producers (skip reductions)
    if (producer.getNumParallelLoops() != producer.getNumLoops())
      return false;

    // Only fuse if producer result has a single use (avoid duplicating)
    if (!fusedOperand->get().hasOneUse())
      return false;

    fusedCount++;
    return true;
  };

  {
    RewritePatternSet fusionPatterns(&getContext());
    populateElementwiseOpsFusionPatterns(fusionPatterns, controlFn);
    if (failed(applyPatternsGreedily(module, std::move(fusionPatterns)))) {
      llvm::errs() << "Fusion failed\n";
      signalPassFailure();
      return;
    }
  }

  llvm::outs() << "  Fused " << fusedCount << " element-wise generic(s)"
               << " into their consumers\n";

  // ---- Step 3: Stats -----------------------------------------------------
  int genericsAfter = 0;
  module.walk([&](linalg::GenericOp) { genericsAfter++; });

  llvm::outs() << "\n--- POST-FUSION STATS ---\n";
  llvm::outs() << "  linalg.generic ops remaining: " << genericsAfter << "\n";

  // Count reduction generics (these should remain unfused)
  int reductionOps = 0;
  module.walk([&](linalg::GenericOp generic) {
    if (generic.getNumParallelLoops() != generic.getNumLoops())
      reductionOps++;
  });
  llvm::outs() << "  Of which are reductions:     " << reductionOps << "\n";
  llvm::outs() << "  Of which are pure parallel:  "
               << (genericsAfter - reductionOps) << "\n";

  llvm::outs() << "\n==================================================\n";
  llvm::outs() << "DONE.\n";
}

} // namespace tutorial
} // namespace mlir
