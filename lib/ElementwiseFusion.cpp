//===----------------------------------------------------------------------===//
// ElementwiseFusion.cpp
//
// Fuses adjacent element-wise linalg.generic ops where both are pure
// parallel with identity-like indexing maps. This is safe because:
//   - Only element-wise ops are fused (no reduction consumers)
//   - No init-tensor producers (linalg.fill stays separate)
//   - Single-use producers only (no compute duplication)
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

/// Return true if all indexing maps are identity-like (each dim maps to
/// exactly one operand dim with stride 1), meaning pure element-wise.
static bool isElementwiseLike(GenericOp op) {
  for (auto map : op.getIndexingMapsArray()) {
    if (map.getNumResults() != op.getNumLoops())
      return false;
    for (unsigned i = 0; i < map.getNumResults(); i++) {
      auto expr = map.getResult(i);
      if (!isa<AffineDimExpr>(expr))
        return false;
      if (cast<AffineDimExpr>(expr).getPosition() != i)
        return false;
    }
  }
  return true;
}

void ElementwiseFusionPass::runOnOperation() {
  ModuleOp module = getOperation();

  int fusedCount = 0;
  int skippedReduction = 0;
  int skippedNonElem = 0;
  int skippedMultiUse = 0;

  auto controlFn = [&](OpOperand *fusedOperand) -> bool {
    auto producer = fusedOperand->get().getDefiningOp<GenericOp>();
    if (!producer)
      return false;

    // Producer must be all-parallel
    if (producer.getNumParallelLoops() != producer.getNumLoops())
      return false;

    // Producer must have a single use
    if (!fusedOperand->get().hasOneUse()) {
      skippedMultiUse++;
      return false;
    }

    // Consumer must be all-parallel (no fusing into reductions)
    auto consumer = dyn_cast<GenericOp>(fusedOperand->getOwner());
    if (!consumer || consumer.getNumParallelLoops() != consumer.getNumLoops()) {
      skippedReduction++;
      return false;
    }

    // Both must be element-wise (identity-like maps)
    if (!isElementwiseLike(producer) || !isElementwiseLike(consumer)) {
      skippedNonElem++;
      return false;
    }

    fusedCount++;
    return true;
  };

  llvm::outs() << "Fusing element-wise generics into consumers...\n";

  RewritePatternSet fusionPatterns(&getContext());
  populateElementwiseOpsFusionPatterns(fusionPatterns, controlFn);
  if (failed(applyPatternsGreedily(module, std::move(fusionPatterns)))) {
    llvm::errs() << "Fusion failed\n";
    signalPassFailure();
    return;
  }

  llvm::outs() << "  Fused:           " << fusedCount << "\n";
  llvm::outs() << "  Skipped non-elem:" << skippedNonElem << "\n";
  llvm::outs() << "  Skipped reduct.: " << skippedReduction << "\n";
  llvm::outs() << "  Skipped multi-use:" << skippedMultiUse << "\n";

  // Stats
  int generics = 0, matmuls = 0, batchMat = 0;
  module.walk([&](GenericOp) { generics++; });
  module.walk([&](MatmulOp) { matmuls++; });
  module.walk([&](BatchMatmulOp) { batchMat++; });

  llvm::outs() << "\n--- POST-FUSION ---\n";
  llvm::outs() << "  linalg.generic:   " << generics << "\n";
  llvm::outs() << "  linalg.matmul:    " << matmuls << "\n";
  llvm::outs() << "  linalg.batch_mat: " << batchMat << "\n";
  llvm::outs() << "  DONE.\n";
}

} // namespace tutorial
} // namespace mlir
