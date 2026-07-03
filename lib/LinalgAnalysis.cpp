//===----------------------------------------------------------------------===//
// LinalgAnalysis.cpp — strict fusion legality check
//
// Applies MLIR's exact `areElementwiseOpsFusable` conditions:
//   1. Producer generic → all parallel iterators
//   2. Consumer generic (or matmul/batch_matmul — extended check)
//   3. Fused operand is INPUT to consumer
//   4. Indexing map compatibility: consumer map results = producer loops
//   5. Producer result map is a permutation
//   6. Pure tensor semantics
//
// Run: tutorial-opt --linalg-analysis gpt2_clean.mlir
//===----------------------------------------------------------------------===//

#include "lib/LinalgAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir {
namespace tutorial {

void LinalgAnalysisPass::runOnOperation() {
  ModuleOp module = getOperation();

  llvm::outs() << "==================================================\n";
  llvm::outs() << "LINALG FUSION STRICT LEGALITY CHECK\n";
  llvm::outs() << "==================================================\n\n";

  // Statistics
  int totalProducers = 0;     // generic ops that could be producer
  int singleUseResults = 0;   // results with exactly 1 use
  int inputToConsumer = 0;    // that single use is as INPUT to consumer
  int allParallelPass = 0;    // producer is all-parallel
  int tensorSemanticsPass = 0; // pure tensor semantics
  int mapResultsMatch = 0;    // consumer index map results = producer loops
  int producerMapPerm = 0;    // producer result map is a permutation
  int fullyFusible = 0;       // passed ALL checks

  // Breakdown by consumer type
  int fusibleIntoMatmul = 0;
  int fusibleIntoBatchMatmul = 0;
  int fusibleIntoGeneric = 0;

  // Failures tracked
  SmallVector<std::string> failureReasons;

  module.walk([&](linalg::GenericOp producer) {
    // Skip if producer has no tensor semantics
    if (!producer.hasPureTensorSemantics())
      return;

    for (auto [resIdx, result] : llvm::enumerate(producer.getResults())) {
      totalProducers++;

      // --- Check 1: Single use ---
      if (!result.hasOneUse())
        continue;
      singleUseResults++;

      OpOperand &use = *result.getUses().begin();
      Operation *consumerOp = use.getOwner();
      Value usedValue = use.get();

      // --- Check 2: Consumer is a linalg op ---
      // Extended: accept linalg::GenericOp, MatmulOp, BatchMatmulOp
      bool isGenericConsumer = isa<linalg::GenericOp>(consumerOp);
      bool isMatmulConsumer = isa<linalg::MatmulOp>(consumerOp);
      bool isBatchMatmulConsumer = isa<linalg::BatchMatmulOp>(consumerOp);

      if (!isGenericConsumer && !isMatmulConsumer && !isBatchMatmulConsumer)
        continue;

      // --- Check 3: Fused operand is INPUT to consumer ---
      // For GenericOp: use isDpsInput
      // For MatmulOp/BatchMatmulOp: check if operand is one of the inputs
      bool isInput = false;
      AffineMap consumerIndexMap;
      int consumerNumLoops = 0;

      if (auto consumerGeneric = dyn_cast<linalg::GenericOp>(consumerOp)) {
        isInput = consumerGeneric.isDpsInput(&use);
        if (isInput) {
          consumerIndexMap =
              consumerGeneric.getMatchingIndexingMap(&use);
          consumerNumLoops = consumerGeneric.getNumLoops();
        }
      } else if (auto matmul = dyn_cast<linalg::MatmulOp>(consumerOp)) {
        // matmul: first 2 operands are inputs, 3rd is output
        isInput = (usedValue == matmul.getInputs()[0] ||
                   usedValue == matmul.getInputs()[1]);
        if (isInput) {
          consumerIndexMap =
              matmul.getMatchingIndexingMap(&use);
          consumerNumLoops = matmul.getNumLoops();
        }
      } else if (auto bm = dyn_cast<linalg::BatchMatmulOp>(consumerOp)) {
        isInput = (usedValue == bm.getInputs()[0] ||
                   usedValue == bm.getInputs()[1]);
        if (isInput) {
          consumerIndexMap =
              bm.getMatchingIndexingMap(&use);
          consumerNumLoops = bm.getNumLoops();
        }
      }
      if (!isInput)
        continue;
      inputToConsumer++;

      // --- Check 4: Producer has ALL parallel iterators ---
      if (producer.getNumParallelLoops() != producer.getNumLoops())
        continue;
      allParallelPass++;

      // --- Check 5: Tensor semantics ---
      if (!isa<RankedTensorType>(usedValue.getType()))
        continue;
      tensorSemanticsPass++;

      // --- Check 6: Consumer index map results = producer loops ---
      if (consumerIndexMap.getNumResults() != producer.getNumLoops())
        continue;
      mapResultsMatch++;

      // --- Check 7: Producer result map is a permutation ---
      auto producerResult = cast<OpResult>(usedValue);
      AffineMap producerResultMap =
          producer.getIndexingMapMatchingResult(producerResult);
      if (!producerResultMap.isPermutation())
        continue;
      producerMapPerm++;

      // --- ALL CHECKS PASSED ---
      fullyFusible++;

      if (isMatmulConsumer)
        fusibleIntoMatmul++;
      else if (isBatchMatmulConsumer)
        fusibleIntoBatchMatmul++;
      else
        fusibleIntoGeneric++;
    }
  });

  // ---- Print results ------------------------------------------------------
  llvm::outs() << "FUSION LEGALITY CHECK (MLIR's areElementwiseOpsFusable rules)\n";
  llvm::outs() << "-------------------------------------------------------\n\n";
  llvm::outs() << "Total generic results checked:          " << totalProducers << "\n";
  llvm::outs() << "  Passed: single use:                   " << singleUseResults << "\n";
  llvm::outs() << "  Passed: is input to linalg consumer:  " << inputToConsumer << "\n";
  llvm::outs() << "  Passed: producer all-parallel:        " << allParallelPass << "\n";
  llvm::outs() << "  Passed: tensor semantics:             " << tensorSemanticsPass << "\n";
  llvm::outs() << "  Passed: map results = producer loops: " << mapResultsMatch << "\n";
  llvm::outs() << "  Passed: producer result map is perm:  " << producerMapPerm << "\n";
  llvm::outs() << "\n";
  llvm::outs() << "  >>> STRICTLY FUSIBLE:                " << fullyFusible << " <<<\n";
  llvm::outs() << "\n";
  llvm::outs() << "  Fusible into linalg.generic:          " << fusibleIntoGeneric << "\n";
  llvm::outs() << "  Fusible into linalg.matmul:           " << fusibleIntoMatmul << "\n";
  llvm::outs() << "  Fusible into linalg.batch_matmul:     " << fusibleIntoBatchMatmul << "\n";

  // ---- Print breakdown of what fails where --------------------------------
  int failSingleUse = totalProducers - singleUseResults;
  int failConsumer = singleUseResults - inputToConsumer;
  int failReduction = inputToConsumer - allParallelPass;
  int failTensor = allParallelPass - tensorSemanticsPass;
  int failMap = tensorSemanticsPass - mapResultsMatch;
  int failPerm = mapResultsMatch - producerMapPerm;

  llvm::outs() << "\n--- FAILURE BREAKDOWN ---\n";
  llvm::outs() << "  Not single-use:                " << failSingleUse << "\n";
  llvm::outs() << "  Consumer not linalg op:        " << failConsumer << "\n";
  llvm::outs() << "  Producer has reduction loops:  " << failReduction << "\n";
  llvm::outs() << "  Not tensor semantics:          " << failTensor << "\n";
  llvm::outs() << "  Map results != producer loops: " << failMap << "\n";
  llvm::outs() << "  Producer map not permutation:  " << failPerm << "\n";

  llvm::outs() << "\n==================================================\n";
  llvm::outs() << "DONE.\n";
}

} // namespace tutorial
} // namespace mlir
