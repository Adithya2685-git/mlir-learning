#ifndef LIB_GPT2SLICEANALYSIS_H
#define LIB_GPT2SLICEANALYSIS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tutorial {

struct GPT2SliceAnalysisPass
    : public PassWrapper<GPT2SliceAnalysisPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPT2SliceAnalysisPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, linalg::LinalgDialect>();
  }

  void runOnOperation() override;

  StringRef getArgument() const final { return "gpt2-slice-analysis"; }

  StringRef getDescription() const final {
    return "Slice analysis of GPT-2: trace op dependencies, count operations "
           "per subgraph";
  }
};

} // namespace tutorial
} // namespace mlir

#endif // LIB_GPT2SLICEANALYSIS_H
