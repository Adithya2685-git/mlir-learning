#ifndef LIB_LINALGANALYSIS_H
#define LIB_LINALGANALYSIS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tutorial {

struct LinalgAnalysisPass
    : public PassWrapper<LinalgAnalysisPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgAnalysisPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, linalg::LinalgDialect>();
  }

  void runOnOperation() override;

  StringRef getArgument() const final { return "linalg-analysis"; }
  StringRef getDescription() const final {
    return "Analyze linalg ops in GPT-2: fusion opportunities, "
           "producer-consumer chains, tiling candidates";
  }
};

} // namespace tutorial
} // namespace mlir

#endif
