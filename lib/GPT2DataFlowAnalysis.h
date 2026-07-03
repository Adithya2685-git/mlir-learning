#ifndef LIB_GPT2DATAFLOWANALYSIS_H
#define LIB_GPT2DATAFLOWANALYSIS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tutorial {

struct GPT2DataFlowAnalysisPass
    : public PassWrapper<GPT2DataFlowAnalysisPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPT2DataFlowAnalysisPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, linalg::LinalgDialect, arith::ArithDialect,
                tensor::TensorDialect>();
  }

  void runOnOperation() override;

  StringRef getArgument() const final { return "gpt2-dataflow"; }
  StringRef getDescription() const final {
    return "Sparse forward dataflow: propagate known dimensions, constant "
           "folding, and value ranges through GPT-2";
  }
};

} // namespace tutorial
} // namespace mlir

#endif
