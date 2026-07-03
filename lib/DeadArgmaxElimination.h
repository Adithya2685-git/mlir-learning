#ifndef LIB_DEADARGMAXELIMINATION_H
#define LIB_DEADARGMAXELIMINATION_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tutorial {

struct DeadArgmaxEliminationPass
    : public PassWrapper<DeadArgmaxEliminationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DeadArgmaxEliminationPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, linalg::LinalgDialect>();
  }

  void runOnOperation() override;

  StringRef getArgument() const final { return "eliminate-dead-argmax"; }
  StringRef getDescription() const final {
    return "Eliminate dead linalg.generic results identified by dataflow "
           "liveness analysis";
  }
};

} // namespace tutorial
} // namespace mlir

#endif
