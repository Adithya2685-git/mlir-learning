#ifndef LIB_ELEMENTWISEFUSION_H
#define LIB_ELEMENTWISEFUSION_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tutorial {

struct ElementwiseFusionPass
    : public PassWrapper<ElementwiseFusionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ElementwiseFusionPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, linalg::LinalgDialect>();
  }

  void runOnOperation() override;

  StringRef getArgument() const final { return "fuse-elementwise"; }
  StringRef getDescription() const final {
    return "Fuse element-wise linalg.generic ops into their consumers "
           "(non-reduction only, top-down traversal)";
  }
};

} // namespace tutorial
} // namespace mlir

#endif
