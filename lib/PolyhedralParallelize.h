#ifndef LIB_POLYHEDRALPARALLELIZE_H
#define LIB_POLYHEDRALPARALLELIZE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tutorial {

struct PolyhedralParallelizePass
    : public PassWrapper<PolyhedralParallelizePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PolyhedralParallelizePass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, linalg::LinalgDialect, scf::SCFDialect>();
  }

  void runOnOperation() override;

  StringRef getArgument() const final { return "polyhedral-parallelize"; }
  StringRef getDescription() const final {
    return "Tiles linalg.generic ops using polyhedral analysis. Computes tile "
           "sizes from iteration domains and iterator_types, applies via "
           "linalg::tileLinalgOp with ParallelLoops for GPU mapping.";
  }
};

} // namespace tutorial
} // namespace mlir

#endif
