#ifndef LIB_POLYHEDRALANALYSIS_H
#define LIB_POLYHEDRALANALYSIS_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tutorial {

struct PolyhedralAnalysisPass
    : public PassWrapper<PolyhedralAnalysisPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PolyhedralAnalysisPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, linalg::LinalgDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override;

  StringRef getArgument() const final { return "polyhedral-analysis"; }
  StringRef getDescription() const final {
    return "Polyhedral analysis on linalg.generic ops: iteration domains, "
           "access relations, dependence polyhedra, tile sizes, fusibility";
  }
};

} // namespace tutorial
} // namespace mlir

#endif
