//===----------------------------------------------------------------------===//
// PolyhedralParallelize.cpp
//
// Tiles linalg.generic ops on buffers using polyhedral analysis:
//   1. Computes polyhedral tile sizes from indexing_maps + iterator_types
//   2. Applies linalg::tileLinalgOp with ParallelLoops for GPU mapping
//   3. Generates scf.parallel (outer tiles) + linalg.generic (inner tile)
//
// Pipeline: bufferize → this pass → convert-linalg-to-parallel-loops
//            → gpu-map-parallel-loops → convert-parallel-loops-to-gpu → NVVM
//===----------------------------------------------------------------------===//

#include "lib/PolyhedralParallelize.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::linalg;

namespace mlir {
namespace tutorial {

// --- Tile size computation ------------------------------------------------
static SmallVector<int64_t> computeOpTileSizes(LinalgOp op) {
  unsigned numLoops = op.getNumLoops();
  SmallVector<int64_t> tileSizes(numLoops, 0);
  auto iterTypes = op.getIteratorTypesArray();
  auto maps = op.getIndexingMapsArray();

  for (unsigned d = 0; d < numLoops; d++) {
    int64_t dimSize = -1;

    for (unsigned i = 0; i < maps.size(); i++) {
      for (unsigned r = 0; r < maps[i].getNumResults(); r++) {
        auto e = maps[i].getResult(r);
        if (auto dimExpr = dyn_cast<AffineDimExpr>(e)) {
          if (dimExpr.getPosition() == d) {
            auto type = op->getOperand(i).getType();
            if (auto ranked = dyn_cast<RankedTensorType>(type)) {
              if (r < ranked.getShape().size())
                dimSize = ranked.getShape()[r];
            } else if (auto memref = dyn_cast<MemRefType>(type)) {
              if (r < memref.getShape().size())
                dimSize = memref.getShape()[r];
            }
          }
        }
      }
    }

    if (dimSize <= 0) continue;

    if (iterTypes[d] == utils::IteratorType::parallel) {
      for (int64_t ts : {32, 16, 8, 4, 2}) {
        if (dimSize % ts == 0 && ts <= dimSize) {
          tileSizes[d] = ts; break;
        }
      }
      if (tileSizes[d] == 0 && dimSize >= 32)
        tileSizes[d] = 32;
    } else {
      for (int64_t ts : {8, 4, 2}) {
        if (dimSize % ts == 0 && ts <= dimSize) {
          tileSizes[d] = ts; break;
        }
      }
    }
  }
  return tileSizes;
}

// --- Tile size computation for matmul (M,N,K) ----------------------------
// For GPU: tile M,N for block/thread distribution, K for register reuse
static SmallVector<int64_t> computeMatmulTileSizes(LinalgOp op) {
  unsigned numLoops = op.getNumLoops();
  SmallVector<int64_t> tileSizes(numLoops, 0);
  auto iterTypes = op.getIteratorTypesArray();
  auto maps = op.getIndexingMapsArray();

  for (unsigned d = 0; d < numLoops; d++) {
    int64_t dimSize = -1;
    for (unsigned i = 0; i < maps.size(); i++) {
      for (unsigned r = 0; r < maps[i].getNumResults(); r++) {
        auto e = maps[i].getResult(r);
        if (auto dimExpr = dyn_cast<AffineDimExpr>(e)) {
          if (dimExpr.getPosition() == d) {
            auto type = op->getOperand(i).getType();
            if (auto memref = dyn_cast<MemRefType>(type)) {
              if (r < memref.getShape().size())
                dimSize = memref.getShape()[r];
            }
          }
        }
      }
    }
    if (dimSize <= 0) continue;

    if (iterTypes[d] == utils::IteratorType::parallel) {
      // Tile M,N for block-level distribution
      for (int64_t ts : {128, 64, 32, 16}) {
        if (dimSize % ts == 0 && ts <= dimSize) {
          tileSizes[d] = ts; break;
        }
      }
    } else {
      // Skip reduction dims (K) — tiling them creates scf.for loops
      // that cause nested iter_arg issues with the GPU pipeline
    }
  }
  return tileSizes;
}

// --- Main pass ------------------------------------------------------------
void PolyhedralParallelizePass::runOnOperation() {
  ModuleOp module = getOperation();

  llvm::outs() << "==================================================\n";
  llvm::outs() << "POLYHEDRAL PARALLELIZATION — linalg.generic tiling\n";
  llvm::outs() << "==================================================\n\n";

  int beforeOps = 0;
  module.walk([&](GenericOp) { beforeOps++; });

  SmallVector<GenericOp> ops;
  module.walk([&](GenericOp op) { ops.push_back(op); });

  IRRewriter rewriter(&getContext());
  int tiledCount = 0, skippedRed = 0, skippedTensor = 0;

  for (auto genericOp : ops) {
    // scf.parallel only works on buffers (no iter_args)
    if (!genericOp.hasPureBufferSemantics()) {
      skippedTensor++;
      continue;
    }

    // Only pure parallel (scf.parallel doesn't support reductions)
    bool hasRed = llvm::any_of(genericOp.getIteratorTypesArray(),
                               [](auto t) {
                                 return t == utils::IteratorType::reduction;
                               });
    if (hasRed) {
      skippedRed++;
      continue;
    }

    auto tileSizes = computeOpTileSizes(genericOp);
    if (llvm::all_of(tileSizes, [](int64_t ts) { return ts == 0; }))
      continue;

    LinalgTilingOptions options;
    options.setTileSizes(tileSizes)
           .setLoopType(LinalgTilingLoopType::ParallelLoops);

    rewriter.setInsertionPoint(genericOp);
    FailureOr<TiledLinalgOp> tiled =
        tileLinalgOp(rewriter, genericOp, options);
    if (succeeded(tiled)) {
      // Erase original: on buffers there are no tensor results to replace
      rewriter.eraseOp(genericOp);
      tiledCount++;
    }
  }

  int afterOps = 0, parallelLoops = 0;
  module.walk([&](GenericOp) { afterOps++; });
  module.walk([&](scf::ParallelOp) { parallelLoops++; });

  // ---- Tile matmuls (block-level) ---------------------------------------
  int matmulsTiled = 0, batchMatsTiled = 0;
  auto tileNamed = [&](auto namedOp) {
    if (!namedOp.template hasPureBufferSemantics()) return;
    if (!namedOp) return;
    auto tileSizes = computeMatmulTileSizes(namedOp);
    if (llvm::all_of(tileSizes, [](int64_t ts) { return ts == 0; }))
      return;

    LinalgTilingOptions options;
    options.setTileSizes(tileSizes)
           .setLoopType(LinalgTilingLoopType::Loops);

    rewriter.setInsertionPoint(namedOp);
    FailureOr<TiledLinalgOp> tiled = tileLinalgOp(rewriter, namedOp, options);
    if (succeeded(tiled)) {
      rewriter.eraseOp(namedOp);
      if (isa<MatmulOp>(namedOp.getOperation())) matmulsTiled++;
      else batchMatsTiled++;
    }
  };

  SmallVector<MatmulOp> matmuls;
  SmallVector<BatchMatmulOp> batchMats;
  module.walk([&](MatmulOp op) { matmuls.push_back(op); });
  module.walk([&](BatchMatmulOp op) { batchMats.push_back(op); });
  for (auto op : matmuls) tileNamed(op);
  for (auto op : batchMats) tileNamed(op);

  // Count final
  afterOps = 0; parallelLoops = 0;
  module.walk([&](GenericOp) { afterOps++; });
  module.walk([&](scf::ParallelOp) { parallelLoops++; });

  llvm::outs() << "  linalg.generic before:   " << beforeOps << "\n";
  llvm::outs() << "  linalg.generic after:    " << afterOps << "\n";
  llvm::outs() << "  scf.parallel generated:  " << parallelLoops << "\n";
  llvm::outs() << "  Generics tiled:          " << tiledCount << "\n";
  llvm::outs() << "  Matmuls tiled:           " << matmulsTiled << "\n";
  llvm::outs() << "  BatchMats tiled:         " << batchMatsTiled << "\n";
  llvm::outs() << "  Skipped (tensor):        " << skippedTensor << "\n";
  llvm::outs() << "  Skipped (reduction):     " << skippedRed << "\n";
  llvm::outs() << "\nDONE.\n";
}

} // namespace tutorial
} // namespace mlir
