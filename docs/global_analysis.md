# MLIR Global Analysis Inventory

Complete inventory of every analysis available in MLIR (LLVM 23.0.0git), organized by category. All headers live under `/home/adi/Projects/Honours/MLIR/llvm-project/mlir/include/mlir/`.

---

## 1. Core Analyses (`Analysis/`)

### 1.1 Control Flow & Structure

| Analysis | Header | What it computes |
|---|---|---|
| CallGraph | `Analysis/CallGraph.h` | Call graph across all functions in a module. Nodes = callable ops, edges = CallOp/CallInterface |
| CFGLoopInfo | `Analysis/CFGLoopInfo.h` | Natural loop detection on CFG blocks (LLVM LoopInfo adapted for MLIR) |
| Liveness (block-level) | `Analysis/Liveness.h` | Live-in/live-out SSA value sets per basic block via fixpoint iteration |
| SymbolTableAnalysis | `Analysis/SymbolTableAnalysis.h` | Symbol table lookups — visibility, nested symbol resolution |

### 1.2 Dataflow Graph

| Analysis | Header | What it computes |
|---|---|---|
| SliceAnalysis | `Analysis/SliceAnalysis.h` | Forward/backward transitive use-def slices through operation DAGs. Configurable filter to stop propagation |
| SliceWalk | `Analysis/SliceWalk.h` | Walk along use-def chains with configurable filters |
| TopologicalSortUtils | `Analysis/TopologicalSortUtils.h` | Sort ops in a block in topological (producer-before-consumer) order |

### 1.3 Memory & Alias

| Analysis | Header | What it computes |
|---|---|---|
| AliasAnalysis | `Analysis/AliasAnalysis.h` | Memory alias queries between two values: NoAlias, MayAlias, PartialAlias, MustAlias |
| LocalAliasAnalysis | `Analysis/AliasAnalysis/LocalAliasAnalysis.h` | Local (intra-block) alias analysis for memref values |
| DataLayoutAnalysis | `Analysis/DataLayoutAnalysis.h` | Data layout (size, alignment, endianness) queries per operation |

### 1.4 Integer Sets & Constraints

| Analysis | Header | What it computes |
|---|---|---|
| FlatLinearValueConstraints | `Analysis/FlatLinearValueConstraints.h` | Affine constraint system over SSA values — extends IntegerPolyhedron with an AffineExpr-based API |

---

## 2. Dataflow Framework (`Analysis/DataFlow/`)

### 2.1 Framework Core

| Class | Header | Purpose |
|---|---|---|
| DataFlowSolver | `Analysis/DataFlowFramework.h` | Generic sparse dataflow solver. Manages program points, lattice states, worklist |
| AbstractSparseLattice | `Analysis/DataFlowFramework.h` | Base lattice class for sparse analysis — one lattice per SSA value |
| SparseForwardDataFlowAnalysis | `Analysis/DataFlow/SparseAnalysis.h` | Forward sparse propagation through ops, blocks, and callgraph |
| SparseBackwardDataFlowAnalysis | `Analysis/DataFlow/SparseAnalysis.h` | Backward sparse propagation |
| DenseForwardDataFlowAnalysis | `Analysis/DataFlow/DenseAnalysis.h` | Dense (block-level, one state per program point) forward analysis |
| DenseBackwardDataFlowAnalysis | `Analysis/DataFlow/DenseAnalysis.h` | Dense backward analysis |

### 2.2 Pre-built Dataflow Analyses

| Analysis | Header | What it computes |
|---|---|---|
| ConstantPropagationAnalysis | `Analysis/DataFlow/ConstantPropagationAnalysis.h` | Sparse constant folding — which SSA values are statically known constants |
| LivenessAnalysis (dataflow) | `Analysis/DataFlow/LivenessAnalysis.h` | Backward dataflow: values are "live" if they have memory effects or contribute to live outputs |
| DeadCodeAnalysis | `Analysis/DataFlow/DeadCodeAnalysis.h` | Which blocks/edges are executable (live code detection) |
| IntegerRangeAnalysis | `Analysis/DataFlow/IntegerRangeAnalysis.h` | Integer value ranges [min, max] for arith ops, with widening for loops |
| IntegerDivisibilityAnalysis | `Analysis/DataFlow/IntegerDivisibilityAnalysis.h` | Which integer values are divisible by which constants |
| StridedMetadataRangeAnalysis | `Analysis/DataFlow/StridedMetadataRangeAnalysis.h` | Stride/offset/size ranges for memref values |

---

## 3. Presburger Library (`Analysis/Presburger/`)

The polyhedral math engine. Operates on integer sets defined by affine constraints.

### 3.1 Core Polyhedra

| Class | Header | Purpose |
|---|---|---|
| IntegerRelation | `Analysis/Presburger/IntegerRelation.h` | Set of integer tuples satisfying affine equality/inequality constraints. A convex polyhedron over Z |
| IntegerPolyhedron | (in IntegerRelation.h) | IntegerRelation where all vars are domain (no range/domain distinction) |
| PresburgerRelation | `Analysis/Presburger/PresburgerRelation.h` | Union of IntegerRelations (non-convex sets) |
| PresburgerSpace | `Analysis/Presburger/PresburgerSpace.h` | Dimensionality: number of domain/range/symbol/local variables |

### 3.2 Algorithms

| Class | Header | Purpose |
|---|---|---|
| Simplex | `Analysis/Presburger/Simplex.h` | Simplex algorithm for linear programming on IntegerRelations. Finds sample points, checks emptiness |
| Barvinok | `Analysis/Presburger/Barvinok.h` | Barvinok's algorithm — counts integer points in polyhedra via generating functions |
| GeneratingFunction | `Analysis/Presburger/GeneratingFunction.h` | Formal power series representation of counting functions |
| QuasiPolynomial | `Analysis/Presburger/QuasiPolynomial.h` | Quasi-polynomials (periodic coefficients) from Barvinok output |

### 3.3 Functions & Mappings

| Class | Header | Purpose |
|---|---|---|
| PWMAFunction | `Analysis/Presburger/PWMAFunction.h` | Piecewise multi-affine function over Presburger sets |
| MultiAffineFunction | (in PWMAFunction.h) | Multi-dimensional affine function |
| LinearTransform | `Analysis/Presburger/LinearTransform.h` | Linear transformations on Presburger spaces |

### 3.4 Utilities

| Class | Header | Purpose |
|---|---|---|
| Matrix | `Analysis/Presburger/Matrix.h` | Arbitrary-precision integer matrix |
| Fraction | `Analysis/Presburger/Fraction.h` | Arbitrary-precision rational numbers |

---

## 4. Dialect-specific Analyses

### 4.1 Affine Dialect (`Dialect/Affine/Analysis/`)

| Analysis | Header | What it computes |
|---|---|---|
| AffineAnalysis | `Dialect/Affine/Analysis/AffineAnalysis.h` | Dependence analysis, dependence vectors, access functions for affine.for nests |
| AffineStructures | `Dialect/Affine/Analysis/AffineStructures.h` | Build IntegerRelations from affine.for/affine.if bodies |
| LoopAnalysis | `Dialect/Affine/Analysis/LoopAnalysis.h` | Trip counts, loop bounds, parallel loop detection for affine.for |
| NestedMatcher | `Dialect/Affine/Analysis/NestedMatcher.h` | Pattern matcher for nested affine loop structures |
| Utils | `Dialect/Affine/Analysis/Utils.h` | Affine utility analysis — loop fusion legality, tiling validity, memref promotion |

### 4.2 Bufferization Dialect (`Dialect/Bufferization/Transforms/`)

| Analysis | Header | What it computes |
|---|---|---|
| BufferViewFlowAnalysis | `Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h` | Tracks which buffer views alias or overlap |
| OneShotAnalysis | `Dialect/Bufferization/Transforms/OneShotAnalysis.h` | One-shot bufferization — determines where to insert copies, which tensors become in-place buffers |

### 4.3 Shape Dialect (`Dialect/Shape/Analysis/`)

| Analysis | Header | What it computes |
|---|---|---|
| ShapeMappingAnalysis | `Dialect/Shape/Analysis/ShapeMappingAnalysis.h` | Maps between shape dialect ops and the SSA values they describe |

---

## 5. Conversion & Rewriting Infrastructure

Not analysis per se, but the core transformation machinery analyses are built on.

| Component | Header | Purpose |
|---|---|---|
| DialectConversion | `Transforms/DialectConversion.h` | Dialect conversion driver: legal/illegal ops, TypeConverter, applyPartialConversion |
| PatternMatch | `Rewrite/PatternMatch.h` | Pattern matching engine: RewritePattern, matchAndRewrite, RewritePatternSet |
| PatternApplicator | `Rewrite/PatternApplicator.h` | Applies pattern sets to IR regions |
| GreedyPatternRewriteDriver | `Transforms/GreedyPatternRewriteDriver.h` | Greedy application of patterns to convergence |

---

## 6. Key Documentation

| Topic | Path |
|---|---|
| Pass infrastructure | `docs/PassManagement.md` |
| Pattern rewriting | `docs/PatternRewriter.md` |
| Dialect conversion | `docs/DialectConversion.md` |
| Declarative rewrites (TableGen) | `docs/DeclarativeRewrites.md` |
| Quickstart rewrites | `docs/Tutorials/QuickstartRewrites.md` |
| Dataflow analysis tutorial | `docs/Tutorials/DataFlowAnalysis.md` |
| Affine dialect | `docs/Dialects/Affine.md` |
| Polyhedral rationale | `docs/Rationale/RationaleSimplifiedPolyhedralForm.md` |
| Linalg dialect | `docs/Dialects/Linalg/_index.md` |
| Creating a dialect | `docs/Tutorials/CreatingADialect.md` |
| MLIR Language Reference | `docs/LangRef.md` |

---

## 7. How to Write a Pass

### Minimal pass skeleton

**Header:**
```cpp
struct MyPass : public PassWrapper<MyPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override;

  StringRef getArgument() const final { return "my-pass"; }
  StringRef getDescription() const final { return "Does something"; }
};
```

**Implementation (read-only analysis):**
```cpp
void MyPass::runOnOperation() {
  ModuleOp module = getOperation();
  module.walk([&](linalg::MatmulOp matmul) {
    // inspect, count, record — don't modify
  });
  llvm::outs() << "Found N matmuls\n";
}
```

**Implementation (rewrite):**
```cpp
void MyPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<...>();
  target.addIllegalOp<linalg::MatmulOp>();

  RewritePatternSet patterns(&getContext());
  patterns.add<MyRewritePattern>(&getContext());

  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

struct MyRewritePattern : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    // check preconditions, create new ops, erase old op
    return success();
  }
};
```

**Registration:**
```cpp
// In tutorial-opt.cpp or similar:
mlir::PassRegistration<MyPass>();
```

### Existing working example

This project has a complete working pass at:
- `lib/ConvertMatMulToBlas.h` — pass declaration
- `lib/ConvertMatMulToBlas.cpp` — pattern-based rewrite: replaces `linalg.matmul` with `cblas_sgemm` calls
- `tools/tutorial-opt.cpp` — pipeline registration
