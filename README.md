# GPT-2 GPU Optimization with MLIR Polyhedral Compilation

Lowering GPT-2 from PyTorch to NVIDIA GPU binaries through MLIR's linalg dialect with custom polyhedral-based optimization passes. End-to-end verified against PyTorch reference (max error < 10⁻⁵).

## Overview

```
PyTorch GPT-2 (124M params, 12 layers, 768 hidden, 50257 vocab)
  │  torch-mlir (torch.export + fx)
  ▼
linalg-on-tensors MLIR (950MB, 49 matmuls + 24 batch_matmuls + 786 generics)
  │
  ├── --eliminate-dead-argmax     removes 12 dead softmax argmax results
  ├── --canonicalize              folds constants, simplifies
  ├── --fuse-elementwise          fuses 172 element-wise generics
  ├── --polyhedral-analysis       dependence analysis, tile size computation
  ├── one-shot-bufferize          tensor → memref
  ├── --polyhedral-parallelize    tiles 419 generics + 48 matmuls for GPU
  │
  ▼
scf.parallel + linalg.generic on buffers
  │
  ├── convert-linalg-to-parallel-loops
  ├── gpu-map-parallel-loops
  ├── convert-parallel-loops-to-gpu
  ├── gpu-kernel-outlining
  ├── gpu-lower-to-nvvm-pipeline  (sm_89, PTX 8.0, O3)
  ├── gpu-to-llvm + gpu-module-to-binary
  ▼
ELF object file (481MB, cubin embedded)
  │
  └── g++ host program + libmlir_cuda_runtime + libcudart → executable
```

## Custom Passes

### `--eliminate-dead-argmax`
GPT-2's softmax attention produces both the softmax output and argmax indices. Only the softmax output is used downstream. This pass identifies the 12 dead argmax results (one per transformer layer) and eliminates them via `linalg.generic` result pruning + canonicalization cleanup.

### `--fuse-elementwise`
Fuses chains of element-wise `linalg.generic` ops using MLIR's `populateElementwiseOpsFusionPatterns`. Strict control function ensures only identity-mapped, all-parallel, single-use producers are fused — never into reductions. Eliminates 172 intermediate tensors, reducing kernel launch count.

### `--polyhedral-analysis`
Polyhedral analysis operating directly on `linalg.generic` ops:
- Parses `indexing_maps` as affine access relations and `iterator_types` as schedule labels
- Builds iteration domains from operand shapes
- Detects dependence edges (RAW, WAR, WAW) between ops via memory overlap
- Computes tile sizes per dimension: parallel dims → GPU-friendly (32, 16, 8, ...), reduction dims → register reuse (8, 4, 2)
- Classifies ops: pure parallel (343), mixed parallel+reduction (184), matmul-like (98)

### `--polyhedral-parallelize`
Applies tiling decisions from polyhedral analysis using `linalg::tileLinalgOp`:
- Pure parallel generics → `scf.parallel` loops (312 generated) for GPU thread mapping
- Matmuls / batch_matmuls → tiled on M,N parallel dims (48 + 12) for GPU block distribution
- Uses `linalg::tileLinalgOp` with `LinalgTilingLoopType::ParallelLoops` for generics, `Loops` for matmuls
- Original ops erased after tiling (buffer semantics — no tensor results to replace)

## Verification

All passes verified end-to-end against PyTorch GPT-2 reference (HuggingFace `gpt2` model). Input: "The capital of France is" → tokens [464, 3139, 286, 4881, 318].

```
PyTorch reference top-5:
  #1: token=262  (" the")     logit=-100.2498
  #2: token=783  (" now")     logit=-100.8175
  #3: token=257  (" a")       logit=-100.8555
  #4: token=4881 (" France")  logit=-101.2102
  #5: token=6342 (" Paris")   logit=-101.2143

Our optimized (fusion + tiling):
  #1: token=262  logit=-100.2498  ← exact match
  #2: token=783  logit=-100.8175  ← exact match
  #3: token=257  logit=-100.8555  ← exact match
  #4: token=4881 logit=-101.2102  ← exact match
  #5: token=6342 logit=-101.2143  ← exact match

Max absolute error across all 50257 logits: 8×10⁻⁶
```

## Performance (RTX 4080 Laptop GPU, sm_89)

| Pipeline | Wall time | vs Base |
|----------|----------|---------|
| Base GPU (no optimizations) | 2.71s | — |
| + Dead-argmax + Tiling | 2.72s | ~0% |
| + Fusion (+172 fused) | **2.44s** | **+10% faster** |
| + Fusion + MatmulTiling | 2.95s | -9% slower |

Fusion eliminates 172 intermediate kernels, reducing launch overhead. Matmul M,N tiling without K-dimension shared memory adds launch overhead without locality benefit — K-tiling requires shared memory promotion which is blocked by MLIR scf.for iter_arg handling across scf.parallel boundaries.

## Known Limitations

- K-dimension tiling for matmuls crashes GPU pipeline (nested `scf.for` iter_args not supported across `scf.parallel`)
- Multi-level tiling (block + thread decomposition) not implemented
- Shared memory promotion (software-managed cache tiling) not implemented
- 797 separate kernel launches — kernel fusion would consolidate these

## Files

```
lib/
├── DeadArgmaxElimination.cpp/h    — dead argmax removal pass
├── ElementwiseFusion.cpp/h         — element-wise fusion pass
├── PolyhedralAnalysis.cpp/h        — linalg.generic polyhedral analysis
├── PolyhedralParallelize.cpp/h     — linalg+matmul tiling pass
└── CMakeLists.txt                  — build configuration

tools/
├── tutorial-opt.cpp                — pass registration
└── CMakeLists.txt

src/gpt/
├── gpt2_linalg.mlir                — GPT-2 in linalg dialect (950MB)
├── lower_gpt_model.py              — PyTorch → MLIR export script
├── gpt2_call.cpp                   — GPU host program
└── pytorch_ref.py                  — PyTorch reference for comparison
```

## Build

Requires LLVM/MLIR built with NVPTX backend and CUDA runner enabled.

```bash
mkdir build-ninja && cd build-ninja
cmake -G Ninja .. \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm
ninja tutorial-opt
```

## Run

```bash
# Full pipeline
tutorial-opt --eliminate-dead-argmax gpt2_linalg.mlir | \
  mlir-opt --canonicalize | \
  tutorial-opt --fuse-elementwise | \
  mlir-opt --one-shot-bufferize=... --buffer-deallocation-pipeline --convert-bufferization-to-memref | \
  tutorial-opt --polyhedral-parallelize | \
  mlir-opt --convert-linalg-to-parallel-loops --gpu-map-parallel-loops \
           --convert-parallel-loops-to-gpu --gpu-kernel-outlining \
           --gpu-lower-to-nvvm-pipeline="cubin-chip=sm_89 cubin-features=+ptx80 opt-level=3" \
           --gpu-to-llvm=... --gpu-module-to-binary --reconcile-unrealized-casts \
  -o gpt2_gpu.mlir

mlir-translate -mlir-to-llvmir gpt2_gpu.mlir -o gpt2.ll
llc --filetype=obj -O3 gpt2.ll -o gpt2.o

# Host program
g++ -o gpt2_run gpt2_call.cpp gpt2.o \
  -L/opt/cuda/lib64 -lcudart -lcuda \
  -L/path/to/llvm-project/build/lib -lmlir_cuda_runtime -lmlir_c_runner_utils
./gpt2_run
```

## References

- [MLIR Linalg Dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)
- [GPU Compilation with MLIR — Stephen Diehl](https://www.stephendiehl.com/posts/mlir_gpu/)
- [Polygeist: Raising C to Polyhedral MLIR](https://c.wsmoses.com/papers/Polygeist_PACT.pdf)
- Base project: [DavidGinten/ML-compiler-exercise](https://github.com/DavidGinten/ML-compiler-exercise)
