mlir-opt flan_linalg.mlir \
  --convert-tensor-to-linalg \
  --linalg-generalize-named-ops \
  --linalg-fuse-elementwise-ops \
  --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
  --buffer-deallocation-pipeline \
  --convert-bufferization-to-memref \
  --llvm-request-c-wrappers \
  --convert-linalg-to-parallel-loops \
  --gpu-map-parallel-loops \
  --convert-parallel-loops-to-gpu \
  --canonicalize \
  --cse \
| mlir-opt \
  --pass-pipeline='builtin.module(func.func(affine-loop-invariant-code-motion))' \
  --pass-pipeline='builtin.module(func.func(convert-affine-for-to-gpu))' \
| mlir-opt \
  --gpu-kernel-outlining \
  --lower-affine \
  --gpu-decompose-memrefs \
  --expand-strided-metadata \
  --normalize-memrefs \
  --convert-index-to-llvm \
  --arith-expand \
  --memref-expand \
  --gpu-lower-to-nvvm-pipeline="cubin-chip=sm_90 cubin-features=+ptx80 opt-level=3" \
  --convert-nvvm-to-llvm \
  --reconcile-unrealized-casts \
  --gpu-to-llvm='use-bare-pointers-for-host=false use-bare-pointers-for-kernels=false' \
  --gpu-module-to-binary \
  -o flan_nvvm.mlir

mlir-translate -mlir-to-llvmir flan_nvvm.mlir -o flan.ll

llc -filetype=obj -O3 flan.ll