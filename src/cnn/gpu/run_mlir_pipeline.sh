# Play with tile size for performance (16, 32, 64, 128, 256, ...) 
mlir-opt cnn_model_linalg.mlir \
  --convert-tensor-to-linalg \
  --linalg-generalize-named-ops \
  --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
  --buffer-deallocation-pipeline \
  --convert-bufferization-to-memref \
  --llvm-request-c-wrappers \
  --convert-linalg-to-affine-loops \
  --affine-loop-tile="tile-size=64" \
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
  --gpu-to-llvm='use-bare-pointers-for-host=true use-bare-pointers-for-kernels=true' \
  --gpu-module-to-binary \
  -o cnn_nvvm.mlir

mlir-translate -mlir-to-llvmir cnn_nvvm.mlir -o cnn.ll

llc -filetype=obj -O3 cnn.ll