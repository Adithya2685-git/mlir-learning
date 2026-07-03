mlir-opt sample_model_linalg.mlir \
  --convert-tensor-to-linalg \
  --linalg-generalize-named-ops \
  --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
  --buffer-deallocation-pipeline \
  --convert-bufferization-to-memref \
  --llvm-request-c-wrappers \
  --convert-linalg-to-affine-loops \
  --affine-loop-tile="tile-size=4" \
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
  -o sample_nvvm.mlir

mlir-translate -mlir-to-llvmir sample_nvvm.mlir -o sample.ll

llc -filetype=obj -O3 sample.ll






#llc -march=nvptx64 -mcpu=sm_90 square.ll
#| mlir-opt \
#  --pass-pipeline='builtin.module(gpu.module(convert-gpu-to-nvvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=true}))' \
#| mlir-opt \
#  --nvvm-attach-target="chip=sm_90 features=+ptx80 O=3" \
#  -o gpu_out_2.mlir