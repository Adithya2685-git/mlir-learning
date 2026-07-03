###  Pipeline to get from linalg to llvm ir  ###

torch-mlir-opt -torch-backend-to-linalg-on-tensors-backend-pipeline resnet18_model_torch.mlir > resnet18_model_linalg.mlir

torch-mlir-opt -torch-backend-to-linalg-on-tensors-backend-pipeline resnet18_model_linalg.mlir \
| ../../build-ninja/tools/tutorial-opt -linalg-to-bufferization \
| ../../build-ninja/tools/tutorial-opt -llvm-request-c-wrappers \
| ../../build-ninja/tools/tutorial-opt -bufferization-to-llvm > resnet18_llvm.mlir

mlir-translate -mlir-to-llvmir resnet18_llvm.mlir \
| llc --filetype=obj -O3 -o resnet18_llvm_ir.o