# First lower to TORCH (lower_bert_model.py), then use torch-mlir-opt -torch-backend-to-linalg-on-tensors-backend-pipeline bert_torch.mlir
# to get to Linalg on Tensors.
# This is because the direct fx.export_and_import to Linalg_on_Tensors does currently not work for the model.

torch-mlir-opt -torch-backend-to-linalg-on-tensors-backend-pipeline bert_torch_both.mlir \
| ../../build-ninja/tools/tutorial-opt -linalg-to-bufferization \
| ../../build-ninja/tools/tutorial-opt -llvm-request-c-wrappers \
| ../../build-ninja/tools/tutorial-opt -bufferization-to-llvm \
| mlir-translate -mlir-to-llvmir \
| llc --filetype=obj -O3 -o bert_llvm_ir.o