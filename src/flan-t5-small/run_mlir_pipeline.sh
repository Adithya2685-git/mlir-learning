###  Pipeline to get from linalg to llvm dialect  ###
# The object file is position independent code (PIC) and can be linked into a shared library.
# This is to be able to call the function form python using cpython.

torch-mlir-opt -torch-backend-to-linalg-on-tensors-backend-pipeline flan_linalg_test.mlir \
| ../../build-ninja/tools/tutorial-opt -linalg-to-bufferization \
| ../../build-ninja/tools/tutorial-opt -llvm-request-c-wrappers \
| ../../build-ninja/tools/tutorial-opt -bufferization-to-llvm \
| mlir-translate -mlir-to-llvmir \
| llc --filetype=obj -O3 -relocation-model=pic -o flan_llvm_test_ir.o