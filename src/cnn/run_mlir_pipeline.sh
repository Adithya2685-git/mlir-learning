###  Pipeline to get from linalg to llvm dialect  ###

# For BLAS integration (Todo: Can be merged)
../../build-ninja/tools/tutorial-opt --linalg-to-bufferization $PWD/cnn_model_linalg.mlir > $PWD/cnn_model_buf_linalg.mlir
../../build-ninja/tools/tutorial-opt --llvm-request-c-wrappers --bufferization-to-llvm $PWD/cnn_model_buf_linalg.mlir > $PWD/cnn_model_llvm.mlir

###  Use mlir-translate to get from mlir to llvm ir  ###
mlir-translate -mlir-to-llvmir $PWD/cnn_model_llvm.mlir > $PWD/cnn_model_llvm_ir.ll

###  Create .obj file  ###
llc --filetype=obj $PWD/cnn_model_llvm_ir.ll

###  Compile  ###
g++ -c cnn_call.cpp -o cnn_call.o && g++ cnn_call.o cnn_model_llvm_ir.o -o a.out \
	-lopenblas \
	-lm -no-pie