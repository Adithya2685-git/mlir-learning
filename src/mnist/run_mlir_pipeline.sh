###  Pipeline to get from linalg to llvm dialect  ###

# With BLAS integration (Todo: merge)
../../build-ninja/tools/tutorial-opt --linalg-to-bufferization $PWD/mnist_model_linalg.mlir > $PWD/mnist_model_buf_linalg.mlir
../../build-ninja/tools/tutorial-opt --llvm-request-c-wrappers --bufferization-to-llvm $PWD/mnist_model_buf_linalg.mlir > $PWD/mnist_model_llvm.mlir

###  Use mlir-translate to get from mlir to llvm ir  ###
mlir-translate -mlir-to-llvmir $PWD/mnist_model_llvm.mlir > $PWD/mnist_model_llvm_ir.ll

###  Create .obj file  ###
llc --filetype=obj $PWD/mnist_model_llvm_ir.ll

###  Compile  ###
g++ -c mnist_call.cpp -o mnist_call.o && g++ mnist_call.o mnist_model_llvm_ir.o -o a.out \
	-L../../llvm-project/build/lib -lmlir_c_runner_utils \
	-lopenblas \
	-Wl,-rpath=../../llvm-project/build/lib \
	-no-pie