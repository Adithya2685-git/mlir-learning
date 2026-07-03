###  Pipeline to get from linalg to x86 executable  ###

# With BLAS integration (Todo: merge)
../../build-ninja/tools/tutorial-opt --linalg-to-bufferization $PWD/sample_model_linalg.mlir > $PWD/sample_model_buf_linalg.mlir
../../build-ninja/tools/tutorial-opt --llvm-request-c-wrappers --bufferization-to-llvm $PWD/sample_model_buf_linalg.mlir > $PWD/sample_model_llvm.mlir

###  Use mlir-translate to get from mlir to llvm ir  ###
mlir-translate -mlir-to-llvmir $PWD/sample_model_llvm.mlir > $PWD/sample_model_llvm_ir.ll

###  Create .obj file  ###
llc --filetype=obj $PWD/sample_model_llvm_ir.ll

###  Compile  ###
g++ -c sample_call.cpp -o sample_call.o && g++ sample_call.o sample_model_llvm_ir.o -o a.out -lopenblas -lm -no-pie
