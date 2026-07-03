g++ -c cnn_call.cpp -o cnn_call.o && g++ cnn_call.o cnn.o -o a.out \
	-L../../../llvm-project/build/lib -lmlir_runner_utils -lmlir_cuda_runtime \
	-L/opt/cuda/lib64 -lcuda -lcudart \
	-lm \
	-Wl,-rpath,'./../../../llvm-project/build/lib' \
	-Wl,-rpath,'/opt/cuda/lib64'