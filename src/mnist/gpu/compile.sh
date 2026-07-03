g++ -c mnist_call.cpp -o mnist_call.o && g++ mnist_call.o mnist.o -o a.out \
	-L../../../llvm-project/build/lib -lmlir_runner_utils -lmlir_cuda_runtime \
	-L/opt/cuda/lib64 -lcuda -lcudart \
	-Wl,-rpath,'./../../../llvm-project/build/lib' \
	-Wl,-rpath,'/opt/cuda/lib64'