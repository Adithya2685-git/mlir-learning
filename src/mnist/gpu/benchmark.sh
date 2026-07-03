g++ -O3 -c mnist_call_benchmark.cpp -o mnist_call_benchmark.o && g++ -O3 mnist_call_benchmark.o mnist.o -o bench.out \
	-L../../../llvm-project/build/lib -lmlir_runner_utils -lmlir_cuda_runtime \
	-L/opt/cuda/lib64 -lcuda -lcudart \
	-Wl,-rpath,'./../../../llvm-project/build/lib' \
	-Wl,-rpath,'/opt/cuda/lib64'