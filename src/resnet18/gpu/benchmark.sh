g++ -O3 -c resnet18_call_benchmark.cpp -o resnet18_call_benchmark.o && g++ -O3 resnet18_call_benchmark.o resnet18.o -o bench.out \
	-L../../../llvm-project/build/lib -lmlir_c_runner_utils -lmlir_cuda_runtime \
	-L/opt/cuda/lib64 -lcuda -lcudart \
	-Wl,-rpath,'./../../../llvm-project/build/lib' \
	-Wl,-rpath,'/opt/cuda/lib64'