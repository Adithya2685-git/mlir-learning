g++ -O3 -c sample_call_benchmark.cpp -o sample_call_benchmark.o && g++ -O3 sample_call_benchmark.o sample.o -o bench.out \
	-L../../../llvm-project/build/lib -lmlir_runner_utils -lmlir_cuda_runtime \
	-L/opt/cuda/lib64 -lcuda -lcudart \
	-Wl,-rpath,'./../../../llvm-project/build/lib' \
	-Wl,-rpath,'/opt/cuda/lib64'