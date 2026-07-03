g++ -O3 -c cnn_call_benchmark.cpp -o cnn_call_benchmark.o && g++ -O3 cnn_call_benchmark.o cnn.o -o bench.out\
	-L../../../llvm-project/build/lib -lmlir_runner_utils -lmlir_cuda_runtime \
	-L/opt/cuda/lib64 -lcuda -lcudart \
	-lm \
	-Wl,-rpath,'./../../../llvm-project/build/lib' \
	-Wl,-rpath,'/opt/cuda/lib64'