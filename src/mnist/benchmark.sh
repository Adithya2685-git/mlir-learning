g++ -O3 -c mnist_call_benchmark.cpp -o mnist_call_benchmark.o && g++ mnist_call_benchmark.o mnist_model_llvm_ir.o -o bench.out \
	-L../../llvm-project/build/lib -lmlir_c_runner_utils \
	-lopenblas \
	-Wl,-rpath=../../llvm-project/build/lib \
	-no-pie