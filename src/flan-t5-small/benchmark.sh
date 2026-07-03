g++ -O3 -c flan_call_benchmark.cpp -o flan_call_benchmark.o && g++ flan_call_benchmark.o flan_llvm_ir.o -o bench.out\
	-L../../llvm-project/build/lib -lmlir_c_runner_utils \
	-lopenblas \
	-Wl,-rpath=../../llvm-project/build/lib