g++ -O3 -c sample_call_benchmark.cpp -o sample_call_benchmark.o && g++ sample_call_benchmark.o sample_model_llvm_ir.o -o bench.out \
	-lopenblas -lm -no-pie