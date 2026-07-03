g++ -O3 -c cnn_call_benchmark.cpp -o cnn_call_benchmark.o && g++ cnn_call_benchmark.o cnn_model_llvm_ir.o -o bench.out\
	-lopenblas \
	-lm -no-pie \