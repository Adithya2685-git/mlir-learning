g++ -O3 -c resnet18_call_benchmark.cpp -o resnet18_call_benchmark.o && g++ resnet18_call_benchmark.o resnet18_model_llvm_ir.o -o bench.out \
	-L../../llvm-project/build/lib -lmlir_c_runner_utils \
	-lopenblas \
	-Wl,-rpath=../../llvm-project/build/lib \
	-no-pie