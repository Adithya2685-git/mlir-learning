g++ -c resnet18_call.cpp -o resnet18_call.o && g++ resnet18_call.o resnet18_llvm_ir.o -o a.out \
	-L../../llvm-project/build/lib -lmlir_c_runner_utils \
	-lopenblas \
	-Wl,-rpath=../../llvm-project/build/lib \
	-no-pie