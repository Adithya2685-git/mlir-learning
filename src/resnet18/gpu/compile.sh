g++ -c resnet18_call.cpp -o resnet18_call.o && g++ resnet18_call.o resnet18.o -o a.out \
	-L../../../llvm-project/build/lib -lmlir_c_runner_utils -lmlir_cuda_runtime \
	-L/opt/cuda/lib64 -lcuda -lcudart \
	-Wl,-rpath,'./../../../llvm-project/build/lib' \
	-Wl,-rpath,'/opt/cuda/lib64'