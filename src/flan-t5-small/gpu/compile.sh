g++ -c flan_call.cpp -o flan_call.o && g++ flan_call.o flan.o -o a.out \
	-L../../../llvm-project/build/lib -lmlir_runner_utils -lmlir_cuda_runtime \
	-L/opt/cuda/lib64 -lcuda -lcudart \
	-Wl,-rpath,'./../../../llvm-project/build/lib' \
	-Wl,-rpath,'/opt/cuda/lib64'