clang++ -c call_bert.cpp -o call_bert.o && clang++ -lm call_bert.o bert_llvm_ir.o -o a.out \
    -L../../llvm-project/build/lib \
    -lmlir_c_runner_utils \
    -lopenblas \
    -Wl,-rpath=../../llvm-project/build/lib \
    -no-pie