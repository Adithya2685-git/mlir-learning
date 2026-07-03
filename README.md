# ML compiler exercise

## Abstract

This is an open-source, online tutorial that provides an end-to-end introduction to MLIR, demonstrating how deep learning models can be lowered from a machine learning framework to executable binaries. Aimed at newcomers and university students, the tutorial focuses on conveying the core concepts of the MLIR compilation flow rather than on performance optimization. Using torch-mlir, upstream MLIR passes, and a custom lowering pass targeting OpenBLAS, the talk illustrates how to build and extend a practical ML compilation pipeline targeting both CPU and NVIDIA GPU backends. The tutorial is designed to lower the barrier to entry and broaden participation in the MLIR community. From a newcomer to other newcomers, so to speak.

## Tutorial content

The repo provides some PyTorch models (small sample models and real world models) and provides the following:
1. Import models from PyTorch and HF to MLIR using [torch-mlir](https://github.com/llvm/torch-mlir)
2. Use existing MLIR passes to lower from entry level IR's (linalg, arith, ...) to llvm ir
3. Create the corrsponding object file
4. Call the model via a function call in C++

Additionally, I created a pass that converts linalg.matmul's to [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) matrix-multiplication function calls. Further, I target an Nvidia GPU (sm_90) to launch specific kenrels on the GPU (This currently only works for sample and the mnist model). 

*Warning*: Some instructions are RWTH cluster specific, e.g. paths.

1. [Introduction](https://github.com/DavidGinten/ML-compiler-exercise/blob/main/docs/Chapter1.md)
2. [Getting started and project setup](https://github.com/DavidGinten/ML-compiler-exercise/blob/main/docs/Chapter2.md)
3. [Importing PyTorch models to torch-mlir](https://github.com/DavidGinten/ML-compiler-exercise/blob/main/docs/Chapter3.md)
4. [Lowering models to x86 machine code](https://github.com/DavidGinten/ML-compiler-exercise/blob/main/docs/Chapter4.md)
5. [Integration of OpenBLAS for Matrix Multiplications](https://github.com/DavidGinten/ML-compiler-exercise/blob/main/docs/Chapter5.md)
6. [Targeting an Nvidia GPU](https://github.com/DavidGinten/ML-compiler-exercise/blob/main/docs/Chapter6.md)

*Appendix*: [An overview of IREE](https://github.com/DavidGinten/ML-compiler-exercise/blob/main/docs/iree_appendix.md)

## Build
See [Chapter 2](https://github.com/DavidGinten/ML-compiler-exercise/blob/main/docs/Chapter2.md) in the docs.

## References
- [Official MLIR website](https://mlir.llvm.org/) 
- [MLIR for Beginners](https://www.jeremykun.com/2023/08/10/mlir-getting-started/) by Jeremy Kun
- [MLIR tutorial with GPU compilation](https://www.stephendiehl.com/tags/mlir/) by Stephen Diehl
