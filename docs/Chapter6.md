# Chapter 6: Targeting an Nvidia GPU

First, a complete example of a square matrix operation in Python, including the corresponding kernel launch, can be found [here](https://github.com/DavidGinten/ML-compiler-exercise/tree/main/src/python/gpu). This example is taken from Stephan Diehl's [GPU compilation with MLIR](https://www.stephendiehl.com/posts/mlir_gpu/) and can be used to test the GPU availability.

However, we will stick with our own pipeline. I'll now consider the example of the sample model. This time, I have just created a [shell script](https://github.com/DavidGinten/ML-compiler-exercise/blob/main/src/sample/gpu/run_mlir_pipeline.sh) that executes all the relevant passes on our input linalg-on-tensor model. The given pipeline works for our models (T5 does not work yet). However, the tile size needs to be properly selected. You can play around with it and compare the performance. Smaller models need a smaller tile size, bigger models, and larger ones.

When [compiling](https://github.com/DavidGinten/ML-compiler-exercise/blob/main/src/sample/gpu/compile.sh), we link our MLIR compiled code and, of course, need to link the CUDA runtime (The path where your CUDA is stored probably differs).

After compiling, we can simply run our C++ code again as if targeting the CPU. MLIR does all the kernel launches for us.