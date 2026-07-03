# A GPU compilation pipeline completely in Python

This is basically a copy of [Stephan Diehl's GPU pipeline](https://github.com/sdiehl/gpu-offload/tree/main). 

- It can be used to check if all tools for Nvidia GPU deployment are available and working.
- It's also an example of a full, automatic MLIR-linalg to GPU kernel launch pipeline completely in Python.   

## Requirements:

`pip install mlir_python_bindings -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest`

`pip install cuda-python==12.6.0` (install cuda-python 12.x.x, if CUDA 12.x is installed, otherwise `pip install cuda-python` is sufficient)