- `python lower_mnist_model.py` to import the Pytorch model to mlir using torch-mlir
- `python run_mnist_model.py` to print the output with sample input
- `python benchmark_mnist_model.py` to benchmark the sample model
- `sh run_mlir_pipeline.sh` to get the executable that runs the with MLIR lowered PyTorch model.

### Benchnmark results:
- PyTorch avg. inference time (CPU): 0.000151 sec
- MLIR pipeline avg. inference time (CPU): 0.000853 sec
- MLIR pipeline avg. inference time (GPU): 0.005568 sec