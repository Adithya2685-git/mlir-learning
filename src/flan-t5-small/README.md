- `python lower_flan_model.py` to import the Pytorch model to mlir using torch-mlir
- `python run_flan_model.py` to print the output with sample input
- `python benchmark_flan_model.py` to benchmark the sample model
- `sh run_mlir_pipeline.sh` to get the executable that runs the with MLIR lowered PyTorch model.
- `sh generate_and_decode.sh` calls the compiled FLAN T5 small model and decodes the output sequence.

### Benchnmark results:
- PyTorch avg. inference time (CPU): 0.1178 sec
- MLIR pipeline avg. inference time (CPU): 0.397444 sec