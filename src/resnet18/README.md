Currently, when impoting the resnet model to torch-mlir (using torch.export()) 
many params of the model are expected to be passed as input. The number of input tensors 
increases from 1 to over 500. My solution by now isn't pretty, but it works. 
Basically I extract the params from the model and manually insert them into the mlir model.

### Benchnmark results:
- PyTorch avg. inference time (CPU): 0.045876 sec
- MLIR pipeline avg. inference time (CPU): 6.889084 sec
- MLIR pipeline avg. inference time (GPU): 0.619498 sec


TODO: Usually, in our MLIR code every third argument is !torch.vtensor<[],si64> and they are nowhere used in the code.
I couldn't find a pass that cleans up those dead/unused arguments.
