# Chapter 4: Lowering models to x86 machine code

As mentioned in the tutorials, we will use existing passes to lower our models to the LLVM exit dialect (GPU later). We create our own pipeline with all the passes, so at the end, we only have to run one pass that runs them all. This pipeline is defined in [tools/tutorials-opt.cpp](https://github.com/DavidGinten/ML-compiler-exercise/blob/main/tools/tutorial-opt.cpp). I created two pipelines for demonstration, but they could also be merged. 

The first one creates a bufferized version of the model.

``` C++
void linalgToBufferizationPipelineBuilder(mlir::OpPassManager &manager) {
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createConvertTensorToLinalgPass());

  // One-shot bufferize
  mlir::bufferization::OneShotBufferizePassOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  manager.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
  mlir::bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  mlir::bufferization::buildBufferDeallocationPipeline(manager,
                                                       deallocationOptions);
}
```

In the second pipeline, we lower the bufferized version of the program down to the LLVM dialect that we can then use to exit the MLIR space, thereby getting closer to our object file. The first pass adds our own, newly created pass, converting Matmuls to OpenBLAS library call (More on that and what the pass looks like later).

``` C++
std::unique_ptr<mlir::Pass> createConvertMatmulToBlasLibraryCallPass() {
  return std::make_unique<mlir::tutorial::ConvertMatmulToBlasLibraryCallPass>();
}

void BufferizationToLLVMPipelineBuilder(mlir::OpPassManager &manager) {
  // CRITICAL: Replace matmuls with BLAS calls AFTER bufferization but BEFORE
  // other LLVM conversions
  manager.addPass(createConvertMatmulToBlasLibraryCallPass());

  // Convert remaining linalg ops to loops
  manager.addPass(mlir::createConvertLinalgToLoopsPass());

  // Standard LLVM lowering pipeline
  manager.addPass(mlir::memref::createExpandStridedMetadataPass());
  manager.addPass(mlir::createLowerAffinePass());
  manager.addPass(mlir::affine::createLoopFusionPass());
  manager.addPass(mlir::affine::createAffineVectorize());
  manager.addPass(mlir::createSCFToControlFlowPass());

  // Convert to LLVM - order matters here
  manager.addPass(mlir::createArithToLLVMConversionPass());
  manager.addPass(mlir::createConvertMathToLLVMPass());
  manager.addPass(
      mlir::createConvertMathToLibmPass()); // For bert model to lower math.erf
  manager.addPass(mlir::createConvertControlFlowToLLVMPass());
  manager.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  manager.addPass(mlir::createConvertFuncToLLVMPass());
  manager.addPass(mlir::createReconcileUnrealizedCastsPass());

  // Cleanup
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createSCCPPass());
  manager.addPass(mlir::createCSEPass());
  manager.addPass(mlir::createSymbolDCEPass());
}
```

In the main function, we first need to register the dialect and the passes (including our own). We then register our pipelines, giving them names for mlir-opt and a description:

``` C++
  mlir::PassPipelineRegistration<>(
      "linalg-to-bufferization",
      "Run passes to lower the linalg dialect to bufferization",
      linalgToBufferizationPipelineBuilder);

  mlir::PassPipelineRegistration<>(
      "bufferization-to-llvm", "Run passes to lower bufferized code to LLVM",
      BufferizationToLLVMPipelineBuilder);
```

After building (using build.sh), our passes/pipelines are available through the use of tutorial-opt, our own extended version of mlir-opt. (Talking about mlir-opt, torch-mlir also has its own torch-mlir-opt. For example, it converts the torch-mlir specific torch dialect into linalg-on-tensor, which we use for our resnet model --> See further down).

Now we can run the whole pipeline in one step, lowering our model from linalg-on-tensor directly to an executable object file. Thereby, we first call our `linalg-to-bufferization` and `bufferization-to-llvm` pass. As you notice, I also added the `--llvm-request-c-wrappers` pass. That emits a C-friendly callable function, prefixed with _mlir_ciface_ (@_mlir_ciface_sample_model() for our sample model). Through this function, we can call the model from C/C++ in a rather compact way with a so-called MemRefDescriptor struct. Especially for large models (e.g., resnet18), we can just pass this struct containing and wrapping all the inputs. Otherwise, we would need to pass the components of the struct individually.

```shell
# With BLAS integration
../../build-ninja/tools/tutorial-opt --linalg-to-bufferization $PWD/sample_model_linalg.mlir > $PWD/sample_model_buf_linalg.mlir
../../build-ninja/tools/tutorial-opt --llvm-request-c-wrappers --bufferization-to-llvm $PWD/sample_model_buf_linalg.mlir > $PWD/sample_model_llvm.mlir

###  Use mlir-translate to get from mlir to llvm ir  ###
mlir-translate -mlir-to-llvmir $PWD/sample_model_llvm.mlir > $PWD/sample_model_llvm_ir.ll

###  Create .obj file  ###
llc --filetype=obj $PWD/sample_model_llvm_ir.ll

###  Compile  ###
g++ -c sample_call.cpp -o sample_call.o && g++ sample_call.o sample_model_llvm_ir.o -o a.out -L../../lib -lopenblas -lm
```

In our C++ program [sample_call.cpp](https://github.com/DavidGinten/ML-compiler-exercise/blob/main/src/sample/sample_call.cpp), we initialize the input tensor, set offset, size, and stride, i.e., we initialize our input and output MemRefDescriptor. We can then call the sample_model using the C wrapper interface. We pass both input and output Descriptors to the function and can thus access the output data, for example, via `float *output = (float *)outputMemRef.aligned;`. With the correct memory access (consider size and stride), we can print the output.    

Definition of the MemRefDescriptor struct:
``` C++
template <typename T, int N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};
```

The function signature of our model looks as follows:
``` C++
void _mlir_ciface_sample_model(MemRefDescriptor<float, 2> *output,
                               MemRefDescriptor<float, 2> *input);
```

We have already compiled and linked the code with our model code.

Execute: `./a.out`

The same procedure is also applied for the other dummy models: mnist and cnn

For the other models, we have to do a little bit more..

## Passing buffers as inputs
When importing resnet, bert, gpt2 to torch-mlir (using torch.export()) some buffers of the model are moved out and are now part of the input. This seems to be the current default in torch-mlir/torch.export, as they also extract the buffers before executing the model in [this](https://github.com/llvm/torch-mlir/blob/main/projects/pt1/examples/fximporter_resnet18.py#L43) example.   
- For the resnet18 model, I extract the params from the PyTorch model ((get_buffers_in_mlir_format.py)[https://github.com/DavidGinten/ML-compiler-exercise/blob/main/src/resnet18/get_buffers_in_mlir_format.py]) and hard code them into the mlir model. 
- For the bert model, I [extract](https://github.com/DavidGinten/ML-compiler-exercise/blob/main/src/bert-base-uncased/get_buffers.py) those buffers and directly write them to a .csv file. That file is later loaded in our C++ program. Here it is just two tensors, whereas for the resnet is about 60 arguments. Also, for the resnet model, in the MLIR code every third argument is of type !torch.vtensor<[],si64> and they are nowhere used in the code. GPT2 has also up to 20 arguments that are not used in the code. I couldn't find a pass that cleans up those dead/unused arguments. Neither in torch-mlir-opt, nor in mlir-opt. 

## Execution of bert-base-uncased
The basic structure of [call_bert.cpp](https://github.com/DavidGinten/ML-compiler-exercise/blob/main/src/bert-base-uncased/call_bert.cpp) is the same as before. 

But a few things have changed:
1. We have a load_tensor() function that loads the buffers that we have previously written in a .csv file. We can now use this data and pass it to    our model as well.

2. The model has two outputs: last_hidden_state and pooler_output. I defined the following struct:
```C++
struct Output {
  MemRefDescriptor<float, 3> output_last_hidden_state_MemRef;
  MemRefDescriptor<float, 2> output_pooler_output_MemRef;
};
```
Here, we can now fit out Memrefs for those two output buffers and pass this struct to the model as the output buffer.

3. In torch-mlir, I added [AtenAnyDimOp](https://github.com/DavidGinten/torch-mlir/blob/97bf9d2e6313abae5eb890748207baa178caeddf/lib/Conversion/TorchToLinalg/Reduction.cpp) in the reduction pass of TorchToLinalg. Without that, the op aten.any.dim.op wouldn't be lowered and that also keeps torch.constant.int op from being lowered to arith.constant. This then results in a `error: failed to legalize operation 'torch.constant.int'`. As mentioned in Chapter 3, the direct export to LINALG_ON_TENSORS still fails with this error, however, first going to TORCH and then to linalg via `-torch-backend-to-linalg-on-tensors-backend-pipeline` works. 

## Execution of google's flan-t5-small
For this model, I created dynmaic input_ids and output_ids. I use pybind11, so that we are able to call our model with varying inputs from python. Then the produced output tokens are also directly decoded into text. **In principle, this can be adapted for all the other models as well.**

So we dynmically get out input tokens and wrap them in memref descriptors. We also initialize the decoder with a start token (0).

We then enter a loop that simulates step-by-step text generation. At each step, we allocate a logit buffer for the vocabulary (size 32,128), call the model with the current decoder sequence, and verify that the model actually writes to the output buffer. We filter invalid values, find the top-5 token predictions by score, print them, and greedily select the highest-scoring token as the next decoder token. Generation stops early if the end-of-sequence token (1) is produced or after 20 steps have been taken. Finally, the full generated token sequence is returned back to python where we print and decode the tokens.

