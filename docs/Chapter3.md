# Chapter 3: Importing PyTorch models to torch-mlir
We will now look at how to import PyTorch models to torch-mlir and how it outputs MLIR code in the torch dialect (no upstream MLIR dialect) and in linalg-on-tensor form. (There's also the RAW output, but we will not use it here.)
Thereby, linalg-on-tensor is a combination of linalg, arith, and other base dialects.
The procedure is basically the same for all models (except for the transformer; see further below). It's implemented in the "lower_\<model_name\>_python.py" files. Here we'll take a look at [lower_sample_model.py](https://github.com/DavidGinten/ML-compiler-exercise/blob/main/src/sample/lower_sample_model.py).

The simple sample model looks like this:
```python
class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return self.linear(x + self.param).clamp(min=0.0, max=1.0)
```
This model basically adds a bias term to x, performs a linear transformation (y = xW^T + b), and clamps the output to 0 and 1 (0 < y < 1). The output shape will be 3x5.

## Execute the models with PyTorch
You can execute [run_sample_model.py](https://github.com/DavidGinten/ML-compiler-exercise/blob/main/src/sample/run_sample_model.py) to run the model with random but constant inputs. This file also exists for all other models.

## Benchmark the models in PyTorch
Run [benchmark_sample_model.py](https://github.com/DavidGinten/ML-compiler-exercise/blob/main/src/sample/benchmark_sample_model.py) to benchmark the model with random but constant inputs. After a warm-up phase, it calls the model 100 times and reports the average inference time. This file also exists for all other models. For the transformer model, the number of iterations is dropped to 10 due to its size.

## Importing a transformer model
The following refers to this [file](https://github.com/DavidGinten/ML-compiler-exercise/blob/main/src/transformer/lower_flan_model.py).
The T5 model I use here in generall has a complex forward() signature:

```python
model(
    input_ids=...,
    attention_mask=...,
    decoder_input_ids=...,
    labels=...,
    output_hidden_states=...,
    output_attentions=...,
    return_dict=True,
    ...
)
```

The model’s output also contains various attributes that we are not interested in, such as encoder_hidden_states or cross_attentions. The problem is that the torch.export system cannot easily export models with:

- multiple optional keyword arguments,

- dictionary-style outputs,

- side data (states, caches, hidden states), or

- multiple internal model heads.

To avoid these issues, we create a simple wrapper that exposes a clean forward() signature:
`forward(input_ids, attention_mask, decoder_input_ids)`

It returns only a single tensor: the decoder output logits. This simplifies the graph and ensures that the exported model fits into the constraints of torch.export.

In export_model(), we first load the tokenizer and construct an example input, which gives us input_ids and attention_mask. These serve as fixed sample inputs that torch.export uses to trace the model.
We then load the HuggingFace model:
`model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")`

and wrap it:
`wrapped_model = Wrapper(model)`

This produces a simple interface for the exported function:
`(input_ids, attention_mask, decoder_input_ids) → logits`

Next, we define a decoder start sequence. In this example, we use two <pad> tokens because dynamic shapes require at least a small starting sequence; there may be cleaner alternatives, but this works for now.

Calling torch.export produces a TorchExportedProgram, a graph-like program representation that is ready for lowering. We explicitly call torch.export here, but the same process could be performed inside the torch-mlir export_and_import() function—torch.export would simply run internally.

In our export call, we provide the model, the example inputs, and the dynamic shape specifications. We fix the batch size to 1, allow the encoder sequence length to vary up to 10, and mark the decoder sequence length as dynamic so that it can grow during runtime (dec_len ≤ 16).

After exporting, we call .run_decompositions() on the graph. This step decomposes high-level PyTorch operations into smaller primitive ops, removes composite kernels, and produces a cleaner, lower-level graph that can be translated into MLIR’s Linalg-on-Tensors dialect.

With that, the model is ready to be imported into MLIR.

## Importing the model to MLIR
We now use torch-mlir to import our model into an MLIR representation.
The recommended API for this is the FX-based exporter, available through fx.export_and_import
(torch_mlir.compile() is deprecated).
To use it, we first import the FX utilities:

`from torch_mlir import fx`

The function export_and_import accepts several parameters, but for this example we only need four.
A typical call looks like this:

```python
m = fx.export_and_import(
    MyModule(),
    torch.randn(3, 4),
    output_type=OutputType.TORCH,
    func_name="sample_model",
)
```
Here we pass:
- the model instance,

- an example input tensor,

- the desired output dialects used, e.g., TORCH, LINALG_ON_TENSORS (If it works, we use LINALG_ON_TENSORS so we directly leave torch-mlir behind),

- and the function name that will appear in the generated MLIR.

The resulting MLIR module can then be saved to a .mlir file.
It yields:

```
module {
  func.func @main(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,5],f32> {
    %int0 = torch.constant.int 0
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %float0.000000e00 = torch.constant.float 0.000000e+00
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_5_torch.float32> : tensor<5xf32>) : !torch.vtensor<[5],f32>
    %1 = torch.vtensor.literal(dense_resource<torch_tensor_5_4_torch.float32> : tensor<5x4xf32>) : !torch.vtensor<[5,4],f32>
    %2 = torch.vtensor.literal(dense_resource<torch_tensor_3_4_torch.float32> : tensor<3x4xf32>) : !torch.vtensor<[3,4],f32>
    %int1 = torch.constant.int 1
    %3 = torch.aten.add.Tensor %arg0, %2, %int1 : !torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],f32>, !torch.int -> !torch.vtensor<[3,4],f32>
    %4 = torch.aten.transpose.int %1, %int0, %int1 : !torch.vtensor<[5,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[4,5],f32>
    %5 = torch.aten.mm %3, %4 : !torch.vtensor<[3,4],f32>, !torch.vtensor<[4,5],f32> -> !torch.vtensor<[3,5],f32>
    %6 = torch.aten.add.Tensor %5, %0, %int1 : !torch.vtensor<[3,5],f32>, !torch.vtensor<[5],f32>, !torch.int -> !torch.vtensor<[3,5],f32>
    %7 = torch.aten.clamp %6, %float0.000000e00, %float1.000000e00 : !torch.vtensor<[3,5],f32>, !torch.float, !torch.float -> !torch.vtensor<[3,5],f32>
    return %7 : !torch.vtensor<[3,5],f32>
  }
}
```

And for the LINALG_ON_TENSOR version, we get:
``` 
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @sample_model(%arg0: tensor<3x4xf32>) -> tensor<3x5xf32> {
    %cst = arith.constant dense_resource<torch_tensor_3_4_torch.float32> : tensor<3x4xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant dense_resource<torch_tensor_5_4_torch.float32> : tensor<5x4xf32>
    %cst_3 = arith.constant dense_resource<torch_tensor_5_torch.float32> : tensor<5xf32>
    %0 = tensor.empty() : tensor<3x4xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %cst : tensor<3x4xf32>, tensor<3x4xf32>) outs(%0 : tensor<3x4xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %8 = arith.addf %in, %in_4 : f32
      linalg.yield %8 : f32
    } -> tensor<3x4xf32>
    %2 = tensor.empty() : tensor<4x5xf32>
    %transposed = linalg.transpose ins(%cst_2 : tensor<5x4xf32>) outs(%2 : tensor<4x5xf32>) permutation = [1, 0] 
    %3 = tensor.empty() : tensor<3x5xf32>
    %4 = linalg.fill ins(%cst_0 : f32) outs(%3 : tensor<3x5xf32>) -> tensor<3x5xf32>
    %5 = linalg.matmul ins(%1, %transposed : tensor<3x4xf32>, tensor<4x5xf32>) outs(%4 : tensor<3x5xf32>) -> tensor<3x5xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%5, %cst_3 : tensor<3x5xf32>, tensor<5xf32>) outs(%3 : tensor<3x5xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %8 = arith.addf %in, %in_4 : f32
      linalg.yield %8 : f32
    } -> tensor<3x5xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<3x5xf32>) outs(%3 : tensor<3x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.cmpf ult, %in, %cst_0 : f32
      %9 = arith.select %8, %cst_0, %in : f32
      %10 = arith.cmpf ugt, %9, %cst_1 : f32
      %11 = arith.select %10, %cst_1, %9 : f32
      linalg.yield %11 : f32
    } -> tensor<3x5xf32>
    return %7 : tensor<3x5xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_3_4_torch.float32: "0x04000000001F723E8C0E683E8CEF4C3F883F2D3EA0A8873E41A1453FCC43033E59C63E3F97F14D3FD5BD223F1AF1163F6D7B313F",
      torch_tensor_5_4_torch.float32: "0x04000000D8A9C13E809B263DFA4EB8BE4635EC3E64B9BB3E80CE3DBC20A895BEB04E46BE5409E2BE7AA7A93E70B931BD6661B03E70C7403E4025CBBD6AB2A3BE7031323E9842A4BE4892E3BDEC2F983EDE668BBE",
      torch_tensor_5_torch.float32: "0x040000006061C8BC30FC8ABD20AE6A3DA27BF63E38A6953E"
    }
  }
#-}
```

Looking at this code, we can see that it has no torch ops anymore, so it only depends on upstream MLIR dialects. With this code, we can continue with the lowering process in the next chapter.


## When the import to LINALG_ON_TENSORS breaks
Currently, when trying to lower bert and gpt2 to LINALG_ON_TENSORS, it fails with `error: failed to legalize operation 'torch.constant.int'`. However, when you first lower with torch-mlir to the torch dialect (i.e. TORCH) and then use `-torch-backend-to-linalg-on-tensors-backend-pipeline` we get to the representation we want (i.e. LINALG_ON_TENSORS):

```shell
# torch-mlir: Convert from torch dialect to linalg dialect
torch-mlir-opt --torch-backend-to-linalg-on-tensors-backend-pipeline resnet18_model_torch.mlir > resnet18_model_linalg.mlir
```