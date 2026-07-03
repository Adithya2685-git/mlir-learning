""" Sample code to demonstrate lowering a PyTorch model to various IRs using torch-mlir. """

from model import ConvolutionalNetwork

import torch
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType

torch.manual_seed(41)

def run(f):
    def wrapper():
        print(f"{f.__name__}")
        print("-" * len(f.__name__))
        f()
        print()
    return wrapper

torch.manual_seed(41)


@run
def lower_pytorch_to_torch_fx():
    from torch.fx import symbolic_trace
    # Symbolic tracing frontend - captures the semantics of the module
    symbolic_traced: torch.fx.GraphModule = symbolic_trace(ConvolutionalNetwork())

    # High-level intermediate representation (IR) - Graph representation
    print(symbolic_traced.graph)


@run
def lower_pytorch_to_raw_output():
    # Export model to torch-mlir
    m = fx.export_and_import(ConvolutionalNetwork(), torch.randn(4, 1, 28, 28), output_type=OutputType.RAW,
                             func_name = "cnn_model")

    mlir_str = str(m)

    """ Print without builtin dialect_resources """
    #mlir_ir = mlir_str.split("{-#")[0].strip()
    #print(mlir_ir)

    with open("cnn_model_raw.mlir", "w") as f:
        f.write(mlir_str)

@run
def lower_pytorch_to_torch_mlir():
    # Export model to torch-mlir
    m = fx.export_and_import(ConvolutionalNetwork(), torch.randn(4, 1, 28, 28), output_type=OutputType.TORCH,
                             func_name = "cnn_model")

    mlir_str = str(m)

    """ Print without builtin dialect_resources """
    #mlir_ir = mlir_str.split("{-#")[0].strip()
    #print(mlir_ir)

    with open("cnn_model_torch.mlir", "w") as f:
        f.write(mlir_str)

@run
def lower_pytorch_to_linalg_on_tensors():
    # Export model to torch-mlir
    m = fx.export_and_import(ConvolutionalNetwork(), torch.randn(4, 1, 28, 28), output_type=OutputType.LINALG_ON_TENSORS, 
                             func_name = "cnn_model")
    
    mlir_str = str(m)

    """ Print without builtin dialect_resources """
    #mlir_ir = mlir_str.split("{-#")[0].strip()
    #print(mlir_ir)

    with open("cnn_model_linalg.mlir", "w") as f:
        f.write(mlir_str)

if __name__ == "__main__":
    lower_pytorch_to_linalg_on_tensors()
    #lower_pytorch_to_torch_mlir()
    #lower_pytorch_to_raw_output()
    #lower_pytorch_to_torch_fx()  