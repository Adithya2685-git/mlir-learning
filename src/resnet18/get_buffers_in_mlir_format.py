" Get ResNet-18 model buffers and write them to a file in hex format. "

import struct
from transformers import AutoModelForImageClassification
import torch.utils._pytree as pytree

# --- Load model ---
resnet18 = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
resnet18.eval()

# --- Helper decorator ---
def run(f):
    def wrapper():
        print(f"{f.__name__}")
        print("-" * len(f.__name__))
        f()
        print()
    return wrapper

# --- Float tensor --> hex string ---
def tensor_to_hex(tensor):
    """Convert a float32 tensor into a LLVM-style hex literal."""
    tensor = tensor.float().detach().cpu().numpy()
    hex_parts = [struct.pack("<f", v).hex().upper() for v in tensor]
    return "0x04000000" + "".join(hex_parts)

# --- Get model parameters ---
@run
def get_params():
    params = dict(resnet18.named_buffers(remove_duplicate=False))
    params_flat, _ = pytree.tree_flatten(params)

    with open("resnet18_params.txt", "w") as f:
        tensor_sizes = []

        for i, tensor in enumerate(params_flat):
            if tensor.dim() == 0:  # Skip scalars
                continue

            hex_blob = tensor_to_hex(tensor)
            f.write(f'torch_tensor_custom_{i}.float32: "{hex_blob}",\n')

            num_elems = (len(hex_blob) - 10) // 8  # each float32 = 8 hex chars
            tensor_sizes.append((i, num_elems))

        for i, n in tensor_sizes:
            f.write(
                f"%arg{i} = torch.vtensor.literal("
                f"dense_resource<torch_tensor_custom_{i}.float32> "
                f": tensor<{n}xf32>) : !torch.vtensor<[{n}],f32>\n"
            )

if __name__ == "__main__":
    get_params()