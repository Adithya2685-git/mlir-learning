from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

params = dict(model.named_buffers(remove_duplicate=False))

import numpy as np

with open('tensor.csv', 'w') as f:
    for value in params.values():
        if hasattr(value, 'numpy'):
            value = value.numpy()
        np.savetxt(f, [value.flatten()], delimiter=',', fmt='%d')
