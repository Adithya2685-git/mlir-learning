"""
    Print the output of a resnet18 model given a tensor of ones as input.
"""

import torch
import torch.nn.functional as F

# Limit to single thread for reproducibility
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

torch.manual_seed(41)

from transformers import AutoModelForImageClassification
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
model.eval()

def print_output():
    x = torch.ones((1, 3, 224, 224))
    output = model(x)
    probabilities = F.softmax(output.logits , dim=1)
    top_class = probabilities.argmax(dim=1)

    print("Input shape: ", x.shape)
    print("Output logits shape:", output.logits.shape)
    print("Logits: ", output.logits)
    print("Top class index:", top_class.item())

if __name__ == "__main__":
   print_output()