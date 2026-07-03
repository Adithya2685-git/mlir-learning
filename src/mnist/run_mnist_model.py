"""
    Print the output of a sample model given a tensor of ones as input.
"""

import torch
from model import MnistNetwork

# Limit to single thread for reproducibility
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

torch.manual_seed(41)
    

def print_output():
    model = MnistNetwork()
    input = torch.ones((1, 1, 28, 28))
    output = model(input)
    print("Input shape: ", input.shape)
    print("Output shape: ", output.shape)
    print("Output: ", output)

if __name__ == "__main__":
   print_output()