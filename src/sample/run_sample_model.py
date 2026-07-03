"""
    Print the output of a sample model given a tensor of ones as input.
"""

import torch
from model import Sample

# Limit to single thread for reproducibility
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

torch.manual_seed(41)
    

def print_output():
    model = Sample()
    input = torch.ones((3, 4))
    output = model(input)
    print("Input: ", input)
    print("Output: ", output)
    
if __name__ == "__main__":
   print_output()