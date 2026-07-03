"""
    Benchmark the sample model inference time
"""

import torch, time

# Limit to single thread for reproducibility
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

torch.manual_seed(41)

# Load ResNet-18 model from Hugging Face Transformers
from transformers import AutoModelForImageClassification
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")

def benchmark():
    model.eval()
    
    x = torch.ones((1, 3, 224, 224))
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)

    # Timed loop
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(x)
    end = time.time()

    print(f"Avg inference time: {(end - start) / 100:.6f} sec")


if __name__ == "__main__":
   benchmark()