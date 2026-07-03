"""
    Benchmark the sample model inference time
"""

import time
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

input_text = "translate English to German: How are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Warmup
for _ in range(3):
    model.generate(input_ids)

# Benchmark
runs = 10
times = []

for _ in range(runs):
    start = time.time()
    model.generate(input_ids)
    end = time.time()
    times.append(end - start)

print(f"Avg inference time: {sum(times)/len(times):.4f} s")
print(f"Min: {min(times):.4f} s, Max: {max(times):.4f} s")
