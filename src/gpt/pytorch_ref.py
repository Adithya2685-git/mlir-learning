"""Run GPT-2 inference via PyTorch — reference output for comparison."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

input_text = "The capital of France is"
inputs = tokenizer(input_text, return_tensors="pt")

print(f"Input: {input_text}")
print(f"Token IDs: {inputs['input_ids'].tolist()}")
print(f"Attention mask: {inputs['attention_mask'].tolist()}")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits  # shape: [1, seq_len, vocab]

print(f"\nOutput shape: {logits.shape}")

# Last token logits
last_logits = logits[0, -1, :]
print(f"\nLast token logits (first 10):")
for i in range(10):
    print(f"  vocab[{i:5d}] = {last_logits[i].item():14.6f}")

# Top-5
top5 = torch.topk(last_logits, 5)
print(f"\nTop-5 predictions:")
for i in range(5):
    print(f"  #{i+1}: token_id={top5.indices[i].item():5d}  logit={top5.values[i].item():.4f}")
    print(f"        token='{tokenizer.decode([top5.indices[i].item()])}'")

# Save reference outputs for comparison
torch.save(logits, "/home/adi/tmp/pytorch_gpt2_logits.pt")
print(f"\nSaved reference logits to /home/adi/tmp/pytorch_gpt2_logits.pt")
