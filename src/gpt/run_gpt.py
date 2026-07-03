from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Optional but common for GPT-2
tokenizer.pad_token = tokenizer.eos_token

input_text = "What is the capital of France?"

# Tokenize (this already creates attention_mask)
inputs = tokenizer(input_text, return_tensors="pt")

print(f'Input text: "{input_text}"')
print("Input IDs:", inputs["input_ids"])
print("Attention mask:", inputs["attention_mask"])
print(inputs)

# Pass attention_mask to generate
outputs = model.generate(
    **inputs,
	max_new_tokens=10,
    pad_token_id=tokenizer.eos_token_id
)

print("Output IDs:", outputs)
print("Decoded output:\n", tokenizer.decode(outputs[0], skip_special_tokens=True))
