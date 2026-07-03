"""
    Print the output of a sample model given an example text input
	using the MLIR lowered version of the model.
"""
from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

input_text = "translate English to German: What is your name?"
input = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
print(f'Input text: "{input_text}"')
print("Corresponding input IDs: ", input.input_ids)
print("Corresponding attention mask: ", input.attention_mask)

import flan_call
result = flan_call.generate_tokens(input.input_ids.tolist()[0], input.attention_mask.tolist()[0])
print("Generated tokens: ", result)
print(tokenizer.decode(result))