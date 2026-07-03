from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
text = "This is a sample input for BERT model export."
encoded_input = tokenizer(text, return_tensors='pt')

print("Input: ", encoded_input)

output = model(**encoded_input)

print("Output: ", output)
print(output.last_hidden_state.shape)
print(output.pooler_output.shape)