from transformers import BertTokenizer, BertModel
import torch
from torch_mlir import fx
from torch_mlir.fx import OutputType

""" A simple wrapper around the bert model to adapt its forward method for export."""
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state , outputs.pooler_output

""" Exports the bert model to Linalg on Tensors dialect in MLIR."""
def export_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    text = "This is a sample input for BERT model export."
    encoded_input = tokenizer(text, return_tensors='pt')

    wrapped_model = Wrapper(model)

    ep = torch.export.export(
        wrapped_model,
        (
            encoded_input["input_ids"],
            encoded_input["token_type_ids"],
            encoded_input["attention_mask"],
        ),
    )
    
    ep = ep.run_decompositions()    

    m = fx.export_and_import(
        ep,
        output_type=OutputType.TORCH,
        func_name="bert_model"
    )

    mlir_str = str(m)
    with open("bert_torch.mlir", "w") as f:
        f.write(mlir_str)

if __name__ == "__main__":
    export_model()
