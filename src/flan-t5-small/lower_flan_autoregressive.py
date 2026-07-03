"""
Full autoregressive text generation with dynamically-exported Flan-T5.

This demonstrates:
1. Exporting the model with dynamic decoder dimensions
2. Using the exported model for step-by-step token generation
3. Maintaining autoregressive generation without KV cache

Key insight: Even though we don't use KV cache, we can still generate
tokens one at a time (or in batches) by repeatedly calling the exported model
with growing decoder_input_ids.
"""

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch_mlir import fx
from torch_mlir.fx import OutputType
import torch.nn.functional as F


class SimpleDecoderWrapper(torch.nn.Module):
    """
    Wrapper that returns only logits for the LAST token in the sequence.
    
    This is useful for autoregressive generation where you only care about
    the next token prediction, not the full sequence of logits.
    """
    def __init__(self, model, return_all_logits=False):
        super().__init__()
        self.model = model
        self.return_all_logits = return_all_logits
    
    def forward(self, input_ids, attention_mask, decoder_input_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids
        )
        # Return only the logits for the last token if not returning all
        if self.return_all_logits:
            return outputs.logits
        else:
            return outputs.logits[:, -1:, :]  # (batch, 1, vocab_size)


def export_for_autoregressive():
    """Export Flan-T5 with dynamic decoder sequence dimension."""
    print("=" * 70)
    print("Exporting Flan-T5 for autoregressive generation")
    print("=" * 70)
    
    # Setup
    print("\n1. Loading model and tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    tokenizer.pad_token = tokenizer.eos_token
    
    encoded_input = tokenizer(
        "translate English to German: How are you?",
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    model.eval()
    model.config.use_cache = False
    
    # Warm up
    print("2. Warming up model...")
    with torch.no_grad():
        model(
            input_ids=encoded_input["input_ids"],
            attention_mask=encoded_input["attention_mask"],
            decoder_input_ids=torch.tensor([[tokenizer.pad_token_id]])
        )
    
    # Wrap and export
    print("3. Creating wrapper and exporting...")
    wrapped_model = SimpleDecoderWrapper(model, return_all_logits=True)
    
    # Dynamic dimension: decoder can accept 1 to 32 tokens
    encoder_seq_len = torch.export.Dim("encoder_seq_len", max=128)
    decoder_seq_len = torch.export.Dim("decoder_seq_len", max=32)
    decoder_start = torch.tensor([[tokenizer.pad_token_id, tokenizer.pad_token_id]])
    
    print(f"   Decoder dimension: 1 to 32 tokens")
    
    ep = torch.export.export(
        wrapped_model,
        (
            encoded_input["input_ids"],
            encoded_input["attention_mask"],
            decoder_start
        ),
		dynamic_shapes={
			"input_ids": {1: encoder_seq_len},                                # ✓ Include all args
			"attention_mask": {1: encoder_seq_len},                          # ✓ Even if static
			"decoder_input_ids": {1: decoder_seq_len}      # ✓ The dynamic one
		}
    )
    print("   ✓ Export succeeded!")
    
    print("4. Running decompositions...")
    ep = ep.run_decompositions()
    
    print("5. Importing to MLIR...")
    m = fx.export_and_import(
        ep,
        output_type=OutputType.LINALG_ON_TENSORS,
        func_name="flan_decoder"
    )
    
    mlir_str = str(m)
    with open("flan_linalg_test.mlir", "w") as f:
        f.write(mlir_str)
    print(f"   ✓ MLIR written ({len(mlir_str)} chars)")
    

def main():
    print("\nAutoregressive Flan-T5 Generation with Dynamic Export")
    print("=" * 70)
    
    # Export
    export_for_autoregressive()


if __name__ == "__main__":
    main()