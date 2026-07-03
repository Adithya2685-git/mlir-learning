"""
Corrected autoregressive text generation with dynamically-exported GPT-2.

KEY DIFFERENCES FROM THE BROKEN VERSION:
- GPT-2 is a CAUSAL language model, not an encoder-decoder model
- It only takes: input_ids, attention_mask (optional)
- No decoder_input_ids parameter!
- For autoregressive generation, you append new tokens to input_ids

The wrapper should only accept input_ids and attention_mask.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch_mlir import fx
from torch_mlir.fx import OutputType
import torch.nn.functional as F


class GPT2Wrapper(torch.nn.Module):
    """
    Wrapper for GPT-2 that only returns logits.
    
    GPT-2 takes:
    - input_ids: (batch, seq_len) - the token sequence
    - attention_mask: (batch, seq_len) - which tokens to attend to
    
    Returns:
    - logits: (batch, seq_len, vocab_size) - predictions for each position
    """
    def __init__(self, model, return_last_logits_only=False):
        super().__init__()
        self.model = model
        self.return_last_logits_only = return_last_logits_only
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        
        # Option 1: Return all logits (batch, seq_len, vocab_size)
        if not self.return_last_logits_only:
            return logits
        
        # Option 2: Return only logits for the last token (batch, 1, vocab_size)
        # Useful for autoregressive generation where you only care about next token
        else:
            return logits[:, -1:, :]


def export_gpt2_for_autoregressive(return_last_only=False):
    """
    Export GPT-2 with DYNAMIC sequence length for autoregressive generation.
    
    Args:
        return_last_only: If True, only return logits for the last position.
                         If False, return logits for all positions.
    """
    print("=" * 70)
    print("Exporting GPT-2 for Autoregressive Generation")
    print("=" * 70)
    
    # Setup tokenizer and model
    print("\n1. Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create sample input
    input_text = "The capital of France is"
    encoded_input = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    
    print(f"   Input text: '{input_text}'")
    print(f"   input_ids shape: {encoded_input['input_ids'].shape}")
    print(f"   Vocabulary size: {tokenizer.vocab_size}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.eval()
    model.config.use_cache = False  # Important: disable caching for exportable MLIR
    
    print(f"   Model config:")
    print(f"     - hidden_size: {model.config.hidden_size}")
    print(f"     - num_layers: {model.config.num_hidden_layers}")
    print(f"     - num_attention_heads: {model.config.num_attention_heads}")
    
    # Warm up the model
    print("\n2. Warming up model...")
    with torch.no_grad():
        _ = model(
            input_ids=encoded_input["input_ids"],
            attention_mask=encoded_input["attention_mask"]
        )
    print("   ✓ Warm-up complete")
    
    # Create wrapper
    print("\n3. Creating wrapper...")
    wrapped_model = GPT2Wrapper(model, return_last_logits_only=return_last_only)
    
    # Define dynamic dimension for sequence length
    print("\n4. Setting up dynamic sequence length dimension...")
    seq_len_dim = torch.export.Dim("seq_len", max=128)
    
    print(f"   Sequence can be: 1 to 128 tokens")
    print(f"   This allows autoregressive generation of long sequences")
    
    # Export the model
    print("\n5. Exporting with torch.export.export()...")
    print(f"   Input shapes:")
    print(f"     - input_ids: {encoded_input['input_ids'].shape}")
    print(f"     - attention_mask: {encoded_input['attention_mask'].shape}")
    
    try:
        ep = torch.export.export(
            wrapped_model,
            (
                encoded_input["input_ids"],
                encoded_input["attention_mask"]
            ),
            dynamic_shapes={
                "input_ids": {1: seq_len_dim},           # dimension 1 (seq_len) is dynamic
                "attention_mask": {1: seq_len_dim}       # same dynamic dimension
            }
        )
        print("   ✓ Export succeeded!")
        
    except Exception as e:
        print(f"   ✗ Export failed: {type(e).__name__}")
        print(f"   Error: {str(e)}")
        raise
    
    # Run decompositions
    print("\n6. Running decompositions...")
    ep = ep.run_decompositions()
    print("   ✓ Decompositions complete")
    
    # Import to MLIR
    print("\n7. Importing to MLIR...")
    try:
        m = fx.export_and_import(
            ep,
            output_type=OutputType.TORCH,
            func_name="gpt2_decoder"
        )
        print("   ✓ MLIR import succeeded")
    except Exception as e:
        print(f"   ✗ MLIR import failed: {type(e).__name__}")
        print(f"   Error: {str(e)}")
        raise
    
    # Save MLIR
    print("\n8. Saving MLIR to file...")
    mlir_str = str(m)
    output_file = "gpt2_torch.mlir"
    with open(output_file, "w") as f:
        f.write(mlir_str)
    
    print(f"   ✓ MLIR written to {output_file}")
    print(f"   File size: {len(mlir_str):,} bytes")
    
    return wrapped_model, encoded_input, tokenizer, ep


if __name__ == "__main__":
    print("\nCorrected GPT-2 Export for Autoregressive Generation")
    print("=" * 70)
    
    try:
        # Export the model
        wrapped_model, encoded_input, tokenizer, ep = export_gpt2_for_autoregressive(
            return_last_only=False
        )
        
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"""
✓ Export successful!
✓ MLIR file: gpt2_torch.mlir
✓ Supports dynamic sequence length (1-128 tokens)

For autoregressive generation:
1. Export once with dynamic sequence length
2. Each inference call:
   - input_ids: (batch, 1..128) ← grows with each token
   - attention_mask: (batch, 1..128) ← same length as input_ids
3. Extract logits[:, -1, :] to get next token
4. Append token to input_ids, repeat

The MLIR will work for any sequence length up to the max (128).
        """)
        
    except Exception as e:
        print(f"\n✗ Error: {type(e).__name__}")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()