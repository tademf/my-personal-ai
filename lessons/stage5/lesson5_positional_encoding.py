"""
=============================================================================
STAGE FIVE - LESSON 5: POSITIONAL ENCODINGS
File: lessons/stage5/lesson5_positional_encoding.py
=============================================================================
Unlike Recurrent Neural Networks (RNNs) that read text sequentially word-by-word,
Transformers process all tokens in a sequence simultaneously in parallel.

This parallelism gives Transformers incredible speed, but has a fatal flaw:
THE ATTENTION MECHANISM IS PERMUTATION INVARIANT!
Without positional information, the sentences:
  "Dog bites man"  and  "Man bites dog"
would produce identical mathematical representations!

To fix this, we must inject word order information by adding a Positional
Encoding (PE) vector to each token's embedding:
  final_input_vector = token_embedding + positional_encoding

Vaswani et al. (2017) introduced Sinusoidal Positional Encoding:
  PE(pos, 2i)   = sin( pos / (10000^(2i / d_model)) )
  PE(pos, 2i+1) = cos( pos / (10000^(2i / d_model)) )
=============================================================================
"""

import sys
import math
from pathlib import Path
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


class SinusoidalPositionalEncoding(nn.Module):
    """
    Computes sinusoidal positional encodings for sequence positions 0 to max_len.
    """
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        self.d_model = d_model

        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)

        # Compute the division term: 10000^(2i / d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sin to even indices (2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices (2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (persistent state that is NOT a learnable parameter)
        self.register_buffer("pe", pe.unsqueeze(0))  # Shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, seq_len, d_model)
        Returns: x + pe[:, :seq_len, :]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


def demonstrate_positional_encoding():
    """Demonstrates how positional encoding differentiates identical words in different positions."""
    print("=" * 75)
    print("STAGE FIVE - LESSON 5: POSITIONAL ENCODING")
    print("=" * 75)

    torch.manual_seed(42)
    d_model = 6
    seq_len = 4

    pe_module = SinusoidalPositionalEncoding(d_model=d_model, max_len=20)
    pe_matrix = pe_module.pe.squeeze(0)[:seq_len]  # (4 positions, 6 dimensions)

    print(f"Positional Encoding Matrix for first {seq_len} positions (d_model={d_model}):")
    print(pe_matrix)

    # Consider two identical tokens placed at position 0 vs position 2
    # E.g., in "dog bites dog", the word "dog" appears twice.
    token_embed = torch.tensor([[0.5, -0.2, 0.8, -0.5, 0.1, 0.9]])  # (1, 6)

    # Replicate token at position 0 and position 2
    input_embeddings = torch.zeros(1, 4, d_model)
    input_embeddings[0, 0] = token_embed  # "dog" at index 0
    input_embeddings[0, 2] = token_embed  # "dog" at index 2

    # Add positional encoding
    encoded_sequence = pe_module(input_embeddings)

    vec_pos_0 = encoded_sequence[0, 0]
    vec_pos_2 = encoded_sequence[0, 2]

    # Verify they are no longer identical
    distance = torch.norm(vec_pos_0 - vec_pos_2).item()
    print(f"\nComparing the same token ('dog') at position 0 vs position 2:")
    print(f"  Vector at position 0: {vec_pos_0.tolist()[:4]} ...")
    print(f"  Vector at position 2: {vec_pos_2.tolist()[:4]} ...")
    print(f"  Euclidean distance between them: {distance:.4f}")
    assert distance > 0.5, "Positional encoding failed to differentiate positions!"

    print("\nTakeaway: The model now has a distinct geometric signature for every position,")
    print("allowing it to distinguish 'dog bites man' from 'man bites dog'!")

    return pe_module, pe_matrix


if __name__ == "__main__":
    demonstrate_positional_encoding()
    print("\n" + "=" * 75)
    print("LESSON 5 COMPLETED SUCCESSFULLY!")
    print("=" * 75)
