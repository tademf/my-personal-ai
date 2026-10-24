"""
=============================================================================
STAGE FIVE - LESSON 6: SCALED DOT-PRODUCT ATTENTION
File: lessons/stage5/lesson6_scaled_dot_product_attention.py
=============================================================================
Attention is the core breakthrough of the Transformer architecture.
It allows every token in a sentence to look at ("attend to") every other token
and dynamically decide which words are most relevant.

The Formula (Vaswani et al. 2017):
  Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(d_k) ) @ V

Intuitive Roles of Q, K, V:
- Query (Q): What am I looking for? (e.g. "I am a verb, where is my subject?")
- Key   (K): What information do I contain? (e.g. "I am a noun subject.")
- Value (V): The actual representation transferred if the Query matches the Key.

Why scale by 1 / sqrt(d_k)?
As the embedding dimension d_k grows, dot products grow larger in magnitude.
Large values push the Softmax function into flat saturation zones where gradients
are near-zero (the vanishing gradient problem). Dividing by sqrt(d_k) stabilizes
variance back to 1.0.
=============================================================================
"""

import sys
import math
from pathlib import Path
from typing import Optional, Tuple
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Scaled Dot-Product Attention.
    Inputs:
      Q: (batch, seq_len_q, d_k)
      K: (batch, seq_len_k, d_k)
      V: (batch, seq_len_k, d_v)
      mask: Optional tensor of shape (batch, seq_len_q, seq_len_k) with 1s for valid and 0s for masked
    Outputs:
      output: (batch, seq_len_q, d_v)
      attention_weights: (batch, seq_len_q, seq_len_k)
    """
    d_k = Q.size(-1)

    # 1. Compute raw affinity scores: (Q @ K^T) / sqrt(d_k)
    # Shape: (batch, seq_len_q, seq_len_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # 2. Apply mask if provided (e.g., set masked positions to -1e9 so softmax assigns 0.0)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 3. Softmax along the last dimension to get attention probabilities (sum to 1.0)
    attention_weights = torch.softmax(scores, dim=-1)

    # 4. Multiply attention weights by values: (batch, seq_len_q, d_v)
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


def demonstrate_attention():
    """Demonstrates attention calculation with sample Query, Key, and Value tensors."""
    print("=" * 75)
    print("STAGE FIVE - LESSON 6: SCALED DOT-PRODUCT ATTENTION")
    print("=" * 75)

    torch.manual_seed(42)

    # Batch of 1 sequence with 3 tokens (e.g., ["The", "cat", "slept"])
    # Each token has dimension d_k = 4
    batch_size = 1
    seq_len = 3
    d_k = 4

    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)

    print(f"Query (Q) Shape: {Q.shape}")
    print(f"Key   (K) Shape: {K.shape}")
    print(f"Value (V) Shape: {V.shape}")

    output, weights = scaled_dot_product_attention(Q, K, V)

    print(f"\nAttention Weights Matrix Shape: {weights.shape}")
    print("Attention Weights (Row i shows how much Token i attends to every Token j):")
    weights_matrix = weights.squeeze(0)
    for i in range(seq_len):
        row_str = " ".join(f"{w:.4f}" for w in weights_matrix[i].tolist())
        row_sum = weights_matrix[i].sum().item()
        print(f"  Token {i} attention distribution: [{row_str}] -> Sum = {row_sum:.4f}")

    print(f"\nFinal Attention Output Shape: {output.shape}")
    print("Output Vector for Token 0 (Weighted blend of all Values):")
    print(output[0, 0].tolist())

    # Verification: Each row of attention weights must sum to exactly 1.0
    for i in range(seq_len):
        assert abs(weights_matrix[i].sum().item() - 1.0) < 1e-5

    print("\nVerification: Every row sums to 1.0 (valid probability distribution)!")
    return output, weights


if __name__ == "__main__":
    demonstrate_attention()
    print("\n" + "=" * 75)
    print("LESSON 6 COMPLETED SUCCESSFULLY!")
    print("=" * 75)
