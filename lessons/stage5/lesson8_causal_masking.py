"""
=============================================================================
STAGE FIVE - LESSON 8: CAUSAL (LOOK-AHEAD) MASKING
File: lessons/stage5/lesson8_causal_masking.py
=============================================================================
In an autoregressive language model (like GPT, Claude, or Llama), the model's
objective is NEXT-TOKEN PREDICTION.

THE CHEATING PROBLEM:
If token 2 ("cat") could attend to token 3 ("slept"), the model would simply
cheat by reading ahead rather than learning to predict!

THE CAUSAL MASK SOLUTION:
We apply a lower-triangular mask to the attention matrix before Softmax:
- Allowed positions (j <= i) keep their normal attention scores.
- Forbidden future positions (j > i) are filled with -infinity (-1e9).
- When Softmax is applied, e^(-infinity) = 0.0000!
- Result: Token i has exactly 0.0 probability of seeing future token j > i.
=============================================================================
"""

import sys
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from lessons.stage5.lesson6_scaled_dot_product_attention import scaled_dot_product_attention


def generate_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Creates a lower-triangular binary mask of shape (1, seq_len, seq_len).
    1 means allowed to attend; 0 means blocked/masked.
    """
    # torch.tril creates a lower triangular matrix
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.uint8))
    return mask.unsqueeze(0)  # Shape: (1, seq_len, seq_len)


def demonstrate_causal_masking():
    """Demonstrates causal masking blocking future attention."""
    print("=" * 75)
    print("STAGE FIVE - LESSON 8: CAUSAL (LOOK-AHEAD) MASKING")
    print("=" * 75)

    seq_len = 4
    mask = generate_causal_mask(seq_len)

    print(f"Lower-Triangular Causal Mask (Seq_Len = {seq_len}):")
    print(mask.squeeze(0))
    print("Legend: 1 = Allowed to attend, 0 = Blocked (future token)")

    # Test attention with causal mask
    torch.manual_seed(42)
    Q = torch.randn(1, seq_len, 4)
    K = torch.randn(1, seq_len, 4)
    V = torch.randn(1, seq_len, 4)

    output, weights = scaled_dot_product_attention(Q, K, V, mask=mask)

    print("\nResulting Causal Attention Probability Matrix:")
    attn_matrix = weights.squeeze(0)
    for i in range(seq_len):
        row_str = " ".join(f"{w:.4f}" for w in attn_matrix[i].tolist())
        print(f"  Token {i} attention: [{row_str}]")

    # Strict Verification: For all j > i, attention must be exactly 0.0
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert attn_matrix[i, j].item() == 0.0, f"Leakage! Token {i} attended to future token {j}"

    print("\nVerification Passed:")
    print("1. Token 0 attends ONLY to itself (1.0000). Future tokens are strictly 0.0000.")
    print("2. Token 1 attends to tokens 0 and 1. Future tokens 2 and 3 are strictly 0.0000.")
    print("3. No information from the future can leak into earlier positions!")

    return mask, weights


if __name__ == "__main__":
    demonstrate_causal_masking()
    print("\n" + "=" * 75)
    print("LESSON 8 COMPLETED SUCCESSFULLY!")
    print("=" * 75)
