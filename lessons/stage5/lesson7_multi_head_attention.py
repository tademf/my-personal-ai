"""
=============================================================================
STAGE FIVE - LESSON 7: MULTI-HEAD ATTENTION AND TRANSFORMER BLOCKS
File: lessons/stage5/lesson7_multi_head_attention.py
=============================================================================
In single-head attention, all information is averaged into a single attention map.
Multi-Head Attention (MHA) splits the model's embedding dimension into `h`
independent heads (subspaces):
- Head 1 might learn subject-verb relationships.
- Head 2 might track pronoun references ("it" -> "the database").
- Head 3 might attend to nearby neighboring punctuation.

A Complete Transformer Block consists of:
1. Multi-Head Self-Attention
2. Residual Add & Layer Normalization (Add & Norm)
3. Position-Wise Feed-Forward Network (FFN: Linear -> GELU -> Linear)
4. Second Residual Add & Layer Normalization (Add & Norm)
=============================================================================
"""

import sys
import math
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from lessons.stage5.lesson6_scaled_dot_product_attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    Projects inputs into Q, K, V for each head, computes parallel attention,
    concatenates results, and projects back through W_out.
    """
    def __init__(self, d_model: int = 16, num_heads: int = 2):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Final output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # 1. Linear projections: (batch, seq_len, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 2. Reshape into multiple heads:
        # (batch, seq_len, num_heads, head_dim) -> transpose to (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Scaled dot-product attention across all heads in parallel
        # Q, K, V are 4D tensors: batch and num_heads are treated as batch dimensions
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        head_outputs = torch.matmul(attn_weights, V)  # (batch, num_heads, seq_len, head_dim)

        # 4. Concatenate heads back into (batch, seq_len, d_model)
        concat_heads = head_outputs.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # 5. Final linear projection
        return self.out_proj(concat_heads)


class TransformerBlock(nn.Module):
    """
    A full Transformer Decoder Block:
    Multi-Head Attention + Residual Connection + LayerNorm + FFN + Residual Connection + LayerNorm.
    """
    def __init__(self, d_model: int = 16, num_heads: int = 2, d_ff: int = 32):
        super().__init__()
        self.attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        # Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Self-Attention with Residual connection & LayerNorm
        attn_out = self.attn(x, mask=mask)
        x = self.norm1(x + attn_out)

        # 2. Feed-Forward Network with Residual connection & LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


def demonstrate_transformer_block():
    """Demonstrates input passing through a full Transformer Block."""
    print("=" * 75)
    print("STAGE FIVE - LESSON 7: MULTI-HEAD ATTENTION & TRANSFORMER BLOCK")
    print("=" * 75)

    torch.manual_seed(42)
    batch_size = 2
    seq_len = 5
    d_model = 16
    num_heads = 2

    block = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=32)
    print("Transformer Block Architecture:\n", block)

    dummy_input = torch.randn(batch_size, seq_len, d_model)
    print(f"\nInput Tensor Shape:  {dummy_input.shape} [batch={batch_size}, seq_len={seq_len}, d_model={d_model}]")

    output = block(dummy_input)
    print(f"Output Tensor Shape: {output.shape} (Dimensions perfectly preserved!)")

    assert output.shape == dummy_input.shape, "Transformer Block altered tensor dimensions!"
    print("\nTakeaways:")
    print("1. Multi-Head Attention projected input into 2 separate heads of dimension 8 each.")
    print("2. Residual connections ('x + sublayer(x)') prevent vanishing gradients, allowing depth.")
    print("3. LayerNorm normalizes activations across feature dimensions for stable learning.")

    return block, output


if __name__ == "__main__":
    demonstrate_transformer_block()
    print("\n" + "=" * 75)
    print("LESSON 7 COMPLETED SUCCESSFULLY!")
    print("=" * 75)
