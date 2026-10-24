"""
=============================================================================
STAGE SIX - REUSABLE MODEL MODULE (lessons/stage6/model.py)
=============================================================================
This module implements a complete, self-contained, decoder-only Transformer
language model (GPT architecture) built from scratch using PyTorch.

Architecture:
- Character-Level Tokenizer with special tokens (<PAD>, <UNK>, <BOS>, <EOS>).
- Token and Positional Embedding layers.
- Stack of Causal Transformer Decoder Blocks with:
  * Multi-Head Causal Self-Attention
  * Residual skip connections
  * Pre-Layer Normalization
  * Position-Wise Feed-Forward Network with GELU activation
- Final Layer Normalization and Linear Vocabulary Projection Head.
=============================================================================
"""

import sys
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import torch
import torch.nn as nn

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


@dataclass
class TransformerLMConfig:
    """Hyperparameter configuration for Decoder-Only Transformer."""
    vocab_size: int
    context_length: int = 48
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.0

    def validate(self):
        """Validates all architectural assumptions."""
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.context_length <= 0:
            raise ValueError(f"context_length must be positive, got {self.context_length}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")


class CharTokenizer:
    """
    Educational Character-Level Tokenizer.
    Maps distinct characters to unique integer IDs with special tokens.
    """
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self):
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]

        for idx, token in enumerate(self.special_tokens):
            self.char_to_id[token] = idx
            self.id_to_char[idx] = token

        self.pad_id = self.char_to_id[self.PAD_TOKEN]
        self.unk_id = self.char_to_id[self.UNK_TOKEN]
        self.bos_id = self.char_to_id[self.BOS_TOKEN]
        self.eos_id = self.char_to_id[self.EOS_TOKEN]

    def fit(self, text: str):
        """Constructs vocabulary from unique characters in raw text."""
        unique_chars = sorted(list(set(text)))
        curr_id = len(self.char_to_id)
        for ch in unique_chars:
            if ch not in self.char_to_id:
                self.char_to_id[ch] = curr_id
                self.id_to_char[curr_id] = ch
                curr_id += 1

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encodes string into integer token IDs."""
        ids = [self.char_to_id.get(ch, self.unk_id) for ch in text]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decodes integer token IDs back to a reconstructed string."""
        chars = []
        for i in ids:
            ch = self.id_to_char.get(i, self.UNK_TOKEN)
            if skip_special_tokens and ch in self.special_tokens:
                continue
            chars.append(ch)
        return "".join(chars)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes tokenizer state."""
        return {
            "char_to_id": self.char_to_id,
            "id_to_char": {int(k): v for k, v in self.id_to_char.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CharTokenizer":
        """Reconstructs tokenizer from serialized dictionary."""
        tok = cls()
        tok.char_to_id = data["char_to_id"]
        tok.id_to_char = {int(k): v for k, v in data["id_to_char"].items()}
        tok.pad_id = tok.char_to_id[cls.PAD_TOKEN]
        tok.unk_id = tok.char_to_id[cls.UNK_TOKEN]
        tok.bos_id = tok.char_to_id[cls.BOS_TOKEN]
        tok.eos_id = tok.char_to_id[cls.EOS_TOKEN]
        return tok

    def __len__(self) -> int:
        return len(self.char_to_id)


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal (Autoregressive) Self-Attention."""
    def __init__(self, config: TransformerLMConfig):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads

        # Linear projections for Queries, Keys, and Values
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

        # Register lower-triangular causal mask buffer
        mask = torch.tril(torch.ones(config.context_length, config.context_length))
        self.register_buffer("causal_mask", mask.view(1, 1, config.context_length, config.context_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Projections: (B, T, d_model) -> (B, num_heads, T, head_dim)
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask: mask positions where j > i with -inf
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Multiply by values
        out = torch.matmul(attn_weights, V)  # (B, num_heads, T, head_dim)

        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """
    Standard Transformer Decoder Block using Pre-LayerNorm architecture:
    x = x + Attn(LayerNorm(x))
    x = x + FFN(LayerNorm(x))
    """
    def __init__(self, config: TransformerLMConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)

        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyTransformerLM(nn.Module):
    """
    A Decoder-Only Transformer Language Model built from scratch.
    """
    def __init__(self, config: TransformerLMConfig):
        super().__init__()
        config.validate()
        self.config = config

        # Token embedding and learnable positional embedding tables
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.context_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Stack of Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])

        # Final LayerNorm before language modeling head
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def count_parameters(self) -> int:
        """Returns total number of trainable model parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        input_ids: (batch_size, seq_len)
        targets:   (batch_size, seq_len) optional ground truth token IDs for loss calculation.
        """
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.context_length:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds context_length ({self.config.context_length})"
            )

        # 1. Positional indices: [0, 1, 2, ..., seq_len - 1]
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)

        # 2. Sum Token Embeddings and Positional Embeddings
        tok_emb = self.token_embed(input_ids)          # (B, T, d_model)
        pos_emb = self.pos_embed(positions)           # (T, d_model)
        x = self.dropout(tok_emb + pos_emb)

        # 3. Pass through stacked Transformer Blocks
        for block in self.blocks:
            x = block(x)

        # 4. Final normalization and vocabulary projection
        x = self.ln_f(x)
        logits = self.lm_head(x)                      # (B, T, vocab_size)

        # 5. Compute Cross-Entropy Loss if targets are provided
        loss = None
        if targets is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        stop_id: Optional[int] = None
    ) -> List[int]:
        """
        Autoregressively generates new token IDs conditioned on prompt_ids.
        """
        self.eval()
        generated = list(prompt_ids)

        for _ in range(max_new_tokens):
            # Crop to the most recent context_length tokens
            context_window = generated[-self.config.context_length:]
            input_tensor = torch.tensor([context_window], dtype=torch.long)

            logits, _ = self.forward(input_tensor)
            # Take logits from the last position
            next_token_logits = logits[0, -1, :]

            # Apply temperature scaling
            if temperature > 0:
                scaled_logits = next_token_logits / temperature
            else:
                scaled_logits = next_token_logits

            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                k = min(top_k, scaled_logits.size(-1))
                top_values, _ = torch.topk(scaled_logits, k)
                scaled_logits = torch.where(
                    scaled_logits < top_values[-1],
                    torch.tensor(-float("inf")),
                    scaled_logits
                )

            # Sample or greedy argmax
            if temperature <= 0.05:
                next_id = int(torch.argmax(scaled_logits).item())
            else:
                probs = torch.softmax(scaled_logits, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())

            generated.append(next_id)

            if stop_id is not None and next_id == stop_id:
                break

        return generated
