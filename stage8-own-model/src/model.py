"""
Tiny decoder-only Transformer language model.

This is a from-scratch PyTorch implementation, initialized with random
weights. No pretrained model weights are loaded.

Architecture (GPT-style, decoder-only):
    - Token embedding
    - Learned positional embedding
    - N Transformer blocks, each with:
        * Pre-LayerNorm
        * Causal multi-head self-attention
        * Residual connection
        * Pre-LayerNorm
        * Position-wise feed-forward (MLP with GELU)
        * Residual connection
    - Final LayerNorm
    - Language-model head: Linear(d_model, vocab_size)

Trained with next-token prediction (cross-entropy loss) in train.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Config ────────────────────────────────────────────────────


@dataclass
class ModelConfig:
    """
    Model hyperparameters.

    Default values give ~830K parameters — the target for Stage 8.
    Adjust `d_model` and `n_layers` to scale up.
    """
    vocab_size: int = 128         # set this from tokenizer.vocab_size
    max_seq_len: int = 128        # longest input the model accepts
    d_model: int = 128            # embedding dimension
    n_heads: int = 4              # attention heads
    n_layers: int = 4             # Transformer blocks
    d_ff: int = 512               # feed-forward hidden size (4 * d_model)
    dropout: float = 0.1


# ── Building blocks ───────────────────────────────────────────


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Given a sequence of embeddings, each position attends to all
    positions at or before it (never to the future). This is what makes
    the model "causal" — it can only use left context when predicting.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, \
            "d_model must be divisible by n_heads"

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.d_model = config.d_model

        # One linear layer produces Q, K, V for all heads at once.
        # Shape: [d_model, 3 * d_model] → split into three [d_model, d_model] slices.
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)

        # Project the concatenated head outputs back to d_model.
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: lower-triangular matrix of 1s, registered as a buffer
        # so it moves to the right device when the model does.
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, config.max_seq_len, config.max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model]
        """
        B, T, C = x.shape  # batch, sequence length, channels (d_model)

        # Project and split into Q, K, V
        qkv = self.qkv(x)                     # [B, T, 3*C]
        q, k, v = qkv.chunk(3, dim=-1)        # each: [B, T, C]

        # Reshape to [B, n_heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale       # [B, n_heads, T, T]

        # Apply causal mask: set future positions to -inf so softmax ≈ 0
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Weighted sum of values
        out = attn @ v                                  # [B, n_heads, T, head_dim]

        # Concatenate heads back to d_model
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        out = self.resid_dropout(self.out_proj(out))
        return out


class FeedForward(nn.Module):
    """
    Position-wise MLP: Linear → GELU → Linear → Dropout.

    Expands from d_model to d_ff (typically 4×), applies nonlinearity,
    then projects back. Runs independently at every position.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One decoder block: attention + feed-forward, both with residuals
    and pre-LayerNorm (the modern GPT-2/GPT-3 style).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN residual: x = x + Attn(LN(x))
        x = x + self.attn(self.ln1(x))
        # Pre-LN residual: x = x + FFN(LN(x))
        x = x + self.ff(self.ln2(x))
        return x


# ── The full model ────────────────────────────────────────────


class TinyLLM(nn.Module):
    """
    The full language model.

    Forward returns raw logits over the vocabulary at every position:
        [batch, seq_len, vocab_size]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_final = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: the LM head shares weights with the token embedding.
        # This is standard in GPT-style models and saves vocab_size × d_model params.
        self.lm_head.weight = self.token_embedding.weight

        # Weight initialization (GPT-2 style)
        self.apply(self._init_weights)

        # Print param count on creation
        n_params = sum(p.numel() for p in self.parameters())
        print(f"TinyLLM initialized: {n_params:,} parameters")

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            input_ids: [batch, seq_len] long tensor of token IDs
            targets:   [batch, seq_len] long tensor, or None

        Returns:
            logits: [batch, seq_len, vocab_size]
            loss:   scalar cross-entropy loss if `targets` provided, else None

        The loss uses the standard next-token-prediction shift:
            logits[:, :-1, :] predicts targets[:, 1:]
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, \
            f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"

        # Token + positional embeddings
        tok_emb = self.token_embedding(input_ids)               # [B, T, d_model]
        positions = torch.arange(T, device=input_ids.device)    # [T]
        pos_emb = self.position_embedding(positions)            # [T, d_model]
        x = self.embed_dropout(tok_emb + pos_emb)               # [B, T, d_model]

        # Transformer blocks
        x = self.blocks(x)                                      # [B, T, d_model]
        x = self.ln_final(x)                                    # [B, T, d_model]

        # Project to vocabulary
        logits = self.lm_head(x)                                # [B, T, vocab_size]

        loss = None
        if targets is not None:
            # Flatten for cross-entropy: [B*T, vocab_size] vs [B*T]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,   # -100 is the standard "ignore" label
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Args:
            idx: [batch, seq_len] starting token IDs
            max_new_tokens: how many tokens to append
            temperature: sampling temperature (>1 = more random)
            top_k: if set, restrict sampling to top-k most likely tokens

        Returns:
            [batch, seq_len + max_new_tokens]
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Truncate to max_seq_len if we've grown too long
            idx_cond = idx[:, -self.config.max_seq_len:]

            logits, _ = self(idx_cond)
            # Focus on the last time step
            logits = logits[:, -1, :] / max(temperature, 1e-8)   # [B, vocab]

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, k)
                cutoff = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < cutoff, float("-inf"), logits)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)     # [B, 1]
            idx = torch.cat([idx, next_id], dim=1)

        return idx


# ── Sanity test (run this file directly) ──────────────────────


if __name__ == "__main__":
    # Build a tiny config and check that forward + backward work.
    cfg = ModelConfig(vocab_size=64, max_seq_len=32, d_model=64, n_heads=4, n_layers=2, d_ff=256)
    model = TinyLLM(cfg)

    # Fake batch: 2 sequences of length 16
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))

    logits, loss = model(input_ids, targets)
    print("logits shape:", logits.shape)         # expect [2, 16, 64]
    print("loss:", loss.item())

    # Backward pass should not error
    loss.backward()
    print("backward OK")

    # Generation smoke test
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(prompt, max_new_tokens=8, temperature=1.0, top_k=10)
    print("generated shape:", out.shape)         # expect [1, 12]
    print("generated:", out.tolist())