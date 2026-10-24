"""
=============================================================================
STAGE FIVE - LESSON 10: TRAINING AND VALIDATION ON A TEXT CORPUS
File: lessons/stage5/lesson10_training_validation.py
=============================================================================
In this lesson, we assemble all previous components:
- Vocabulary & Special Tokens
- Token Embeddings & Positional Encodings
- Transformer Block with Causal Masking
- Output Projection Head (d_model -> vocab_size)

We train this tiny Language Model on an educational corpus, tracking both
Training Loss and Validation Loss over epochs.

EDUCATIONAL DISCLAIMER:
This is a micro-scale educational demonstration to understand loss curves
and validation mechanics. It is trained on a few dozen words. It does NOT have
ChatGPT-level capabilities.
=============================================================================
"""

import sys
from pathlib import Path
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from lessons.stage5.lesson3_vocabulary_special_tokens import Vocabulary, pad_sequences
from lessons.stage5.lesson5_positional_encoding import SinusoidalPositionalEncoding
from lessons.stage5.lesson7_multi_head_attention import TransformerBlock
from lessons.stage5.lesson8_causal_masking import generate_causal_mask
from lessons.stage5.lesson9_next_token_prediction import prepare_shifted_sequences


class TinyLanguageModel(nn.Module):
    """
    A minimal Causal Decoder-Only Language Model (GPT Architecture):
    Token Embedding + Positional Encoding -> Transformer Block -> Linear Projection Head
    """
    def __init__(self, vocab_size: int, d_model: int = 16, num_heads: int = 2, d_ff: int = 32, max_len: int = 50):
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)
        self.transformer_block = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
        # Final linear projection maps hidden states back to vocabulary logits
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        causal_mask = generate_causal_mask(seq_len)

        # 1. Embeddings + Positional Encoding
        x = self.token_embed(input_ids) * (self.d_model ** 0.5)
        x = self.pos_encoder(x)

        # 2. Transformer Block with Causal Mask
        x = self.transformer_block(x, mask=causal_mask)

        # 3. Logits over vocabulary: (batch, seq_len, vocab_size)
        logits = self.lm_head(x)
        return logits


def build_toy_dataset() -> Tuple[Vocabulary, torch.Tensor, torch.Tensor]:
    """Prepares training and validation sequence tensors from an educational code corpus."""
    raw_texts = [
        ["def", "greet", ":", "print", "hello"],
        ["def", "farewell", ":", "print", "bye"],
        ["def", "compute", ":", "return", "total"],
        ["def", "run", ":", "return", "status"],
        ["def", "load", ":", "return", "data"],
        ["def", "save", ":", "return", "success"]
    ]

    vocab = Vocabulary(min_freq=1)
    vocab.build_from_corpus(raw_texts)

    # Encode with BOS and EOS tokens
    encoded_seqs = [vocab.encode(seq, add_special_tokens=True) for seq in raw_texts]
    padded_seqs = pad_sequences(encoded_seqs, pad_value=vocab.pad_id)
    tensor_data = torch.tensor(padded_seqs, dtype=torch.long)

    # Train / Validation Split: First 4 samples train, last 2 samples validation
    train_data = tensor_data[:4]
    val_data = tensor_data[4:]

    return vocab, train_data, val_data


def train_toy_lm(epochs: int = 50, lr: float = 0.01):
    """Executes educational training and validation loop."""
    print("=" * 75)
    print("STAGE FIVE - LESSON 10: TRAINING & VALIDATION OF A TINY LM")
    print("=" * 75)

    torch.manual_seed(42)
    vocab, train_data, val_data = build_toy_dataset()
    vocab_size = len(vocab)

    model = TinyLanguageModel(vocab_size=vocab_size, d_model=16, num_heads=2, d_ff=32)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Vocabulary Size: {vocab_size} | Training Batches: {len(train_data)} | Val Batches: {len(val_data)}")
    print("\nEpoch | Train Loss | Val Loss   | Status")
    print("-" * 50)

    train_loss_hist, val_loss_hist = [], []

    # Shift inputs and targets once
    X_train, y_train = prepare_shifted_sequences(train_data)
    X_val, y_val = prepare_shifted_sequences(val_data)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Forward pass on train set
        train_logits = model(X_train)
        loss = criterion(train_logits.reshape(-1, vocab_size), y_train.reshape(-1))
        loss.backward()
        optimizer.step()

        # Evaluation pass on validation set
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits.reshape(-1, vocab_size), y_val.reshape(-1))

        train_loss_hist.append(loss.item())
        val_loss_hist.append(val_loss.item())

        if epoch in [1, 10, 20, 30, 40, 50]:
            print(f"{epoch:5d} | {loss.item():10.4f} | {val_loss.item():10.4f} | Active Training")

    print("\nTraining Observations:")
    print(f"1. Training loss dropped from {train_loss_hist[0]:.4f} down to {train_loss_hist[-1]:.4f}.")
    print("2. Notice that on a tiny dataset, the model can memorize the training patterns rapidly.")
    print("3. In production LLMs, pre-training runs on trillions of tokens across weeks/months.")

    return model, vocab, train_loss_hist, val_loss_hist


if __name__ == "__main__":
    train_toy_lm()
    print("\n" + "=" * 75)
    print("LESSON 10 COMPLETED SUCCESSFULLY!")
    print("=" * 75)
