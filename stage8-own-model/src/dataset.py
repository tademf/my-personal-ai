"""
Dataset loader for the tiny LLM.

Loads a text file, fits a CharTokenizer on it, encodes everything to
one long tensor, splits into train / val, and serves random batches.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Make `src` importable when running as a script
sys.path.insert(0, str(Path(__file__).parent))
from tokenizer import CharTokenizer


class TextDataset:
    """Holds the tokenized corpus and serves random training batches."""

    def __init__(
        self,
        text_file: str | Path,
        tokenizer: CharTokenizer | None = None,
        train_split: float = 0.9,
        seed: int = 42,
    ):
        path = Path(text_file)
        if not path.exists():
            raise FileNotFoundError(f"Training text not found: {path}")

        self.text = path.read_text(encoding="utf-8")
        print(f"Loaded {len(self.text):,} characters from {path.name}")

        if tokenizer is None:
            self.tokenizer = CharTokenizer()
            self.tokenizer.fit(self.text)
        else:
            self.tokenizer = tokenizer
        print(f"Tokenizer vocab size: {self.tokenizer.vocab_size}")

        all_ids = self.tokenizer.encode(self.text)
        self.data = torch.tensor(all_ids, dtype=torch.long)
        print(f"Total tokens: {len(self.data):,}")

        n = len(self.data)
        split_idx = int(n * train_split)
        self.train_data = self.data[:split_idx]
        self.val_data = self.data[split_idx:]
        print(f"Train tokens: {len(self.train_data):,}")
        print(f"Val tokens:   {len(self.val_data):,}")

        self.rng = torch.Generator().manual_seed(seed)

    def get_batch(
        self,
        split: str,
        batch_size: int = 16,
        seq_len: int = 128,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if split == "train":
            data = self.train_data
        elif split == "val":
            data = self.val_data
        else:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        max_start = len(data) - seq_len - 1
        if max_start <= 0:
            raise ValueError(
                f"Split {split!r} has {len(data)} tokens, too short for seq_len={seq_len}"
            )

        starts = torch.randint(0, max_start, (batch_size,), generator=self.rng)
        input_ids = torch.stack([data[s : s + seq_len] for s in starts])
        targets = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts])
        return input_ids, targets


if __name__ == "__main__":
    dataset = TextDataset("data/raw/shakespeare.txt")
    xb, yb = dataset.get_batch("train", batch_size=2, seq_len=16)
    print(f"\ninput_ids shape: {tuple(xb.shape)}")
    print(f"targets shape:   {tuple(yb.shape)}")
    print(f"Decoded: {dataset.tokenizer.decode(xb[0].tolist())!r}")
    assert torch.equal(xb[0, 1:], yb[0, :-1]), "targets not 1-shifted"
    print("✓ shift check passed")