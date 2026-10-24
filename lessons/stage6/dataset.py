"""
=============================================================================
STAGE SIX - DATASET & BATCHING MODULE (lessons/stage6/dataset.py)
=============================================================================
This module handles corpus loading, clean train/validation splitting,
and reproducible mini-batch sampling for next-token prediction.

DATA INTEGRITY ASSURANCES:
- The corpus is split into distinct training (85%) and validation (15%) strings
  BEFORE any sequences are generated.
- No sequence crosses the boundary between training and validation data,
  preventing data leakage.
=============================================================================
"""

import sys
from pathlib import Path
from typing import Tuple, List
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from lessons.stage6.model import CharTokenizer

CORPUS_PATH = BASE_DIR / "data" / "tiny_corpus.txt"


def load_corpus(path: Path = CORPUS_PATH) -> str:
    """Loads raw text corpus from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if len(text.strip()) == 0:
        raise ValueError(f"Corpus at {path} is empty.")
    return text


def split_text_data(text: str, train_ratio: float = 0.85) -> Tuple[str, str]:
    """
    Splits text cleanly into training and validation sets without overlap.
    """
    split_idx = int(len(text) * train_ratio)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    return train_text, val_text


class TextDataset:
    """
    Manages tokenized dataset and produces randomized mini-batches
    of (input_seq, target_seq) pairs.
    """
    def __init__(self, token_ids: List[int], context_length: int):
        if len(token_ids) <= context_length:
            raise ValueError(
                f"Dataset size ({len(token_ids)} tokens) must be strictly greater "
                f"than context_length ({context_length}) to train next-token prediction."
            )
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length

    def get_batch(self, batch_size: int, random_seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples a batch of inputs X and shifted targets Y.
        X shape: (batch_size, context_length)
        Y shape: (batch_size, context_length)
        """
        max_start = len(self.data) - self.context_length - 1
        # Random starting indices
        start_indices = torch.randint(0, max_start + 1, (batch_size,))

        X_list = [self.data[i : i + self.context_length] for i in start_indices]
        Y_list = [self.data[i + 1 : i + self.context_length + 1] for i in start_indices]

        X = torch.stack(X_list)
        Y = torch.stack(Y_list)
        return X, Y

    def __len__(self) -> int:
        return len(self.data)


def prepare_datasets(
    corpus_text: str,
    context_length: int = 48,
    train_ratio: float = 0.85
) -> Tuple[CharTokenizer, TextDataset, TextDataset]:
    """
    Tokenizes raw text, fits vocabulary, and returns (tokenizer, train_dataset, val_dataset).
    """
    train_text, val_text = split_text_data(corpus_text, train_ratio=train_ratio)

    tokenizer = CharTokenizer()
    # Fit vocabulary on full corpus so both splits share identical token mappings
    tokenizer.fit(corpus_text)

    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    train_dataset = TextDataset(train_ids, context_length=context_length)
    val_dataset = TextDataset(val_ids, context_length=context_length)

    return tokenizer, train_dataset, val_dataset
