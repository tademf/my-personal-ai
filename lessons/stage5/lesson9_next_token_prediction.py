"""
=============================================================================
STAGE FIVE - LESSON 9: NEXT-TOKEN PREDICTION & SHIFTED TARGETS
File: lessons/stage5/lesson9_next_token_prediction.py
=============================================================================
How does an autoregressive language model learn from raw text?
Through Self-Supervised Learning using SHIFTED SEQUENCES.

Given a tokenized sentence of length T:
  Full:   [<BOS>,  "the",  "cat",  "sat",  "on",  <EOS>]

We split it into two shifted sequences of length T-1:
  Input  (X): [<BOS>,  "the",  "cat",  "sat",  "on" ]
  Target (Y): [ "the",  "cat",  "sat",  "on",  <EOS>]

Notice the 1-to-1 alignment at each timestep:
- Given [<BOS>],           predict "the"
- Given [<BOS>, "the"],    predict "cat"
- Given [... "the", "cat"], predict "sat"
- Given [... "cat", "sat"], predict "on"
- Given [... "sat", "on"],  predict <EOS>

In a single forward pass, the model trains on EVERY prefix simultaneously!
=============================================================================
"""

import sys
from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def prepare_shifted_sequences(sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a sequence of token IDs of shape (batch, seq_len),
    returns (input_seq, target_seq) shifted by 1 position.
    Inputs:
      input_seq:  tokens from index 0 to seq_len - 2
      target_seq: tokens from index 1 to seq_len - 1
    """
    if sequence.size(1) < 2:
        raise ValueError("Sequence length must be at least 2 to produce shifted pairs.")

    input_seq = sequence[:, :-1].contiguous()
    target_seq = sequence[:, 1:].contiguous()
    return input_seq, target_seq


def demonstrate_next_token_prediction():
    """Demonstrates shifted sequence preparation and vocabulary loss calculation."""
    print("=" * 75)
    print("STAGE FIVE - LESSON 9: NEXT-TOKEN PREDICTION & SHIFTED TARGETS")
    print("=" * 75)

    # Let token IDs represent: [<BOS>=2, "the"=4, "cat"=5, "sat"=6, <EOS>=3]
    tokens = torch.tensor([[2, 4, 5, 6, 3]], dtype=torch.long)
    print(f"Original Token Sequence:\n  {tokens.tolist()[0]}")

    inputs, targets = prepare_shifted_sequences(tokens)
    print(f"\nModel Input  (X): {inputs.tolist()[0]} (Length {inputs.size(1)})")
    print(f"Model Target (Y): {targets.tolist()[0]} (Length {targets.size(1)})")

    # Verify shift alignment
    for pos in range(inputs.size(1)):
        print(f"  Step {pos}: When given token ID {inputs[0, pos].item()}, target next token is ID {targets[0, pos].item()}")

    # Demonstrate Cross-Entropy Loss computation over a mini-vocabulary (vocab_size = 10)
    vocab_size = 10
    batch_size = 1
    seq_len = inputs.size(1)

    # Simulated model output logits: (batch, seq_len, vocab_size)
    torch.manual_seed(42)
    simulated_logits = torch.randn(batch_size, seq_len, vocab_size)

    criterion = nn.CrossEntropyLoss()

    # PyTorch CrossEntropyLoss expects:
    # - Logits: (N, vocab_size) where N = batch_size * seq_len
    # - Targets: (N)
    flat_logits = simulated_logits.view(-1, vocab_size)
    flat_targets = targets.view(-1)

    loss = criterion(flat_logits, flat_targets)
    print(f"\nComputed Next-Token Cross-Entropy Loss: {loss.item():.4f}")
    assert loss.item() > 0.0

    return inputs, targets, loss.item()


if __name__ == "__main__":
    demonstrate_next_token_prediction()
    print("\n" + "=" * 75)
    print("LESSON 9 COMPLETED SUCCESSFULLY!")
    print("=" * 75)
