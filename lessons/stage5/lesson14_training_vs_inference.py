"""
=============================================================================
STAGE FIVE - LESSON 14: TRAINING VERSUS INFERENCE
File: lessons/stage5/lesson14_training_vs_inference.py
=============================================================================
Operating a language model during TRAINING is fundamentally different from
operating it during INFERENCE (Generation).

CRITICAL DIFFERENCES:

1. COMPUTATIONAL GRAPH & MEMORY:
   - Training: Must store all intermediate activations in RAM so backpropagation
     can compute gradients via the chain rule. High memory consumption!
   - Inference: No backward pass is needed. Wrapped in `with torch.no_grad():`,
     saving massive amounts of RAM and compute time.

2. MODEL MODES:
   - `model.train()`: Enables stochastic regularization layers (Dropout, BatchNorm).
   - `model.eval()`: Disables Dropout (freezes weights for deterministic predictions)
     and uses accumulated global normalization statistics.

3. KV-CACHING (KEY-VALUE CACHING):
   - Without caching: Generating token 100 re-computes attention over tokens 1 to 99!
   - With KV-Caching: The model stores computed Key and Value matrices in RAM,
     only computing Q for the newly generated token. This cuts generation latency by 10x!
=============================================================================
"""

import sys
import time
from pathlib import Path
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from lessons.stage5.lesson10_training_validation import TinyLanguageModel


def compare_training_vs_inference_modes():
    """Demonstrates model.train() vs model.eval() and torch.no_grad()."""
    print("=" * 75)
    print("STAGE FIVE - LESSON 14: TRAINING VERSUS INFERENCE MODES")
    print("=" * 75)

    vocab_size = 20
    model = TinyLanguageModel(vocab_size=vocab_size, d_model=32, num_heads=4, d_ff=64)

    # Input sequence of 10 tokens
    dummy_input = torch.randint(0, vocab_size, (1, 10))

    # 1. Training Mode (Grad tracking ON)
    model.train()
    out_train = model(dummy_input)
    print("1. Training Mode (`model.train()`):")
    print(f"   requires_grad on output: {out_train.requires_grad}")
    print(f"   grad_fn attached:        {out_train.grad_fn} (Graph actively tracked)")

    # 2. Inference Mode (Grad tracking OFF)
    model.eval()
    with torch.no_grad():
        out_eval = model(dummy_input)
    print("\n2. Inference Mode (`model.eval()` + `torch.no_grad()`):")
    print(f"   requires_grad on output: {out_eval.requires_grad}")
    print(f"   grad_fn attached:        {out_eval.grad_fn} (Zero graph overhead!)")

    assert out_eval.requires_grad is False

    # 3. Deterministic Inference Verification
    with torch.no_grad():
        out_eval_2 = model(dummy_input)
    max_diff = torch.max(torch.abs(out_eval - out_eval_2)).item()
    print(f"\n3. Determinism Check: Discrepancy between two identical eval runs: {max_diff:.1e}")
    assert max_diff == 0.0, "Inference was not deterministic!"
    print("   Verification: model.eval() guarantees bit-exact reproducible responses.")

    print("\n4. Understanding KV-Caching (Key-Value Caching):")
    print("   - When you prompt an LLM: 'The quick brown fox'")
    print("   - Generates 'jumps'. In naive decoding, the next step would recompute Q, K, V")
    print("     for 'The', 'quick', 'brown', 'fox', AND 'jumps'.")
    print("   - In KV-Caching, K and V for the past tokens are preserved in memory.")
    print("     The model only computes Q for the new word, vastly speeding up generation.")


if __name__ == "__main__":
    compare_training_vs_inference_modes()
    print("\n" + "=" * 75)
    print("LESSON 14 COMPLETED SUCCESSFULLY!")
    print("=" * 75)
