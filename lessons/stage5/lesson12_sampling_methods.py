"""
=============================================================================
STAGE FIVE - LESSON 12: SAMPLING METHODS (TEMPERATURE, TOP-K, AND TOP-P)
File: lessons/stage5/lesson12_sampling_methods.py
=============================================================================
If a language model always greedily selects the #1 highest-probability token
(argmax), the output becomes repetitive, stiff, and prone to endless loops.
Conversely, if it samples purely at random, it will occasionally pick rare,
nonsensical tokens from the long tail of the vocabulary.

Modern LLMs control generation quality using three complementary techniques:

1. TEMPERATURE (T):
   Scales logits before Softmax: Softmax(logits / T).
   - T -> 0 (e.g., 0.1): Spikes probabilities on top choices (conservative, focused).
   - T = 1.0: Normal calibrated model probabilities.
   - T > 1.0 (e.g., 1.8): Flattens the distribution (creative, unpredictable, riskier).

2. TOP-K SAMPLING:
   Restricts sampling strictly to the top K most likely tokens.
   All tokens outside the top K are masked to -infinity.

3. TOP-P (NUCLEUS) SAMPLING:
   Selects the dynamic smallest subset of tokens whose cumulative probability
   reaches P (e.g., 0.90 = 90% of the probability mass).
   Unlike Top-K which is fixed, Top-P expands when the model is uncertain and
   contracts when the model is confident.
=============================================================================
"""

import sys
from pathlib import Path
from typing import List
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Scales logits by temperature. Temperature must be > 0."""
    if temperature <= 0.0:
        raise ValueError("Temperature must be strictly positive.")
    return logits / temperature


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Masks all logits outside the top-k highest values with -infinity."""
    if k <= 0:
        return logits
    k = min(k, logits.size(-1))
    # Find the threshold value of the k-th largest logit
    top_values, _ = torch.topk(logits, k)
    min_top_val = top_values[-1]
    # Set anything below the k-th value to -infinity
    filtered_logits = torch.where(logits < min_top_val, torch.tensor(-float("inf")), logits)
    return filtered_logits


def apply_top_p(logits: torch.Tensor, p: float = 0.90) -> torch.Tensor:
    """
    Nucleus sampling: Keeps the top tokens whose cumulative probability <= p.
    """
    if p >= 1.0 or p <= 0.0:
        return logits

    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Find tokens that exceed the cumulative threshold p
    sorted_indices_to_remove = cumulative_probs > p
    # Shift indices right so the first token that exceeds p is still kept
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = False

    # Mask removed indices with -inf
    sorted_logits[sorted_indices_to_remove] = -float("inf")

    # Restore original token order
    filtered_logits = torch.zeros_like(logits).scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return filtered_logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0
) -> int:
    """Applies temperature, Top-K, and Top-P filtering, then samples a token ID."""
    scaled = apply_temperature(logits, temperature)
    if top_k > 0:
        scaled = apply_top_k(scaled, top_k)
    if top_p < 1.0:
        scaled = apply_top_p(scaled, top_p)

    probs = torch.softmax(scaled, dim=-1)
    sampled_id = torch.multinomial(probs, num_samples=1).item()
    return int(sampled_id)


def demonstrate_sampling():
    """Demonstrates how temperature, Top-K, and Top-P shape token probabilities."""
    print("=" * 75)
    print("STAGE FIVE - LESSON 12: SAMPLING METHODS")
    print("=" * 75)

    vocab = ["Paris", "Rome", "Berlin", "London", "Madrid", "Tokyo", "Banana"]
    # Simulated model logits for "The capital of France is"
    # Paris has highest logit, followed by Rome, Berlin, with "Banana" having a tiny logit
    raw_logits = torch.tensor([5.0, 3.5, 3.0, 2.5, 2.0, 1.0, -2.0])

    print("Candidate Vocabulary & Raw Logits:")
    for word, logit in zip(vocab, raw_logits.tolist()):
        print(f"  {word:<10}: logit={logit:+.2f}")

    # 1. Temperature Impact
    print("\n--- 1. Temperature Impact on Probabilities ---")
    for temp in [0.2, 1.0, 2.5]:
        scaled = apply_temperature(raw_logits, temp)
        probs = torch.softmax(scaled, dim=-1).tolist()
        desc = "Sharp/Focused (T=0.2)" if temp == 0.2 else "Standard (T=1.0)" if temp == 1.0 else "Flat/Creative (T=2.5)"
        print(f"\nTemperature {temp} ({desc}):")
        for word, prob in zip(vocab, probs):
            print(f"  {word:<10}: {prob * 100:5.2f}%")

    # 2. Top-K Impact (e.g. k=3)
    print("\n--- 2. Top-K Filtering (k=3) ---")
    top_k_logits = apply_top_k(raw_logits.clone(), k=3)
    top_k_probs = torch.softmax(top_k_logits, dim=-1).tolist()
    print("Filtered to top 3 choices only (all other words zeroed out):")
    for word, prob in zip(vocab, top_k_probs):
        status = "KEPT" if prob > 0 else "ZEROED"
        print(f"  {word:<10}: {prob * 100:5.2f}% [{status}]")

    # 3. Top-P Nucleus Impact (e.g. p=0.85)
    print("\n--- 3. Top-P Nucleus Filtering (p=0.85) ---")
    top_p_logits = apply_top_p(raw_logits.clone(), p=0.85)
    top_p_probs = torch.softmax(top_p_logits, dim=-1).tolist()
    print("Filtered to smallest subset reaching 85% cumulative probability mass:")
    for word, prob in zip(vocab, top_p_probs):
        status = "IN NUCLEUS" if prob > 0 else "PRUNED"
        print(f"  {word:<10}: {prob * 100:5.2f}% [{status}]")

    return raw_logits


if __name__ == "__main__":
    demonstrate_sampling()
    print("\n" + "=" * 75)
    print("LESSON 12 COMPLETED SUCCESSFULLY!")
    print("=" * 75)
