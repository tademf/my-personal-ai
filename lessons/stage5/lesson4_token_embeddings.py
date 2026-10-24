"""
=============================================================================
STAGE FIVE - LESSON 4: TOKEN EMBEDDINGS (torch.nn.Embedding)
File: lessons/stage5/lesson4_token_embeddings.py
=============================================================================
Token IDs are integers (e.g. 5, 12, 89). But integers have an arbitrary order:
token 12 is NOT "greater than" token 5, nor is token 6 "closer" to 7 than to 100.

An Embedding Layer (`torch.nn.Embedding`) solves this by mapping each discrete
token ID to a continuous, high-dimensional vector in a semantic vector space.

Key Concept:
`torch.nn.Embedding(vocab_size, embedding_dim)` is simply a learnable lookup matrix
of shape `(vocab_size, embedding_dim)`:
- Row `i` of the matrix is the vector representing token `i`.
- Input shape:  `(batch_size, sequence_length)` of integer IDs
- Output shape: `(batch_size, sequence_length, embedding_dim)` of floating-point vectors.
=============================================================================
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def demonstrate_token_embeddings():
    """Demonstrates token ID conversion to continuous dense vectors."""
    print("=" * 75)
    print("STAGE FIVE - LESSON 4: TOKEN EMBEDDINGS")
    print("=" * 75)

    torch.manual_seed(42)

    # Let vocab size be 10 tokens, and embedding dimension be 4
    vocab_size = 10
    embedding_dim = 4

    embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

    print(f"Embedding Lookup Table Shape: {embedding_layer.weight.shape}")
    print("Underlying learnable weights (10 tokens x 4 dimensions):")
    print(embedding_layer.weight.data)

    # Input: A batch of 2 sequences, each of length 3 token IDs
    token_ids = torch.tensor([
        [2, 5, 8],  # Sequence 1: tokens 2, 5, 8
        [1, 9, 0]   # Sequence 2: tokens 1, 9, 0
    ], dtype=torch.long)

    print(f"\nInput Token IDs (Batch=2, Seq_Len=3):\n{token_ids}")

    # Forward lookup: Look up corresponding rows in the weight table
    embedded_vectors = embedding_layer(token_ids)

    print(f"\nOutput Embedded Tensor Shape: {embedded_vectors.shape}")
    print("Shape Explanation: [batch_size=2, sequence_length=3, embedding_dim=4]")
    print("\nEmbedded Representation of Sequence 1 (Token 2, Token 5, Token 8):")
    print(embedded_vectors[0])

    # Verify that the vector for token 2 matches row 2 of the embedding table
    diff = torch.norm(embedded_vectors[0, 0] - embedding_layer.weight[2]).item()
    assert diff < 1e-6, "Embedding lookup does not match table row!"
    print(f"\nVerification: Token 2 embedded vector exactly matches table row 2 (diff={diff:.1e})!")

    print("\nWhy is this essential for Transformers?")
    print("1. It replaces arbitrary integer IDs with rich floating-point coordinates.")
    print("2. During pre-training, backpropagation updates these vectors so that semantically")
    print("   related tokens ('king' and 'queen', or 'python' and 'javascript') cluster together.")

    return embedding_layer, embedded_vectors


if __name__ == "__main__":
    demonstrate_token_embeddings()
    print("\n" + "=" * 75)
    print("LESSON 4 COMPLETED SUCCESSFULLY!")
    print("=" * 75)
