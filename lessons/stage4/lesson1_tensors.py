"""
=============================================================================
STAGE FOUR - LESSON 1: TENSORS IN PYTORCH
File: lessons/stage4/lesson1_tensors.py
=============================================================================
A Tensor is the foundational data structure of deep learning.
Conceptually, a tensor is a generalized multi-dimensional array of numbers:
- 0D Tensor: Scalar (a single number, e.g., 42.0)
- 1D Tensor: Vector (a 1D array of numbers, e.g., [1.0, 2.0, 3.0])
- 2D Tensor: Matrix (a table of rows and columns, e.g., tabular features)
- 3D Tensor: Cube of numbers (e.g., batches of text tokens: [batch, sequence_length, embedding_dim])
- 4D Tensor: (e.g., computer vision images: [batch, channels, height, width])

PyTorch tensors are virtually identical to NumPy ndarrays, with two superpowers:
1. They can run lightning-fast on accelerators (GPUs, TPUs).
2. They support automatic differentiation (Autograd) for computing derivatives.
=============================================================================
"""

import torch
import numpy as np


def demonstrate_tensor_creation():
    """Demonstrates scalar, vector, matrix, and higher-dimensional tensors."""
    print("=" * 75)
    print("PART 1: TENSOR CREATION AND DIMENSIONS")
    print("=" * 75)

    # 0D Tensor (Scalar)
    scalar = torch.tensor(3.14159)
    print(f"0D Tensor (Scalar): value={scalar.item():.4f}, shape={scalar.shape}, dim={scalar.ndim}")

    # 1D Tensor (Vector)
    vector = torch.tensor([10.0, 20.0, 30.0, 40.0])
    print(f"1D Tensor (Vector): values={vector.tolist()}, shape={vector.shape}, dim={vector.ndim}")

    # 2D Tensor (Matrix: 3 rows, 2 columns)
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    print(f"2D Tensor (Matrix 3x2):\n{matrix}\nshape={matrix.shape}, dim={matrix.ndim}")

    # 3D Tensor (Batch of sequences: 2 batches, 3 tokens each, 4 embedding dimensions)
    batch_tokens = torch.randn(2, 3, 4)
    print(f"3D Tensor (LLM-style shape [batch=2, seq_len=3, embed_dim=4]):\nshape={batch_tokens.shape}")

    # Interoperability with NumPy
    np_arr = np.array([1.5, 2.5, 3.5])
    t_from_np = torch.from_numpy(np_arr)
    print(f"\nCreated tensor from NumPy array: {t_from_np}, dtype={t_from_np.dtype}")

    return scalar, vector, matrix, batch_tokens


def demonstrate_indexing_and_reshaping():
    """Demonstrates slicing, reshaping, viewing, squeezing, and unsqueezing."""
    print("\n" + "=" * 75)
    print("PART 2: INDEXING, SLICING, AND SHAPE MANIPULATION")
    print("=" * 75)

    data = torch.tensor([
        [10, 20, 30, 40],
        [50, 60, 70, 80],
        [90, 100, 110, 120]
    ], dtype=torch.float32)

    print(f"Original 3x4 Matrix:\n{data}")
    # Slicing: Row 1, Columns 2 to 3 (0-indexed)
    sliced = data[1, 2:4]
    print(f"Slice [row 1, cols 2:4]: {sliced.tolist()}")

    # Reshaping: 3x4 -> 2x6
    reshaped = data.view(2, 6)
    print(f"Reshaped with .view(2, 6):\n{reshaped}")

    # Flattening into 1D
    flattened = data.flatten()
    print(f"Flattened to 1D (length {flattened.shape[0]}): {flattened.tolist()}")

    # Unsqueeze (adds a dimension of size 1, common for batching single inputs)
    unsqueezed = flattened.unsqueeze(0)  # Shape (1, 12)
    print(f"Unsqueeze dimension 0: shape changed from {flattened.shape} -> {unsqueezed.shape}")

    # Squeeze (removes all singleton dimensions)
    squeezed = unsqueezed.squeeze()
    print(f"Squeeze back: shape changed from {unsqueezed.shape} -> {squeezed.shape}")

    return sliced, reshaped, flattened


def demonstrate_math_operations():
    """Demonstrates element-wise operations and matrix multiplication."""
    print("\n" + "=" * 75)
    print("PART 3: TENSOR MATHEMATICS AND MATRIX MULTIPLICATION")
    print("=" * 75)

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    # 1. Element-wise operations
    add_res = a + b
    mult_elem = a * b
    print("Element-wise Addition (A + B):\n", add_res)
    print("Element-wise Multiplication (A * B - Hadamard product):\n", mult_elem)

    # 2. Matrix Multiplication (@ or torch.matmul)
    # Essential for neural network dense layers: (batch x in_features) @ (in_features x out_features)
    matmul_res = a @ b
    print("\nMatrix Multiplication (A @ B):\n", matmul_res)
    print("Calculation check: Row 0 x Col 0 = (1*5 + 2*7) = 19. Matches output!")

    # 3. Reduction operations
    sum_val = a.sum().item()
    mean_val = a.mean().item()
    mean_by_column = a.mean(dim=0).tolist()
    print(f"\nReduction Ops: Sum={sum_val:.1f}, Mean={mean_val:.2f}, Column Means={mean_by_column}")

    return matmul_res


if __name__ == "__main__":
    print("STAGE FOUR - LESSON 1: TENSORS IN PYTORCH")
    demonstrate_tensor_creation()
    demonstrate_indexing_and_reshaping()
    demonstrate_math_operations()
    print("\n" + "=" * 75)
    print("LESSON 1 COMPLETED SUCCESSFULLY!")
    print("=" * 75)
