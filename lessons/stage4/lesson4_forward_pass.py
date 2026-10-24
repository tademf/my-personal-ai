"""
=============================================================================
STAGE FOUR - LESSON 4: THE FORWARD PASS WITH torch.nn.Linear
File: lessons/stage4/lesson4_forward_pass.py
=============================================================================
In PyTorch, neural network architectures are defined by subclassing `torch.nn.Module`.
A forward pass is the process of feeding input tensors through the network's layers
to produce output predictions.

This lesson explains:
1. `torch.nn.Linear`: Implements the affine transformation y = x @ W^T + b.
2. Building a Multi-Layer Perceptron (MLP) class.
3. Tracking tensor dimensions at each stage of the forward pass.
4. Batch processing: How neural networks process multiple samples simultaneously.
=============================================================================
"""

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """
    A 2-layer Multi-Layer Perceptron (MLP):
    Input Layer (in_features=4) -> Hidden Layer (hidden_dim=8) -> ReLU -> Output Layer (out_features=2)
    """
    def __init__(self, in_features: int = 4, hidden_dim: int = 8, out_features: int = 2):
        super().__init__()
        # Layer 1: Transforms 4 input features to 8 hidden activations
        self.fc1 = nn.Linear(in_features, hidden_dim)
        # Non-linear activation
        self.relu = nn.ReLU()
        # Layer 2: Transforms 8 hidden features to 2 output logits
        self.fc2 = nn.Linear(hidden_dim, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the computational pipeline for input tensor x.
        x shape: (batch_size, in_features)
        """
        # Step 1: Linear transformation Layer 1
        z1 = self.fc1(x)       # (batch_size, 8)
        # Step 2: Non-linear activation
        a1 = self.relu(z1)     # (batch_size, 8)
        # Step 3: Linear transformation Layer 2
        logits = self.fc2(a1)  # (batch_size, 2)
        return logits


def demonstrate_forward_pass():
    """Demonstrates passing single samples and batches through SimpleMLP."""
    print("=" * 75)
    print("DEMONSTRATING A PYTORCH FORWARD PASS")
    print("=" * 75)

    torch.manual_seed(42)
    model = SimpleMLP(in_features=4, hidden_dim=8, out_features=2)
    print("Model Architecture:\n", model)

    # Inspect learned parameters and their shapes
    print("\nLayer Parameters:")
    for name, param in model.named_parameters():
        print(f"  {name:<15} shape={tuple(param.shape)} requires_grad={param.requires_grad}")

    # Case 1: Single sample input (shape [1, 4])
    single_input = torch.tensor([[1.2, -0.5, 2.3, 0.8]])
    single_output = model(single_input)
    print(f"\nSingle Input Shape:  {single_input.shape}")
    print(f"Single Output Shape: {single_output.shape}")
    print(f"Raw Output Logits:   {single_output.detach().tolist()}")

    # Case 2: Batch processing (Batch of 5 samples: shape [5, 4])
    batch_input = torch.randn(5, 4)
    batch_output = model(batch_input)
    print(f"\nBatch Input Shape:   {batch_input.shape} (5 samples in parallel)")
    print(f"Batch Output Shape:  {batch_output.shape} (5 output predictions)")
    print("Predictions for all 5 samples computed in a single vectorized GPU/CPU operation!")

    # Applying Softmax to convert raw logits into valid probabilities
    probabilities = torch.softmax(batch_output, dim=1)
    print("\nProbabilities after Softmax (rows sum to 1.0):")
    for idx, prob in enumerate(probabilities.detach().tolist()):
        print(f"  Sample {idx}: Class 0 = {prob[0]:.4f}, Class 1 = {prob[1]:.4f} (Sum = {sum(prob):.4f})")

    return model, batch_output


if __name__ == "__main__":
    print("STAGE FOUR - LESSON 4: FORWARD PASS IMPLEMENTATION")
    demonstrate_forward_pass()
    print("\n" + "=" * 75)
    print("LESSON 4 COMPLETED SUCCESSFULLY!")
    print("=" * 75)
