"""
=============================================================================
STAGE FOUR - LESSON 3: NEURAL NETWORK BASICS & ACTIVATION FUNCTIONS
File: lessons/stage4/lesson3_neural_network_basics.py
=============================================================================
A Deep Neural Network is composed of stacked layers of artificial neurons:
1. Input Layer: Accepts the raw feature vector (e.g., [x1, x2, ... xn]).
2. Hidden Layers: Learn increasingly abstract hierarchical representations.
3. Output Layer: Produces final predictions (probabilities or continuous numbers).

Inside each artificial neuron:
  Step 1 (Linear Combination):  z = (w1*x1 + w2*x2 + ... + wn*xn) + b = W*x + b
  Step 2 (Non-linear Activation): a = Activation(z)

WHY ARE ACTIVATION FUNCTIONS MANDATORY?
Without non-linear activations, stacking 100 linear layers is mathematically
identical to a SINGLE linear layer:
  Layer2(Layer1(x)) = W2 * (W1 * x + b1) + b2 = (W2 * W1) * x + (W2 * b1 + b2)
                    = W_combined * x + b_combined
Activation functions allow neural networks to model non-linear boundaries.
=============================================================================
"""

import torch
import torch.nn as nn


def demonstrate_activation_functions():
    """Demonstrates ReLU, Sigmoid, Tanh, and GELU on a test range."""
    print("=" * 75)
    print("PART 1: COMMON ACTIVATION FUNCTIONS IN DEEP LEARNING & LLMs")
    print("=" * 75)

    # Test inputs spanning negative, zero, and positive values
    x = torch.tensor([-3.0, -1.5, 0.0, 1.5, 3.0])

    relu = nn.ReLU()
    sigmoid = nn.Sigmoid()
    tanh = nn.Tanh()
    gelu = nn.GELU()  # Modern standard in GPT / BERT / Transformers

    y_relu = relu(x)
    y_sigmoid = sigmoid(x)
    y_tanh = tanh(x)
    y_gelu = gelu(x)

    print(f"{'Input (x)':<12} | {'ReLU':<10} | {'Sigmoid':<10} | {'Tanh':<10} | {'GELU (LLMs)':<12}")
    print("-" * 75)
    for i in range(len(x)):
        print(f"{x[i].item():<12.1f} | {y_relu[i].item():<10.4f} | {y_sigmoid[i].item():<10.4f} | {y_tanh[i].item():<10.4f} | {y_gelu[i].item():<12.4f}")

    print("\nKey Insights:")
    print("1. ReLU (Rectified Linear Unit, max(0, x)): Simple, fast, solves vanishing gradients in positives.")
    print("2. Sigmoid (1 / (1 + e^-x)): Squeezes into (0, 1). Ideal for binary probabilities.")
    print("3. Tanh: Squeezes into (-1, +1). Zero-centered.")
    print("4. GELU (Gaussian Error Linear Unit): Smooth probabilistic activation used in modern Transformers.")

    return y_relu, y_sigmoid, y_gelu


def prove_linearity_collapse():
    """
    Mathematical Proof: Stacking two linear layers without activation
    collapses into an equivalent single linear layer.
    """
    print("\n" + "=" * 75)
    print("PART 2: MATHEMATICAL PROOF OF LINEARITY COLLAPSE")
    print("=" * 75)

    torch.manual_seed(42)

    # Define two sequential linear transformations: 2 -> 3 -> 1
    W1 = torch.tensor([[1.5, -0.5], [0.8, 2.0], [-1.0, 0.5]])  # 3x2
    b1 = torch.tensor([0.2, -0.1, 0.5])                         # 3

    W2 = torch.tensor([[0.7, -1.2, 0.4]])                        # 1x3
    b2 = torch.tensor([1.0])                                    # 1

    x = torch.tensor([3.0, 4.0])                                # 2

    # Step-by-step two-layer forward pass WITHOUT activation
    h = W1 @ x + b1       # Layer 1
    y_two_layers = W2 @ h + b2  # Layer 2

    # Equivalent collapsed single layer:
    # W_combined = W2 @ W1 (1x2 matrix)
    # b_combined = W2 @ b1 + b2 (scalar)
    W_combined = W2 @ W1
    b_combined = W2 @ b1 + b2
    y_single_layer = W_combined @ x + b_combined

    print(f"Two Layer Output:   {y_two_layers.item():.6f}")
    print(f"Single Layer Output:{y_single_layer.item():.6f}")
    diff = abs(y_two_layers.item() - y_single_layer.item())
    print(f"Absolute Difference:{diff:.8f}")
    print("Result: Exactly identical! Without activations, depth adds ZERO expressive power.")


if __name__ == "__main__":
    print("STAGE FOUR - LESSON 3: NEURAL NETWORK BASICS")
    demonstrate_activation_functions()
    prove_linearity_collapse()
    print("\n" + "=" * 75)
    print("LESSON 3 COMPLETED SUCCESSFULLY!")
    print("=" * 75)
