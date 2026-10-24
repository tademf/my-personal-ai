"""
=============================================================================
STAGE FOUR - LESSON 5: BACKPROPAGATION AND PARAMETER UPDATES
File: lessons/stage4/lesson5_backpropagation.py
=============================================================================
Backpropagation is the process of computing gradients across all layers using
the Chain Rule of Calculus, starting from the loss and flowing backwards
to the network's input weights.

The Standard 5-Step PyTorch Training Recipe:
1. `optimizer.zero_grad()`  -> Reset accumulated gradients to zero.
2. `outputs = model(inputs)` -> Forward pass to compute predictions.
3. `loss = criterion(outputs, targets)` -> Evaluate loss against ground truth.
4. `loss.backward()`        -> Backpropagation (compute dLoss/dWeight for all params).
5. `optimizer.step()`        -> Update weights: w := w - lr * grad.

This lesson inspects weight values and gradients before and after each step.
=============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim


def demonstrate_backprop_step():
    """Traces parameters, loss, gradients, and optimizer updates numerically."""
    print("=" * 75)
    print("STEP-BY-STEP NUMERICAL TRACE OF BACKPROPAGATION")
    print("=" * 75)

    torch.manual_seed(42)

    # 1. Simple Single-Layer Model: 2 inputs -> 1 output
    model = nn.Linear(2, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    print(f"Initial Weight (fc.weight): {model.weight.data.tolist()}")
    print(f"Initial Bias   (fc.bias):   {model.bias.data.tolist()}")
    print(f"Initial Gradients:          {model.weight.grad} (None before backward)")

    # Synthetic sample: Input [2.0, 3.0], Target [10.0]
    inputs = torch.tensor([[2.0, 3.0]])
    targets = torch.tensor([[10.0]])

    print("\n--- Executing The 5-Step Training Step ---")

    # Step 1: Zero gradients
    optimizer.zero_grad()
    print("Step 1: optimizer.zero_grad() -> Clears any residual gradients.")

    # Step 2: Forward pass
    outputs = model(inputs)
    print(f"Step 2: Forward Pass Output: {outputs.item():.4f} (Target is {targets.item():.1f})")

    # Step 3: Compute Loss
    loss = criterion(outputs, targets)
    print(f"Step 3: Loss (MSE):          {loss.item():.4f}")

    # Step 4: Backpropagation
    loss.backward()
    print("Step 4: loss.backward() executed.")
    print(f"  -> Computed weight gradient: {model.weight.grad.tolist()}")
    print(f"  -> Computed bias gradient:   {model.bias.grad.tolist()}")

    # Step 5: Optimizer Step
    pre_weight = model.weight.data.clone()
    optimizer.step()
    post_weight = model.weight.data.clone()

    print("Step 5: optimizer.step() executed (weight = weight - lr * gradient).")
    print(f"  -> Previous Weight: {pre_weight.tolist()}")
    print(f"  -> Updated Weight:  {post_weight.tolist()}")

    # Verify manual calculation: new_w = old_w - (0.1 * grad)
    expected_new_w = pre_weight - (0.1 * model.weight.grad)
    diff = torch.norm(post_weight - expected_new_w).item()
    assert diff < 1e-6
    print(f"\nVerification: Updated weights match exact gradient update formula (diff={diff:.1e})!")

    return model, loss.item()


if __name__ == "__main__":
    print("STAGE FOUR - LESSON 5: BACKPROPAGATION")
    demonstrate_backprop_step()
    print("\n" + "=" * 75)
    print("LESSON 5 COMPLETED SUCCESSFULLY!")
    print("=" * 75)
