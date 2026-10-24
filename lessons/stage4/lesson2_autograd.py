"""
=============================================================================
STAGE FOUR - LESSON 2: AUTOMATIC DIFFERENTIATION (AUTOGRAD)
File: lessons/stage4/lesson2_autograd.py
=============================================================================
In Stage Three, we manually derived the calculus gradients (dw and db) by hand.
In deep neural networks with millions of parameters, deriving derivatives by hand
is impossible.

PyTorch's Autograd engine automates this:
1. When a tensor has `requires_grad=True`, PyTorch records every mathematical
   operation applied to it inside a dynamic Computational Graph.
2. When you call `.backward()` on a final scalar loss, Autograd automatically traverses
   the graph backwards using the Calculus Chain Rule to compute the exact gradient
   of the loss with respect to every parameter: dLoss / dParam.
3. The computed gradients are stored in the `.grad` attribute of each tensor.
=============================================================================
"""

import torch


def demonstrate_single_variable_autograd():
    """
    Computes derivative of y = 3x^2 + 2x + 5 at x = 4.0.
    Calculus: dy/dx = 6x + 2.
    At x = 4.0: dy/dx = 6(4) + 2 = 26.0.
    """
    print("=" * 75)
    print("PART 1: SINGLE VARIABLE AUTOGRAD DERIVATIVE")
    print("=" * 75)

    x = torch.tensor(4.0, requires_grad=True)
    # Forward computation
    y = 3.0 * (x ** 2) + 2.0 * x + 5.0

    print(f"Input x:                 {x.item():.2f}")
    print(f"Evaluated y = 3x^2+2x+5: {y.item():.2f}")
    print(f"Pre-backward gradient:   {x.grad}")

    # Backward pass: Compute dy/dx
    y.backward()

    print(f"Computed x.grad:         {x.grad.item():.2f}")
    print("Analytical dy/dx = 6x + 2 = 6(4) + 2 = 26.00")
    assert abs(x.grad.item() - 26.0) < 1e-5
    print("Verification: Autograd matches exact analytical calculus!")

    return x.grad.item()


def demonstrate_multivariable_autograd():
    """
    Computes partial derivatives of z = 2*a^2 + 3*b^3 at a = 2.0, b = 3.0.
    Partial derivative dz/da = 4*a -> 4(2) = 8.0.
    Partial derivative dz/db = 9*b^2 -> 9(3^2) = 81.0.
    """
    print("\n" + "=" * 75)
    print("PART 2: MULTIVARIATE PARTIAL DERIVATIVES")
    print("=" * 75)

    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)

    z = 2.0 * (a ** 2) + 3.0 * (b ** 3)
    z.backward()

    print(f"Inputs: a = {a.item():.1f}, b = {b.item():.1f}")
    print(f"Function z = 2a^2 + 3b^3 = {z.item():.1f}")
    print(f"Partial derivative dz/da: computed={a.grad.item():.2f}, expected=8.00")
    print(f"Partial derivative dz/db: computed={b.grad.item():.2f}, expected=81.00")

    return a.grad.item(), b.grad.item()


def demonstrate_gradient_accumulation_and_zeroing():
    """
    Demonstrates that PyTorch accumulates gradients by default (+ instead of =),
    and why zeroing gradients (.grad.zero_()) is mandatory before each training step.
    """
    print("\n" + "=" * 75)
    print("PART 3: GRADIENT ACCUMULATION AND ZEROING")
    print("=" * 75)

    w = torch.tensor(5.0, requires_grad=True)

    # First iteration: loss = 2 * w -> dw = 2
    loss1 = 2.0 * w
    loss1.backward()
    print(f"Step 1: loss = 2*w -> w.grad = {w.grad.item():.2f}")

    # Second iteration without zeroing: loss = 2 * w
    loss2 = 2.0 * w
    loss2.backward()
    print(f"Step 2 (Without zero_grad): w.grad accumulated to = {w.grad.item():.2f} (2 + 2 = 4!)")

    # Correct practice: Zero gradients before backward pass
    w.grad.zero_()
    loss3 = 2.0 * w
    loss3.backward()
    print(f"Step 3 (After w.grad.zero_()): w.grad correctly reset to = {w.grad.item():.2f}")


def demonstrate_no_grad_context():
    """
    Demonstrates torch.no_grad(): Disables computational graph construction
    during evaluation or inference to save RAM and computation time.
    """
    print("\n" + "=" * 75)
    print("PART 4: DISABLING AUTOGRAD FOR INFERENCE (torch.no_grad)")
    print("=" * 75)

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    # Under torch.no_grad(), operations do not track history
    with torch.no_grad():
        y = x * 2.0

    print(f"Input x requires_grad:  {x.requires_grad}")
    print(f"Output y requires_grad: {y.requires_grad}")
    print(f"Output y grad_fn:       {y.grad_fn} (None indicates no graph tracking)")


if __name__ == "__main__":
    print("STAGE FOUR - LESSON 2: AUTOMATIC DIFFERENTIATION (AUTOGRAD)")
    demonstrate_single_variable_autograd()
    demonstrate_multivariable_autograd()
    demonstrate_gradient_accumulation_and_zeroing()
    demonstrate_no_grad_context()
    print("\n" + "=" * 75)
    print("LESSON 2 COMPLETED SUCCESSFULLY!")
    print("=" * 75)
