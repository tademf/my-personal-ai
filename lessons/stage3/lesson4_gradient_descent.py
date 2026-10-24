"""
=============================================================================
STAGE THREE - LESSON 4: GRADIENT DESCENT OPTIMIZATION
File: lessons/stage3/lesson4_gradient_descent.py
=============================================================================
Gradient Descent is the universal optimization algorithm powering almost all
modern machine learning and deep learning (including Transformers & LLMs).

Core Equation:
  weight = weight - (learning_rate * gradient)
  w := w - alpha * (dL / dw)

This script demonstrates:
1. 1D Convex Optimization: Finding the minimum of f(w) = w^2 - 4w + 7 (analytical min at w=2).
2. The impact of the learning rate (alpha):
   - Optimal alpha: Smooth, steady convergence.
   - Too small alpha: Painfully slow progress.
   - Too large alpha: Overshooting, oscillation, and divergence!
3. Multivariate Gradient Descent: Training a Linear Regression model from scratch
   (learning weight and bias iteratively) without any scikit-learn help.
4. Generates a convergence trajectory visualization.
=============================================================================
"""

from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def objective_function(w: float) -> float:
    """A simple convex bowl: f(w) = w^2 - 4w + 7. Minimum occurs at w = 2.0."""
    return (w ** 2) - (4.0 * w) + 7.0


def objective_gradient(w: float) -> float:
    """The exact derivative: df/dw = 2w - 4."""
    return (2.0 * w) - 4.0


def run_1d_gradient_descent(
    initial_w: float = 8.0,
    learning_rate: float = 0.1,
    n_iterations: int = 15
) -> Tuple[List[float], List[float]]:
    """Tracks w and loss values over iterations."""
    w = initial_w
    history_w = [w]
    history_loss = [objective_function(w)]

    for step in range(1, n_iterations + 1):
        grad = objective_gradient(w)
        # Parameter update step: move against the gradient slope
        w = w - (learning_rate * grad)
        loss = objective_function(w)

        history_w.append(w)
        history_loss.append(loss)

    return history_w, history_loss


def demonstrate_learning_rates():
    """
    Compares 3 learning rates:
    - Optimal (alpha = 0.15)
    - Too small (alpha = 0.02)
    - Too large / overshooting (alpha = 1.05)
    """
    print("=" * 75)
    print("LESSON 4 - PART 1: 1D GRADIENT DESCENT & LEARNING RATE DYNAMICS")
    print("=" * 75)
    print("Function to minimize: f(w) = w^2 - 4w + 7")
    print("Analytical minimum: w = 2.0, f(2.0) = 3.0\n")

    initial_w = 7.0
    iterations = 10

    # Optimal
    w_hist_opt, loss_hist_opt = run_1d_gradient_descent(initial_w, 0.15, iterations)
    # Too slow
    w_hist_slow, loss_hist_slow = run_1d_gradient_descent(initial_w, 0.02, iterations)
    # Overshooting
    w_hist_over, loss_hist_over = run_1d_gradient_descent(initial_w, 1.05, iterations)

    print(f"{'Step':<5} | {'Optimal (alpha=0.15) w':<24} | {'Slow (alpha=0.02) w':<20} | {'Overshooting (alpha=1.05) w'}")
    print("-" * 80)
    for step in range(len(w_hist_opt)):
        print(f"{step:<5} | {w_hist_opt[step]:<24.4f} | {w_hist_slow[step]:<20.4f} | {w_hist_over[step]:<20.4f}")

    print("\nObservation:")
    print("  - Optimal (0.15): Rapidly converges toward 2.0000.")
    print("  - Slow (0.02): Barely moves toward 2.0 after 10 steps.")
    print("  - Overshooting (1.05): Flips sign, oscillates with growing magnitude, and explodes!")

    # Plot convergence trajectories
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Curve plot
    w_vals = np.linspace(-4, 9, 300)
    loss_vals = [objective_function(val) for val in w_vals]
    ax1.plot(w_vals, loss_vals, "k-", linewidth=2, label="f(w) = w^2 - 4w + 7")
    ax1.scatter(w_hist_opt, loss_hist_opt, color="green", s=50, zorder=5, label="Optimal Path (α=0.15)")
    ax1.plot(w_hist_opt, loss_hist_opt, "g--", alpha=0.8)
    ax1.scatter([2.0], [3.0], color="red", marker="*", s=200, zorder=6, label="Global Minimum (2, 3)")
    ax1.set_xlabel("Parameter Value (w)")
    ax1.set_ylabel("Loss f(w)")
    ax1.set_title("Optimization Landscape & Trajectory")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Loss vs step plot
    steps = list(range(len(loss_hist_opt)))
    ax2.plot(steps, loss_hist_opt, "g-o", label="Optimal (alpha=0.15)")
    ax2.plot(steps, loss_hist_slow, "b-s", label="Too Slow (alpha=0.02)")
    ax2.set_xlabel("Optimization Step")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss Reduction per Iteration")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_path = Path(__file__).resolve().parent / "gradient_descent_convergence.png"
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"Convergence plot saved to: {output_path}")


def train_linear_regression_from_scratch():
    """
    Part 2: 2D Gradient Descent for Linear Regression (y = w * x + b).
    Shows how weights (w) and biases (b) are updated using MSE loss gradients.
    dL/dw = -(2/n) * sum(x * (y - y_hat))
    dL/db = -(2/n) * sum(y - y_hat)
    """
    print("\n" + "=" * 75)
    print("LESSON 4 - PART 2: LINEAR REGRESSION TRAINED WITH GRADIENT DESCENT FROM SCRATCH")
    print("=" * 75)

    rng = np.random.default_rng(seed=42)
    # Synthetic ground truth: y = 3.5 * x + 10.0 + noise
    x = rng.uniform(1.0, 10.0, size=50)
    y = 3.5 * x + 10.0 + rng.normal(0, 0.5, size=50)

    # Initialize parameters randomly
    w = 0.0
    b = 0.0
    learning_rate = 0.01
    epochs = 1200
    n = len(x)

    print(f"Ground Truth Formula: y = 3.5 * x + 10.0")
    print(f"Initial Parameters:   w = {w:.2f}, b = {b:.2f}")

    for epoch in range(1, epochs + 1):
        # 1. Forward pass (compute prediction)
        y_hat = (w * x) + b

        # 2. Compute Loss (MSE)
        mse = np.mean((y - y_hat) ** 2)

        # 3. Compute Gradients analytically
        dw = -(2.0 / n) * np.sum(x * (y - y_hat))
        db = -(2.0 / n) * np.sum(y - y_hat)

        # 4. Parameter update (backprop step)
        w = w - (learning_rate * dw)
        b = b - (learning_rate * db)

        if epoch in [1, 50, 150, 300, 600, 900, 1200]:
            print(f"Epoch {epoch:4d} | MSE: {mse:8.4f} | w: {w:6.4f} (target 3.5) | b: {b:6.4f} (target 10.0)")

    print(f"\nFinal Learned Formula: y = {w:.3f} * x + {b:.3f}")
    print("Takeaway: This is the exact mechanism that trains Deep Neural Networks and LLMs.")
    print("In PyTorch, 'loss.backward()' computes dw and db, and 'optimizer.step()' updates w and b.")

    return w, b


if __name__ == "__main__":
    demonstrate_learning_rates()
    train_linear_regression_from_scratch()
    print("\n" + "=" * 75)
    print("LESSON 4 COMPLETED SUCCESSFULLY!")
    print("=" * 75)
