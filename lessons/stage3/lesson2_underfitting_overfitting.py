"""
=============================================================================
STAGE THREE - LESSON 2: UNDERFITTING AND OVERFITTING
File: lessons/stage3/lesson2_underfitting_overfitting.py
=============================================================================
This script demonstrates the Bias-Variance tradeoff:
1. Underfitting (High Bias): Model is too simple to capture the underlying pattern.
2. Overfitting (High Variance): Model is overly complex and memorizes noise.
3. Good Generalization (Balanced): Model captures the true signal and ignores noise.

It produces quantitative error metrics (MAE and MSE) comparing training vs.
testing error, and saves a side-by-side comparison plot to disk.
=============================================================================
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for headless scripts
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


def generate_nonlinear_data(n_samples: int = 40, noise_std: float = 0.2, random_state: int = 42):
    """
    Generates synthetic data from a true non-linear mathematical curve:
    y = cos(1.5 * pi * x) + Gaussian noise
    Domain: x in [0, 1]
    """
    rng = np.random.default_rng(seed=random_state)
    X = np.sort(rng.uniform(0, 1, n_samples))
    y_clean = np.cos(1.5 * np.pi * X)
    y_noisy = y_clean + rng.normal(0, noise_std, n_samples)

    return X[:, np.newaxis], y_noisy, y_clean


def run_bias_variance_experiment():
    """
    Trains 3 models with varying degrees of polynomial complexity:
    - Degree 1: Linear model (Underfitting)
    - Degree 4: Quadratic/quartic model (Balanced Generalization)
    - Degree 15: 15th-order polynomial (Overfitting)
    """
    print("=" * 75)
    print("LESSON 2: UNDERFITTING VS. OVERFITTING (BIAS-VARIANCE TRADEOFF)")
    print("=" * 75)

    # 1. Generate Training Data (30 samples) and Test Data (30 unseen samples)
    X_train, y_train, _ = generate_nonlinear_data(n_samples=30, noise_std=0.18, random_state=42)
    X_test, y_test, _ = generate_nonlinear_data(n_samples=30, noise_std=0.18, random_state=99)

    # Dense points for plotting the true mathematical curve
    X_dense = np.linspace(0, 1, 200)[:, np.newaxis]
    y_true_dense = np.cos(1.5 * np.pi * X_dense)

    degrees = [1, 4, 15]
    descriptions = [
        "Underfitting (Degree 1, High Bias)",
        "Balanced Fit (Degree 4, Optimal)",
        "Overfitting (Degree 15, High Variance)"
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    results = {}

    print(f"{'Model Complexity':<35} | {'Train MSE':<12} | {'Test MSE':<12} | {'Diagnosis'}")
    print("-" * 75)

    for i, (degree, desc) in enumerate(zip(degrees, descriptions)):
        # Construct pipeline: Polynomial expansion followed by Linear Regression
        model = Pipeline([
            ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
            ("linear_regression", LinearRegression())
        ])
        model.fit(X_train, y_train)

        # Evaluate on both training set and unseen test set
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        train_mse = mean_squared_error(y_train, train_preds)
        test_mse = mean_squared_error(y_test, test_preds)

        if degree == 1:
            diagnosis = "High Train & High Test Error"
        elif degree == 4:
            diagnosis = "Low Train & Low Test Error (Ideal)"
        else:
            diagnosis = "Near-Zero Train, Massive Test Error"

        results[degree] = {"train_mse": train_mse, "test_mse": test_mse}
        print(f"{desc:<35} | {train_mse:<12.4f} | {test_mse:<12.4f} | {diagnosis}")

        # Plot predictions
        ax = axes[i]
        ax.plot(X_dense, y_true_dense, "k--", label="True Function", alpha=0.7)
        ax.scatter(X_train, y_train, color="royalblue", edgecolor="k", s=35, label="Train Samples")
        ax.scatter(X_test, y_test, color="crimson", marker="x", s=30, alpha=0.6, label="Unseen Test")
        
        y_dense_pred = model.predict(X_dense)
        ax.plot(X_dense, y_dense_pred, "darkorange", linewidth=2.5, label=f"Degree {degree} Fit")
        ax.set_ylim(-2.0, 2.0)
        ax.set_title(desc, fontsize=11, fontweight="bold")
        ax.set_xlabel("Input Feature (x)")
        if i == 0:
            ax.set_ylabel("Target (y)")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Bias-Variance Tradeoff: Underfitting vs. Balanced vs. Overfitting", fontsize=14, y=1.02)
    plt.tight_layout()

    output_path = Path(__file__).resolve().parent / "underfitting_overfitting.png"
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 75)
    print("KEY LESSON TAKEAWAYS:")
    print("=" * 75)
    print("1. Low training error DOES NOT equal a good model.")
    print("   Notice Degree 15 has the lowest Train MSE, but explodes on unseen Test data.")
    print("2. The goal of machine learning is NOT memorizing past samples.")
    print("   The goal is GENERALIZATION to future, unseen inputs.")
    print(f"3. Visualization saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_bias_variance_experiment()
