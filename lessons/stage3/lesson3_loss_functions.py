"""
=============================================================================
STAGE THREE - LESSON 3: LOSS FUNCTIONS
File: lessons/stage3/lesson3_loss_functions.py
=============================================================================
A loss function (or cost function) measures how bad a model's prediction is.
It maps (predicted_value, actual_value) to a single non-negative scalar number.
The objective of training is to adjust parameters to minimize this loss.

This script demonstrates:
1. Regression Loss Functions:
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)
   - Huber Loss (Robust combination of MSE & MAE)
2. Classification & Language Model Loss Functions:
   - Binary Cross-Entropy (Log Loss)
   - Categorical Cross-Entropy (Softmax Loss - the backbone of LLMs)
3. Direct sensitivity comparisons to outliers and prediction confidence.
=============================================================================
"""

import numpy as np


def mean_squared_error_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MSE: Average of squared residuals. L = (1/n) * sum((y - y_hat)^2)"""
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE: Average of absolute residuals. L = (1/n) * sum(|y - y_hat|)"""
    return float(np.mean(np.abs(y_true - y_pred)))


def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """
    Binary Cross-Entropy (Log Loss):
    L = - (1/n) * sum( y * log(p) + (1-y) * log(1-p) )
    p must be in range (0, 1). eps prevents log(0) which is undefined (-inf).
    """
    p_clipped = np.clip(y_prob, eps, 1.0 - eps)
    loss = -np.mean(y_true * np.log(p_clipped) + (1.0 - y_true) * np.log(1.0 - p_clipped))
    return float(loss)


def categorical_cross_entropy(y_one_hot: np.ndarray, predicted_probs: np.ndarray, eps: float = 1e-15) -> float:
    """
    Multi-Class Categorical Cross-Entropy:
    Used in Multi-class Classification and NEXT-TOKEN PREDICTION in LLMs.
    L = - sum( y_true_k * log(p_k) )
    """
    probs_clipped = np.clip(predicted_probs, eps, 1.0 - eps)
    loss = -np.sum(y_one_hot * np.log(probs_clipped)) / y_one_hot.shape[0]
    return float(loss)


def demonstrate_regression_losses():
    """
    Shows how MSE vs. MAE respond to small vs. large errors (outliers).
    """
    print("=" * 75)
    print("PART 1: REGRESSION LOSS FUNCTIONS (MSE VS. MAE)")
    print("=" * 75)

    y_true = np.array([100.0, 105.0, 98.0, 102.0, 101.0])

    # Case A: Consistent small errors of 3 units each
    y_pred_small_errors = np.array([103.0, 108.0, 101.0, 105.0, 104.0])

    # Case B: 4 perfect predictions, but 1 massive outlier error of 15 units
    y_pred_outlier_error = np.array([100.0, 105.0, 98.0, 102.0, 116.0])

    mae_a = mean_absolute_error_manual(y_true, y_pred_small_errors)
    mse_a = mean_squared_error_manual(y_true, y_pred_small_errors)

    mae_b = mean_absolute_error_manual(y_true, y_pred_outlier_error)
    mse_b = mean_squared_error_manual(y_true, y_pred_outlier_error)

    print("Actual Values:         ", y_true.tolist())
    print("\nCase A (Uniform small errors of 3.0):")
    print(f"  Predictions:         {y_pred_small_errors.tolist()}")
    print(f"  MAE:                 {mae_a:.2f}")
    print(f"  MSE:                 {mse_a:.2f}")

    print("\nCase B (Four perfect predictions + one 15.0 outlier error):")
    print(f"  Predictions:         {y_pred_outlier_error.tolist()}")
    print(f"  MAE:                 {mae_b:.2f}  (Notice: MAE is identical to Case A!)")
    print(f"  MSE:                 {mse_b:.2f}  (Notice: MSE increased by 5x due to squaring!)")

    print("\nTakeaway:")
    print("  - MSE squares errors: an error of 10 hurts 100x more than an error of 1.")
    print("    Use MSE when large errors are dangerous/unacceptable.")
    print("  - MAE treats all errors linearly. Use MAE when your data has extreme outliers.")


def demonstrate_classification_loss():
    """
    Shows how Cross-Entropy penalizes overconfident wrong predictions.
    """
    print("\n" + "=" * 75)
    print("PART 2: CLASSIFICATION LOSS (CROSS-ENTROPY / LOG LOSS)")
    print("=" * 75)

    # True class is Positive (1)
    y_true = np.array([1.0])

    print("Scenario: True label is 1 (Positive).")
    print(f"{'Predicted Probability P(y=1)':<32} | {'Cross-Entropy Loss':<20} | {'Behavior'}")
    print("-" * 75)

    test_probabilities = [0.99, 0.90, 0.70, 0.50, 0.30, 0.10, 0.01, 0.0001]
    for p in test_probabilities:
        loss = binary_cross_entropy(y_true, np.array([p]))
        if p >= 0.90:
            behavior = "Confident & Correct -> Near 0 Loss"
        elif p >= 0.50:
            behavior = "Uncertain but right -> Moderate Loss"
        elif p >= 0.10:
            behavior = "Wrong direction -> High Loss"
        else:
            behavior = "CONFIDENTLY WRONG -> Massive Loss Penalty!"
        print(f"{p:<32.4f} | {loss:<20.4f} | {behavior}")

    print("\nTakeaway:")
    print("  Cross-Entropy heavily penalizes models that are confidently wrong.")
    print("  This property makes it ideal for training neural networks and language models,")
    print("  because it drives gradients forcefully away from confident mistakes.")


def demonstrate_categorical_cross_entropy_for_llms():
    """
    Connects Cross-Entropy to how Language Models learn vocabulary next-token prediction.
    """
    print("\n" + "=" * 75)
    print("PART 3: CONNECTION TO LLMs (NEXT-TOKEN CATEGORICAL CROSS-ENTROPY)")
    print("=" * 75)

    # Suppose prompt is: "The capital of France is"
    # Target next token in vocabulary is "Paris" (Index 2 in a 4-token mini-vocab: ["Rome", "Berlin", "Paris", "London"])
    vocab = ["Rome", "Berlin", "Paris", "London"]
    y_one_hot = np.array([[0, 0, 1, 0]])  # Target is "Paris" (index 2)

    # Scenario 1: Model assigns 85% probability to "Paris"
    prob_good = np.array([[0.05, 0.05, 0.85, 0.05]])
    loss_good = categorical_cross_entropy(y_one_hot, prob_good)

    # Scenario 2: Model assigns only 2% probability to "Paris" and 90% to "London"
    prob_bad = np.array([[0.04, 0.04, 0.02, 0.90]])
    loss_bad = categorical_cross_entropy(y_one_hot, prob_bad)

    print(f"Target word: 'Paris' (One-hot vector: {y_one_hot[0].tolist()})")
    print(f"Good LLM output:  P('Paris') = 0.85 -> Categorical Cross-Entropy Loss = {loss_good:.4f}")
    print(f"Poor LLM output:  P('Paris') = 0.02 -> Categorical Cross-Entropy Loss = {loss_bad:.4f}")
    print("Every modern LLM (GPT, Llama, Claude) minimizes this exact loss during pre-training!")


if __name__ == "__main__":
    demonstrate_regression_losses()
    demonstrate_classification_loss()
    demonstrate_categorical_cross_entropy_for_llms()
    print("\n" + "=" * 75)
    print("LESSON 3 COMPLETED SUCCESSFULLY!")
    print("=" * 75)
