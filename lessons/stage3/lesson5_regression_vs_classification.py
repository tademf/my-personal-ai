"""
=============================================================================
STAGE THREE - LESSON 5: REGRESSION VERSUS CLASSIFICATION
File: lessons/stage3/lesson5_regression_vs_classification.py
=============================================================================
This script provides two parallel, end-to-end demonstrations:
1. Regression Task: Predicting continuous server response time (milliseconds).
   Evaluated with: MAE, MSE, RMSE, R^2.
2. Classification Task: Predicting discrete server status (Healthy vs. Overloaded).
   Evaluated with: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

It explains why regression and classification metrics are NOT interchangeable.
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def run_regression_task():
    """
    Continuous Prediction: Predict API latency (ms) based on:
    - active_connections: Number of concurrent users
    - cpu_utilization_pct: CPU usage (0-100%)
    """
    print("=" * 75)
    print("TASK 1: REGRESSION (CONTINUOUS VALUE PREDICTION)")
    print("=" * 75)

    rng = np.random.default_rng(seed=42)
    n = 200

    connections = rng.integers(10, 1000, size=n)
    cpu_pct = rng.uniform(5.0, 95.0, size=n)
    # Ground truth formula with noise
    latency_ms = 10.0 + (0.05 * connections) + (0.8 * cpu_pct) + rng.normal(0, 5.0, size=n)

    X = pd.DataFrame({"connections": connections, "cpu_pct": cpu_pct})
    y = pd.Series(latency_ms, name="latency_ms")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Appropriate Regression Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("Regression Evaluation Metrics:")
    print(f"  Target Range:           {y.min():.1f} ms to {y.max():.1f} ms")
    print(f"  MAE (Mean Absolute):    {mae:.2f} ms (average linear error distance)")
    print(f"  RMSE (Root Mean Sq):    {rmse:.2f} ms (quadratic penalty for large errors)")
    print(f"  R^2 (Variance Ratio):   {r2:.4f} (proportion of variance explained; 1.0 is perfect)")

    # Sample predictions
    sample = pd.DataFrame([{"connections": 500, "cpu_pct": 75.0}])
    pred = model.predict(sample)[0]
    print(f"\nSample Prediction for 500 connections & 75% CPU: {pred:.1f} ms\n")

    return {"mae": mae, "rmse": rmse, "r2": r2}


def run_classification_task():
    """
    Discrete Prediction: Classify whether a server is 'FAILING/ALERT' (1) or 'HEALTHY' (0)
    based on error rate and memory usage.
    """
    print("=" * 75)
    print("TASK 2: CLASSIFICATION (DISCRETE CATEGORY PREDICTION)")
    print("=" * 75)

    rng = np.random.default_rng(seed=123)
    n = 200

    error_rate = rng.uniform(0.0, 15.0, size=n)   # Percent error responses
    memory_pct = rng.uniform(20.0, 99.0, size=n)  # Memory utilization

    # Ground truth: Alert triggered if error_rate > 5% or memory > 90%
    is_failing = ((error_rate > 5.0) | (memory_pct > 90.0)).astype(int)

    X = pd.DataFrame({"error_rate": error_rate, "memory_pct": memory_pct})
    y = pd.Series(is_failing, name="is_failing")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Appropriate Classification Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Classification Evaluation Metrics:")
    print(f"  Accuracy:  {acc * 100:.1f}% (total correct predictions / total test instances)")
    print(f"  Precision: {prec * 100:.1f}% (when model says 'Alert', how often is it actually failing?)")
    print(f"  Recall:    {rec * 100:.1f}% (of all truly failing servers, how many did we catch?)")
    print(f"  F1-Score:  {f1:.3f} (harmonic mean of Precision & Recall; robust against class imbalance)")
    print("\nConfusion Matrix:")
    print(f"  [True Negative={cm[0,0]}  False Positive={cm[0,1]}]")
    print(f"  [False Negative={cm[1,0]} True Positive={cm[1,1]}]")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def explain_why_metrics_cannot_be_swapped():
    """Explains why regression and classification metrics cannot be interchanged."""
    print("\n" + "=" * 75)
    print("WHY CAN'T WE SWAP REGRESSION AND CLASSIFICATION METRICS?")
    print("=" * 75)
    print("""
1. R-squared ($R^2$) and MAE make no sense on class labels:
   If classes are [Cat=0, Dog=1, Bird=2], predicting 1.5 has no mathematical meaning.
   A "dog" is not halfway between a cat and a bird. Classes are discrete identities,
   not continuous numerical distances.

2. Accuracy is dangerous on continuous numbers:
   If house price is $450,000.00 and the model predicts $450,000.01, exact equality
   accuracy gives 0% (False), even though the prediction is practically flawless.

3. Accuracy is deceptive on imbalanced datasets:
   If 99% of emails are legitimate and 1% are spam, a lazy model that ALWAYS predicts
   'legitimate' achieves 99% accuracy while failing 100% of its real mission.
   Precision, Recall, and F1-Score expose this failure instantly.
    """)


if __name__ == "__main__":
    run_regression_task()
    run_classification_task()
    explain_why_metrics_cannot_be_swapped()
    print("=" * 75)
    print("LESSON 5 COMPLETED SUCCESSFULLY!")
    print("=" * 75)
