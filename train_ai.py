"""
=============================================================================
STAGE TWO: MY FIRST MACHINE-LEARNING MODEL (train_ai.py)
=============================================================================
This script builds, trains, evaluates, saves, reloads, and validates a
classical linear regression model using scikit-learn and joblib.

ILLUSTRATIVE DATASET DISCLAIMER:
The dataset generated and used here is strictly synthetic and illustrative.
It simulates the relationship between project complexity, team experience,
and the total delivery time (in hours) of a software task. It is created
for educational purposes to demonstrate how supervised machine learning works,
and should never be used for real-world managerial, staffing, or academic decisions.
=============================================================================
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "project_estimator.joblib"


def generate_illustrative_data(n_samples: int = 300, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generates synthetic illustrative data for supervised regression.

    Features:
      1. complexity_points: Algorithmic & functional complexity score (10.0 to 100.0).
      2. team_experience_years: Average years of relevant team experience (1.0 to 12.0).

    Target:
      delivery_hours: Time required to complete the task in hours.
      Ground truth relationship (with random Gaussian noise):
      delivery_hours = 15.0 + (1.6 * complexity_points) - (2.4 * team_experience_years) + noise
    """
    rng = np.random.default_rng(seed=random_state)

    complexity = rng.uniform(10.0, 100.0, size=n_samples)
    experience = rng.uniform(1.0, 12.0, size=n_samples)

    # Underlying formula with realistic Gaussian measurement noise (std = 4.0)
    noise = rng.normal(loc=0.0, scale=4.0, size=n_samples)
    hours = 15.0 + (1.6 * complexity) - (2.4 * experience) + noise

    # Ensure hours are non-negative
    hours = np.clip(hours, a_min=1.0, a_max=None)

    features_df = pd.DataFrame({
        "complexity_points": np.round(complexity, 2),
        "team_experience_years": np.round(experience, 2)
    })
    target_series = pd.Series(np.round(hours, 2), name="delivery_hours")

    return features_df, target_series


def validate_input(input_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Validates user input before inference.
    Checks:
      - Required feature keys exist.
      - Types are numeric (int or float).
      - Values are within sensible non-negative boundaries.

    Returns:
      A pandas DataFrame matching the feature schema fitted during training.
    """
    required_features = ["complexity_points", "team_experience_years"]

    for feature in required_features:
        if feature not in input_data:
            raise ValueError(f"Missing required input feature: '{feature}'")

        val = input_data[feature]
        if not isinstance(val, (int, float)) or np.isnan(val):
            raise TypeError(f"Feature '{feature}' must be a valid number, got {type(val).__name__} ({val})")

    complexity = float(input_data["complexity_points"])
    experience = float(input_data["team_experience_years"])

    if complexity < 1.0 or complexity > 500.0:
        raise ValueError(f"complexity_points ({complexity}) must be between 1.0 and 500.0.")

    if experience < 0.0 or experience > 50.0:
        raise ValueError(f"team_experience_years ({experience}) must be between 0.0 and 50.0.")

    return pd.DataFrame([[complexity, experience]], columns=required_features)


def train_and_evaluate_model() -> Dict[str, Any]:
    """
    Executes the full training, evaluation, and serialization workflow.
    """
    print("\n" + "=" * 70)
    print("STEP 1: GENERATING ILLUSTRATIVE TRAINING DATA")
    print("=" * 70)

    X, y = generate_illustrative_data(n_samples=300, random_state=42)
    print(f"Total samples: {len(X)}")
    print("Sample feature values (first 5 rows):")
    print(X.head())
    print("\nCorresponding target values (delivery_hours):")
    print(y.head().to_dict())

    print("\n" + "=" * 70)
    print("STEP 2: SPLITTING INTO TRAINING AND TEST DATASETS")
    print("=" * 70)
    # 80% used for model parameter learning; 20% held out to test generalization
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Training set size: {len(X_train)} samples (80%)")
    print(f"Test set size:     {len(X_test)} samples (20%)")
    print("Why this split matters: The test set acts as unseen real-world data.")
    print("Evaluating on test data guarantees the model is not merely memorizing.")

    print("\n" + "=" * 70)
    print("STEP 3: TRAINING THE MODEL WITH model.fit()")
    print("=" * 70)
    # LinearRegression finds optimal weights (coefficients) that minimize the sum of squared residuals
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("Training finished successfully.")
    print(f"Learned Intercept (bias, b):           {model.intercept_:.4f}")
    print(f"Learned Weight for complexity_points:  {model.coef_[0]:.4f} (ground truth: ~1.6)")
    print(f"Learned Weight for experience_years:   {model.coef_[1]:.4f} (ground truth: ~-2.4)")

    print("\n" + "=" * 70)
    print("STEP 4: EVALUATING MODEL ON HELD-OUT TEST DATA")
    print("=" * 70)
    y_test_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_test_pred)

    print(f"Mean Absolute Error (MAE):     {mae:.3f} hours (average error per prediction)")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f} hours")
    print(f"R-squared Score (R^2):         {r2:.4f} (1.0 = perfect explanation of variance)")

    print("\n" + "=" * 70)
    print("STEP 5: SAVING THE TRAINED MODEL WITH joblib")
    print("=" * 70)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Bundle the model with metadata for safety
    artifact = {
        "model": model,
        "features": list(X.columns),
        "target": "delivery_hours",
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
        "description": "Illustrative linear regression model for software delivery estimation"
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Model and metadata saved to: {MODEL_PATH}")
    print(f"File size on disk: {MODEL_PATH.stat().st_size} bytes")

    return artifact


def run_inference_demo():
    """
    Loads the saved model and runs new predictions with input validation.
    """
    print("\n" + "=" * 70)
    print("STEP 6: RELOADING MODEL AND PERFORMING NEW INFERENCE")
    print("=" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    loaded_artifact = joblib.load(MODEL_PATH)
    model: LinearRegression = loaded_artifact["model"]
    print(f"Successfully reloaded model: {type(model).__name__}")
    print(f"Expected features: {loaded_artifact['features']}")

    # Case 1: Moderate task, senior engineer
    sample_input_1 = {"complexity_points": 45.0, "team_experience_years": 8.0}
    features_vec_1 = validate_input(sample_input_1)
    prediction_1 = model.predict(features_vec_1)[0]
    print(f"\nInference Case 1:")
    print(f"  Inputs: {sample_input_1}")
    print(f"  Predicted Delivery Time: {prediction_1:.2f} hours")

    # Case 2: Highly complex task, junior engineer
    sample_input_2 = {"complexity_points": 90.0, "team_experience_years": 1.5}
    features_vec_2 = validate_input(sample_input_2)
    prediction_2 = model.predict(features_vec_2)[0]
    print(f"\nInference Case 2:")
    print(f"  Inputs: {sample_input_2}")
    print(f"  Predicted Delivery Time: {prediction_2:.2f} hours")

    # Case 3: Test input validation with invalid inputs
    print(f"\nInference Case 3 (Input Validation Test):")
    invalid_input = {"complexity_points": -20.0, "team_experience_years": 3.0}
    print(f"  Attempting inference with invalid input: {invalid_input}")
    try:
        validate_input(invalid_input)
    except (ValueError, TypeError) as err:
        print(f"  Validation correctly rejected input: {err}")


if __name__ == "__main__":
    print("=" * 70)
    print("PERSONAL AI PROJECT - STAGE TWO: MACHINE LEARNING PIPELINE")
    print("=" * 70)
    train_and_evaluate_model()
    run_inference_demo()
    print("\n" + "=" * 70)
    print("STAGE TWO EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
