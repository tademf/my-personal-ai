"""
=============================================================================
STAGE THREE - LESSON 6: DATA QUALITY AND GENERALIZATION
File: lessons/stage3/lesson6_data_quality_generalization.py
=============================================================================
"Garbage in, garbage out" (GIGO) is the golden rule of artificial intelligence.
Even the most sophisticated neural network or LLM will fail completely if its
training data suffers from leakage, corruption, or distribution shift.

This script demonstrates 6 critical real-world data challenges:
1. Impact of extreme outliers on Ordinary Least Squares vs. Huber Regressor.
2. Missing value imputation strategies (Mean vs. Median vs. Dropping).
3. Data Leakage (scaling before split vs. scaling strictly on training split).
4. Distribution Shift (train data distribution != production test data).
5. The 3-way split: Training, Validation, and Test sets.
6. The illusion of synthetic data (why clean synthetic data can be deceptive).
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


def demonstrate_outlier_corruption():
    """
    Demonstrates how a single catastrophic outlier corrupts OLS Linear Regression,
    and how robust loss/estimators (Huber) withstand it.
    """
    print("=" * 75)
    print("1. OUTLIER IMPACT: ORDINARY LEAST SQUARES (OLS) VS. HUBER REGRESSOR")
    print("=" * 75)

    rng = np.random.default_rng(seed=42)
    # 20 clean data points along y = 2.0 * x + 5.0
    x_clean = rng.uniform(1.0, 10.0, size=20)
    y_clean = 2.0 * x_clean + 5.0 + rng.normal(0, 0.3, size=20)

    # Add 1 massive sensor malfunction outlier
    x_corrupted = np.append(x_clean, [9.0])
    y_corrupted = np.append(y_clean, [-150.0])  # Catastrophic negative outlier

    X_mat = x_corrupted[:, np.newaxis]

    # Standard OLS (minimizes squared error)
    ols = LinearRegression()
    ols.fit(X_mat, y_corrupted)

    # Robust Huber Regressor (uses linear penalty for large deviations)
    huber = HuberRegressor()
    huber.fit(X_mat, y_corrupted)

    print(f"True slope should be: ~2.0000")
    print(f"OLS learned slope:     {ols.coef_[0]:.4f} (severely corrupted by single outlier)")
    print(f"Huber learned slope:   {huber.coef_[0]:.4f} (robust; ignores outlier weight)")


def demonstrate_data_leakage():
    """
    Demonstrates Data Leakage: Calculating normalization statistics (mean/std)
    across the entire dataset BEFORE train/test split.
    """
    print("\n" + "=" * 75)
    print("2. DATA LEAKAGE: WHY YOU MUST NEVER SCALE BEFORE SPLITTING")
    print("=" * 75)

    rng = np.random.default_rng(seed=42)
    raw_data = rng.normal(loc=50.0, scale=10.0, size=(100, 1))

    # Flawed Workflow (LEAKAGE):
    scaler_leaked = StandardScaler()
    data_scaled_leaked = scaler_leaked.fit_transform(raw_data)
    train_leak, test_leak = train_test_split(data_scaled_leaked, test_size=0.2, random_state=42)

    # Correct Workflow (NO LEAKAGE):
    train_raw, test_raw = train_test_split(raw_data, test_size=0.2, random_state=42)
    scaler_clean = StandardScaler()
    train_clean = scaler_clean.fit_transform(train_raw)       # fit ONLY on train
    test_clean = scaler_clean.transform(test_raw)             # transform test using train parameters

    print("Flawed (Leakage): Scaler computed mean/variance using BOTH train AND test data.")
    print("  -> Information from the future test set leaked into training normalization!")
    print("Correct (Clean):  Scaler computed mean/variance exclusively on training set.")
    print(f"  -> Clean Train Mean: {scaler_clean.mean_[0]:.3f}, Clean Train Std: {np.sqrt(scaler_clean.var_[0]):.3f}")


def demonstrate_distribution_shift():
    """
    Demonstrates Covariate Shift / Concept Drift:
    Model is trained on normal operating temperatures (15C to 25C),
    but deployed into extreme winter weather (-10C to 5C).
    """
    print("\n" + "=" * 75)
    print("3. DISTRIBUTION SHIFT (TRAINING VS. PRODUCTION MISMATCH)")
    print("=" * 75)

    rng = np.random.default_rng(seed=42)
    # Training environment: Moderate temperatures
    temp_train = rng.uniform(15.0, 25.0, size=100)
    battery_life_train = 100.0 - (1.2 * temp_train) + rng.normal(0, 1.0, size=100)

    model = LinearRegression()
    model.fit(temp_train[:, np.newaxis], battery_life_train)

    # In-distribution test (similar temperatures)
    temp_test_indist = rng.uniform(15.0, 25.0, size=40)
    battery_test_indist = 100.0 - (1.2 * temp_test_indist) + rng.normal(0, 1.0, size=40)
    pred_indist = model.predict(temp_test_indist[:, np.newaxis])
    mae_indist = mean_absolute_error(battery_test_indist, pred_indist)

    # Out-of-distribution production (freezing weather causes chemical battery drop)
    temp_test_shift = rng.uniform(-10.0, 5.0, size=40)
    # In reality, freezing cold causes severe non-linear chemical degradation:
    battery_test_shift = 100.0 - (1.2 * temp_test_shift) - (0.5 * (10.0 - temp_test_shift) ** 1.5)
    pred_shift = model.predict(temp_test_shift[:, np.newaxis])
    mae_shift = mean_absolute_error(battery_test_shift, pred_shift)

    print(f"In-Distribution Test MAE:  {mae_indist:.2f} hours (Model looks great in test lab)")
    print(f"Distribution Shift MAE:    {mae_shift:.2f} hours (Model fails in real freezing deployment)")
    print("Lesson: High offline test accuracy fails when real-world input distributions drift.")


def explain_the_three_way_split():
    """
    Explains the necessity of Train vs. Validation vs. Test splits.
    """
    print("\n" + "=" * 75)
    print("4. THE THREE-WAY SPLIT: TRAIN VS. VALIDATION VS. TEST")
    print("=" * 75)
    print("""
Why two splits (Train / Test) are NOT enough when tuning hyperparameters:
  - If you test 50 different model hyperparameters (e.g. learning rate, tree depth,
    polynomial degree) against the Test set, you are effectively using the Test set
    to guide your decisions.
  - Over time, your hyperparameter choices OVERFIT to that specific Test set!
  - Solution: The 3-Way Split:
    1. Training Set (60-70%): Used to compute gradients and fit weights.
    2. Validation Set (15-20%): Used to evaluate hyperparameter choices and pick the best architecture.
    3. Test Set (15-20%): Kept in a vault! Evaluated EXACTLY ONCE at the very end to
       estimate true, unbiased real-world performance.
    """)


def explain_synthetic_data_illusions():
    """
    Explains why synthetic data can produce deceptively high scores.
    """
    print("=" * 75)
    print("5. WHY SYNTHETIC DATA GIVES MISLEADINGLY GOOD RESULTS")
    print("=" * 75)
    print("""
Notice that in Stage Two, our illustrative model scored an R^2 of 0.9905!
Why was it so high?
  - Synthetic data is generated by an explicit mathematical formula with simple Gaussian noise.
  - A linear regression model assumes an identical linear formula, so it fits near-perfectly.
  - Real-world human, code, and language data is messy:
    * Unmeasured confounders (e.g., developer fatigue, vague specifications).
    * Sensor failure and missing records.
    * Highly non-linear, unpredictable interactions.
  - In real machine learning, an R^2 of 0.60 to 0.85 is often considered strong.
    Never assume production performance will match clean synthetic experiments!
    """)


if __name__ == "__main__":
    demonstrate_outlier_corruption()
    demonstrate_data_leakage()
    demonstrate_distribution_shift()
    explain_the_three_way_split()
    explain_synthetic_data_illusions()
    print("=" * 75)
    print("LESSON 6 COMPLETED SUCCESSFULLY!")
    print("=" * 75)
