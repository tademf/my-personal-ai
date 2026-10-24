"""
=============================================================================
STAGE FOUR - LESSON 6: TRAINING A COMPLETE NEURAL NETWORK
File: lessons/stage4/lesson6_train_neural_network.py
=============================================================================
This script builds and trains a Multi-Layer Perceptron (MLP) neural network
using PyTorch on CPU.

PROBLEM SCENARIO (SYNTHETIC ILLUSTRATIVE DATASET):
Classifying software code components as:
  Class 0: Maintainable (Good quality)
  Class 1: High Tech-Debt (Needs Refactoring)
Features (3 inputs):
  1. cyclomatic_complexity (1.0 to 50.0)
  2. code_churn_lines (10.0 to 500.0 lines modified per month)
  3. unit_test_coverage_pct (0.0% to 100.0%)

DISCLAIMER:
This dataset is synthetic and illustrative. It is designed to teach neural network
training dynamics on personal CPU hardware. It does NOT represent ChatGPT, an LLM,
or a commercial software static-analysis tool.
=============================================================================
"""

from typing import Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class CodeQualityClassifier(nn.Module):
    """
    3-Layer Multi-Layer Perceptron for Binary Classification:
    Input (3 features) -> Hidden 1 (16, ReLU) -> Hidden 2 (8, ReLU) -> Output (2 logits)
    """
    def __init__(self, in_features: int = 3, hidden_dim1: int = 16, hidden_dim2: int = 8, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def generate_synthetic_data(n_samples: int = 600, random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generates synthetic code quality features and target labels."""
    rng = np.random.default_rng(seed=random_seed)

    complexity = rng.uniform(1.0, 50.0, size=n_samples)
    churn = rng.uniform(10.0, 500.0, size=n_samples)
    coverage = rng.uniform(0.0, 100.0, size=n_samples)

    # Ground truth non-linear decision rule:
    # High complexity + high churn - coverage
    score = (0.08 * complexity) + (0.005 * churn) - (0.04 * coverage) + rng.normal(0, 0.3, size=n_samples)
    # Binary label: 1 if score > 0 (high risk), 0 otherwise
    labels = (score > 0).astype(np.int64)

    X = np.column_stack([complexity, churn, coverage])
    return X, labels


def validate_raw_input(complexity: float, churn: float, coverage: float):
    """Validates real-world input values before passing to network."""
    if not (1.0 <= complexity <= 200.0):
        raise ValueError(f"Complexity ({complexity}) must be in range [1.0, 200.0].")
    if not (0.0 <= churn <= 10000.0):
        raise ValueError(f"Churn lines ({churn}) must be in range [0.0, 10000.0].")
    if not (0.0 <= coverage <= 100.0):
        raise ValueError(f"Coverage ({coverage}) must be in range [0.0, 100.0]%.")


def train_neural_network(
    epochs: int = 60,
    learning_rate: float = 0.01,
    random_seed: int = 42
) -> Tuple[CodeQualityClassifier, StandardScaler, Dict[str, Any]]:
    """
    Executes reproducible neural network training on CPU:
    - Data generation and 80/20 train/test split.
    - Zero data-leakage standardization.
    - Mini-batch gradient descent loop using PyTorch Adam optimizer.
    """
    print("=" * 75)
    print("STAGE FOUR - LESSON 6: TRAINING NEURAL NETWORK ON CPU")
    print("=" * 75)

    # 1. Set seeds for complete reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    X_raw, y_raw = generate_synthetic_data(n_samples=600, random_seed=random_seed)
    X_train_raw, X_test_raw, y_train_np, y_test_np = train_test_split(
        X_raw, y_raw, test_size=0.20, random_state=random_seed, stratify=y_raw
    )

    # 2. Prevent Data Leakage: fit scaler ONLY on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # 3. Convert NumPy arrays to PyTorch Tensors
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.int64)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.int64)

    # 4. Initialize Network, Loss, and Optimizer
    model = CodeQualityClassifier(in_features=3, hidden_dim1=16, hidden_dim2=8, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Training Samples: {len(X_train)} | Test Samples: {len(X_test)}")
    print(f"Optimizer: Adam (lr={learning_rate}) | Loss Function: CrossEntropyLoss")
    print("\n--- Training Loop Starting ---")

    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()  # Set model to training mode

        # Step 1: Zero gradients
        optimizer.zero_grad()

        # Step 2: Forward pass
        logits = model(X_train)

        # Step 3: Compute Loss
        loss = criterion(logits, y_train)

        # Step 4: Backpropagation
        loss.backward()

        # Step 5: Optimizer step
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 10 == 0 or epoch == 1:
            # Calculate training accuracy
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                train_acc = (preds == y_train).float().mean().item() * 100.0
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {loss.item():.4f} | Train Accuracy: {train_acc:.1f}%")

    print("\nTraining completed successfully!")
    print(f"Initial Loss (Epoch 1):  {loss_history[0]:.4f}")
    print(f"Final Loss   (Epoch {epochs}): {loss_history[-1]:.4f}")
    assert loss_history[-1] < loss_history[0], "Training failed to reduce loss!"

    metadata = {
        "loss_history": loss_history,
        "X_test": X_test,
        "y_test": y_test,
        "X_test_raw": X_test_raw
    }

    return model, scaler, metadata


if __name__ == "__main__":
    train_neural_network()
    print("\n" + "=" * 75)
    print("LESSON 6 COMPLETED SUCCESSFULLY!")
    print("=" * 75)
