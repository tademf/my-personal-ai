"""
=============================================================================
STAGE FOUR - LESSON 7: EVALUATION, SAVING, AND RELOADING WEIGHTS
File: lessons/stage4/lesson7_evaluate_and_save.py
=============================================================================
In PyTorch, the standard and safest method to save a model is by saving its
`state_dict` (an ordered dictionary containing every learnable weight and bias tensor).

This lesson demonstrates:
1. Evaluating the trained model on held-out test data (never seen during training).
2. Saving weights and scaler metadata to `models/code_quality_nn.pt`.
3. Instantiating a clean new model and reloading weights from disk.
4. Verifying mathematical identity between the in-memory and reloaded models.
5. Making real-world inference predictions with input validation.
6. Real-world generalization limitations.
=============================================================================
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path for direct script execution
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from lessons.stage4.lesson6_train_neural_network import (
    CodeQualityClassifier,
    train_neural_network,
    validate_raw_input
)

MODELS_DIR = BASE_DIR / "models"
CHECKPOINT_PATH = MODELS_DIR / "code_quality_nn.pt"


def evaluate_model_on_test_data(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor) -> Dict[str, float]:
    """
    Evaluates model in eval mode under torch.no_grad().
    """
    print("=" * 75)
    print("PART 1: EVALUATING NEURAL NETWORK ON HELD-OUT TEST DATA")
    print("=" * 75)

    model.eval()  # Set to evaluation mode (freezes dropout/batchnorm)
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).numpy()
        y_true = y_test.numpy()

    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    cm = confusion_matrix(y_true, preds)

    print(f"Test Accuracy:  {acc * 100:.1f}%")
    print(f"Test Precision: {prec * 100:.1f}%")
    print(f"Test Recall:    {rec * 100:.1f}%")
    print(f"Test F1-Score:  {f1:.3f}")
    print("\nConfusion Matrix:")
    print(f"  [TN={cm[0,0]}  FP={cm[0,1]}]")
    print(f"  [FN={cm[1,0]}  TP={cm[1,1]}]")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def save_checkpoint(model: nn.Module, scaler: Any, save_path: Path):
    """Saves model state_dict and scaler parameters to a PyTorch .pt file."""
    print("\n" + "=" * 75)
    print("PART 2: SAVING MODEL WEIGHTS TO .pt FILE")
    print("=" * 75)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "features": ["cyclomatic_complexity", "code_churn_lines", "unit_test_coverage_pct"],
        "classes": ["Maintainable", "High Tech-Debt / Refactor"],
        "architecture": "CodeQualityClassifier(3 -> 16 -> 8 -> 2)"
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint successfully saved to: {save_path}")
    print(f"File size on disk: {save_path.stat().st_size} bytes")


def reload_and_verify(checkpoint_path: Path, reference_model: nn.Module, X_test: torch.Tensor):
    """Reloads model weights into a fresh instance and verifies output identity."""
    print("\n" + "=" * 75)
    print("PART 3: RELOADING WEIGHTS INTO A CLEAN MODEL INSTANCE")
    print("=" * 75)

    # 1. Instantiate clean model architecture
    fresh_model = CodeQualityClassifier(in_features=3, hidden_dim1=16, hidden_dim2=8, num_classes=2)

    # 2. Load checkpoint dictionary from disk
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    fresh_model.load_state_dict(checkpoint["model_state_dict"])
    fresh_model.eval()

    print(f"Loaded checkpoint for: {checkpoint['architecture']}")
    print(f"Target classes: {checkpoint['classes']}")

    # 3. Test mathematical equivalence
    with torch.no_grad():
        ref_output = reference_model(X_test)
        reloaded_output = fresh_model(X_test)

    max_diff = torch.max(torch.abs(ref_output - reloaded_output)).item()
    print(f"Max absolute discrepancy between in-memory and reloaded models: {max_diff:.1e}")
    assert max_diff < 1e-6, "Reloaded model outputs do not match reference model!"
    print("Verification: Reloaded weights produce bit-exact identical predictions.")

    return fresh_model, checkpoint


def predict_single_component(model: nn.Module, checkpoint: Dict[str, Any], complexity: float, churn: float, coverage: float):
    """Performs inference on a single raw input with validation and scaling."""
    validate_raw_input(complexity, churn, coverage)

    # Apply saved scaler transformations
    raw_vec = np.array([[complexity, churn, coverage]], dtype=np.float32)
    mean = checkpoint["scaler_mean"]
    scale = checkpoint["scaler_scale"]
    scaled_vec = (raw_vec - mean) / scale

    tensor_input = torch.tensor(scaled_vec, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        logits = model(tensor_input)
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
        pred_class = int(torch.argmax(logits, dim=1).item())

    class_names = checkpoint["classes"]
    print(f"\nInference for: Complexity={complexity}, Churn={churn} lines, Coverage={coverage}%")
    print(f"  -> Prediction: {class_names[pred_class].upper()}")
    print(f"  -> Confidence: Maintainable: {probs[0]*100:.1f}%, Refactor: {probs[1]*100:.1f}%")

    return pred_class, probs


def run_full_evaluation_pipeline():
    """Runs complete training, evaluation, saving, reloading, and inference."""
    model, scaler, metadata = train_neural_network()
    metrics = evaluate_model_on_test_data(model, metadata["X_test"], metadata["y_test"])
    save_checkpoint(model, scaler, CHECKPOINT_PATH)
    fresh_model, checkpoint = reload_and_verify(CHECKPOINT_PATH, model, metadata["X_test"])

    print("\n" + "=" * 75)
    print("PART 4: SAMPLE REAL-WORLD INFERENCE DEMOS")
    print("=" * 75)
    # Case 1: Low complexity, low churn, 90% test coverage -> Should be Maintainable
    predict_single_component(fresh_model, checkpoint, complexity=5.0, churn=20.0, coverage=90.0)

    # Case 2: Catastrophic complexity, huge churn, 5% test coverage -> Needs Refactoring
    predict_single_component(fresh_model, checkpoint, complexity=45.0, churn=450.0, coverage=5.0)

    print("\n" + "=" * 75)
    print("CRITICAL MACHINE LEARNING TAKEAWAY:")
    print("=" * 75)
    print("Notice how high the test accuracy is on our synthetic dataset.")
    print("In real-world software engineering, static analysis is noisy: legacy systems")
    print("have hidden business constraints, tests can be flaky, and code style varies.")
    print("Never confuse synthetic test metrics with real production reliability!")

    return metrics


if __name__ == "__main__":
    run_full_evaluation_pipeline()
    print("\n" + "=" * 75)
    print("LESSON 7 COMPLETED SUCCESSFULLY!")
    print("=" * 75)
