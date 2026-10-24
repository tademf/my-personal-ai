"""
=============================================================================
STAGE FOUR - AUTOMATED TESTS
File: lessons/stage4/test_stage4.py
=============================================================================
Unit tests verifying all Stage Four neural network modules:
- Tensor shape operations
- Autograd dynamic derivatives
- Model forward dimension tracking
- Loss reduction during training
- Checkpoint persistence and reload equivalence
- Input validation rejection
=============================================================================
"""

import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn as nn
import numpy as np

from lessons.stage4.lesson1_tensors import (
    demonstrate_tensor_creation,
    demonstrate_indexing_and_reshaping,
    demonstrate_math_operations
)
from lessons.stage4.lesson2_autograd import (
    demonstrate_single_variable_autograd,
    demonstrate_multivariable_autograd
)
from lessons.stage4.lesson3_neural_network_basics import (
    demonstrate_activation_functions
)
from lessons.stage4.lesson4_forward_pass import (
    SimpleMLP,
    demonstrate_forward_pass
)
from lessons.stage4.lesson5_backpropagation import (
    demonstrate_backprop_step
)
from lessons.stage4.lesson6_train_neural_network import (
    CodeQualityClassifier,
    train_neural_network,
    validate_raw_input
)
from lessons.stage4.lesson7_evaluate_and_save import (
    evaluate_model_on_test_data,
    save_checkpoint,
    reload_and_verify,
    predict_single_component,
    CHECKPOINT_PATH
)


class TestStage4NeuralNetworks(unittest.TestCase):

    def test_01_tensor_operations_and_shapes(self):
        """Verify tensor creation, slicing, reshaping, and matrix multiplication shapes."""
        scalar, vector, matrix, batch_tokens = demonstrate_tensor_creation()
        self.assertEqual(scalar.shape, torch.Size([]))
        self.assertEqual(vector.shape, torch.Size([4]))
        self.assertEqual(matrix.shape, torch.Size([3, 2]))
        self.assertEqual(batch_tokens.shape, torch.Size([2, 3, 4]))

        sliced, reshaped, flattened = demonstrate_indexing_and_reshaping()
        self.assertEqual(sliced.shape, torch.Size([2]))
        self.assertEqual(reshaped.shape, torch.Size([2, 6]))
        self.assertEqual(flattened.shape, torch.Size([12]))

        matmul_res = demonstrate_math_operations()
        self.assertEqual(matmul_res.shape, torch.Size([2, 2]))

    def test_02_autograd_calculates_exact_derivatives(self):
        """Verify autograd produces exact analytical derivatives."""
        grad_single = demonstrate_single_variable_autograd()
        self.assertAlmostEqual(grad_single, 26.0, places=4)

        grad_a, grad_b = demonstrate_multivariable_autograd()
        self.assertAlmostEqual(grad_a, 8.0, places=4)
        self.assertAlmostEqual(grad_b, 81.0, places=4)

    def test_03_activation_and_model_dimensions(self):
        """Verify model outputs match specified batch and class dimensions."""
        model = SimpleMLP(in_features=4, hidden_dim=8, out_features=2)
        dummy_batch = torch.randn(7, 4)
        output = model(dummy_batch)
        self.assertEqual(output.shape, torch.Size([7, 2]))

    def test_04_training_reduces_loss(self):
        """Verify training loop reduces cross-entropy loss and achieves >80% accuracy."""
        model, scaler, metadata = train_neural_network(epochs=40, learning_rate=0.01)
        loss_hist = metadata["loss_history"]
        self.assertLess(loss_hist[-1], loss_hist[0], "Training failed to reduce loss")

        # Evaluate on test set
        metrics = evaluate_model_on_test_data(model, metadata["X_test"], metadata["y_test"])
        self.assertGreater(metrics["accuracy"], 0.80)

    def test_05_checkpoint_save_and_reload(self):
        """Verify checkpoint file exists, weights reload cleanly, and output matches exactly."""
        model, scaler, metadata = train_neural_network(epochs=20)
        save_checkpoint(model, scaler, CHECKPOINT_PATH)
        self.assertTrue(CHECKPOINT_PATH.exists())
        self.assertGreater(CHECKPOINT_PATH.stat().st_size, 0)

        fresh_model, checkpoint = reload_and_verify(CHECKPOINT_PATH, model, metadata["X_test"])
        self.assertIn("model_state_dict", checkpoint)
        self.assertIn("scaler_mean", checkpoint)

        # Verify inference on single component
        pred_class, probs = predict_single_component(
            fresh_model, checkpoint, complexity=10.0, churn=50.0, coverage=80.0
        )
        self.assertIn(pred_class, [0, 1])
        self.assertEqual(len(probs), 2)
        self.assertAlmostEqual(sum(probs), 1.0, places=4)

    def test_06_input_validation(self):
        """Verify input validation rejects out-of-range parameters."""
        with self.assertRaises(ValueError):
            validate_raw_input(complexity=-5.0, churn=100.0, coverage=50.0)

        with self.assertRaises(ValueError):
            validate_raw_input(complexity=10.0, churn=-20.0, coverage=50.0)

        with self.assertRaises(ValueError):
            validate_raw_input(complexity=10.0, churn=100.0, coverage=150.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
