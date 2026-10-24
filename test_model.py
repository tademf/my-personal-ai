"""
=============================================================================
STAGE TWO: AUTOMATED VERIFICATION TESTS (test_model.py)
=============================================================================
This script verifies that the trained model saved by train_ai.py can be loaded
in a completely clean Python session, passes input validation, and produces
valid predictions with genuine model inference.
=============================================================================
"""

import unittest
from pathlib import Path
import numpy as np
import joblib

from train_ai import validate_input, MODEL_PATH


class TestModelArtifactsAndInference(unittest.TestCase):

    def test_01_model_file_exists(self):
        """Verify the joblib file exists on disk and is non-empty."""
        self.assertTrue(MODEL_PATH.exists(), f"Model file missing at {MODEL_PATH}")
        self.assertGreater(MODEL_PATH.stat().st_size, 0, "Model file is 0 bytes")

    def test_02_model_loads_cleanly(self):
        """Verify model artifact can be unpickled/loaded and has expected keys."""
        artifact = joblib.load(MODEL_PATH)
        self.assertIn("model", artifact)
        self.assertIn("features", artifact)
        self.assertIn("metrics", artifact)
        self.assertEqual(artifact["features"], ["complexity_points", "team_experience_years"])
        self.assertIn("mae", artifact["metrics"])
        self.assertLess(artifact["metrics"]["mae"], 10.0, "MAE is unexpectedly high")

    def test_03_valid_prediction_monotonicity(self):
        """
        Verify genuine model inference:
        - Output is a float.
        - Higher complexity should result in higher predicted hours.
        - Higher experience should reduce predicted hours.
        """
        artifact = joblib.load(MODEL_PATH)
        model = artifact["model"]

        # Base case
        x_base = validate_input({"complexity_points": 50.0, "team_experience_years": 5.0})
        pred_base = float(model.predict(x_base)[0])

        # Higher complexity case
        x_hard = validate_input({"complexity_points": 90.0, "team_experience_years": 5.0})
        pred_hard = float(model.predict(x_hard)[0])

        # Higher experience case
        x_senior = validate_input({"complexity_points": 50.0, "team_experience_years": 10.0})
        pred_senior = float(model.predict(x_senior)[0])

        self.assertGreater(pred_hard, pred_base, "More complexity must require more delivery hours")
        self.assertLess(pred_senior, pred_base, "More experience must reduce delivery hours")

    def test_04_input_validation_rejections(self):
        """Verify input validation catches missing keys, non-numbers, and out-of-bound values."""
        # Missing feature
        with self.assertRaises(ValueError):
            validate_input({"complexity_points": 50.0})

        # Non-numeric type
        with self.assertRaises(TypeError):
            validate_input({"complexity_points": "very_complex", "team_experience_years": 5.0})

        # Negative complexity
        with self.assertRaises(ValueError):
            validate_input({"complexity_points": -10.0, "team_experience_years": 5.0})

        # Negative experience
        with self.assertRaises(ValueError):
            validate_input({"complexity_points": 50.0, "team_experience_years": -2.0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
