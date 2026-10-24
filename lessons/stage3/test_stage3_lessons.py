"""
=============================================================================
STAGE THREE - AUTOMATED LESSON TESTS
File: lessons/stage3/test_stage3_lessons.py
=============================================================================
Automated test suite verifying that all Stage Three lessons, loss formulas,
optimizers, and models execute cleanly and pass mathematical invariants.
=============================================================================
"""

import unittest
from pathlib import Path
import numpy as np

# Import functions from all 6 Stage 3 lessons
from lessons.stage3.lesson1_supervised_vs_unsupervised import (
    demonstrate_supervised_learning,
    demonstrate_unsupervised_learning
)
from lessons.stage3.lesson2_underfitting_overfitting import (
    run_bias_variance_experiment
)
from lessons.stage3.lesson3_loss_functions import (
    mean_squared_error_manual,
    mean_absolute_error_manual,
    binary_cross_entropy,
    categorical_cross_entropy
)
from lessons.stage3.lesson4_gradient_descent import (
    objective_function,
    objective_gradient,
    run_1d_gradient_descent,
    train_linear_regression_from_scratch
)
from lessons.stage3.lesson5_regression_vs_classification import (
    run_regression_task,
    run_classification_task
)
from lessons.stage3.lesson6_data_quality_generalization import (
    demonstrate_outlier_corruption,
    demonstrate_data_leakage,
    demonstrate_distribution_shift
)


class TestStage3Lessons(unittest.TestCase):

    def test_lesson1_supervised_and_unsupervised(self):
        """Verify supervised classifier trains with >80% accuracy and KMeans discovers 3 centroids."""
        clf, acc = demonstrate_supervised_learning()
        self.assertGreater(acc, 0.80, "Supervised classifier accuracy is too low")

        kmeans, centroids = demonstrate_unsupervised_learning()
        self.assertEqual(len(centroids), 3, "K-Means should discover exactly 3 cluster centers")

    def test_lesson2_bias_variance_experiment(self):
        """Verify that Degree 1 has high test error and Degree 4 achieves balanced low test error."""
        results = run_bias_variance_experiment()
        # Degree 1 (Underfitting) has higher test error than Degree 4 (Balanced)
        self.assertGreater(results[1]["test_mse"], results[4]["test_mse"])
        # Generated image must exist on disk
        img_path = Path(__file__).resolve().parent / "underfitting_overfitting.png"
        self.assertTrue(img_path.exists(), "underfitting_overfitting.png was not generated")
        self.assertGreater(img_path.stat().st_size, 1000)

    def test_lesson3_loss_function_mathematics(self):
        """Verify manual loss formulas against analytical expectations."""
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([12.0, 16.0])

        # Residuals: [-2, +4]. Absolute: [2, 4] -> MAE = 3.0. Squared: [4, 16] -> MSE = 10.0
        self.assertAlmostEqual(mean_absolute_error_manual(y_true, y_pred), 3.0)
        self.assertAlmostEqual(mean_squared_error_manual(y_true, y_pred), 10.0)

        # Cross entropy of perfect prediction should be near zero
        bce_near_zero = binary_cross_entropy(np.array([1.0]), np.array([0.9999]))
        self.assertLess(bce_near_zero, 0.01)

        # Cross entropy of wrong prediction should be high
        bce_high = binary_cross_entropy(np.array([1.0]), np.array([0.01]))
        self.assertGreater(bce_high, 4.0)

        # Categorical cross entropy
        one_hot = np.array([[0, 1, 0]])
        probs = np.array([[0.05, 0.90, 0.05]])
        cce = categorical_cross_entropy(one_hot, probs)
        self.assertLess(cce, 0.20)

    def test_lesson4_gradient_descent_convergence(self):
        """Verify 1D gradient descent converges within tolerance of the global minimum (w=2.0)."""
        w_hist, loss_hist = run_1d_gradient_descent(initial_w=8.0, learning_rate=0.15, n_iterations=20)
        final_w = w_hist[-1]
        self.assertAlmostEqual(final_w, 2.0, places=2)
        self.assertAlmostEqual(loss_hist[-1], 3.0, places=2)

        # Multivariate linear regression from scratch
        learned_w, learned_b = train_linear_regression_from_scratch()
        # Ground truth: w ~ 3.5, b ~ 10.0
        self.assertAlmostEqual(learned_w, 3.5, delta=0.5)
        self.assertAlmostEqual(learned_b, 10.0, delta=1.5)

        # Generated plot must exist
        plot_path = Path(__file__).resolve().parent / "gradient_descent_convergence.png"
        self.assertTrue(plot_path.exists())

    def test_lesson5_regression_vs_classification(self):
        """Verify metric dictionaries return sensible values for both tasks."""
        reg_metrics = run_regression_task()
        self.assertIn("mae", reg_metrics)
        self.assertGreater(reg_metrics["r2"], 0.85)

        clf_metrics = run_classification_task()
        self.assertIn("accuracy", clf_metrics)
        self.assertGreater(clf_metrics["f1"], 0.80)

    def test_lesson6_data_quality(self):
        """Verify data quality demonstration functions execute cleanly without exceptions."""
        demonstrate_outlier_corruption()
        demonstrate_data_leakage()
        demonstrate_distribution_shift()


if __name__ == "__main__":
    unittest.main(verbosity=2)
