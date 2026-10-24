"""
=============================================================================
STAGE THREE - LESSON 1: SUPERVISED VS. UNSUPERVISED LEARNING
File: lessons/stage3/lesson1_supervised_vs_unsupervised.py
=============================================================================
Machine learning algorithms fundamentally fall into two primary learning paradigms:
1. Supervised Learning: Learning with a teacher (labeled ground-truth targets).
2. Unsupervised Learning: Learning without a teacher (discovering intrinsic patterns).

This script provides executable, minimal implementations of both paradigms
using real scikit-learn models.
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


def explain_core_concepts():
    """
    Prints foundational explanations of supervised vs. unsupervised learning.
    """
    print("=" * 75)
    print("CORE CONCEPTS: SUPERVISED VS. UNSUPERVISED LEARNING")
    print("=" * 75)
    print("""
1. SUPERVISED LEARNING (Learning from Labeled Examples):
   - You provide inputs (Features, X) AND the correct answers (Labels/Targets, y).
   - Goal: Learn a mathematical mapping function f(X) -> y so the model can
     accurately predict y for new, previously unseen inputs X.
   - Two Main Sub-types:
     * Regression: The target is a continuous numerical value (e.g., house price, temperature).
     * Classification: The target is a discrete category/class (e.g., spam vs. not spam, dog vs. cat).
   - Real-World Examples:
     * Medical diagnosis (X = patient vitals, y = disease present / absent).
     * Code bug prediction (X = code complexity metrics, y = bug probability).

2. UNSUPERVISED LEARNING (Discovering Hidden Structure):
   - You provide inputs (Features, X) ONLY. There are NO labels (no y).
   - Goal: Find underlying patterns, natural groupings, or geometric structure in data.
   - Two Main Sub-types:
     * Clustering: Grouping similar data points together (e.g., customer segmentation).
     * Dimensionality Reduction: Compressing high-dimensional features while preserving variance (e.g., PCA).
   - Real-World Examples:
     * Customer segmentation (grouping users by shopping habits without prior tags).
     * Anomaly / fraud detection (identifying unusual transactions that don't fit any cluster).
    """)


def demonstrate_supervised_learning():
    """
    Demonstrates Supervised Learning: Binary Classification.
    Scenario: Predicting whether a user support request is 'URGENT' (1) or 'NORMAL' (0)
    based on two features: message length and keyword severity score.
    """
    print("-" * 75)
    print("PART 1: SUPERVISED LEARNING DEMONSTRATION (CLASSIFICATION)")
    print("-" * 75)

    rng = np.random.default_rng(seed=42)
    n_samples = 120

    # Features (X):
    # - message_words: Word count (10 to 150 words)
    # - urgency_keyword_score: Frequency of words like 'urgent', 'error', 'broken' (0 to 10)
    message_words = rng.integers(10, 150, size=n_samples)
    urgency_score = rng.integers(0, 11, size=n_samples)

    # Ground-truth labels (y):
    # Rule: If urgency_score >= 6 OR (urgency_score >= 4 and words > 80), mark as urgent (1)
    is_urgent = ((urgency_score >= 6) | ((urgency_score >= 4) & (message_words > 80))).astype(int)

    X = pd.DataFrame({
        "message_words": message_words,
        "urgency_score": urgency_score
    })
    y = pd.Series(is_urgent, name="is_urgent")

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Supervised model: Logistic Regression (maps inputs to class probabilities)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Dataset: {n_samples} customer tickets with features {list(X.columns)}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Supervised Classification Accuracy on Test Set: {accuracy * 100:.1f}%\n")

    # Predict on 2 sample new tickets
    sample_queries = pd.DataFrame([
        {"message_words": 120, "urgency_score": 8},  # Clearly urgent
        {"message_words": 25,  "urgency_score": 1}   # Routine inquiry
    ])
    sample_preds = model.predict(sample_queries)
    sample_probs = model.predict_proba(sample_queries)[:, 1]

    for idx, row in sample_queries.iterrows():
        pred_label = "URGENT" if sample_preds[idx] == 1 else "NORMAL"
        print(f"Ticket {idx + 1}: Words={row['message_words']}, Urgency Score={row['urgency_score']}")
        print(f"  -> Prediction: {pred_label} (Urgency Probability: {sample_probs[idx]:.2f})")

    return model, accuracy


def demonstrate_unsupervised_learning():
    """
    Demonstrates Unsupervised Learning: Clustering with K-Means.
    Scenario: Discovering customer personas based on:
    - annual_login_days (10 to 350)
    - code_snippets_generated (5 to 500)
    NO labels are provided to the algorithm!
    """
    print("\n" + "-" * 75)
    print("PART 2: UNSUPERVISED LEARNING DEMONSTRATION (K-MEANS CLUSTERING)")
    print("-" * 75)

    rng = np.random.default_rng(seed=101)
    # Generate 3 distinct synthetic user cohorts
    # Group A: Occasional users
    g_a = rng.normal(loc=[30, 20], scale=[8, 8], size=(40, 2))
    # Group B: Regular weekly developers
    g_b = rng.normal(loc=[150, 180], scale=[15, 25], size=(40, 2))
    # Group C: Power users
    g_c = rng.normal(loc=[300, 420], scale=[20, 30], size=(40, 2))

    X_data = np.vstack([g_a, g_b, g_c])
    X = pd.DataFrame(X_data, columns=["annual_login_days", "code_snippets_generated"])

    print(f"Unlabeled Dataset: {len(X)} users. Notice there are NO 'y' labels provided.")
    print("The algorithm must inspect feature geometry and group similar users autonomously.")

    # Apply K-Means with k=3
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    X["assigned_cluster"] = cluster_labels
    centroids = kmeans.cluster_centers_

    print("\nDiscovered Cluster Centers (Centroids):")
    for i, centroid in enumerate(centroids):
        count = (cluster_labels == i).sum()
        print(f"  Cluster {i}: Logins/year ~ {centroid[0]:.1f}, Snippets ~ {centroid[1]:.1f} ({count} users)")

    print("\nTakeaway: The algorithm successfully partitioned users into meaningful personas")
    print("without any human ever annotating which group a user belonged to.")
    return kmeans, centroids


if __name__ == "__main__":
    explain_core_concepts()
    demonstrate_supervised_learning()
    demonstrate_unsupervised_learning()
    print("\n" + "=" * 75)
    print("LESSON 1 COMPLETED SUCCESSFULLY!")
    print("=" * 75)
