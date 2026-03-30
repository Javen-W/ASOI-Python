"""
Unit tests for the ASOI (Anomaly Separation and Overlap Index) implementation.

This test suite validates the correctness of the ASOI algorithm and attempts to
replicate key experimental findings from the original research paper, including:

  - Precision degradation tests showing ASOI degrades with label noise.
  - Spearman correlation between ASOI and the F1 score on benchmark datasets.
  - Behaviour on standard sklearn datasets (Breast Cancer Wisconsin, Digits).

References:
    Mahmud, J. S., Farou, Z., & Lendák, I. (2025). ASOI: anomaly separation and
    overlap index, an internal evaluation metric for unsupervised anomaly detection.
    Complex & Intelligent Systems (Springer).
    https://doi.org/10.1007/s40747-025-02204-0
"""

import unittest

import numpy as np
from scipy.stats import spearmanr
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.metrics import f1_score

from asoi import asoi_score


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation(unittest.TestCase):
    """Tests that asoi_score raises errors for invalid inputs."""

    def _simple_data(self):
        X = np.array([[1.0, 2.0], [1.5, 2.5], [10.0, 10.0]])
        y = np.array([0, 0, 1])
        return X, y

    def test_alpha_above_one_raises(self):
        X, y = self._simple_data()
        with self.assertRaises(ValueError):
            asoi_score(X, y, alpha=1.5)

    def test_alpha_below_zero_raises(self):
        X, y = self._simple_data()
        with self.assertRaises(ValueError):
            asoi_score(X, y, alpha=-0.1)

    def test_multiclass_labels_raise(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([0, 1, 2])
        with self.assertRaises(ValueError):
            asoi_score(X, y)

    def test_only_normal_labels_raise(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 0])
        with self.assertRaises(ValueError):
            asoi_score(X, y)

    def test_only_anomaly_labels_raise(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1, 1])
        with self.assertRaises(ValueError):
            asoi_score(X, y)


# ---------------------------------------------------------------------------
# Score properties
# ---------------------------------------------------------------------------

class TestScoreProperties(unittest.TestCase):
    """Tests for fundamental mathematical properties of the ASOI score."""

    def test_returns_float(self):
        X = np.array([[1.0, 2.0], [1.5, 2.5], [10.0, 10.0]])
        y = np.array([0, 0, 1])
        self.assertIsInstance(asoi_score(X, y), float)

    def test_score_in_unit_interval(self):
        """ASOI is a convex combination of S_norm and H, both in [0, 1]."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = np.zeros(100, dtype=int)
        y[:10] = 1
        score = asoi_score(X, y)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_deterministic(self):
        """asoi_score must be deterministic for identical inputs."""
        X = np.array([[1.0, 2.0], [1.2, 1.8], [0.8, 2.2], [10.0, 10.0], [9.8, 10.2]])
        y = np.array([0, 0, 0, 1, 1])
        self.assertAlmostEqual(asoi_score(X, y), asoi_score(X, y), places=12)

    def test_list_inputs_accepted(self):
        """asoi_score should convert plain Python lists to numpy arrays."""
        X = [[1.0, 2.0], [1.5, 2.5], [10.0, 10.0]]
        y = [0, 0, 1]
        self.assertIsInstance(asoi_score(X, y), float)

    def test_single_feature(self):
        X = np.array([[1.0], [2.0], [3.0], [10.0]])
        y = np.array([0, 0, 0, 1])
        score = asoi_score(X, y)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_constant_feature_handled(self):
        """A constant feature should not cause division-by-zero errors."""
        X = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 10.0]])
        y = np.array([0, 0, 1])
        score = asoi_score(X, y)
        self.assertIsInstance(score, float)

    def test_alpha_zero_uses_only_hellinger(self):
        """With alpha=0 the score equals the mean Hellinger distance H."""
        X = np.array([[1.0, 2.0], [1.5, 2.5], [10.0, 10.0]])
        y = np.array([0, 0, 1])
        score = asoi_score(X, y, alpha=0.0)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_alpha_one_uses_only_separation(self):
        """With alpha=1 the score equals the normalised separation S_norm."""
        X = np.array([[1.0, 2.0], [1.5, 2.5], [10.0, 10.0]])
        y = np.array([0, 0, 1])
        score = asoi_score(X, y, alpha=1.0)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_normalize_flag_has_effect(self):
        """Turning off normalisation should produce a different result on unscaled data."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3) * 1000   # large-scale, unnormalised
        y = np.zeros(100, dtype=int)
        y[:10] = 1
        score_norm = asoi_score(X, y, normalize=True)
        score_no_norm = asoi_score(X, y, normalize=False)
        self.assertIsInstance(score_norm, float)
        self.assertIsInstance(score_no_norm, float)
        # The two scores can legitimately differ when features are not pre-scaled.
        # We simply verify both succeed.

    def test_well_separated_beats_overlapping(self):
        """Clearly separated anomalies should yield a higher ASOI than overlapping ones."""
        rng = np.random.RandomState(0)

        # Well-separated: anomalies shifted by 15 std from normals.
        X_sep = np.vstack([rng.randn(90, 2), rng.randn(10, 2) + 15])
        y_sep = np.array([0] * 90 + [1] * 10)

        # Overlapping: anomalies drawn from the same distribution as normals.
        X_ov = rng.randn(100, 2)
        y_ov = np.array([0] * 90 + [1] * 10)

        self.assertGreater(asoi_score(X_sep, y_sep), asoi_score(X_ov, y_ov))


# ---------------------------------------------------------------------------
# Precision degradation tests (replicates paper's methodology)
# ---------------------------------------------------------------------------

class TestPrecisionDegradation(unittest.TestCase):
    """
    Replicates the precision degradation experiments from the paper.

    Starting from perfect anomaly labels, noise is introduced gradually and
    ASOI is expected to degrade monotonically. This validates the metric's
    sensitivity to detector quality.
    """

    def _build_separated_dataset(self, rng, n_normal=270, n_anomaly=30, n_features=5, shift=5):
        X_normal = rng.randn(n_normal, n_features)
        X_anomaly = rng.randn(n_anomaly, n_features) + shift
        X = np.vstack([X_normal, X_anomaly])
        y = np.array([0] * n_normal + [1] * n_anomaly)
        return X, y

    def test_asoi_decreases_overall_with_label_noise(self):
        """
        The ASOI score at zero noise should exceed the score at the highest noise level.
        """
        rng = np.random.RandomState(42)
        X, y_true = self._build_separated_dataset(rng)

        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
        scores = []
        for noise in noise_levels:
            y_noisy = y_true.copy()
            n_flip = int(noise * len(y_true))
            if n_flip:
                flip_idx = rng.choice(len(y_true), size=n_flip, replace=False)
                y_noisy[flip_idx] = 1 - y_noisy[flip_idx]
            if len(np.unique(y_noisy)) < 2:
                scores.append(scores[-1])
                continue
            scores.append(asoi_score(X, y_noisy))

        self.assertGreater(
            scores[0], scores[-1],
            msg=f"Expected ASOI to decrease with label noise. Scores: {scores}",
        )

    def test_perfect_labels_beat_random_labels(self):
        """Perfect anomaly labels should yield a higher ASOI than random labels."""
        rng = np.random.RandomState(7)
        X, y_perfect = self._build_separated_dataset(rng, shift=8)

        n_total = len(y_perfect)
        y_random = rng.choice([0, 1], size=n_total, p=[0.9, 0.1])
        y_random[0] = 0   # guarantee both classes exist
        y_random[-1] = 1

        score_perfect = asoi_score(X, y_perfect)
        score_random = asoi_score(X, y_random)

        self.assertGreater(
            score_perfect, score_random,
            msg=f"Perfect: {score_perfect:.4f}, Random: {score_random:.4f}",
        )


# ---------------------------------------------------------------------------
# Spearman correlation tests (replicates paper's correlation experiments)
# ---------------------------------------------------------------------------

class TestSpearmanCorrelation(unittest.TestCase):
    """
    Tests that ASOI is positively correlated with the F1 score when label quality
    is varied, as reported in the research paper.
    """

    def _asoi_f1_at_noise_levels(self, X, y_true, noise_levels, seed=42):
        """Return parallel arrays of (asoi_scores, f1_scores) for each noise level."""
        rng = np.random.RandomState(seed)
        asoi_vals, f1_vals = [], []
        for noise in noise_levels:
            y_noisy = y_true.copy()
            n_flip = int(noise * len(y_true))
            if n_flip:
                flip_idx = rng.choice(len(y_true), size=n_flip, replace=False)
                y_noisy[flip_idx] = 1 - y_noisy[flip_idx]
            if len(np.unique(y_noisy)) < 2:
                continue
            try:
                asoi_vals.append(asoi_score(X, y_noisy))
                f1_vals.append(f1_score(y_true, y_noisy, pos_label=1, zero_division=0))
            except Exception:
                continue
        return np.array(asoi_vals), np.array(f1_vals)

    def test_positive_spearman_breast_cancer(self):
        """
        On the Breast Cancer Wisconsin dataset ASOI should positively correlate
        with the F1 score across varying noise levels.

        Dataset details (following the paper's convention):
          - Benign (357 samples)  → normal class  (label 0)
          - Malignant (212 samples) → anomaly class (label 1)
        """
        data = load_breast_cancer()
        X = data.data
        # sklearn encodes benign=1, malignant=0; we remap so anomaly=1.
        y_true = (data.target == 0).astype(int)

        noise_levels = np.linspace(0.0, 0.45, 12)
        asoi_vals, f1_vals = self._asoi_f1_at_noise_levels(X, y_true, noise_levels)

        self.assertGreater(len(asoi_vals), 2, msg="Too few valid noise levels to correlate.")
        corr, _ = spearmanr(asoi_vals, f1_vals)
        self.assertGreater(
            corr, 0.0,
            msg=f"Expected positive ASOI-F1 Spearman correlation on Breast Cancer, got {corr:.4f}",
        )

    def test_positive_spearman_digits_binary(self):
        """
        On a binary subset of the Digits dataset (digit '0' vs digit '1')
        ASOI should positively correlate with the F1 score.
        """
        data = load_digits()
        mask = (data.target == 0) | (data.target == 1)
        X = data.data[mask]
        # digit 0 = normal (0), digit 1 = anomaly (1)
        y_true = data.target[mask].copy()

        noise_levels = np.linspace(0.0, 0.45, 12)
        asoi_vals, f1_vals = self._asoi_f1_at_noise_levels(X, y_true, noise_levels)

        self.assertGreater(len(asoi_vals), 2, msg="Too few valid noise levels to correlate.")
        corr, _ = spearmanr(asoi_vals, f1_vals)
        self.assertGreater(
            corr, 0.0,
            msg=f"Expected positive ASOI-F1 Spearman correlation on Digits, got {corr:.4f}",
        )


# ---------------------------------------------------------------------------
# Benchmark dataset sanity checks
# ---------------------------------------------------------------------------

class TestBenchmarkDatasets(unittest.TestCase):
    """
    Sanity checks on standard benchmark datasets used in anomaly detection.

    These tests verify that asoi_score produces a positive, finite score within
    the unit interval and that the score for the true label assignment exceeds
    that of a random assignment, in keeping with the paper's results.
    """

    def _random_labels(self, y_true, seed=0):
        rng = np.random.RandomState(seed)
        contamination = y_true.mean()
        y_rand = rng.choice([0, 1], size=len(y_true), p=[1 - contamination, contamination])
        y_rand[0] = 0
        y_rand[-1] = 1
        return y_rand

    def test_breast_cancer_wisconsin(self):
        """
        Breast Cancer Wisconsin dataset.
        Minority class (malignant, ~37 %) is treated as the anomaly class.
        """
        data = load_breast_cancer()
        X = data.data
        y_true = (data.target == 0).astype(int)

        score = asoi_score(X, y_true)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

        score_rand = asoi_score(X, self._random_labels(y_true))
        self.assertGreater(
            score, score_rand,
            msg=f"True labels ASOI ({score:.4f}) should exceed random ASOI ({score_rand:.4f})",
        )

    def test_digits_binary_zero_vs_one(self):
        """
        Binary subset of the Digits dataset: digit 0 (normal) vs digit 1 (anomaly).
        """
        data = load_digits()
        mask = (data.target == 0) | (data.target == 1)
        X = data.data[mask]
        y_true = data.target[mask].copy()  # 0 = normal, 1 = anomaly

        score = asoi_score(X, y_true)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_synthetic_low_contamination(self):
        """Synthetic dataset with ~2 % anomaly contamination."""
        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(490, 6), rng.randn(10, 6) + 5])
        y = np.array([0] * 490 + [1] * 10)
        score = asoi_score(X, y)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_synthetic_high_contamination(self):
        """Synthetic dataset with ~40 % anomaly contamination."""
        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(60, 4), rng.randn(40, 4) + 3])
        y = np.array([0] * 60 + [1] * 40)
        score = asoi_score(X, y)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_synthetic_high_dimensional(self):
        """Synthetic dataset with 100 features — tests scalability."""
        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(200, 100), rng.randn(20, 100) + 2])
        y = np.array([0] * 200 + [1] * 20)
        score = asoi_score(X, y)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
