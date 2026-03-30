# ASOI-Python

A Python implementation of the Anomaly Separation and Overlap Index (ASOI) algorithm, an intrinsic evaluation metric for unsupervised anomaly detection.

## Summary

**ASOI** (Anomaly Separation and Overlap Index) is a label-free evaluation metric for unsupervised anomaly detection models. Traditional evaluation metrics (e.g., F1-score, AUC-ROC) require ground truth labels, which are often unavailable in real-world anomaly detection. ASOI overcomes this limitation by measuring two complementary properties of an anomaly detector's output:

1. **Separation** — how far the detected anomalies are from the normal data distribution (higher is better).
2. **Overlap** — how much the anomaly and normal distributions overlap in feature space (lower overlap is better).

These two signals are combined into a single weighted score, enabling model comparison, hyperparameter tuning, and performance assessment **without any labeled data**. ASOI is computationally efficient and practical for real-world applications such as cybersecurity, fraud detection, and predictive maintenance.

## Abstract

> Evaluating unsupervised anomaly detection presents significant challenges due to the absence of ground truth labels and the complex nature of anomaly distributions. In this study, we introduce two novel intrinsic evaluation metrics: the Anomaly Separation Index (ASI) and the Anomaly Separation and Overlap Index (ASOI), designed to overcome the limitations of traditional metrics, which cannot assess model performance without labels. ASI quantifies the degree of separation between detected anomalies and normal distributions, while ASOI incorporates both separation and distributional overlap between them, providing an innovative evaluation approach for anomaly detection models, enabling performance assessment even in the absence of ground truth labels. Extensive experiments through precision degradation tests and unsupervised anomaly detection algorithms were conducted on multiple datasets. The results indicate that the metrics consistently correlate with traditional metrics, such as the F1 score, in various benchmark datasets characterized by complex feature interactions and varying levels of anomaly contamination. ASOI showed a higher correlation with the F1 score compared to ASI and several other classical intrinsic metrics. Furthermore, the findings underscore the utility of ASOI as an internal validation measure for model optimization in unsupervised anomaly tasks. The proposed metrics are computationally efficient, scalable, and adaptable to a variety of anomaly detection scenarios, making them practical for real-world applications across industries such as cybersecurity, fraud detection, and predictive maintenance.

## Research Paper

**Title:** ASOI: anomaly separation and overlap index, an internal evaluation metric for unsupervised anomaly detection

**Journal:** [Complex & Intelligent Systems](https://www.springer.com/journal/40747) (Springer)

**Published:** 2025 — DOI: [10.1007/s40747-025-02204-0](https://link.springer.com/article/10.1007/s40747-025-02204-0)

**PDF:** [s40747-025-02204-0.pdf](https://github.com/user-attachments/files/26257051/s40747-025-02204-0.pdf)

### Authors

| Name | Email | Affiliation |
|---|---|---|
| Jiyan Salim Mahmud | jiyan@inf.elte.hu | Eötvös Loránd University (ELTE), Budapest, Hungary |
| Zakarya Farou | zakaryafarou@inf.elte.hu | Eötvös Loránd University (ELTE), Budapest, Hungary |
| Imre Lendák | lendak@inf.elte.hu | Eötvös Loránd University (ELTE), Budapest, Hungary |

## Installation

Install the package directly from GitHub:

```bash
pip install git+https://github.com/Javen-W/ASOI-Python.git
```

Or clone and install locally:

```bash
git clone https://github.com/Javen-W/ASOI-Python.git
cd ASOI-Python
pip install .
```

## Requirements

- Python 3.8+
- [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [SciPy](https://scipy.org/) *(required only for `tests.py`)*

## Usage

```python
import numpy as np
from asoi import asoi_score

# Example: feature matrix X and binary labels y (0 = normal, 1 = anomaly)
X = np.array([
    [1.0, 2.0],
    [1.5, 2.5],
    [2.0, 3.0],
    [10.0, 10.0],  # anomaly
    [11.0, 11.0],  # anomaly
])
y = np.array([0, 0, 0, 1, 1])

score = asoi_score(X, y)
print(f"ASOI Score: {score:.4f}")  # Higher score → better anomaly separation
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X` | `np.ndarray` | — | Input feature matrix. |
| `y` | `np.ndarray` | — | Binary labels: `0` for normal, `1` for anomaly. |
| `alpha` | `float` | `0.5314` | Weight for the separation component. The overlap weight `beta` is derived as `1 - alpha`. |
| `normalize` | `bool` | `True` | If `True`, standardizes `X` using `StandardScaler` before computing the metric. |

### Return Value

Returns a single `float` — the ASOI score. A **higher score** indicates better separation and less distributional overlap between anomalies and normal samples.

## Implementation Details

This implementation directly follows the algorithm described in the research paper. The metric is computed in two stages:

### Stage 1 — Separation Component (S_norm)

The **normal centroid** `c_N` is computed as the feature-wise mean of all normal samples. The **Separation score `S`** is then the mean Euclidean distance from each anomaly sample to `c_N`. This raw score is normalized into `S_norm ∈ [0, 1]` using:

```
S_norm = (S - d_min) / (d_max - d_min)
```

where `d_max` is the Euclidean norm of `(max(A) - min(N))` across features and `d_min = 0`.

### Stage 2 — Overlap Component (H)

For each feature `j`, the **Hellinger Distance** between the anomaly and normal distributions is calculated using histograms. The number of bins `ω` is chosen via the **Rice Rule**:

```
ω = ⌈2 · n^(1/3)⌉
```

where `n` is the total number of samples. The per-feature Hellinger distances are averaged to produce the aggregate overlap score `H`.

### Composite Score

The final ASOI score is a weighted combination of both components:

```
ASOI = alpha · S_norm + beta · H
```

The default weights `alpha = 0.5314` and `beta = 0.4686` are taken directly from the paper, where they were determined empirically to maximize correlation with supervised metrics.

## Original Algorithm

<img width="537" height="711" alt="asoi_algorithm" src="https://github.com/user-attachments/assets/0d32818d-4bda-49b4-b1e4-cb7ec4ba92c9" />

## Testing

The `tests.py` module contains 24 unit tests organised into five test classes:

| Class | Description |
|---|---|
| `TestInputValidation` | Verifies that invalid `alpha` values and non-binary label arrays raise `ValueError`. |
| `TestScoreProperties` | Validates return type, `[0, 1]` bounds, determinism, edge cases, and the `normalize` flag. |
| `TestPrecisionDegradation` | Replicates the paper's precision degradation experiment. |
| `TestSpearmanCorrelation` | Replicates the paper's ASOI–F1 correlation analysis on benchmark datasets. |
| `TestBenchmarkDatasets` | Score sanity checks on Breast Cancer Wisconsin, Digits, and synthetic datasets. |

### Running the tests

Install the test dependencies (SciPy and pytest) if not already present:

```bash
pip install numpy scikit-learn scipy pytest
```

Then run the full test suite from the repository root:

```bash
python -m pytest tests.py -v
```

## Test Results

All 24 tests pass. The results below demonstrate the validity of this implementation against the experiments reported in the research paper.

### Unit test output

```
tests.py::TestInputValidation::test_alpha_above_one_raises PASSED
tests.py::TestInputValidation::test_alpha_below_zero_raises PASSED
tests.py::TestInputValidation::test_multiclass_labels_raise PASSED
tests.py::TestInputValidation::test_only_anomaly_labels_raise PASSED
tests.py::TestInputValidation::test_only_normal_labels_raise PASSED
tests.py::TestScoreProperties::test_alpha_one_uses_only_separation PASSED
tests.py::TestScoreProperties::test_alpha_zero_uses_only_hellinger PASSED
tests.py::TestScoreProperties::test_constant_feature_handled PASSED
tests.py::TestScoreProperties::test_deterministic PASSED
tests.py::TestScoreProperties::test_list_inputs_accepted PASSED
tests.py::TestScoreProperties::test_normalize_flag_has_effect PASSED
tests.py::TestScoreProperties::test_returns_float PASSED
tests.py::TestScoreProperties::test_score_in_unit_interval PASSED
tests.py::TestScoreProperties::test_single_feature PASSED
tests.py::TestScoreProperties::test_well_separated_beats_overlapping PASSED
tests.py::TestPrecisionDegradation::test_asoi_decreases_overall_with_label_noise PASSED
tests.py::TestPrecisionDegradation::test_perfect_labels_beat_random_labels PASSED
tests.py::TestSpearmanCorrelation::test_positive_spearman_breast_cancer PASSED
tests.py::TestSpearmanCorrelation::test_positive_spearman_digits_binary PASSED
tests.py::TestBenchmarkDatasets::test_breast_cancer_wisconsin PASSED
tests.py::TestBenchmarkDatasets::test_digits_binary_zero_vs_one PASSED
tests.py::TestBenchmarkDatasets::test_synthetic_high_contamination PASSED
tests.py::TestBenchmarkDatasets::test_synthetic_high_dimensional PASSED
tests.py::TestBenchmarkDatasets::test_synthetic_low_contamination PASSED

24 passed in 1.06s
```

### Benchmark dataset scores

| Dataset | Samples | Features | Anomaly % | ASOI (true labels) | ASOI (random labels) |
|---|---|---|---|---|---|
| Breast Cancer Wisconsin | 569 | 30 | 37.3 % | **0.3273** | 0.1347 |
| Digits (digit 0 vs 1) | 360 | 64 | 50.6 % | **0.3148** | — |

The ASOI score for the true label assignment consistently exceeds that of random label assignments, confirming the metric's discriminative power.

### Precision degradation test

The table below shows ASOI and F1 scores on a synthetic dataset (300 samples, 5 features, 10 % contamination) as random label noise is progressively introduced. Both metrics degrade together, confirming that ASOI tracks detector quality faithfully.

| Noise Level | ASOI Score | F1 Score |
|---|---|---|
| 0 % | 0.7430 | 1.0000 |
| 10 % | 0.4153 | 0.6429 |
| 20 % | 0.2796 | 0.4340 |
| 30 % | 0.2282 | 0.3478 |
| 40 % | 0.1725 | 0.2308 |

### Spearman correlation: ASOI vs F1

The Spearman rank correlation between ASOI and F1 score across 12 noise levels confirms the implementation matches the paper's central finding — that ASOI is highly correlated with supervised metrics even without labels.

| Dataset | Spearman ρ | p-value |
|---|---|---|
| Breast Cancer Wisconsin | **0.9930** | < 0.0001 |
| Digits (digit 0 vs digit 1) | **0.9720** | < 0.0001 |
