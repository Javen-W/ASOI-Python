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

## Requirements

- Python 3.8+
- [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/)

Install dependencies with:

```bash
pip install numpy scikit-learn
```

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
