import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def calculate_asoi(X: np.ndarray, y: np.ndarray, alpha: float = 0.5, normalize: bool = True) -> float:
    """
    Calculates the full Anomaly Separation and Overlap Index (ASOI).
    https://link.springer.com/article/10.1007/s40747-025-02204-0

    :param X: (array-like) Input data (should be standardized/scaled).
    :param y: (array-like) Binary labels (0 for normal, 1 for anomaly).
    :param alpha: (float) Weight for the separation component. Weight for the overlap component (Beta) is automatically derived from alpha.
    :param normalize: Normalizes the input in the range of [0, 1].

    :return asoi: (float) The ASOI metric value (higher means better separation and less overlap).
    """
    # Calculate beta and validate weights.
    if 0.0 < alpha or alpha > 1.0:
        raise ValueError(f"Alpha must be in the range of [0, 1].")
    beta = 1.0 - alpha

    # Convert to numpy arrays.
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    # Validate that we have exactly two clusters.
    unique_labels = np.unique(y)
    if not np.array_equal(np.sort(unique_labels), [0, 1]):
        raise ValueError(
            f"ASOI calculation requires labels to be strictly 0 (normal) and 1 (anomaly). Got: {unique_labels}")

    # Normalize features.
    if normalize:
        X = StandardScaler().fit_transform(X)

    # Split the sets.
    N_set = X[y == 0]
    A_set = X[y == 1]

    n_N, n_A = len(N_set), len(A_set)
    n_features = X.shape[1]

    # --- PART 1: Separation Component (S_norm) ---
    # Normal centroid.
    c_N = np.mean(N_set, axis=0)

    # Mean distance of anomalies to normal centroid (S).
    distances = np.linalg.norm(A_set - c_N, axis=1)
    S = np.mean(distances)

    # Normalization constant (d_max).
    # Norm of the difference between max vector of anomalies and min vector of normals.
    d_max = np.linalg.norm(np.max(A_set, axis=0) - np.min(N_set, axis=0))
    d_min = 0.0

    # Normalize Separation (S_norm).
    S_norm = (S - d_min) / (d_max - d_min) if d_max > 0 else 0.0

    # --- PART 2: Overlap Component (H) ---
    H_features = []

    # Determine the number of bins (Rice Rule).
    n_total = n_N + n_A
    omega = int(np.ceil(2 * (n_total ** (1 / 3))))

    for j in range(n_features):
        # Get feature-specific data.
        feat_A = A_set[:, j]
        feat_N = N_set[:, j]

        # Get global min/max directly from the full set.
        f_min, f_max = X[:, j].min(), X[:, j].max()

        # Handle constant features.
        if f_min == f_max:
            H_features.append(0.0)
            continue

        bins = np.linspace(f_min, f_max, omega + 1)

        # Calculate probabilities P_A and P_N for each bin.
        prob_A, _ = np.histogram(feat_A, bins=bins)
        prob_N, _ = np.histogram(feat_N, bins=bins)

        # Normalize counts to probabilities.
        P_Al = prob_A / n_A
        P_Nl = prob_N / n_N

        # Hellinger Distance for feature j.
        hellinger_j = (1.0 / np.sqrt(2)) * np.sqrt(np.sum((np.sqrt(P_Al) - np.sqrt(P_Nl)) ** 2))
        H_features.append(hellinger_j)

    # Aggregate feature-wise overlap.
    H = np.mean(H_features)

    # Composite Metric.
    asoi = float((alpha * S_norm) + (beta * H))

    return asoi