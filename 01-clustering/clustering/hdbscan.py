"""
hdbscan_cluster.py
-------------------
Utility function for clustering numeric datasets using HDBSCAN.

Assumes input data is numeric-only and already scaled.
"""

import numpy as np
import pandas as pd
import hdbscan
from sklearn.metrics import silhouette_score


def hdbscan_cluster(
        X: pd.DataFrame,
        min_cluster_size: int = 5,
        min_samples: int = None,
        cluster_selection_epsilon: float = 0.0
) -> tuple[np.ndarray, hdbscan.HDBSCAN, float]:
    """
    Run HDBSCAN clustering on numeric, scaled data.

    Parameters
    ----------
    X : pd.DataFrame
        Numeric-only input dataset, already scaled.
        Non-numeric columns must be removed beforehand.

    min_cluster_size : int, default=5
        The minimum size of clusters. Smaller clusters are labeled as noise (-1).

    min_samples : int, optional
        Number of samples in a neighborhood for a point to be considered a core point.
        Defaults to `min_cluster_size` if not specified.

    cluster_selection_epsilon : float, default=0.0
        Distance threshold for cluster selection (can merge small clusters).

    Returns
    -------
    clusters : np.ndarray of shape (n_samples,)
        Cluster labels assigned by HDBSCAN. Noise points are labeled -1.

    model : hdbscan.HDBSCAN
        Fitted HDBSCAN model.

    sil_score : float
        Silhouette score computed on numeric features, excluding noise points.
        Returns np.nan if fewer than 2 clusters are detected.

    Notes
    -----
    - Silhouette score excludes noise points (-1 labels).
    - HDBSCAN is robust to clusters of varying density and does not require specifying k.
    """
    if X.shape[1] == 0:
        raise ValueError("Input X must have at least one numeric column.")

    # Fit HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        prediction_data=True
    )
    labels = clusterer.fit_predict(X)

    # Compute silhouette score (exclude noise)
    mask = labels != -1
    if np.unique(labels[mask]).size >= 2:
        try:
            sil_score = silhouette_score(X[mask], labels[mask])
        except Exception:
            sil_score = np.nan
    else:
        sil_score = np.nan

    return labels, clusterer, sil_score
