"""
spectral_cluster.py
-------------------
Utility function for clustering mixed datasets using Spectral Clustering with Gower distance.

Spectral Clustering works on a similarity (affinity) matrix computed from Gower distances.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import gower

def spectral_cluster(
        X: pd.DataFrame,
        n_clusters: int = 2,
        random_state: int = 42
) -> tuple[np.ndarray, SpectralClustering, float]:
    """
    Run Spectral Clustering on mixed-type data using Gower distance.

    Parameters
    ----------
    X : pd.DataFrame
        Input dataset containing numeric and categorical columns.

    n_clusters : int, default=2
        Number of clusters to find.

    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    clusters : np.ndarray of shape (n_samples,)
        Cluster labels assigned by Spectral Clustering.

    model : sklearn.cluster.SpectralClustering
        Fitted Spectral Clustering model.

    sil_score : float
        Silhouette score computed on numeric columns only.
        Returns np.nan if only one cluster is detected.

    Notes
    -----
    - Uses Gower distance internally to handle mixed numeric and categorical features.
    - Silhouette score is computed on numeric columns only.
    """
    if X.shape[1] == 0:
        raise ValueError("Input X must have at least one column.")

    # Compute Gower distance matrix
    gower_dist = gower.gower_matrix(X)
    np.fill_diagonal(gower_dist, 0)  # required for Spectral Clustering

    # Spectral Clustering with precomputed affinity
    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=random_state,
        assign_labels='kmeans'
    )

    # Use numeric + categorical via Gower
    gower_dist = gower.gower_matrix(X)
    np.fill_diagonal(gower_dist, 0)
    clusters = model.fit_predict(1 - gower_dist)

    # Silhouette using Gower distance
    try:
        sil_score = silhouette_score(gower_dist, clusters, metric='precomputed')
    except Exception:
        sil_score = np.nan

    return clusters, model, sil_score
