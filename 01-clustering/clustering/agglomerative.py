"""
agglomerative_cluster.py
------------------------
Utility function for clustering mixed datasets using Agglomerative Clustering
with Gower distance.

Agglomerative (hierarchical) clustering works directly on a precomputed
dissimilarity matrix, making it suitable for mixed-type data when combined with Gower.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import gower


def agglomerative_cluster(
        X: pd.DataFrame,
        n_clusters: int = 2,
        linkage: str = 'average'
) -> tuple[np.ndarray, AgglomerativeClustering, float]:
    """
    Run Agglomerative Clustering on mixed-type data using Gower distance.

    Parameters
    ----------
    X : pd.DataFrame
        Input dataset containing numeric and categorical columns.

    n_clusters : int, default=2
        Number of clusters to find.

    linkage : {'average', 'complete', 'single'}, default='average'
        Linkage criterion determining cluster merging strategy.
        - 'average' (UPGMA) works best with Gower distance.
        - 'complete' produces compact clusters.
        - 'single' is sensitive to noise (tends to chain clusters).

    Returns
    -------
    clusters : np.ndarray of shape (n_samples,)
        Cluster labels assigned by Agglomerative Clustering.

    model : sklearn.cluster.AgglomerativeClustering
        Fitted Agglomerative Clustering model.

    sil_score : float
        Silhouette score computed on numeric columns only.
        Returns np.nan if only one cluster is detected.

    Notes
    -----
    - Uses Gower distance internally to handle mixed numeric and categorical features.
    - Silhouette score is computed on numeric columns only (not on Gower matrix).
    - For visualization or dendrograms, consider using scipy.cluster.hierarchy.
    """
    if X.shape[1] == 0:
        raise ValueError("Input X must have at least one column.")

    # Compute Gower distance
    gower_dist = gower.gower_matrix(X)
    np.fill_diagonal(gower_dist, 0)

    # Fit Agglomerative Clustering using precomputed distance
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',  # uses Gower distance
        linkage=linkage
    )

    clusters = model.fit_predict(gower_dist)

    # Compute silhouette on numeric columns only
    num_cols = X.select_dtypes(include='number').columns
    if len(num_cols) > 0 and len(np.unique(clusters)) >= 2:
        try:
            sil_score = silhouette_score(X[num_cols], clusters)
        except Exception:
            sil_score = np.nan
    else:
        sil_score = np.nan

    return clusters, model, sil_score
