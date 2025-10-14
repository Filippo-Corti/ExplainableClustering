"""
gmm.py
---------------
Utility function for clustering numeric datasets using Gaussian Mixture Models (GMM).

Assumes input data is numeric-only and already scaled.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


def gmm_cluster(
        X: pd.DataFrame,
        k: int,
        random_state: int = 42
) -> tuple[np.ndarray, GaussianMixture, float]:
    """
    Run Gaussian Mixture Model (GMM) clustering on numeric, scaled data.

    Parameters
    ----------
    X : pd.DataFrame
        Numeric-only input dataset, already scaled.
        Non-numeric columns must be removed beforehand.

    k : int
        Number of mixture components (clusters).

    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    clusters : np.ndarray of shape (n_samples,)
        Hard cluster assignments based on highest posterior probability.

    model : sklearn.mixture.GaussianMixture
        Fitted GMM model (can be used to predict new data).

    sil_score : float
        Silhouette score computed on the input numeric features.
        Returns np.nan if only one cluster is present or silhouette fails.

    Notes
    -----
    - Input must already be scaled because GMM is sensitive to feature ranges.
    - Silhouette score uses Euclidean distance on numeric features.
    """
    if X.shape[1] == 0:
        raise ValueError("Input X must have at least one numeric column.")

    # Fit GMM
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=random_state)
    gmm.fit(X)
    clusters = gmm.predict(X)

    # Compute silhouette score
    if len(np.unique(clusters)) >= 2:
        try:
            sil_score = silhouette_score(X, clusters)
        except Exception:
            sil_score = np.nan
    else:
        sil_score = np.nan

    return clusters, gmm, sil_score
