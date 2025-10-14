"""
kprototype.py
----------
Utility functions for clustering mixed (numerical + categorical) datasets 
using the K-Prototypes algorithm.

K-Prototypes is designed for mixed-type data, combining K-Means (for numeric)
and K-Modes (for categorical) distance computations.
"""

import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def kprototype(
        X: pd.DataFrame,
        k: int,
        init: str = 'Cao',
        random_state: int = 42
) -> tuple[np.ndarray, KPrototypes, float]:
    """
    Run the K-Prototypes clustering algorithm on a mixed dataset.

    Parameters
    ----------
    X : pd.DataFrame
        Input dataset containing both numeric and categorical columns.
        Missing values should be imputed before calling this function.

    k : int
        Number of clusters.

    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    clusters : np.ndarray of shape (n_samples,)
        Cluster labels assigned to each sample.

    model : kmodes.kprototypes.KPrototypes
        The trained K-Prototypes model (can be reused to predict new data).

    sil_score : float
        Silhouette score (computed on numeric columns only, since categorical
        dissimilarities are not supported by silhouette).

    Notes
    -----
    - The algorithm automatically detects categorical columns (dtype == 'object' or 'category').
    - Scaling numeric features may slightly change cluster balance; test both versions.
    - Higher silhouette scores indicate better-defined numeric separation,
      but you should always inspect categorical cluster purity separately.
    """
    # Separate numeric and categorical columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    num_cols = X.select_dtypes(exclude=["object", "category"]).columns

    X_proc = X.copy()

    # Convert DataFrame to numpy array (required by kmodes library)
    X_matrix = X_proc.to_numpy()

    cat_col_idx = [X_proc.columns.get_loc(col) for col in cat_cols]

    # Train K-Prototypes
    model = KPrototypes(n_clusters=k, init=init, n_init=10, random_state=random_state)
    clusters = model.fit_predict(X_matrix, categorical=cat_col_idx)

    # Compute silhouette on numeric subset (if applicable)
    if len(num_cols) > 0:
        try:
            sil_score = silhouette_score(X_proc[num_cols], clusters)
        except Exception:
            sil_score = np.nan
    else:
        sil_score = np.nan

    return clusters, model, sil_score

