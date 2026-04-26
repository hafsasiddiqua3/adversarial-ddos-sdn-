"""SHAP-based feature selection for the CNN-MLP model.

IMPORTANT: SHAP feature selection requires a *trained* model, so the workflow is:
  1. Train the model on all features (or a large subset).
  2. Call compute_shap_values() with the trained model.
  3. Call get_top_features() to identify the top-k features.
  4. Call select_features() to subset X arrays.
  5. Optionally retrain on the reduced feature set.

A correlation/variance-based fallback (select_k_best_variance) is also provided
for cases where SHAP is too slow or unavailable.
"""

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  SHAP-based selection                                                        #
# --------------------------------------------------------------------------- #

def compute_shap_values(
    model: nn.Module,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    device: str,
    max_samples: int = 1000,
) -> np.ndarray:
    """Compute SHAP values using DeepExplainer for the CNN-MLP.

    DeepExplainer is chosen because it can back-propagate through convolutional
    layers without requiring a differentiable surrogate. For models that do not
    support DeepExplainer fall back to KernelExplainer automatically.

    Args:
        model: Trained CNN_MLP instance.
        X_background: Background dataset for SHAP baseline (numpy, float32).
            Should be a representative sample (e.g., 200–500 rows from train).
        X_explain: Samples to explain (numpy, float32).
            Capped internally at ``max_samples``.
        device: Torch device string ('cuda' or 'cpu').
        max_samples: Maximum number of rows from X_explain to process.

    Returns:
        SHAP values array of shape (n_explain_samples, n_features).
        If the model produces 2 output logits, returns values for class 1.
    """
    import shap

    model.eval()
    model.to(device)

    # Cap explain set
    if len(X_explain) > max_samples:
        idx = np.random.choice(len(X_explain), max_samples, replace=False)
        X_explain = X_explain[idx]

    try:
        logger.info("[feature_selection] Attempting SHAP DeepExplainer ...")

        # DeepExplainer needs torch tensors shaped (batch, 1, features)
        bg_tensor = torch.tensor(X_background, dtype=torch.float32).unsqueeze(1).to(device)
        ex_tensor = torch.tensor(X_explain, dtype=torch.float32).unsqueeze(1).to(device)

        explainer = shap.DeepExplainer(model, bg_tensor)
        shap_vals = explainer.shap_values(ex_tensor)

        # shap_vals is a list of arrays [class0, class1] for binary output
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]   # class 1 = attack

        # shap_vals shape: (n_samples, 1, n_features) → squeeze channel dim
        if shap_vals.ndim == 3:
            shap_vals = shap_vals.squeeze(1)

        logger.info(
            f"[feature_selection] DeepExplainer done. "
            f"SHAP shape: {shap_vals.shape}"
        )
        return shap_vals

    except Exception as e:
        logger.warning(
            f"[feature_selection] DeepExplainer failed ({e}). "
            "Falling back to KernelExplainer (slower)."
        )

        # KernelExplainer works on numpy; wrap model in a predict_proba-style fn
        def model_predict(x: np.ndarray) -> np.ndarray:
            t = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(device)
            with torch.no_grad():
                logits = model(t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

        explainer = shap.KernelExplainer(model_predict, X_background[:100])
        shap_vals = explainer.shap_values(X_explain, nsamples=100)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        logger.info(
            f"[feature_selection] KernelExplainer done. "
            f"SHAP shape: {np.array(shap_vals).shape}"
        )
        return np.array(shap_vals)


def get_top_features(
    shap_values: np.ndarray,
    feature_names: List[str],
    k: int,
) -> List[str]:
    """Return the names of the top-k most important features.

    Importance is measured as mean |SHAP value| across all explained samples.

    Args:
        shap_values: Array of shape (n_samples, n_features).
        feature_names: Ordered list of feature names matching axis-1 of shap_values.
        k: Number of top features to return.

    Returns:
        List of feature names sorted by descending importance (length k).
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:k]
    top_features = [feature_names[i] for i in top_indices]
    logger.info(f"[feature_selection] Top-{k} features: {top_features}")
    return top_features


def select_features(
    X: np.ndarray,
    feature_names: List[str],
    top_features: List[str],
) -> np.ndarray:
    """Subset X to retain only the specified features.

    Args:
        X: Feature matrix, shape (n_samples, n_all_features).
        feature_names: Full ordered list of column names for X.
        top_features: Subset of feature_names to keep.

    Returns:
        Subset of X, shape (n_samples, len(top_features)).
    """
    indices = [feature_names.index(f) for f in top_features]
    return X[:, indices]


# --------------------------------------------------------------------------- #
#  Fallback: variance / univariate filter                                      #
# --------------------------------------------------------------------------- #

def select_k_best_variance(
    X_train: np.ndarray,
    feature_names: List[str],
    k: int,
) -> List[str]:
    """Select top-k features by variance computed on the training set.

    This is a model-free fallback when SHAP is unavailable or too slow.
    High-variance features carry more discriminative information under the
    assumption that the feature space is already normalized.

    Args:
        X_train: Training feature matrix (should be Min-Max scaled).
        feature_names: Column names for X_train.
        k: Number of features to select.

    Returns:
        List of top-k feature names by descending variance.
    """
    variances = X_train.var(axis=0)
    top_indices = np.argsort(variances)[::-1][:k]
    top_features = [feature_names[i] for i in top_indices]
    logger.info(
        f"[feature_selection] Variance-based top-{k} features: {top_features}"
    )
    return top_features
