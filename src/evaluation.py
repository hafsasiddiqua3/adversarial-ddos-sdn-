"""Model evaluation, robustness measurement, and visualization.

Public API:
  compute_metrics        — sklearn metrics dict from arrays
  evaluate_model         — run model over a DataLoader, return metrics
  robustness_curve       — accuracy vs epsilon under attack
  plot_robustness_curve  — multi-line plot saved at 400 DPI
  plot_confusion_matrix  — seaborn heatmap saved at 400 DPI
  comparison_table       — DataFrame comparing baseline vs robust, saved as CSV
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Use a clean, publication-friendly style
plt.style.use("seaborn-v0_8-whitegrid")


# --------------------------------------------------------------------------- #
#  Metrics                                                                     #
# --------------------------------------------------------------------------- #

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute classification metrics from arrays.

    Args:
        y_true: Ground-truth integer labels, shape (n,).
        y_pred: Predicted integer labels, shape (n,).
        y_proba: Predicted probability for the positive class, shape (n,).
            Required for ROC-AUC; if None, roc_auc is omitted.

    Returns:
        Dictionary with keys: ``accuracy``, ``precision``, ``recall``,
        ``f1``, and optionally ``roc_auc``.
    """
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = float("nan")

    return metrics


# --------------------------------------------------------------------------- #
#  Full model evaluation                                                       #
# --------------------------------------------------------------------------- #

def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Run a model over an entire DataLoader and return metrics + predictions.

    Args:
        model: Trained CNN_MLP.
        data_loader: DataLoader yielding (X, y) batches shaped (batch, 1, features).
        device: Torch device string.

    Returns:
        Tuple (metrics_dict, y_true, y_pred).
        y_true and y_pred are integer arrays of shape (n_samples,).
        metrics_dict includes accuracy, precision, recall, f1, roc_auc.
    """
    model.eval()
    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_proba: List[np.ndarray] = []

    with torch.no_grad():
        for X_batch, y_batch in tqdm(data_loader, desc="[eval] Inference", leave=False):
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            all_labels.append(y_batch.numpy())
            all_preds.append(preds)
            all_proba.append(probs[:, 1])   # P(attack)

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_proba = np.concatenate(all_proba)

    metrics = compute_metrics(y_true, y_pred, y_proba)
    logger.info(
        f"[eval] accuracy={metrics['accuracy']:.4f}, "
        f"f1={metrics['f1']:.4f}, "
        f"roc_auc={metrics.get('roc_auc', float('nan')):.4f}"
    )
    return metrics, y_true, y_pred


# --------------------------------------------------------------------------- #
#  Robustness curve                                                            #
# --------------------------------------------------------------------------- #

def robustness_curve(
    model: nn.Module,
    data_loader: DataLoader,
    attack_fn: Callable,
    epsilons: List[float],
    device: str,
    **attack_kwargs,
) -> Dict[float, float]:
    """Measure model accuracy under attack at multiple perturbation budgets.

    Args:
        model: Trained CNN_MLP.
        data_loader: DataLoader for the evaluation set.
        attack_fn: ``fgsm_attack`` or ``pgd_attack`` from src.attacks.
        epsilons: List of ε values to evaluate.
        device: Torch device string.
        **attack_kwargs: Additional kwargs for attack_fn (e.g. alpha, num_steps
            for PGD). ``epsilon`` and ``device`` are injected per iteration.

    Returns:
        Dictionary mapping each epsilon to the adversarial accuracy.
    """
    from src.attacks import evaluate_under_attack

    results: Dict[float, float] = {}
    for eps in epsilons:
        kwargs = dict(attack_kwargs)
        kwargs["epsilon"] = eps
        acc = evaluate_under_attack(model, data_loader, attack_fn, kwargs, device)
        results[eps] = acc
        logger.info(f"[robustness_curve] ε={eps:.3f} → acc={acc:.4f}")

    return results


# --------------------------------------------------------------------------- #
#  Plotting                                                                    #
# --------------------------------------------------------------------------- #

def plot_robustness_curve(
    results_dict: Dict[str, Dict[float, float]],
    save_path: Path,
    dpi: int = 400,
) -> None:
    """Plot accuracy vs epsilon for one or more models.

    Args:
        results_dict: Mapping of model_name → {epsilon: accuracy} dict.
            Example: ``{'Baseline': {0.01: 0.95, ...}, 'Robust': {0.01: 0.93, ...}}``
        save_path: Where to save the figure (e.g. results/robustness.png).
        dpi: Output resolution.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    markers = ["o", "s", "^", "D", "v"]
    for i, (name, curve) in enumerate(results_dict.items()):
        eps_vals = sorted(curve.keys())
        acc_vals = [curve[e] for e in eps_vals]
        ax.plot(
            eps_vals, acc_vals,
            marker=markers[i % len(markers)],
            label=name,
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel("Perturbation budget ε (L∞)", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("Robustness Curve: Accuracy vs Adversarial Perturbation Budget", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[eval] Robustness curve saved → {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    dpi: int = 400,
) -> None:
    """Plot and save a seaborn confusion matrix heatmap.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        save_path: Output file path.
        class_names: Tick labels (default: ['BENIGN', 'ATTACK']).
        title: Plot title.
        dpi: Output resolution.
    """
    if class_names is None:
        class_names = ["BENIGN", "ATTACK"]

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=13)

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[eval] Confusion matrix saved → {save_path}")


# --------------------------------------------------------------------------- #
#  Comparison table                                                            #
# --------------------------------------------------------------------------- #

def comparison_table(
    results_baseline: Dict[str, Dict[str, float]],
    results_robust: Dict[str, Dict[str, float]],
    save_path: Path,
) -> pd.DataFrame:
    """Build a comparison DataFrame and save it as CSV.

    Args:
        results_baseline: Dict mapping condition → metrics_dict for the baseline
            model. Expected keys: ``'clean'``, ``'fgsm'``, ``'pgd'``.
            Example:
                {
                    'clean': {'accuracy': 0.98, 'f1': 0.97, ...},
                    'fgsm':  {'accuracy': 0.42, ...},
                    'pgd':   {'accuracy': 0.31, ...},
                }
        results_robust: Same structure for the adversarially trained model.
        save_path: CSV output path.

    Returns:
        Multi-index DataFrame comparing both models across all conditions.
    """
    rows = []
    for condition in ["clean", "fgsm", "pgd"]:
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            b_val = results_baseline.get(condition, {}).get(metric, float("nan"))
            r_val = results_robust.get(condition, {}).get(metric, float("nan"))
            rows.append({
                "Condition": condition.upper(),
                "Metric": metric,
                "Baseline": round(b_val, 4),
                "Adversarially Trained": round(r_val, 4),
            })

    df = pd.DataFrame(rows).set_index(["Condition", "Metric"])
    df.to_csv(save_path)
    logger.info(f"[eval] Comparison table saved → {save_path}")
    return df
