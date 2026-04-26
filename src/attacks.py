"""FGSM and PGD adversarial attacks for tabular (1-D) data.

Both attacks operate under the L-infinity threat model on Min-Max scaled
features, meaning valid feature values lie in [0, 1].

Usage notes:
  - Pass the model in eval() mode — only the input tensor needs gradients.
  - Inputs should be shaped (batch, 1, num_features) to match CNN_MLP.forward().
  - Returned adversarial tensors are detached and still on `device`.
"""

import logging
from typing import Callable, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

_CRITERION = nn.CrossEntropyLoss()


# --------------------------------------------------------------------------- #
#  FGSM                                                                       #
# --------------------------------------------------------------------------- #

def fgsm_attack(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    device: str,
) -> torch.Tensor:
    """Fast Gradient Sign Method (FGSM) — single-step L∞ attack.

    Perturbs each input sample by epsilon in the direction of the gradient of
    the loss with respect to the input:
        X_adv = clip(X + epsilon * sign(∇_X L(X, y)), 0, 1)

    Args:
        model: Trained CNN_MLP (should be in eval() mode before calling).
        X: Input tensor, shape (batch, 1, num_features), values in [0, 1].
        y: True labels, shape (batch,), dtype long.
        epsilon: Maximum L∞ perturbation magnitude.
        device: Torch device string.

    Returns:
        Adversarial input tensor of the same shape as X (detached).
    """
    model.eval()
    X = X.to(device).detach().clone().requires_grad_(True)
    y = y.to(device)

    logits = model(X)
    loss = _CRITERION(logits, y)
    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        X_adv = X + epsilon * X.grad.sign()
        X_adv = torch.clamp(X_adv, 0.0, 1.0)

    return X_adv.detach()


# --------------------------------------------------------------------------- #
#  PGD                                                                        #
# --------------------------------------------------------------------------- #

def pgd_attack(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    alpha: float,
    num_steps: int,
    device: str,
    random_start: bool = True,
) -> torch.Tensor:
    """Projected Gradient Descent (PGD) — iterative L∞ attack (Madry et al., 2018).

    Algorithm:
        1. (Optional) initialise with random perturbation inside the ε-ball.
        2. For each step:
           a. Compute gradient of loss w.r.t. current adversarial input.
           b. Step: X_adv += alpha * sign(grad)
           c. Project back onto ε-ball:  X_adv = clip(X_adv, X - ε, X + ε)
           d. Clip to valid feature range: X_adv = clip(X_adv, 0, 1)

    Args:
        model: Trained CNN_MLP (should be in eval() mode).
        X: Input tensor, shape (batch, 1, num_features), values in [0, 1].
        y: True labels, shape (batch,), dtype long.
        epsilon: Maximum L∞ perturbation magnitude (radius of ε-ball).
        alpha: Step size per iteration.
        num_steps: Number of PGD steps.
        device: Torch device string.
        random_start: If True, initialise from random point in ε-ball.

    Returns:
        Adversarial input tensor of the same shape as X (detached).
    """
    model.eval()
    X_orig = X.to(device).detach()
    y = y.to(device)

    if random_start:
        delta = torch.empty_like(X_orig).uniform_(-epsilon, epsilon)
        X_adv = torch.clamp(X_orig + delta, 0.0, 1.0).detach()
    else:
        X_adv = X_orig.clone().detach()

    for _ in range(num_steps):
        X_adv.requires_grad_(True)

        logits = model(X_adv)
        loss = _CRITERION(logits, y)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            X_adv = X_adv + alpha * X_adv.grad.sign()
            # Project onto L∞ ball around original input
            X_adv = torch.max(torch.min(X_adv, X_orig + epsilon), X_orig - epsilon)
            # Clip to valid feature domain
            X_adv = torch.clamp(X_adv, 0.0, 1.0)

    return X_adv.detach()


# --------------------------------------------------------------------------- #
#  Evaluation under attack                                                     #
# --------------------------------------------------------------------------- #

def evaluate_under_attack(
    model: nn.Module,
    data_loader: DataLoader,
    attack_fn: Callable,
    attack_kwargs: Dict,
    device: str,
) -> float:
    """Evaluate model accuracy when all inputs are replaced with adversarial examples.

    Args:
        model: Trained CNN_MLP.
        data_loader: DataLoader yielding (X, y) batches shaped for CNN_MLP
            (i.e., X has shape (batch, 1, num_features)).
        attack_fn: One of ``fgsm_attack`` or ``pgd_attack``.
        attack_kwargs: Keyword arguments forwarded to attack_fn (excluding
            model, X, y, device which are supplied internally).
        device: Torch device string.

    Returns:
        Adversarial accuracy as a float in [0, 1].
    """
    model.eval()
    correct, total = 0, 0

    pbar = tqdm(data_loader, desc=f"[attack] {attack_fn.__name__}", leave=False)
    for X_batch, y_batch in pbar:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        X_adv = attack_fn(model, X_batch, y_batch, device=device, **attack_kwargs)

        with torch.no_grad():
            logits = model(X_adv)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

        pbar.set_postfix(acc=f"{correct / total:.4f}")

    adv_acc = correct / total
    logger.info(
        f"[attack] {attack_fn.__name__} adversarial accuracy: {adv_acc:.4f}"
    )
    return adv_acc
