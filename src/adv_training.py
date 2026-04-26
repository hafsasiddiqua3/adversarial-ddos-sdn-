"""Adversarial training with combined clean + adversarial loss.

Two training modes are provided:

adversarial_train — standard adversarial training using a single attack (FGSM
    or PGD). Each batch generates adversarial examples and computes:
        L_total = L_clean + adv_loss_weight * L_adv

combined_train — stronger variant that mixes both FGSM and PGD adversarial
    samples in the same batch, forcing the model to be robust against both
    attack families simultaneously.
"""

import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.attacks import fgsm_attack, pgd_attack

logger = logging.getLogger(__name__)

_CRITERION = nn.CrossEntropyLoss()
_ATTACK_MAP = {"fgsm": fgsm_attack, "pgd": pgd_attack}


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _val_accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    """Compute clean validation accuracy."""
    model.eval()
    correct, total = 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        preds = model(X_batch).argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)
    return correct / total


def _adv_val_accuracy(
    model: nn.Module,
    loader: DataLoader,
    attack_fn,
    attack_kwargs: Dict,
    device: str,
) -> float:
    """Compute adversarial validation accuracy."""
    model.eval()
    correct, total = 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        X_adv = attack_fn(model, X_batch, y_batch, device=device, **attack_kwargs)
        with torch.no_grad():
            preds = model(X_adv).argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)
    return correct / total


# --------------------------------------------------------------------------- #
#  Standard adversarial training                                               #
# --------------------------------------------------------------------------- #

def adversarial_train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    attack_type: str,
    epochs: int,
    lr: float,
    attack_kwargs: Dict,
    adv_loss_weight: float,
    device: str,
    seed: Optional[int] = None,
) -> Tuple[nn.Module, Dict]:
    """Adversarial training with a single attack type.

    For each mini-batch:
        1. Forward pass on clean X  → loss_clean
        2. Generate X_adv using the specified attack.
        3. Forward pass on X_adv    → loss_adv
        4. L_total = loss_clean + adv_loss_weight * loss_adv
        5. Backprop and update.

    Validation is performed at the end of each epoch on both clean and
    adversarial examples using the same attack.

    Args:
        model: Initialized CNN_MLP.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        attack_type: ``'fgsm'`` or ``'pgd'``.
        epochs: Number of training epochs.
        lr: Adam learning rate.
        attack_kwargs: Keyword arguments for the chosen attack function
            (e.g. ``{'epsilon': 0.05}`` for FGSM, or
            ``{'epsilon': 0.05, 'alpha': 0.01, 'num_steps': 7}`` for PGD).
        adv_loss_weight: Weight ``w`` on the adversarial loss term.
        device: Torch device string.
        seed: Optional RNG seed.

    Returns:
        Tuple (trained model, history dict).
        History keys: ``train_loss``, ``train_clean_acc``, ``train_adv_acc``,
        ``val_clean_acc``, ``val_adv_acc`` — one value per epoch.
    """
    if seed is not None:
        _set_seeds(seed)

    if attack_type not in _ATTACK_MAP:
        raise ValueError(f"attack_type must be one of {list(_ATTACK_MAP)}, got '{attack_type}'")

    attack_fn = _ATTACK_MAP[attack_type]
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_clean_acc": [],
        "train_adv_acc": [],
        "val_clean_acc": [],
        "val_adv_acc": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        clean_correct, adv_correct, total = 0, 0, 0

        pbar = tqdm(
            train_loader,
            desc=f"[adv/{attack_type}] Epoch {epoch}/{epochs}",
            leave=False,
        )
        for X_batch, y_batch in pbar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Clean forward
            model.train()
            logits_clean = model(X_batch)
            loss_clean = _CRITERION(logits_clean, y_batch)

            # Generate adversarial examples (requires model in eval for stable BN)
            model.eval()
            X_adv = attack_fn(model, X_batch, y_batch, device=device, **attack_kwargs)

            # Adversarial forward
            model.train()
            logits_adv = model(X_adv)
            loss_adv = _CRITERION(logits_adv, y_batch)

            loss_total = loss_clean + adv_loss_weight * loss_adv

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            epoch_loss += loss_total.item() * len(y_batch)
            clean_correct += (logits_clean.argmax(1) == y_batch).sum().item()
            adv_correct += (logits_adv.argmax(1) == y_batch).sum().item()
            total += len(y_batch)

            pbar.set_postfix(
                loss=f"{loss_total.item():.4f}",
                clean_acc=f"{clean_correct / total:.4f}",
                adv_acc=f"{adv_correct / total:.4f}",
            )

        avg_loss = epoch_loss / total
        train_clean_acc = clean_correct / total
        train_adv_acc = adv_correct / total

        val_clean_acc = _val_accuracy(model, val_loader, device)
        val_adv_acc = _adv_val_accuracy(model, val_loader, attack_fn, attack_kwargs, device)

        history["train_loss"].append(avg_loss)
        history["train_clean_acc"].append(train_clean_acc)
        history["train_adv_acc"].append(train_adv_acc)
        history["val_clean_acc"].append(val_clean_acc)
        history["val_adv_acc"].append(val_adv_acc)

        logger.info(
            f"[adv/{attack_type}] Epoch {epoch}/{epochs} | "
            f"loss={avg_loss:.4f} | "
            f"train clean={train_clean_acc:.4f}, adv={train_adv_acc:.4f} | "
            f"val clean={val_clean_acc:.4f}, adv={val_adv_acc:.4f}"
        )

    return model, history


# --------------------------------------------------------------------------- #
#  Combined FGSM + PGD adversarial training                                   #
# --------------------------------------------------------------------------- #

def combined_train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    fgsm_kwargs: Dict,
    pgd_kwargs: Dict,
    adv_loss_weight: float,
    device: str,
    seed: Optional[int] = None,
) -> Tuple[nn.Module, Dict]:
    """Adversarial training mixing FGSM and PGD adversarial samples each batch.

    For each mini-batch, two sets of adversarial examples are generated
    (one via FGSM, one via PGD) and the adversarial loss averages over both:
        L_adv = 0.5 * L_fgsm + 0.5 * L_pgd
        L_total = L_clean + adv_loss_weight * L_adv

    This forces the model to be simultaneously robust against single-step and
    multi-step attacks without requiring twice the training time (the PGD cost
    dominates).

    Args:
        model: Initialized CNN_MLP.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        epochs: Number of training epochs.
        lr: Adam learning rate.
        fgsm_kwargs: Keyword arguments for fgsm_attack (e.g. ``{'epsilon': 0.05}``).
        pgd_kwargs: Keyword arguments for pgd_attack
            (e.g. ``{'epsilon': 0.05, 'alpha': 0.01, 'num_steps': 7}``).
        adv_loss_weight: Weight on the combined adversarial loss.
        device: Torch device string.
        seed: Optional RNG seed.

    Returns:
        Tuple (trained model, history dict).
        History keys: ``train_loss``, ``val_clean_acc``,
        ``val_fgsm_acc``, ``val_pgd_acc``.
    """
    if seed is not None:
        _set_seeds(seed)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_clean_acc": [],
        "val_fgsm_acc": [],
        "val_pgd_acc": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        total = 0

        pbar = tqdm(
            train_loader,
            desc=f"[combined] Epoch {epoch}/{epochs}",
            leave=False,
        )
        for X_batch, y_batch in pbar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            model.train()
            logits_clean = model(X_batch)
            loss_clean = _CRITERION(logits_clean, y_batch)

            model.eval()
            X_fgsm = fgsm_attack(model, X_batch, y_batch, device=device, **fgsm_kwargs)
            X_pgd = pgd_attack(model, X_batch, y_batch, device=device, **pgd_kwargs)

            model.train()
            loss_fgsm = _CRITERION(model(X_fgsm), y_batch)
            loss_pgd = _CRITERION(model(X_pgd), y_batch)
            loss_adv = 0.5 * loss_fgsm + 0.5 * loss_pgd

            loss_total = loss_clean + adv_loss_weight * loss_adv

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            epoch_loss += loss_total.item() * len(y_batch)
            total += len(y_batch)
            pbar.set_postfix(loss=f"{loss_total.item():.4f}")

        avg_loss = epoch_loss / total
        val_clean_acc = _val_accuracy(model, val_loader, device)
        val_fgsm_acc = _adv_val_accuracy(model, val_loader, fgsm_attack, fgsm_kwargs, device)
        val_pgd_acc = _adv_val_accuracy(model, val_loader, pgd_attack, pgd_kwargs, device)

        history["train_loss"].append(avg_loss)
        history["val_clean_acc"].append(val_clean_acc)
        history["val_fgsm_acc"].append(val_fgsm_acc)
        history["val_pgd_acc"].append(val_pgd_acc)

        logger.info(
            f"[combined] Epoch {epoch}/{epochs} | loss={avg_loss:.4f} | "
            f"val clean={val_clean_acc:.4f}, fgsm={val_fgsm_acc:.4f}, pgd={val_pgd_acc:.4f}"
        )

    return model, history
