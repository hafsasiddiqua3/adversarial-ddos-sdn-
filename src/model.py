"""Hybrid CNN-MLP model definition and clean training loop.

Architecture (Mehmood et al., 2025):
  Input (batch, 1, num_features)
    → Conv1d(32) → ReLU → MaxPool1d(2)
    → Conv1d(64) → ReLU → MaxPool1d(2)
    → Flatten
    → Dense(128) → ReLU → Dropout
    → Dense(64)  → ReLU → Dropout
    → Dense(num_classes)  [raw logits]

Loss: CrossEntropyLoss (handles both binary and multi-class).
"""

import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Model                                                                       #
# --------------------------------------------------------------------------- #

class CNN_MLP(nn.Module):
    """Hybrid 1-D CNN + MLP classifier for tabular network traffic features.

    The input feature vector is treated as a 1-D "signal" with a single channel,
    allowing convolutional filters to capture local feature interactions before
    the MLP head learns global patterns.

    Args:
        num_features: Number of input features.
        num_classes: Number of output classes (default 2 for binary DDoS detection).
        conv1_filters: Number of filters in the first Conv1d layer.
        conv2_filters: Number of filters in the second Conv1d layer.
        kernel_size: Convolution kernel size.
        dense1: Neurons in the first dense layer.
        dense2: Neurons in the second dense layer.
        dropout: Dropout probability applied after each dense layer.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        conv1_filters: int = 32,
        conv2_filters: int = 64,
        kernel_size: int = 3,
        dense1: int = 128,
        dense2: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, conv1_filters, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(conv1_filters, conv2_filters, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Infer flattened size after two max-pool(2) operations
        cnn_out_len = num_features // 4  # each pool halves the length
        flat_size = conv2_filters * cnn_out_len

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, dense1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense1, dense2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 1, num_features).

        Returns:
            Logit tensor of shape (batch_size, num_classes).
        """
        x = self.cnn(x)
        x = self.mlp(x)
        return x


# --------------------------------------------------------------------------- #
#  Factory                                                                     #
# --------------------------------------------------------------------------- #

def build_model(
    num_features: int,
    num_classes: int = 2,
    config=None,
) -> CNN_MLP:
    """Construct a CNN_MLP with hyperparameters from config.

    Args:
        num_features: Number of input features.
        num_classes: Number of output classes.
        config: Config object or None (reads from CFG singleton).

    Returns:
        Initialized CNN_MLP instance (on CPU; move to device externally).
    """
    if config is None:
        from config import CFG
        config = CFG

    model = CNN_MLP(
        num_features=num_features,
        num_classes=num_classes,
        conv1_filters=config.CONV1_FILTERS,
        conv2_filters=config.CONV2_FILTERS,
        kernel_size=config.KERNEL_SIZE,
        dense1=config.DENSE1,
        dense2=config.DENSE2,
        dropout=config.DROPOUT,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"[model] Built CNN_MLP: {num_features} features → {num_classes} classes "
        f"| trainable params: {n_params:,}"
    )
    return model


# --------------------------------------------------------------------------- #
#  DataLoader helper                                                           #
# --------------------------------------------------------------------------- #

def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Wrap numpy arrays in a TensorDataset + DataLoader.

    Args:
        X: Feature matrix, float32, shape (n, features).
        y: Labels, int64, shape (n,).
        batch_size: Mini-batch size.
        shuffle: Whether to shuffle each epoch.

    Returns:
        PyTorch DataLoader.
    """
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (n, 1, features)
    y_t = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# --------------------------------------------------------------------------- #
#  Training                                                                    #
# --------------------------------------------------------------------------- #

def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_clean(
    model: CNN_MLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    seed: Optional[int] = None,
) -> Tuple[CNN_MLP, Dict]:
    """Standard (clean) training loop.

    Args:
        model: Initialized CNN_MLP.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        epochs: Number of training epochs.
        lr: Adam learning rate.
        device: Torch device string.
        seed: Optional RNG seed for reproducibility.

    Returns:
        Tuple of (trained model, history dict).
        History contains lists: ``train_loss``, ``train_acc``,
        ``val_loss``, ``val_acc`` — one value per epoch.
    """
    if seed is not None:
        _set_seeds(seed)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"[clean] Epoch {epoch}/{epochs}", leave=False)
        for X_batch, y_batch in pbar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total += len(y_batch)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / train_total
        avg_train_acc = train_correct / train_total

        val_loss, val_acc = _evaluate_loss_acc(model, val_loader, criterion, device)

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(avg_train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info(
            f"[clean] Epoch {epoch}/{epochs} | "
            f"train_loss={avg_train_loss:.4f}, train_acc={avg_train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

    return model, history


def _evaluate_loss_acc(
    model: CNN_MLP,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """Compute average loss and accuracy over a DataLoader."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

    return total_loss / total, correct / total


# --------------------------------------------------------------------------- #
#  Inference                                                                   #
# --------------------------------------------------------------------------- #

def predict(
    model: CNN_MLP,
    X: np.ndarray,
    device: str,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference over a numpy array in mini-batches.

    Args:
        model: Trained CNN_MLP.
        X: Feature matrix, float32, shape (n, features).
        device: Torch device string.
        batch_size: Mini-batch size for inference.

    Returns:
        Tuple (probabilities, predicted_classes).
        probabilities: shape (n, num_classes), softmax-normalised.
        predicted_classes: shape (n,), argmax class index.
    """
    model.eval()
    model.to(device)

    all_probs: List[np.ndarray] = []

    loader = make_loader(X, np.zeros(len(X), dtype=np.int64), batch_size, shuffle=False)
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    probabilities = np.vstack(all_probs)
    predicted_classes = probabilities.argmax(axis=1)
    return probabilities, predicted_classes
