"""Data loading, cleaning, encoding, scaling, and splitting pipeline."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def load_data(path: Path) -> pd.DataFrame:
    path = Path(path)
    logger.info(f"[preprocessing] Loading data from {path}")
    df = pd.read_parquet(path)
    logger.info(f"[preprocessing] Loaded shape: {df.shape}")
    return df


def clean_data(
    df: pd.DataFrame,
    drop_cols: Optional[List[str]] = None,
    label_col: str = "Label_binary",
) -> pd.DataFrame:
    if drop_cols is None:
        from config import CFG
        drop_cols = CFG.DROP_COLS

    df = df.copy()
    df.columns = df.columns.str.strip()

    before = len(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    logger.info(
        f"[preprocessing] Dropped {before - len(df)} rows with NaN/inf "
        f"({len(df)} remaining)"
    )

    cols_to_drop = [c for c in drop_cols if c in df.columns and c != label_col]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        logger.info(f"[preprocessing] Dropped columns: {cols_to_drop}")

    return df


def encode_labels_binary(
    df: pd.DataFrame,
    label_col: str = "Label",
) -> pd.DataFrame:
    df = df.copy()
    df[label_col] = df[label_col].apply(
        lambda x: 0 if str(x).strip().upper() == "BENIGN" else 1
    )
    counts = df[label_col].value_counts()
    logger.info(f"[preprocessing] Label distribution:\n{counts.to_string()}")
    return df


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    val_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, stratify=y_trainval, random_state=random_state,
    )
    logger.info(
        f"[preprocessing] Split sizes — "
        f"train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    logger.info("[preprocessing] Min-Max scaling applied (fit on train only)")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def preprocess_pipeline(
    parquet_path: Path,
    label_col: str = "Label_binary",
    drop_cols: Optional[List[str]] = None,
    test_size: Optional[float] = None,
    val_size: Optional[float] = None,
    random_state: Optional[int] = None,
) -> Dict:
    """End-to-end preprocessing.

    Default label_col is 'Label_binary' to match the dhoogla CICDDoS slice.
    If your dataset uses string labels in 'Label', pass label_col='Label'
    and the encoder will convert them.
    """
    from config import CFG

    test_size = test_size if test_size is not None else CFG.TEST_SIZE
    val_size = val_size if val_size is not None else CFG.VAL_SIZE
    random_state = random_state if random_state is not None else CFG.RANDOM_SEED

    df = load_data(parquet_path)
    df = clean_data(df, drop_cols=drop_cols, label_col=label_col)

    if label_col == "Label_binary":
        # Already 0/1; skip re-encoding
        if df[label_col].dtype != np.int64:
            df[label_col] = df[label_col].astype(np.int64)
        logger.info(
            f"[preprocessing] Using existing binary labels — "
            f"distribution:\n{df[label_col].value_counts().to_string()}"
        )
    else:
        df = encode_labels_binary(df, label_col=label_col)

    # Drop both label columns from features
    feature_names = [c for c in df.columns if c not in ("Label", "Label_binary")]
    X = df[feature_names].values.astype(np.float32)
    y = df[label_col].values.astype(np.int64)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=test_size, val_size=val_size, random_state=random_state
    )
    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test)

    logger.info(
        f"[preprocessing] Pipeline complete. "
        f"Features: {len(feature_names)}, "
        f"Train/Val/Test: {len(y_train)}/{len(y_val)}/{len(y_test)}"
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": feature_names,
    }
