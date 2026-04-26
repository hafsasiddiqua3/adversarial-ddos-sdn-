"""Data loading, cleaning, encoding, scaling, and splitting pipeline.

All heavy lifting for raw parquet → model-ready numpy arrays lives here.
The public entry point is `preprocess_pipeline`.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Loading                                                                     #
# --------------------------------------------------------------------------- #

def load_data(path: Path) -> pd.DataFrame:
    """Load a parquet file into a DataFrame.

    Args:
        path: Absolute path to the .parquet file.

    Returns:
        Raw DataFrame.
    """
    path = Path(path)
    logger.info(f"[preprocessing] Loading data from {path}")
    df = pd.read_parquet(path)
    logger.info(f"[preprocessing] Loaded shape: {df.shape}")
    return df


# --------------------------------------------------------------------------- #
#  Cleaning                                                                    #
# --------------------------------------------------------------------------- #

def clean_data(
    df: pd.DataFrame,
    drop_cols: Optional[List[str]] = None,
    label_col: str = "Label",
) -> pd.DataFrame:
    """Clean raw DataFrame.

    Steps:
        1. Strip whitespace from column names.
        2. Replace ±inf with NaN.
        3. Drop rows with any NaN.
        4. Drop non-feature columns (IPs, ports, timestamps, Flow ID).

    Args:
        df: Raw DataFrame.
        drop_cols: Columns to remove before modelling. Defaults to the list
            in config if None is passed.
        label_col: Name of the target column — kept even if listed in drop_cols.

    Returns:
        Cleaned DataFrame with label column intact.
    """
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
    df.drop(columns=cols_to_drop, inplace=True)
    logger.info(f"[preprocessing] Dropped columns: {cols_to_drop}")

    return df


# --------------------------------------------------------------------------- #
#  Label encoding                                                              #
# --------------------------------------------------------------------------- #

def encode_labels_binary(
    df: pd.DataFrame,
    label_col: str = "Label",
) -> pd.DataFrame:
    """Convert multi-class or string labels to binary (0 = BENIGN, 1 = ATTACK).

    Args:
        df: DataFrame containing a ``label_col`` column.
        label_col: Name of the target column.

    Returns:
        DataFrame with label_col replaced by integer 0/1.
    """
    df = df.copy()
    df[label_col] = df[label_col].apply(
        lambda x: 0 if str(x).strip().upper() == "BENIGN" else 1
    )
    counts = df[label_col].value_counts()
    logger.info(f"[preprocessing] Label distribution:\n{counts.to_string()}")
    return df


# --------------------------------------------------------------------------- #
#  Splitting                                                                   #
# --------------------------------------------------------------------------- #

def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    val_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train / validation / test split.

    The split is performed in two steps:
        1. Hold out ``test_size`` fraction as the test set.
        2. From the remaining data, hold out ``val_size`` fraction as the
           validation set.

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        y: Label vector, shape (n_samples,).
        test_size: Proportion of total data for the test set.
        val_size: Proportion of (train+val) data for the validation set.
        random_state: RNG seed.

    Returns:
        Tuple (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size,
        stratify=y_trainval,
        random_state=random_state,
    )
    logger.info(
        f"[preprocessing] Split sizes — "
        f"train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# --------------------------------------------------------------------------- #
#  Scaling                                                                     #
# --------------------------------------------------------------------------- #

def scale_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Min-Max scale features, fitting only on train to prevent leakage.

    Args:
        X_train: Training features.
        X_val: Validation features.
        X_test: Test features.

    Returns:
        Tuple (X_train_scaled, X_val_scaled, X_test_scaled, fitted_scaler).
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    logger.info("[preprocessing] Min-Max scaling applied (fit on train only)")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# --------------------------------------------------------------------------- #
#  Full pipeline                                                               #
# --------------------------------------------------------------------------- #

def preprocess_pipeline(
    parquet_path: Path,
    label_col: str = "Label",
    drop_cols: Optional[List[str]] = None,
    test_size: Optional[float] = None,
    val_size: Optional[float] = None,
    random_state: Optional[int] = None,
) -> Dict:
    """End-to-end preprocessing: load → clean → encode → split → scale.

    Args:
        parquet_path: Path to the raw parquet dataset.
        label_col: Name of the target column.
        drop_cols: Non-feature columns to remove. Reads from config if None.
        test_size: Test split fraction. Reads from config if None.
        val_size: Validation split fraction. Reads from config if None.
        random_state: RNG seed. Reads from config if None.

    Returns:
        Dictionary with keys:
            - ``X_train``, ``X_val``, ``X_test``: scaled numpy arrays
            - ``y_train``, ``y_val``, ``y_test``: integer label arrays
            - ``scaler``: fitted MinMaxScaler
            - ``feature_names``: list of retained feature column names
    """
    from config import CFG

    test_size = test_size if test_size is not None else CFG.TEST_SIZE
    val_size = val_size if val_size is not None else CFG.VAL_SIZE
    random_state = random_state if random_state is not None else CFG.RANDOM_SEED

    df = load_data(parquet_path)
    df = clean_data(df, drop_cols=drop_cols, label_col=label_col)
    df = encode_labels_binary(df, label_col=label_col)

    feature_names = [c for c in df.columns if c != label_col]
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
