"""Central configuration for the adversarially robust DDoS detection project.

All hyperparameters and paths are defined here. Import this module in other
files instead of hardcoding values.
"""

import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    # ------------------------------------------------------------------ paths
    DATA_PATH: Path = Path("/kaggle/input/your-dataset/traffic.parquet")
    OUTPUT_DIR: Path = Path("/kaggle/working")
    MODELS_DIR: Path = Path("/kaggle/working/models")
    RESULTS_DIR: Path = Path("/kaggle/working/results")

    # --------------------------------------------------------- reproducibility
    RANDOM_SEED: int = 42

    # --------------------------------------------------------- preprocessing
    TEST_SIZE: float = 0.15
    VAL_SIZE: float = 0.15   # fraction of the remaining train set

    # Non-feature columns to drop before modelling
    DROP_COLS: List[str] = field(default_factory=lambda: [
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Source Port",
        "Destination Port",
        "Timestamp",
    ])

    # ----------------------------------------------------- feature selection
    NUM_FEATURES: int = 20   # top SHAP features to retain

    # ------------------------------------------------------- model architecture
    CONV1_FILTERS: int = 32
    CONV2_FILTERS: int = 64
    KERNEL_SIZE: int = 3
    DENSE1: int = 128
    DENSE2: int = 64
    DROPOUT: float = 0.3
    NUM_CLASSES: int = 2

    # ------------------------------------------------------------ training
    BATCH_SIZE: int = 256
    LEARNING_RATE: float = 1e-3
    EPOCHS: int = 10

    # ------------------------------------------------------------ attacks
    FGSM_EPSILON: float = 0.05

    PGD_EPSILON: float = 0.05
    PGD_ALPHA: float = 0.01
    PGD_STEPS: int = 7

    # -------------------------------------------- adversarial training loss
    # L_total = L_clean + ADV_LOSS_WEIGHT * L_adv
    ADV_LOSS_WEIGHT: float = 1.0

    # ----------------------------------------- robustness evaluation curve
    EPSILONS_TO_TEST: List[float] = field(
        default_factory=lambda: [0.01, 0.025, 0.05, 0.075, 0.1]
    )

    # --------------------------------------------------------------- device
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------- plot aesthetics
    PLOT_DPI: int = 400

    def __post_init__(self):
        """Convert string paths to Path objects and create output dirs."""
        self.DATA_PATH = Path(self.DATA_PATH)
        self.OUTPUT_DIR = Path(self.OUTPUT_DIR)
        self.MODELS_DIR = Path(self.MODELS_DIR)
        self.RESULTS_DIR = Path(self.RESULTS_DIR)

        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Module-level singleton so other files can do `from config import CFG`
CFG = Config()
