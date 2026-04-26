# Adversarially Robust DDoS Detection in SDN Networks

## Description

This project implements an adversarially robust DDoS detection system for
Software-Defined Networking (SDN) environments. It extends the baseline
CNN-MLP architecture from Mehmood et al. (2025) — which used SHAP-based
feature selection and Bayesian hyperparameter optimization — by evaluating
and hardening the model against adversarial examples.

The baseline paper demonstrated strong clean-data performance but never tested
whether the model could withstand adversarial perturbations. This project fills
that gap by applying Fast Gradient Sign Method (FGSM) and Projected Gradient
Descent (PGD) attacks, then training a robust variant using adversarial training.

## Architecture Overview

```
Input features (top-20 SHAP-selected)
        │
        ▼
  ┌─────────────────────────────────────┐
  │         CNN Branch                  │
  │  Conv1d(1→32, k=3) → ReLU → Pool   │
  │  Conv1d(32→64, k=3) → ReLU → Pool  │
  │  Flatten                            │
  └───────────────┬─────────────────────┘
                  │
                  ▼
  ┌─────────────────────────────────────┐
  │         MLP Branch                  │
  │  Dense(→128) → ReLU → Dropout(0.3) │
  │  Dense(→64)  → ReLU → Dropout(0.3) │
  │  Dense(→2)   (logits)               │
  └─────────────────────────────────────┘
                  │
        Adversarial Training
    L_total = L_clean + w × L_adv
    where adv samples come from FGSM or PGD
```

## Adversarial Robustness Extension

| Component | Detail |
|-----------|--------|
| Attack 1  | FGSM (L∞, ε = 0.05) |
| Attack 2  | PGD (L∞, ε = 0.05, α = 0.01, 7 steps) |
| Defense   | Adversarial training with combined clean + adversarial loss |
| Ablation  | FGSM-only, PGD-only, Combined FGSM+PGD variants |

## How to Run on Kaggle

1. **Create a Kaggle Notebook** at kaggle.com/notebooks

2. **Attach the dataset** (CIC-IDS-2017 or similar SDN traffic dataset in
   Parquet format) via `+ Add Data` → set the path in `config.py` under
   `DATA_PATH`.

3. **Upload or clone this repo** into the notebook:
   ```python
   # In a notebook cell (uncomment if needed):
   # !git clone https://github.com/<your-username>/info_sec_project.git
   # %cd info_sec_project
   ```

4. **Install dependencies** (first notebook cell):
   ```python
   !pip install -r requirements.txt
   ```

5. **Run `notebooks/main.ipynb`** top to bottom. Each section is self-contained
   after the setup cell.

6. Outputs (models, plots, CSVs) are saved to `/kaggle/working/`.

## Results

| Model | Clean Acc | FGSM Acc (ε=0.05) | PGD Acc (ε=0.05) | F1 |
|-------|-----------|-------------------|------------------|----|
| Baseline CNN-MLP | — | — | — | — |
| Adversarially Trained | — | — | — | — |

*(Results will be filled after Kaggle execution)*

## References

- Mehmood et al. (2025). *CNN-MLP with SHAP feature selection and Bayesian
  optimization for DDoS detection in SDN*. [Citation pending]
- Goodfellow et al. (2015). *Explaining and harnessing adversarial examples.*
  ICLR 2015.
- Madry et al. (2018). *Towards deep learning models resistant to adversarial
  attacks.* ICLR 2018.
