# Project Progress — Adversarially Robust DDoS Detection in SDN
**Last Updated:** 2026-04-27  
**Executed on:** Kaggle (GPU accelerator, T4)  
**Maintainer:** Information Security Project Team

---

## What This File Is

This document traces exactly what was done in `notebooks/main.ipynb`, step by step, in plain language. It is intended to help the paper-writing team understand what was run, what was produced, and where to find every result. It is not a technical reference — for that, see `UPDATE_README.md`.

---

## Step 1 — Environment Setup

The notebook was run on Kaggle. The GitHub repository was cloned directly into the Kaggle working directory:

```
git clone https://github.com/hafsasiddiqua3/adversarial-ddos-sdn-.git /kaggle/working/repo
```

Two files were patched at runtime using `%%writefile` to match the Kaggle dataset paths:
- `config.py` — updated all paths to point to `/kaggle/input/...`
- `src/preprocessing.py` — updated to correctly handle the `Label_binary` column in the CICDDoS dataset

Dependencies installed:
```
pip install torchattacks shap
```

---

## Step 2 — Dataset

**Primary dataset:** CICDDoS-2019 (custom balanced slice)  
- File: `cicddos_balanced_slice.parquet`  
- Hosted on Kaggle under `hafsaaas/cci-ddos-custom-dataset`  
- Pre-balanced (equal benign and attack samples)  
- Features: 77 network traffic flow features  
- Labels: Binary — `0 = BENIGN`, `1 = ATTACK`

**Secondary dataset (cross-dataset evaluation):** InSDN  
- Hosted on Kaggle under `hafsaaas/insdn-dataset`  
- Three CSV files combined: `Normal_data.csv`, `OVS.csv`, `metasploitable-2.csv`  
- Used only for cross-dataset generalization testing — not used in training

**Split applied to both datasets:**
| Split | Size |
|-------|------|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

Splitting was stratified to preserve class balance. Min-Max scaling `[0, 1]` was fit on the training set only and applied to val/test.

---

## Step 3 — Baseline Model Training (CICDDoS-2019)

A clean CNN-MLP model was trained on all 77 features with no adversarial component.

**Architecture:** Hybrid 1-D CNN + MLP  
→ Full architecture details in `UPDATE_README.md` → Section: *Model Architecture*  
→ Code: `src/model.py`

**Training config:**
| Parameter | Value |
|-----------|-------|
| Epochs | 10 |
| Batch Size | 256 |
| Learning Rate | 1e-3 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| Device | CUDA (Kaggle T4) |

**Baseline clean test results:**

| Metric | Value |
|--------|-------|
| Accuracy | **99.76%** |
| (Other metrics) | → see `results/baseline_clean_metrics.json` |

---

## Step 4 — Adversarial Attack on the Baseline

The trained baseline was attacked using FGSM and PGD to measure its vulnerability before any defense was applied.

**Attack parameters:**
| Attack | ε | α | Steps |
|--------|---|---|-------|
| FGSM | 0.05 | — | 1 |
| PGD | 0.05 | 0.01 | 7 |

Both attacks use the L∞ threat model. PGD uses a random start within the ε-ball.

**Results — Baseline under attack:**

| Condition | Accuracy | Drop from Clean |
|-----------|----------|----------------|
| Clean | 99.76% | — |
| FGSM (ε=0.05) | **26.87%** | −72.89 pp |
| PGD (ε=0.05) | **20.83%** | −78.93 pp |

> PGD accuracy of 20.83% is below random guessing for a balanced binary problem — the model completely fails under iterative attack.

→ Full attack results: `results/baseline_attack_results.json`  
→ Confusion matrices: `results/cm_baseline_clean.png`, `results/cm_baseline_pgd.png`

---

## Step 5 — Adversarial Training (Main Contribution)

A fresh model was trained from scratch using PGD adversarial training. The baseline model was **not** fine-tuned — a new model was initialized and trained with the adversarial objective from the start.

**Training objective per batch:**
```
L_total = L_clean + λ · L_adv

where:
  L_clean = CrossEntropyLoss(model(X_clean), y)
  L_adv   = CrossEntropyLoss(model(X_pgd), y)
  λ       = 1.0  (adversarial loss weight)
```

→ Code: `src/adv_training.py` → `adversarial_train()` with `attack_type="pgd"`  
→ Model saved to: `models/robust_pgd_cnn_mlp.pth`

**Robust model results:**

| Condition | Accuracy |
|-----------|----------|
| Clean | → see `results/robust_clean_metrics.json` |
| FGSM (ε=0.05) | → see `results/robust_attack_results.json` |
| PGD (ε=0.05) | → see `results/robust_attack_results.json` |

→ Confusion matrices: `results/cm_robust_clean.png`, `results/cm_robust_pgd.png`

---

## Step 6 — Robustness Curves

Both the baseline and robust model were evaluated across multiple perturbation budgets to show how robustness degrades as attack strength increases.

**Epsilons tested:** `[0.01, 0.025, 0.05, 0.075, 0.1]`  
**Attacks used:** FGSM and PGD separately

→ Raw data: `results/robustness_curves.json`  
→ Plots: `results/robustness_curve_fgsm.png`, `results/robustness_curve_pgd.png`

These plots are the key figures for the paper's robustness analysis section.

---

## Step 7 — Ablation Study

Four training strategies were compared to justify the choice of PGD-only adversarial training as the proposed method:

| Strategy | Description |
|----------|-------------|
| `no_adv_training` | Standard clean training (the baseline) |
| `fgsm_only` | Adversarial training with FGSM examples only |
| `combined` | Adversarial training mixing FGSM + PGD per batch (50/50) — uses `combined_train()` in `src/adv_training.py` |
| `pgd_only` | Adversarial training with PGD examples only **(proposed method)** |

The PGD-only result was reused from Step 5 (not retrained). FGSM-only and Combined were trained fresh.

→ Full ablation results: `results/ablation_study.csv`, `results/ablation_study.json`  
→ Bar chart visualization: `results/ablation_bar_chart.png`

---

## Step 8 — SHAP Feature Importance

SHAP values were computed on the robust model to identify which features it relies on most for its decisions. This replicates the interpretability analysis from the Mehmood et al. (2025) baseline paper.

**Method:** SHAP DeepExplainer (falls back to KernelExplainer if DeepExplainer fails)  
**Background samples:** 200 randomly selected training samples  
**Explained samples:** 500 randomly selected test samples

→ Feature importance ranking: `results/shap_importance.json`  
→ Bar chart: `results/shap_feature_importance.png`

This analysis is useful for the paper's discussion section — connecting which network flow features (e.g., packet length, flow duration) are most discriminative for DDoS detection.

---

## Step 9 — Cross-Dataset Evaluation (InSDN)

The entire pipeline (preprocessing → baseline training → attack → adversarial training → evaluation) was re-run on the InSDN dataset to test whether the approach generalizes beyond CICDDoS-2019.

**InSDN files used:**
- `Normal_data.csv` — benign traffic
- `OVS.csv` — attack traffic (Open vSwitch)
- `metasploitable-2.csv` — attack traffic (Metasploitable-2)

The three files were concatenated, balanced, saved as a parquet file (`insdn_balanced_slice.parquet`), and run through the same `preprocess_pipeline()`.

→ Results: `results/cross_dataset_comparison.csv`, `results/cross_dataset_comparison.png`

---

## Step 10 — Final Outputs Packaged

All results were packaged from `/kaggle/working/results/` for download. The following were also written as standalone markdown files during the notebook run:

- `use_case_writeup.md` — three deployment scenarios (ISP backbone, campus network, cloud provider)
- `conclusion.md` — draft conclusion paragraph for the paper

---

## Summary of All Result Files

| File | What It Contains |
|------|-----------------|
| `results/baseline_clean_metrics.json` | Accuracy, F1, ROC-AUC of baseline on clean test set |
| `results/baseline_attack_results.json` | Baseline accuracy under FGSM and PGD |
| `results/robust_clean_metrics.json` | Robust model accuracy on clean test set |
| `results/robust_attack_results.json` | Robust model accuracy under FGSM and PGD |
| `results/robustness_curves.json` | Accuracy vs ε for baseline and robust model |
| `results/ablation_study.csv` | Ablation: all 4 training strategies compared |
| `results/ablation_study.json` | Same, in JSON format |
| `results/comparison_table.csv` | Side-by-side baseline vs robust across all conditions |
| `results/cross_dataset_comparison.csv` | CICDDoS vs InSDN generalization results |
| `results/shap_importance.json` | Top features ranked by mean absolute SHAP value |
| `results/cm_baseline_clean.png` | Confusion matrix — baseline, clean data |
| `results/cm_baseline_pgd.png` | Confusion matrix — baseline, under PGD |
| `results/cm_robust_clean.png` | Confusion matrix — robust model, clean data |
| `results/cm_robust_pgd.png` | Confusion matrix — robust model, under PGD |
| `results/robustness_curve_fgsm.png` | Plot — accuracy vs ε under FGSM |
| `results/robustness_curve_pgd.png` | Plot — accuracy vs ε under PGD |
| `results/ablation_bar_chart.png` | Bar chart — ablation study comparison |
| `results/shap_feature_importance.png` | Bar chart — top SHAP features |
| `results/cross_dataset_comparison.png` | Plot — cross-dataset generalization |

---

## Reference to Other Documentation

| Document | Purpose |
|----------|---------|
| `README.md` | Original project overview |
| `UPDATE_README.md` | Full technical reference — architecture, attack math, hyperparameters, function signatures |
| `config.py` | Single source of truth for all hyperparameters |
| `notebooks/main.ipynb` | The actual execution — everything above was run here on Kaggle |
