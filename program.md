# autoresearch — Churn Prediction ANN

Autonomous research experiment: improve a customer churn prediction model by modifying `train.py`.

## Setup

1. **Agree on a run tag** based on today's date (e.g. `may14`). Branch `autoresearch/<tag>` must not exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>`
3. **Activate the environment**: `conda activate churn-pred`
4. **Read these files** for full context:
   - `prepare.py` — fixed data pipeline, preprocessing, evaluation harness. **Do not modify.**
   - `train.py` — the ANN model. **Only file you edit.**
5. **Initialize results.tsv** with just the header row (leave untracked by git).
6. **Confirm setup** and begin the experiment loop.

## The goal

**Maximize `f1_churn`** — F1 Score on the churn class (label=1) against the fixed test set defined in `prepare.py`.

Secondary metric to watch: **`recall`** — catching churners is more valuable than avoiding false alarms.

## Dataset context

- **7,043 customers**, 20 features (demographics, services, billing)
- **Class imbalance**: 73% No Churn, 27% Churn
- `prepare.py` handles all preprocessing: label encoding, one-hot encoding, standard scaling, stratified splits
- Fixed splits: 80% train (further split 85/15 train/val), 20% test — same every run

## What you CAN do

- Modify `train.py` — everything is fair game:
  - Model architecture (depth, width, activation functions, BatchNorm, Dropout, skip connections)
  - Optimizer (Adam, AdamW, SGD, learning rate, weight decay, schedulers)
  - Training loop (early stopping, gradient clipping, mixed precision)
  - Loss function (BCEWithLogitsLoss pos_weight, focal loss, label smoothing)
  - Data augmentation (SMOTE via imbalanced-learn, which is installed)
  - Threshold tuning strategy

## What you CANNOT do

- Modify `prepare.py` — fixed data splits and evaluation harness
- Change `SEED`, `TEST_SIZE`, `VAL_SIZE` in `prepare.py`
- Add packages not in the conda environment (`conda activate churn-pred`)

## Output format

```
---
f1_churn:   0.612345
recall:     0.720000
auc_roc:    0.840000
accuracy:   0.810000
threshold:  0.42
```

Extract the key metric:
```
grep "^f1_churn:" run.log
```

## Logging results

Log to `results.tsv` (tab-separated, leave untracked by git):

```
commit	f1_churn	recall	auc_roc	description
```

Example:
```
commit	f1_churn	recall	auc_roc	description
a1b2c3d	0.580000	0.690000	0.820000	baseline 3-layer MLP
b2c3d4e	0.612000	0.720000	0.840000	add BatchNorm + Dropout 0.3
c3d4e5f	0.598000	0.710000	0.835000	discard — worse than prev
d4e5f6g	0.635000	0.750000	0.855000	deeper net + AdamW + LR scheduler
```

## The experiment loop

LOOP FOREVER:

1. Check git state: current branch and commit.
2. Improve `train.py` with one experimental idea.
3. `git commit`
4. Run: `conda run -n churn-pred python train.py > run.log 2>&1`
5. Read results: `grep "^f1_churn:\|^recall:" run.log`
6. If grep is empty → crashed. `tail -n 50 run.log` to diagnose.
7. Log to `results.tsv`.
8. If `f1_churn` improved: keep the commit.
9. If not: `git reset --hard HEAD~1`

## Suggested progression

1. **Baseline** — run as-is to establish the floor (~0.58 f1_churn expected)
2. **Add BatchNorm + Dropout** — biggest single improvement on tabular data
3. **Deeper architecture** — 256 → 128 → 64 → 32 with residual connections
4. **AdamW + ReduceLROnPlateau** — better optimizer + adaptive LR
5. **Focal Loss** — specifically designed for class imbalance (replaces pos_weight)
6. **SMOTE oversampling** — balance training set using imbalanced-learn
7. **Threshold tuning** — find optimal cutoff per run (already implemented in prepare.py)
8. **Ensemble / test-time augmentation** — average predictions from multiple checkpoints

## Key tips

- **pos_weight** in BCEWithLogitsLoss is already set correctly in the baseline — don't remove it
- `find_best_threshold()` in `prepare.py` sweeps [0.30, 0.70] on the val set — always use it
- Overfitting is the main risk on this small dataset (5K train samples) — Dropout is your friend
- A model with f1_churn=0.65 and recall=0.80 is better than one with f1=0.70 and recall=0.55 for a business

## Never stop

Loop indefinitely until manually interrupted. If you run out of ideas, revisit near-misses, try combining approaches, or research focal loss / class-balanced sampling papers.
