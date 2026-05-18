# autoresearch — House Price Regression

Autonomous research experiment: improve house price prediction by modifying `train.py`.

## Setup

1. **Agree on a run tag** based on today's date (e.g. `may18`). Branch `autoresearch/<tag>` must not exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>`
3. **Activate the environment**: `conda activate churn-pred`
4. **Read these files** for full context:
   - `prepare.py` — fixed data pipeline and evaluation harness. **Do not modify.**
   - `train.py` — the model. **Only file you edit.**
5. **Initialize results.tsv** with just the header row (leave untracked by git).
6. **Confirm setup** and begin.

## The goal

**Minimize `rmse`** — Root Mean Squared Error in USD on the fixed test set.

Secondary metric: **`r2`** (higher is better, max 1.0).

**Baseline to beat:** Ridge regression (~$170K RMSE expected).

## Dataset context

- **4,600 King County (Seattle) house sales**
- Features: bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, sqft_above, sqft_basement, house_age, yrs_since_renov, city, zipcode
- **Target**: `price` (USD) — log-transformed in pipeline to handle right skew
- `prepare.py` handles all preprocessing: feature engineering, label encoding, StandardScaler, train/val/test splits

## What you CAN do

- Modify `train.py` — everything is fair game:
  - Model: Ridge, Lasso, RandomForest, XGBRegressor, LGBMRegressor, GradientBoosting, ANN, etc.
  - Hyperparameters: depth, n_estimators, learning rate, regularization
  - Additional feature engineering (transform X_train/val/test inside train.py)
  - Ensembling: average or stack multiple models
  - Early stopping, cross-validation

## What you CANNOT do

- Modify `prepare.py` — fixed splits and evaluation
- Add packages not in the conda environment

## Output format

```
---
rmse:       $152340
r2:         0.821400
mae:        $95200
model:      XGBoost_baseline
```

Extract the key metric:
```
grep "^rmse:" run.log
```

## File output (repo root)

Every run saves to:
- `prediction_plots/pred_<model>_<commit>_<timestamp>.png`
- `saved_models/model_<model>_<commit>_<timestamp>.pkl`
- Appends one line to `metrics.jsonl`

## Logging results

Log to `results.tsv` (tab-separated, leave untracked):

```
commit	rmse	r2	mae	model	description
```

## The experiment loop

LOOP FOREVER:

1. Check git state.
2. Improve `train.py` with one idea.
3. `git commit`
4. Run: `conda run -n churn-pred python train.py > run.log 2>&1`
5. Read: `grep "^rmse:\|^r2:" run.log`
6. If crash: `tail -n 50 run.log`
7. Log to `results.tsv`
8. If `rmse` improved (lower): keep commit.
9. If not: `git reset --hard HEAD~1`

## Suggested progression

1. **Baseline** — Ridge regression (establish floor)
2. **Random Forest** — non-linear, handles interactions
3. **XGBoost / LightGBM** — typically best on tabular regression
4. **Hyperparameter tuning** — tune the best model
5. **Feature engineering** — price per sqft, log transforms of sqft features, etc.
6. **Stacking** — blend best models

## Key tips

- Target is log1p-transformed — `evaluate_model()` converts back to USD for metrics
- RMSE in USD: aim for < $120K (good), < $100K (great)
- R² > 0.85 is solid for this dataset
- Outliers (price > $5M) exist — robust models handle them better

## Never stop

Loop until manually interrupted.
