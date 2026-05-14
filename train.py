"""
Churn prediction — Boosting models.
Only file the agent modifies.
Usage: conda run -n churn-pred python train.py

Files are saved to the repo root:
  confusion_matrices/  — PNG heatmaps
  saved_models/        — serialized model files (.pkl)
  metrics.jsonl        — one line per run
"""

import json
import os
import subprocess
from datetime import datetime

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    f1_score, recall_score, roc_auc_score,
    accuracy_score, confusion_matrix, cohen_kappa_score
)
from xgboost import XGBClassifier

from prepare import load_data, SEED

# ---------------------------------------------------------------------------
# Output dirs — always repo root
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CM_DIR    = os.path.join(REPO_ROOT, 'confusion_matrices')
MODEL_DIR = os.path.join(REPO_ROOT, 'saved_models')
METRICS_F = os.path.join(REPO_ROOT, 'metrics.jsonl')
os.makedirs(CM_DIR,    exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Model — modify this and everything below it
# ---------------------------------------------------------------------------

MODEL_NAME = 'XGBoost_baseline'

def build_model():
    """
    Return an untrained sklearn-compatible model.
    Swap XGBClassifier for LGBMClassifier, CatBoostClassifier,
    GradientBoostingClassifier, StackingClassifier, etc.
    """
    pos_neg_ratio = (1 - 0.265) / 0.265   # 73/27 class split
    return XGBClassifier(
        n_estimators     = 500,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = pos_neg_ratio,
        eval_metric      = 'logloss',
        random_state     = SEED,
        n_jobs           = -1,
    )

# ---------------------------------------------------------------------------
# Helpers (do not modify)
# ---------------------------------------------------------------------------

def find_best_threshold(probs, y_val):
    """Sweep [0.30, 0.70] and return threshold that maximises F1 on val set."""
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.30, 0.70, 0.01):
        f = f1_score(y_val.astype(int), (probs >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_thresh = f, t
    return best_thresh

def save_confusion_matrix(cm, metrics, commit, timestamp, model_name):
    cm_path = os.path.join(CM_DIR, f'cm_{model_name}_{commit}_{timestamp}.png')
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'],
                annot_kws={'size': 14})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(
        f'{model_name} | commit={commit}\n'
        f'F1={metrics["f1_churn"]:.4f}  Recall={metrics["recall"]:.4f}  '
        f'AUC={metrics["auc_roc"]:.4f}  κ={metrics["kappa"]:.4f}', fontsize=10)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=120)
    plt.close(fig)
    return cm_path

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run():
    X_train, X_val, X_test, y_train, y_val, y_test, _ = load_data()

    model       = build_model()
    model.fit(X_train, y_train)

    val_probs   = model.predict_proba(X_val)[:, 1]
    best_thresh = find_best_threshold(val_probs, y_val)

    test_probs  = model.predict_proba(X_test)[:, 1]
    test_preds  = (test_probs >= best_thresh).astype(int)
    labels      = y_test.astype(int)

    metrics = {
        'f1_churn':  float(f1_score(labels, test_preds, zero_division=0)),
        'recall':    float(recall_score(labels, test_preds, zero_division=0)),
        'auc_roc':   float(roc_auc_score(labels, test_probs)),
        'accuracy':  float(accuracy_score(labels, test_preds)),
        'kappa':     float(cohen_kappa_score(labels, test_preds)),
        'threshold': float(best_thresh),
    }

    cm = confusion_matrix(labels, test_preds)
    tn, fp, fn, tp = cm.ravel()

    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = 'unknown'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    cm_path    = save_confusion_matrix(cm, metrics, commit, timestamp, MODEL_NAME)
    model_path = os.path.join(MODEL_DIR, f'model_{MODEL_NAME}_{commit}_{timestamp}.pkl')
    joblib.dump({'model': model, 'threshold': best_thresh, 'metrics': metrics}, model_path)

    record = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'commit': commit, 'model': MODEL_NAME,
        'metrics': {k: round(v, 6) for k, v in metrics.items()},
        'confusion_matrix': {'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
                             'image_path': cm_path},
    }
    with open(METRICS_F, 'a') as f:
        f.write(json.dumps(record) + '\n')

    print("---")
    print(f"f1_churn:   {metrics['f1_churn']:.6f}")
    print(f"recall:     {metrics['recall']:.6f}")
    print(f"auc_roc:    {metrics['auc_roc']:.6f}")
    print(f"accuracy:   {metrics['accuracy']:.6f}")
    print(f"kappa:      {metrics['kappa']:.6f}")
    print(f"threshold:  {metrics['threshold']:.2f}")
    print(f"model:      {MODEL_NAME}")
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"\nConfusion matrix → {cm_path}")
    print(f"Model saved      → {model_path}")
    print(f"Metrics appended → {METRICS_F}")


if __name__ == '__main__':
    run()
