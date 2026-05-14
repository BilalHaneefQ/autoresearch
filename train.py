"""
Churn prediction — Boosting models.
Usage: conda run -n churn-pred python train.py
"""

import json, os, subprocess
from datetime import datetime
import joblib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import f1_score, recall_score, roc_auc_score, accuracy_score, confusion_matrix, cohen_kappa_score
from xgboost import XGBClassifier
from prepare import load_data, SEED

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CM_DIR    = os.path.join(REPO_ROOT, 'confusion_matrices')
MODEL_DIR = os.path.join(REPO_ROOT, 'saved_models')
METRICS_F = os.path.join(REPO_ROOT, 'metrics.jsonl')
os.makedirs(CM_DIR, exist_ok=True); os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_NAME = 'XGBoost_early_stop'

def build_model():
    pos_neg_ratio = (1 - 0.265) / 0.265
    return XGBClassifier(
        n_estimators     = 2000,
        max_depth        = 5,
        learning_rate    = 0.03,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 5,
        scale_pos_weight = pos_neg_ratio,
        eval_metric      = 'aucpr',   # area under precision-recall — better for imbalanced
        early_stopping_rounds = 50,
        random_state     = SEED,
        n_jobs           = -1,
    )

def find_best_threshold(probs, y_val):
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.30, 0.70, 0.01):
        f = f1_score(y_val.astype(int), (probs >= t).astype(int), zero_division=0)
        if f > best_f1: best_f1, best_thresh = f, t
    return best_thresh

def save_cm(cm, metrics, commit, timestamp):
    cm_path = os.path.join(CM_DIR, f'cm_{MODEL_NAME}_{commit}_{timestamp}.png')
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Churn','Churn'], yticklabels=['No Churn','Churn'], annot_kws={'size':14})
    ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'{MODEL_NAME} | commit={commit}\nF1={metrics["f1_churn"]:.4f}  Recall={metrics["recall"]:.4f}  AUC={metrics["auc_roc"]:.4f}  κ={metrics["kappa"]:.4f}', fontsize=10)
    plt.tight_layout(); plt.savefig(cm_path, dpi=120); plt.close(fig)
    return cm_path

def run():
    X_train, X_val, X_test, y_train, y_val, y_test, _ = load_data()
    model = build_model()
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)
    print(f"Best iteration: {model.best_iteration}")
    best_thresh = find_best_threshold(model.predict_proba(X_val)[:,1], y_val)
    test_probs  = model.predict_proba(X_test)[:,1]
    test_preds  = (test_probs >= best_thresh).astype(int)
    labels      = y_test.astype(int)
    metrics = {
        'f1_churn': float(f1_score(labels, test_preds, zero_division=0)),
        'recall':   float(recall_score(labels, test_preds, zero_division=0)),
        'auc_roc':  float(roc_auc_score(labels, test_probs)),
        'accuracy': float(accuracy_score(labels, test_preds)),
        'kappa':    float(cohen_kappa_score(labels, test_preds)),
        'threshold':float(best_thresh),
    }
    cm = confusion_matrix(labels, test_preds); tn,fp,fn,tp = cm.ravel()
    try:    commit = subprocess.check_output(['git','rev-parse','--short','HEAD'],stderr=subprocess.DEVNULL).decode().strip()
    except: commit = 'unknown'
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    cm_path    = save_cm(cm, metrics, commit, ts)
    model_path = os.path.join(MODEL_DIR, f'model_{MODEL_NAME}_{commit}_{ts}.pkl')
    joblib.dump({'model':model,'threshold':best_thresh,'metrics':metrics}, model_path)
    with open(METRICS_F,'a') as f:
        f.write(json.dumps({'timestamp':datetime.now().isoformat(timespec='seconds'),'commit':commit,'model':MODEL_NAME,
            'metrics':{k:round(v,6) for k,v in metrics.items()},
            'confusion_matrix':{'TP':int(tp),'FP':int(fp),'FN':int(fn),'TN':int(tn),'image_path':cm_path}})+'\n')
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

if __name__ == '__main__': run()
