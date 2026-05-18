"""
House price regression — the only file the agent modifies.
Usage: conda run -n churn-pred python train.py

Outputs saved to repo root:
  prediction_plots/  — actual vs predicted scatter plots
  saved_models/      — serialized models (.pkl)
  metrics.jsonl      — one line per run
"""

import json, os, subprocess
from datetime import datetime

import joblib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from prepare import load_data, evaluate_model, SEED

REPO_ROOT  = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR   = os.path.join(REPO_ROOT, 'prediction_plots')
MODEL_DIR  = os.path.join(REPO_ROOT, 'saved_models')
METRICS_F  = os.path.join(REPO_ROOT, 'metrics.jsonl')
os.makedirs(PLOT_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Model — modify this and everything below it
# ---------------------------------------------------------------------------

MODEL_NAME = 'Ridge_baseline'

def build_model():
    """
    Return an untrained sklearn-compatible model.
    Swap for RandomForestRegressor, XGBRegressor, LGBMRegressor,
    GradientBoostingRegressor, MLPRegressor, etc.
    """
    return Ridge(alpha=1.0, random_state=SEED)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_plot(y_true_log, y_pred_log, metrics, commit, timestamp):
    """Actual vs predicted scatter plot in USD scale."""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(np.clip(y_pred_log, 0, None))
    path   = os.path.join(PLOT_DIR, f'pred_{MODEL_NAME}_{commit}_{timestamp}.png')

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true / 1e6, y_pred / 1e6, alpha=0.3, s=15, color='steelblue')
    lim = max(y_true.max(), y_pred.max()) / 1e6 * 1.05
    ax.plot([0, lim], [0, lim], 'r--', lw=1.5, label='Perfect prediction')
    ax.set_xlabel('Actual Price ($M)', fontsize=12)
    ax.set_ylabel('Predicted Price ($M)', fontsize=12)
    ax.set_title(
        f'{MODEL_NAME} | commit={commit}\n'
        f'RMSE=${metrics["rmse"]/1e3:.1f}K  R²={metrics["r2"]:.4f}  MAE=${metrics["mae"]/1e3:.1f}K',
        fontsize=10)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run():
    X_train, X_val, X_test, y_train, y_val, y_test, _ = load_data()

    model = build_model()
    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)
    metrics    = evaluate_model(y_test, y_pred_log)

    try:    commit = subprocess.check_output(['git','rev-parse','--short','HEAD'],stderr=subprocess.DEVNULL).decode().strip()
    except: commit = 'unknown'
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    plot_path  = save_plot(y_test, y_pred_log, metrics, commit, ts)
    model_path = os.path.join(MODEL_DIR, f'model_{MODEL_NAME}_{commit}_{ts}.pkl')
    joblib.dump({'model': model, 'metrics': metrics}, model_path)

    with open(METRICS_F, 'a') as f:
        f.write(json.dumps({
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'commit': commit, 'model': MODEL_NAME,
            'metrics': {k: round(v, 4) for k, v in metrics.items()},
            'plot': plot_path,
        }) + '\n')

    print("---")
    print(f"rmse:       ${metrics['rmse']:,.0f}")
    print(f"r2:         {metrics['r2']:.6f}")
    print(f"mae:        ${metrics['mae']:,.0f}")
    print(f"model:      {MODEL_NAME}")
    print(f"\nPlot saved  → {plot_path}")
    print(f"Model saved → {model_path}")
    print(f"Metrics     → {METRICS_F}")


if __name__ == '__main__':
    run()
