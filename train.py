"""
House price regression — the only file the agent modifies.
Usage: conda run -n churn-pred python train.py
"""
import json, os, subprocess
from datetime import datetime
import joblib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from prepare import evaluate_model, SEED

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR  = os.path.join(REPO_ROOT, 'prediction_plots')
MODEL_DIR = os.path.join(REPO_ROOT, 'saved_models')
METRICS_F = os.path.join(REPO_ROOT, 'metrics.jsonl')
os.makedirs(PLOT_DIR, exist_ok=True); os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_NAME = 'Stack_TargetEnc_tuned'

def load_and_engineer():
    df = pd.read_csv('dataset/data.csv')
    df = df[df['price'] > 0].dropna(subset=['price'])
    df.drop(columns=['date','street','country'], inplace=True)
    df['zipcode']         = df['statezip'].str.extract(r'(\d+)').astype(float)
    df['house_age']       = 2025 - df['yr_built']
    df['yrs_since_renov'] = np.where(df['yr_renovated']>0, 2025-df['yr_renovated'], df['house_age'])
    df.drop(columns=['statezip','yr_built','yr_renovated'], inplace=True)
    le = LabelEncoder()
    df['city_label'] = le.fit_transform(df['city'].astype(str))
    df.drop(columns=['city'], inplace=True)

    y = np.log1p(df['price'].values).astype(np.float32)
    X_raw = df.drop(columns=['price'])

    idx = np.arange(len(y))
    idx_tv, idx_test  = train_test_split(idx, test_size=0.20, random_state=SEED)
    idx_train, idx_val = train_test_split(idx_tv, test_size=0.15, random_state=SEED)

    def target_enc(col, train_idx, y_tr, smoothing=10):
        global_mean = y_tr.mean()
        vals = X_raw[col].values
        enc_map = {}
        for cat in np.unique(vals[train_idx]):
            mask = vals[train_idx] == cat
            n, mean = mask.sum(), y_tr[mask].mean()
            enc_map[cat] = (n*mean + smoothing*global_mean) / (n+smoothing)
        return np.array([enc_map.get(v, global_mean) for v in vals])

    y_train = y[idx_train]
    city_enc = target_enc('city_label', idx_train, y_train, smoothing=5)
    zip_enc  = target_enc('zipcode',    idx_train, y_train, smoothing=10)

    X_raw['city_enc'] = city_enc
    X_raw['zip_enc']  = zip_enc
    X_raw.drop(columns=['city_label','zipcode'], inplace=True)

    X_raw['bath_per_bed']   = X_raw['bathrooms'] / (X_raw['bedrooms'] + 1e-6)
    X_raw['sqft_per_room']  = X_raw['sqft_living'] / (X_raw['bedrooms']+X_raw['bathrooms']+1e-6)
    X_raw['basement_ratio'] = X_raw['sqft_basement'] / (X_raw['sqft_living']+1e-6)
    X_raw['living_sq']      = X_raw['sqft_living'] ** 2
    X_raw['age_sq']         = X_raw['house_age'] ** 2
    X_raw['view_water']     = X_raw['view'] * X_raw['waterfront']
    X_raw['quality']        = X_raw['view'] + X_raw['condition'] + X_raw['waterfront']*3
    X_raw['renov_rec']      = 1.0 / (X_raw['yrs_since_renov'] + 1)

    X = X_raw.values.astype(np.float32)
    scaler = StandardScaler()
    X[idx_train] = scaler.fit_transform(X[idx_train])
    X[idx_val]   = scaler.transform(X[idx_val])
    X[idx_test]  = scaler.transform(X[idx_test])

    return (X[idx_train], X[idx_val], X[idx_test],
            y[idx_train], y[idx_val],  y[idx_test], scaler)

def build_model():
    # More aggressive tuning on best feature set
    lgbm = LGBMRegressor(n_estimators=5000, max_depth=10, learning_rate=0.003,
                         num_leaves=255, subsample=0.7, colsample_bytree=0.7,
                         min_child_samples=8, reg_alpha=0.05, reg_lambda=0.3,
                         random_state=SEED, n_jobs=-1, verbose=-1)
    xgb  = XGBRegressor(n_estimators=5000, max_depth=7, learning_rate=0.005,
                        subsample=0.75, colsample_bytree=0.75, min_child_weight=3,
                        gamma=0.05, reg_alpha=0.05,
                        random_state=SEED, n_jobs=-1, verbosity=0)
    rf   = RandomForestRegressor(n_estimators=800, max_depth=20, min_samples_leaf=2,
                                  random_state=SEED, n_jobs=-1)
    et   = ExtraTreesRegressor(n_estimators=800, max_depth=20, min_samples_leaf=2,
                                random_state=SEED, n_jobs=-1)
    return StackingRegressor(
        estimators=[('lgbm',lgbm),('xgb',xgb),('rf',rf),('et',et)],
        final_estimator=Ridge(alpha=0.1), cv=5, n_jobs=1,
    )

def save_plot(y_true_log, y_pred_log, metrics, commit, timestamp):
    y_true=np.expm1(y_true_log); y_pred=np.expm1(np.clip(y_pred_log,0,None))
    path=os.path.join(PLOT_DIR,f'pred_{MODEL_NAME}_{commit}_{timestamp}.png')
    fig,ax=plt.subplots(figsize=(7,6))
    ax.scatter(y_true/1e6,y_pred/1e6,alpha=0.3,s=15,color='steelblue')
    lim=max(y_true.max(),y_pred.max())/1e6*1.05
    ax.plot([0,lim],[0,lim],'r--',lw=1.5,label='Perfect')
    ax.set_xlabel('Actual ($M)'); ax.set_ylabel('Predicted ($M)')
    ax.set_title(f'{MODEL_NAME} | commit={commit}\nRMSE=${metrics["rmse"]/1e3:.1f}K  R²={metrics["r2"]:.4f}  MAE=${metrics["mae"]/1e3:.1f}K',fontsize=10)
    ax.legend(); plt.tight_layout(); plt.savefig(path,dpi=120); plt.close(fig)
    return path

def run():
    X_train,X_val,X_test,y_train,y_val,y_test,_ = load_and_engineer()
    X_fit=np.vstack([X_train,X_val]); y_fit=np.concatenate([y_train,y_val])
    print(f"Features: {X_train.shape[1]}, Fit samples: {len(y_fit)}")
    model=build_model()
    model.fit(X_fit, y_fit)
    y_pred=model.predict(X_test)
    metrics=evaluate_model(y_test,y_pred)
    try:    commit=subprocess.check_output(['git','rev-parse','--short','HEAD'],stderr=subprocess.DEVNULL).decode().strip()
    except: commit='unknown'
    ts=datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path=save_plot(y_test,y_pred,metrics,commit,ts)
    model_path=os.path.join(MODEL_DIR,f'model_{MODEL_NAME}_{commit}_{ts}.pkl')
    joblib.dump({'model':model,'metrics':metrics},model_path)
    with open(METRICS_F,'a') as f:
        f.write(json.dumps({'timestamp':datetime.now().isoformat(timespec='seconds'),'commit':commit,'model':MODEL_NAME,
            'metrics':{k:round(v,4) for k,v in metrics.items()},'plot':plot_path})+'\n')
    print("---")
    print(f"rmse:       ${metrics['rmse']:,.0f}")
    print(f"r2:         {metrics['r2']:.6f}")
    print(f"mae:        ${metrics['mae']:,.0f}")
    print(f"model:      {MODEL_NAME}")
    print(f"\nPlot  → {plot_path}")
    print(f"Model → {model_path}")

if __name__ == '__main__': run()
