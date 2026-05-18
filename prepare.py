"""
Fixed data pipeline and evaluation harness for house price regression.
Do not modify this file.

Dataset : dataset/data.csv  (King County house sales)
Target  : price  (continuous, USD)
Metric  : RMSE on test set (lower is better)
          Secondary: R² score (higher is better)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

DATA_PATH = 'dataset/data.csv'
SEED      = 42
TEST_SIZE = 0.20
VAL_SIZE  = 0.15   # fraction of remaining train split

# ---------------------------------------------------------------------------
# Data loading and preprocessing (fixed pipeline)
# ---------------------------------------------------------------------------

def load_data():
    """
    Load, clean, encode and split the King County house price dataset.
    Returns X_train, X_val, X_test, y_train, y_val, y_test (numpy float32),
    and the fitted scaler.
    """
    df = pd.read_csv(DATA_PATH)

    # Drop rows where price is 0 or NaN
    df = df[df['price'] > 0].dropna(subset=['price'])

    # Drop non-informative columns
    df.drop(columns=['date', 'street', 'country'], inplace=True)

    # Extract useful features from statezip (zip code as numeric)
    df['zipcode'] = df['statezip'].str.extract(r'(\d+)').astype(float)
    df.drop(columns=['statezip'], inplace=True)

    # Label encode city
    le = LabelEncoder()
    df['city'] = le.fit_transform(df['city'].astype(str))

    # Feature: house age, years since renovation
    df['house_age']        = 2025 - df['yr_built']
    df['yrs_since_renov']  = np.where(
        df['yr_renovated'] > 0,
        2025 - df['yr_renovated'],
        df['house_age']
    )
    df.drop(columns=['yr_built', 'yr_renovated'], inplace=True)

    # Log-transform target to handle right skew
    y = np.log1p(df['price'].values).astype(np.float32)
    X = df.drop(columns=['price']).values.astype(np.float32)

    # Stratified-like split using price quantiles
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, random_state=SEED
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def get_input_dim():
    X_train, *_ = load_data()
    return X_train.shape[1]

# ---------------------------------------------------------------------------
# Evaluation harness (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate_model(y_true_log, y_pred_log):
    """
    Evaluate predictions. Both y_true and y_pred are in log1p space.
    Returns dict with rmse (primary, lower=better), r2, mae — all in original USD scale.
    """
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.clip(y_pred, 0, None)   # no negative prices

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    mae  = float(mean_absolute_error(y_true, y_pred))

    return {'rmse': rmse, 'r2': r2, 'mae': mae}

# ---------------------------------------------------------------------------
# Main — validate pipeline when run directly
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Validating data pipeline...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_data()
    print(f"  Input features : {X_train.shape[1]}")
    print(f"  Train samples  : {len(y_train)}")
    print(f"  Val   samples  : {len(y_val)}")
    print(f"  Test  samples  : {len(y_test)}")
    print(f"  Target range   : ${np.expm1(y_test.min()):,.0f} — ${np.expm1(y_test.max()):,.0f}")
    print(f"  Target mean    : ${np.expm1(y_test.mean()):,.0f}")
    print("Pipeline OK. Ready to train.")
