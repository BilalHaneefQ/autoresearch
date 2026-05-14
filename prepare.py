"""
Fixed data pipeline and evaluation harness for churn prediction autoresearch.
Do not modify this file.

Dataset: Telco Customer Churn (Telco-Customer-Churn.csv)
Target:  Churn (Yes=1, No=0)
Metric:  F1 Score on the churn class (higher is better)
         Secondary: Recall on the churn class
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, recall_score, roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

DATA_PATH  = 'Telco-Customer-Churn.csv'
SEED       = 42
TEST_SIZE  = 0.20
VAL_SIZE   = 0.15   # fraction of remaining train split
BATCH_SIZE = 256

# ---------------------------------------------------------------------------
# Data loading and preprocessing (fixed pipeline)
# ---------------------------------------------------------------------------

def load_data():
    """
    Load, clean, encode and split the Telco churn dataset.
    Returns X_train, X_val, X_test, y_train, y_val, y_test (numpy float32),
    and the fitted scaler.
    """
    df = pd.read_csv(DATA_PATH)
    df.drop(columns=['customerID'], inplace=True)

    # TotalCharges contains spaces for new customers → coerce to float
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Binary columns → label encode (0/1)
    binary_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService',
        'PaperlessBilling', 'Churn'
    ]
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    # Multi-class categoricals → one-hot encode
    ohe_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
    ]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    X = df.drop(columns=['Churn']).values.astype(np.float32)
    y = df['Churn'].values.astype(np.float32)

    # Stratified splits to preserve churn ratio in every split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, random_state=SEED, stratify=y_train
    )

    # Fit scaler on train only, transform all splits
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def get_input_dim():
    """Returns number of input features after preprocessing."""
    X_train, *_ = load_data()
    return X_train.shape[1]


def get_pos_weight(y_train):
    """Class weight for BCEWithLogitsLoss to handle imbalance (73% No, 27% Yes)."""
    import torch
    return torch.tensor(
        [(y_train == 0).sum() / (y_train == 1).sum()],
        dtype=torch.float32
    )

# ---------------------------------------------------------------------------
# Evaluation harness (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate a trained PyTorch model on the test set.
    Primary metric: f1_churn (higher is better).

    Args:
        model:     trained nn.Module
        X_test:    numpy array, preprocessed test features
        y_test:    numpy array, test labels (0/1)
        threshold: classification threshold (default 0.5)

    Returns dict with f1_churn, recall, auc_roc, accuracy, threshold.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    model.eval()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
        batch_size=BATCH_SIZE, shuffle=False
    )
    device = next(model.parameters()).device
    logits_all, labels_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits_all.append(model(xb.to(device)).cpu())
            labels_all.append(yb)

    probs  = torch.sigmoid(torch.cat(logits_all)).numpy()
    labels = torch.cat(labels_all).numpy()
    preds  = (probs >= threshold).astype(int)

    return {
        'f1_churn':  float(f1_score(labels, preds, zero_division=0)),
        'recall':    float(recall_score(labels, preds, zero_division=0)),
        'auc_roc':   float(roc_auc_score(labels, probs)),
        'accuracy':  float(accuracy_score(labels, preds)),
        'threshold': threshold,
    }


def find_best_threshold(model, X_val, y_val):
    """
    Sweep thresholds [0.30, 0.70] on the validation set.
    Returns the threshold that maximises F1 on the churn class.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    model.eval()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=BATCH_SIZE, shuffle=False
    )
    device = next(model.parameters()).device
    logits_all, labels_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits_all.append(model(xb.to(device)).cpu())
            labels_all.append(yb)

    probs  = torch.sigmoid(torch.cat(logits_all)).numpy()
    labels = torch.cat(labels_all).numpy()

    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.30, 0.70, 0.01):
        f = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1     = f
            best_thresh = t
    return best_thresh


# ---------------------------------------------------------------------------
# Main — validate pipeline when run directly
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Validating data pipeline...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_data()
    print(f"  Input features : {X_train.shape[1]}")
    print(f"  Train samples  : {len(y_train)}  (churn {y_train.mean()*100:.1f}%)")
    print(f"  Val   samples  : {len(y_val)}  (churn {y_val.mean()*100:.1f}%)")
    print(f"  Test  samples  : {len(y_test)}  (churn {y_test.mean()*100:.1f}%)")
    print(f"  pos_weight     : {get_pos_weight(y_train).item():.3f}")
    print("Pipeline OK. Ready to train.")
