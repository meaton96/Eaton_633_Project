from sklearn.model_selection import train_test_split
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def group_split(
    df: pd.DataFrame,
    *,
    target_col: str = "target",
    group_col: str = "patient_nbr",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    balance_tolerance: float = 0.01, 
    max_tries: int = 50,
):
    """
    Group-aware split (no patient leakage) into train/val/test.
    Uses two GroupShuffleSplits: (train vs rest), then (val vs test) on the rest.
    Attempts to keep class prevalence close to overall within `balance_tolerance`.
    """

    if not math.isclose(train_size + val_size + test_size, 1.0, rel_tol=1e-9):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    if group_col not in df.columns:
        raise KeyError(f"Expected group column '{group_col}' in df")

    if target_col not in df.columns:
        raise KeyError(f"Expected target column '{target_col}' in df")

    rng = np.random.RandomState(random_state)
    overall_rate = df[target_col].mean()

    def _ok_rate(d: pd.DataFrame) -> bool:
        if d.empty:
            return False
        return abs(d[target_col].mean() - overall_rate) <= balance_tolerance

    # 1) Train vs rest
    tries = 0
    while True:
        gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=rng.randint(0, 10**9))
        tr_idx, rest_idx = next(gss.split(df, df[target_col], groups=df[group_col]))
        train_df = df.iloc[tr_idx].copy()
        rest_df  = df.iloc[rest_idx].copy()
        if _ok_rate(train_df) and _ok_rate(rest_df):
            break
        tries += 1
        if tries >= max_tries:
            break

    # 2) Val vs Test on the remaining groups
    rel_val = val_size / (val_size + test_size)
    tries = 0
    while True:
        gss2 = GroupShuffleSplit(n_splits=1, train_size=rel_val, random_state=rng.randint(0, 10**9))
        val_idx_rel, test_idx_rel = next(gss2.split(rest_df, rest_df[target_col], groups=rest_df[group_col]))
        val_df  = rest_df.iloc[val_idx_rel].copy()
        test_df = rest_df.iloc[test_idx_rel].copy()
        if _ok_rate(val_df) and _ok_rate(test_df):
            break
        tries += 1
        if tries >= max_tries:
            break

    # leak test: no patient overlap
    tr_pat, va_pat, te_pat = set(train_df[group_col]), set(val_df[group_col]), set(test_df[group_col])
    assert tr_pat.isdisjoint(va_pat) and tr_pat.isdisjoint(te_pat) and va_pat.isdisjoint(te_pat), \
        "Patient overlap detected across splits."

    # return X/y splits
    def _xy(d: pd.DataFrame):
        X = d.drop(columns=[target_col])
        y = d[target_col].copy()
        return X, y

    X_train, y_train = _xy(train_df)
    X_val,   y_val   = _xy(val_df)
    X_test,  y_test  = _xy(test_df)

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_val_test_split(df, target_col = 'target', train_size = 0.7):

    X_train, X_, y_train, y_ = train_test_split(
        df.drop(columns=[target_col]), 
        df[target_col], 
        test_size=1-train_size, 
        stratify=df[target_col],
        random_state=42
        )
    
    X_validate, X_test, y_validate, y_test = train_test_split(X_, y_, test_size=0.5, stratify=y_, random_state=42)

    return X_train, X_validate, X_test, y_train, y_validate, y_test

