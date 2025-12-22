"""
Shoebill Prediction Script
--------------------------------------------------------------------
Predict on new samples given exported model.

Usage:
    python shoebill_predict.py \
        --model shoebill_model \
        --input feature.csv \
        --output preds.csv \
        --threshold 0.420 \
"""

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

def load_model(model_dir: str):
    p = Path(model_dir)
    if not p.exists():
        raise FileNotFoundError(f"[ERROR] Model folder not found: {model_dir}")

    model = joblib.load(p / "model.joblib")
    with open(p / "features.json") as f:
        feature_names = json.load(f)
    with open(p / "meta.json") as f:
        meta = json.load(f)
    return model, feature_names, meta

def align_by_feature_names(df: pd.DataFrame, feature_names):
    """Try strict name-based alignment. Raise if missing."""
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise KeyError(missing)
    return df.reindex(columns=feature_names)

def align_by_numeric_mapping(df: pd.DataFrame, feature_names):
    """Try column order-based alignment. Raise if missing."""
    """
    CSV format: [ID | 0 | 1 | 2 | ... | 829]
    Map 'feature_k -> column name k-1' in the numeric block.
    """
    feature_block = df.iloc[:, 1:]  # drop ID

    # Ensure numeric-like column names exactly 0..M-1
    try:
        as_int = [int(c) for c in feature_block.columns]
        if sorted(as_int) != list(range(len(as_int))):
            raise ValueError
    except Exception:
        raise ValueError(
            "[ERROR] Columns 2..end are not numeric 0..N-1; cannot use numeric mapping."
        )

    mapped_numeric_cols = []
    import re
    for fname in feature_names:
        m = re.match(r"^feature_(\d+)$", str(fname))
        if not m:
            raise ValueError(
                f"[ERROR] Can't parse numeric index from feature name: {fname}. "
                "Expected pattern like 'feature_13'."
            )
        k = int(m.group(1))
        mapped_numeric_cols.append(str(k - 1))

    missing_numeric = [c for c in mapped_numeric_cols if c not in feature_block.columns]
    if missing_numeric:
        raise ValueError(
            f"[ERROR] Input CSV missing required numeric columns mapped from feature names: "
            f"{missing_numeric[:10]} ... (total {len(missing_numeric)})"
        )

    X_numeric_ordered = feature_block.loc[:, mapped_numeric_cols]
    X = X_numeric_ordered.copy()
    X.columns = feature_names  # rename back to model feature names
    return X

def main():
    ap = argparse.ArgumentParser(description="Shoebill: STRICT prediction on new samples")
    ap.add_argument("--model", required=True, help="Path to exported model directory")
    ap.add_argument("--input", required=True, help="CSV with new samples")
    ap.add_argument("--output", required=True, help="Output CSV for predictions")
    ap.add_argument("--threshold", type=float, default=None, help="Decision threshold (default = meta.json)")
    args = ap.parse_args()

    # Load model + metadata
    model, feature_names, meta = load_model(args.model)
    thr = float(args.threshold) if args.threshold is not None else float(meta.get("threshold", 0.420))

    # Load samples
    df = pd.read_csv(args.input)
    if df.shape[1] < (1 + len(feature_names)):
        raise ValueError(
            f"[ERROR] Input has {df.shape[1]} columns but needs at least 1(ID)+{len(feature_names)} features."
        )

    # ID column (first column) must be preserved in output
    id_col_name = df.columns[0]
    id_series = df.iloc[:, 0].copy()

    # Try 1: strict name-based alignment
    try:
        X = align_by_feature_names(df, feature_names)
    except KeyError as e:
        # Missing by name → Try 2: numeric mapping
        X = align_by_numeric_mapping(df, feature_names)

    # Strict checks
    if X.isnull().any().any():
        where_nan = np.argwhere(np.isnan(X.to_numpy()))
        raise ValueError(
            f"[ERROR] NaN values detected in aligned feature matrix at positions (row,col): "
            f"{where_nan[:5].tolist()} ..."
        )

    # Predict
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= thr).astype(int)

    # Output: ID first, then proba, then label
    out = pd.DataFrame({
        id_col_name: id_series,
        "pred_proba": proba,
        "pred_label": pred,
    })
    out.to_csv(args.output, index=False)
    print(f"[OK] Predictions saved: {args.output} (threshold={thr:.3f})")


if __name__ == "__main__":
    main()

