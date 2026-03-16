"""
Shoebill SHAP Waterfall Plot Generator
----------------------------------------------
Generate SHAP waterfall plots for each sample in an input CSV.
**必須提供訓練集**當 background，否則 SHAP 值會不準。

Usage:
  python shoebill_shap_waterfall.py \
    --model shoebill_model \
    --input TE_feature.csv \
    --train-data TR_feature_3000.csv \
    --output-dir shap_plots \
    --max-display 11
"""

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ---------- Artifact loader ----------

def load_model(model_dir: str):
    p = Path(model_dir)
    if not p.exists():
        raise FileNotFoundError(f"[ERROR] model folder not found: {model_dir}")

    model = joblib.load(p / "model.joblib")

    with open(p / "features.json") as f:
        feature_names = json.load(f)

    with open(p / "meta.json") as f:
        meta = json.load(f)

    return model, feature_names, meta

# ---------- Alignment helpers ----------

def align_by_feature_names(df: pd.DataFrame, feature_names):
    """Strict name-based alignment. Raises KeyError if missing."""
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise KeyError(missing)
    return df.reindex(columns=feature_names)
        
def align_by_numeric_mapping(df: pd.DataFrame, feature_names):
    """
    Your CSV format: [ID | 0 | 1 | 2 | ... | 829]
    Model features: ['feature_1', 'feature_8', ...]
    """
    feature_block = df.iloc[:, 1:]  # drop ID column

    try:
        as_int = [int(c) for c in feature_block.columns]
        if sorted(as_int) != list(range(len(as_int))):
            raise ValueError
    except Exception:
        raise ValueError(
            "[ERROR] Columns 2..end are not numeric 0..N-1; "
            "cannot use numeric mapping."
        )

    mapped_numeric_cols = []
    for fname in feature_names:
        m = re.match(r"^feature_(\d+)$", str(fname))
        if not m:
            raise ValueError(f"[ERROR] Can't parse numeric index from feature name: {fname}")
        k = int(m.group(1))
        col_name = str(k - 1)
        mapped_numeric_cols.append(col_name)

    missing_numeric = [c for c in mapped_numeric_cols if c not in feature_block.columns]
    if missing_numeric:
        raise ValueError(
            f"[ERROR] Input CSV missing numeric columns: {missing_numeric[:10]} ..."
        )

    X_numeric_ordered = feature_block.loc[:, mapped_numeric_cols]
    X = X_numeric_ordered.copy()
    X.columns = feature_names
    return X

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Shoebill: SHAP waterfall plots")
    ap.add_argument("--model", required=True, help="Path to exported model directory")
    ap.add_argument("--input", required=True, help="CSV with new samples (ID + features)")
    ap.add_argument("--train-data", required=True, help="Training CSV for SHAP background (TR_feature_3000.csv)")
    ap.add_argument("--output-dir", required=True, help="Directory to save SHAP waterfall PDFs")
    ap.add_argument("--max-display", type=int, default=11, help="Max number of features to show in waterfall")
    args = ap.parse_args()

    # 1. Load model + meta
    model, feature_names, meta = load_model(args.model)

    # 2. Load input cases
    df = pd.read_csv(args.input)
    if df.shape[1] < (1 + len(feature_names)):
        raise ValueError(f"[ERROR] Input has {df.shape[1]} columns but needs 1(ID) + {len(feature_names)} features.")

    id_col_name = df.columns[0]
    id_series = df.iloc[:, 0].astype(str).copy()

    # 3. Align features
    try:
        X = align_by_feature_names(df, feature_names)
    except KeyError:
        X = align_by_numeric_mapping(df, feature_names)
    X = X.loc[:, feature_names]

    # 4. Check NaN
    if X.isnull().any().any():
        where_nan = np.argwhere(np.isnan(X.to_numpy()))
        raise ValueError(f"[ERROR] NaN in test data at positions: {where_nan[:5].tolist()} ...")

    # === Training set as background ===
    train_df = pd.read_csv(args.train_data)
    if train_df.shape[1] < (1 + len(feature_names)):
        raise ValueError(f"[ERROR] Train CSV has {train_df.shape[1]} columns but needs 1(ID) + {len(feature_names)} features.")

    try:
        X_train = align_by_feature_names(train_df, feature_names)
    except KeyError:
        X_train = align_by_numeric_mapping(train_df, feature_names)
    X_train = X_train.loc[:, feature_names]

    if X_train.isnull().any().any():
        raise ValueError("[ERROR] NaN detected in training data!")

    bg = shap.sample(X_train, 1000, random_state=42) if len(X_train) > 1000 else X_train

    # 5. Build SHAP explainer（Training set as background）
    explainer = shap.TreeExplainer(
        model,
        data=bg,
        feature_perturbation="interventional",
        model_output="probability",
    )

    shap_values = explainer(X)

    # 6. Draw waterfall
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(len(X)):
        row_exp = shap_values[idx]
        sample_id = id_series.iloc[idx]

        plt.figure()
        shap.plots.waterfall(row_exp, max_display=args.max_display, show=False)
        plt.gca().set_xlabel("Δ Predicted probability")
        plt.tight_layout()

        fname = out_dir / f"Pred_prob_{sample_id}.pdf"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"[OK] Generated {len(X)} SHAP waterfall plots in: {out_dir.resolve()}")

if __name__ == "__main__":
    main()