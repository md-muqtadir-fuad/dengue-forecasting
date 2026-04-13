#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Legacy manuscript-style XGBoost pipeline adapted to the current
prime_dataset_model_input_with_purge.csv-style file.

What stays the same as the previous manuscript workflow:
- one global chronological row order
- direct multi-horizon forecasting (one XGBoost model per horizon)
- horizon target built with shift(-h)
- train/test split by row fraction
- purge gap before test
- validation = last 10% of the training slice
- outbreak weighting from training targets
- NaiveLast and SeasonalNaive12 baselines

What is adapted for the current input:
- drops metadata columns such as District / Date / Month-year / split from predictors
- can derive a raw Month from Date if you want
- can keep current numeric features as-is
- writes clean CSV outputs

IMPORTANT:
This script intentionally preserves the previous manuscript-style logic.
It does NOT implement the newer grouped-district or calendar-merge panel logic.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred))) if len(y_true) else np.nan


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred)) if len(y_true) else np.nan


def r2_safe(y_true, y_pred) -> float:
    if len(y_true) == 0 or len(np.unique(y_true)) <= 1:
        return np.nan
    return float(r2_score(y_true, y_pred))


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    m = denom > 0
    if m.sum() == 0:
        return np.nan
    return float(np.mean(2.0 * np.abs(y_pred[m] - y_true[m]) / denom[m]) * 100.0)


def metrics_dict(y_true, y_pred) -> Dict[str, float]:
    return {
        "n": int(len(y_true)),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": r2_safe(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "MBE": float(np.mean(np.asarray(y_pred) - np.asarray(y_true))) if len(y_true) else np.nan,
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
class Config:
    input_path = "./data/raw/prime_dataset_model_input_with_purge.csv"
    output_dir = "./XGBoost_legacy_operational/outputs"
    target_col = "Log_NoOfDenguePatients"
    horizon = 6
    test_frac = 0.20
    purge_gap = 12
    val_frac_within_train = 0.10
    random_state = 42
    derive_month_from_date = True
    use_current_numeric_features = True
    include_month_even_if_month_sin_cos_present = False

    # XGBoost params kept close to the previous manuscript style
    xgb_params = {
        "objective": "reg:squarederror",
        "n_estimators": 5000,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 200,
        "eval_metric": "rmse",
        "tree_method": "hist",
    }


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------
META_COLS = {"District", "Date", "Month-year", "split"}

def load_and_prepare_dataframe(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.input_path)
    df.columns = [c.strip() for c in df.columns]

    if cfg.target_col not in df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found in input.")

    # Keep current row order unless Date/District sorting is available and the file is clearly unsorted.
    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"], errors="raise")
        except Exception:
            pass

    # Previous manuscript data_csv.csv did not contain metadata columns.
    # To preserve the same modeling logic, we remove them from predictors later,
    # but keep them here for audit outputs.
    if cfg.derive_month_from_date and "Date" in df.columns and "Month" not in df.columns:
        if "Month_sin" not in df.columns or "Month_cos" not in df.columns or cfg.include_month_even_if_month_sin_cos_present:
            df["Month"] = pd.to_datetime(df["Date"]).dt.month

    return df


def build_feature_dataframe(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    feat_df = df.drop(columns=[cfg.target_col], errors="ignore").copy()

    # Drop metadata columns to mimic the old data_csv.csv style.
    drop_cols = [c for c in feat_df.columns if c in META_COLS]
    feat_df = feat_df.drop(columns=drop_cols, errors="ignore")

    # Keep current numeric features; one-hot any categoricals, just like the old notebook.
    cat_cols = feat_df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        feat_df = pd.get_dummies(feat_df, columns=cat_cols, drop_first=True)

    # Ensure all remaining columns are numeric
    non_numeric = [c for c in feat_df.columns if not pd.api.types.is_numeric_dtype(feat_df[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric columns remain after encoding: {non_numeric}")

    return feat_df


# ---------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------
def make_model(cfg: Config) -> xgb.XGBRegressor:
    params = dict(cfg.xgb_params)
    params["random_state"] = cfg.random_state
    return xgb.XGBRegressor(**params)


def compute_split_indices(n: int, cfg: Config) -> Dict[str, int]:
    split_idx = int((1.0 - cfg.test_frac) * n)
    train_end = max(split_idx - cfg.purge_gap, 1)
    return {"n": n, "split_idx": split_idx, "train_end": train_end}


def horizon_row_audit(h: int, n: int, train_end: int, split_idx: int) -> Dict[str, int]:
    train_last = max(train_end - h, 0)
    test_last = max(n - h, split_idx)
    val_start = max(int((1.0 - Config.val_frac_within_train) * train_last), 1) if train_last > 0 else 0
    return {
        "horizon": h,
        "n_rows_total": int(n),
        "train_end_row": int(train_end),
        "split_idx_row": int(split_idx),
        "train_rows_before_val": int(train_last),
        "train_rows_inner": int(max(val_start, 0)),
        "val_rows_inner": int(max(train_last - val_start, 0)),
        "test_rows_outer": int(max(test_last - split_idx, 0)),
    }


def run_pipeline(df: pd.DataFrame, feat_df: pd.DataFrame, cfg: Config, out_dir: Path) -> None:
    X_all = feat_df.values
    y_log_all = df[cfg.target_col].astype(float).values
    y_cnt_all = np.clip(np.expm1(y_log_all), a_min=0, a_max=None)

    feature_names = feat_df.columns.tolist()
    idx = compute_split_indices(len(df), cfg)
    n, split_idx, train_end = idx["n"], idx["split_idx"], idx["train_end"]

    per_h_rows: List[Dict] = []
    row_audit_rows: List[Dict] = []
    split_manifest_rows: List[Dict] = []

    y_true_train_all, y_pred_train_all = [], []
    y_true_val_all, y_pred_val_all = [], []
    y_true_test_all, y_pred_test_all = [], []

    naive_test_all, seasonal_test_all = [], []

    train_pred_rows = []
    val_pred_rows = []
    test_pred_rows = []

    models: Dict[int, xgb.XGBRegressor] = {}

    for h in range(1, cfg.horizon + 1):
        row_audit_rows.append(horizon_row_audit(h, n, train_end, split_idx))

        y_shift_log = pd.Series(y_log_all).shift(-h).values

        train_last = train_end - h
        if train_last <= 5:
            continue

        X_tr = X_all[:train_last]
        y_tr_log = y_shift_log[:train_last]

        # inner validation = last 10% of training
        val_start = max(int((1.0 - cfg.val_frac_within_train) * train_last), 1)
        X_tr_in, y_tr_in = X_tr[:val_start], y_tr_log[:val_start]
        X_va_in, y_va_in = X_tr[val_start:], y_tr_log[val_start:]

        mask_tr = ~np.isnan(y_tr_in)
        mask_va = ~np.isnan(y_va_in)

        if mask_tr.sum() <= 5 or mask_va.sum() <= 1:
            continue

        y_tr_cnt = np.clip(np.expm1(y_tr_in[mask_tr]), a_min=0, a_max=None)
        thr = float(np.quantile(y_tr_cnt, 0.90)) if y_tr_cnt.size else 0.0
        w_tr = np.where(y_tr_cnt >= thr, 3.0, 1.0)

        model = make_model(cfg)
        model.fit(
            X_tr_in[mask_tr],
            y_tr_in[mask_tr],
            sample_weight=w_tr,
            eval_set=[(X_va_in[mask_va], y_va_in[mask_va])],
            verbose=False,
        )
        models[h] = model

        best_iteration = int(getattr(model, "best_iteration", model.n_estimators - 1))
        iter_range = (0, best_iteration + 1)

        # Train block predictions (entire outer-train slice for comparability with old notebook)
        yhat_tr_log = model.predict(X_tr, iteration_range=iter_range)
        yhat_tr = np.clip(np.expm1(yhat_tr_log), a_min=0, a_max=None)
        y_true_tr = np.clip(np.expm1(y_tr_log), a_min=0, a_max=None)
        mtrain = np.isfinite(y_true_tr)

        # Outer test block
        X_te = X_all[split_idx:n - h]
        y_te_log = y_shift_log[split_idx:n - h]
        mtest = ~np.isnan(y_te_log)

        if mtest.sum() == 0:
            continue

        yhat_te_log = model.predict(X_te[mtest], iteration_range=iter_range)
        yhat_te = np.clip(np.expm1(yhat_te_log), a_min=0, a_max=None)
        y_true_te = np.clip(np.expm1(y_te_log[mtest]), a_min=0, a_max=None)

        # Inner validation metrics
        yhat_va_log = model.predict(X_va_in[mask_va], iteration_range=iter_range)
        yhat_va = np.clip(np.expm1(yhat_va_log), a_min=0, a_max=None)
        y_true_va = np.clip(np.expm1(y_va_in[mask_va]), a_min=0, a_max=None)

        train_metrics = metrics_dict(y_true_tr[mtrain], yhat_tr[mtrain])
        val_metrics = metrics_dict(y_true_va, yhat_va)
        test_metrics = metrics_dict(y_true_te, yhat_te)

        # Baselines on test block
        base_idx = np.arange(split_idx, n - h)[mtest]
        naive_last = np.clip(np.expm1(y_log_all[base_idx]), a_min=0, a_max=None)

        seas_idx = base_idx - 12
        seas_valid = seas_idx >= 0
        fallback = naive_last
        seasonal12 = np.where(
            seas_valid,
            np.clip(np.expm1(y_log_all[np.clip(seas_idx, a_min=0, a_max=None)]), a_min=0, a_max=None),
            fallback,
        )

        naive_test_all.append(pd.DataFrame({"horizon": h, "y_true_count": y_true_te, "y_pred_count": naive_last, "baseline": "NaiveLast"}))
        seasonal_test_all.append(pd.DataFrame({"horizon": h, "y_true_count": y_true_te, "y_pred_count": seasonal12, "baseline": "SeasonalNaive12"}))

        # Append long predictions
        train_rows = pd.DataFrame({
            "row_idx": np.arange(train_last),
            "horizon": h,
            "split": "train_outer",
            "y_true_count": y_true_tr,
            "y_pred_count": yhat_tr,
        })
        val_rows = pd.DataFrame({
            "row_idx": np.arange(val_start, train_last)[mask_va],
            "horizon": h,
            "split": "val_inner",
            "y_true_count": y_true_va,
            "y_pred_count": yhat_va,
        })
        test_rows = pd.DataFrame({
            "row_idx": base_idx,
            "horizon": h,
            "split": "test_outer",
            "y_true_count": y_true_te,
            "y_pred_count": yhat_te,
            "naive_last": naive_last,
            "seasonal12": seasonal12,
            "outbreak_threshold": thr,
            "y_pred_is_outbreak": (yhat_te >= thr).astype(int),
            "y_true_is_outbreak": (y_true_te >= thr).astype(int),
        })
        # attach metadata when available
        for meta in ["Date", "District", "Month-year", "split"]:
            if meta in df.columns:
                train_rows[meta] = df.iloc[train_rows["row_idx"].values][meta].values
                val_rows[meta] = df.iloc[val_rows["row_idx"].values][meta].values
                test_rows[meta] = df.iloc[test_rows["row_idx"].values][meta].values

        train_pred_rows.append(train_rows)
        val_pred_rows.append(val_rows)
        test_pred_rows.append(test_rows)

        per_h_rows.append({
            "horizon": h,
            "best_iteration": best_iteration,
            "outbreak_threshold_q90_train": thr,
            "train_n": train_metrics["n"],
            "train_RMSE": train_metrics["RMSE"],
            "train_MAE": train_metrics["MAE"],
            "train_R2": train_metrics["R2"],
            "val_n": val_metrics["n"],
            "val_RMSE": val_metrics["RMSE"],
            "val_MAE": val_metrics["MAE"],
            "val_R2": val_metrics["R2"],
            "test_n": test_metrics["n"],
            "test_RMSE": test_metrics["RMSE"],
            "test_MAE": test_metrics["MAE"],
            "test_R2": test_metrics["R2"],
            "test_sMAPE": test_metrics["sMAPE"],
            "test_MBE": test_metrics["MBE"],
        })

        split_manifest_rows.append({
            "horizon": h,
            "train_end_row": train_end,
            "train_last_row_used": train_last - 1,
            "val_start_row_within_train": val_start,
            "test_start_row": split_idx,
            "test_end_row_used": n - h - 1,
        })

        y_true_train_all.extend(y_true_tr[mtrain].tolist())
        y_pred_train_all.extend(yhat_tr[mtrain].tolist())
        y_true_val_all.extend(y_true_va.tolist())
        y_pred_val_all.extend(yhat_va.tolist())
        y_true_test_all.extend(y_true_te.tolist())
        y_pred_test_all.extend(yhat_te.tolist())

    # overall summaries
    summary_rows = []
    summary_rows.append({"Model": "XGBoost", "Split": "train_outer", **metrics_dict(y_true_train_all, y_pred_train_all)})
    summary_rows.append({"Model": "XGBoost", "Split": "val_inner", **metrics_dict(y_true_val_all, y_pred_val_all)})
    summary_rows.append({"Model": "XGBoost", "Split": "test_outer", **metrics_dict(y_true_test_all, y_pred_test_all)})

    naive_df = pd.concat(naive_test_all, ignore_index=True) if naive_test_all else pd.DataFrame(columns=["y_true_count","y_pred_count"])
    seas_df = pd.concat(seasonal_test_all, ignore_index=True) if seasonal_test_all else pd.DataFrame(columns=["y_true_count","y_pred_count"])

    if len(naive_df):
        summary_rows.append({"Model": "NaiveLast", "Split": "test_outer", **metrics_dict(naive_df["y_true_count"], naive_df["y_pred_count"])})
    if len(seas_df):
        summary_rows.append({"Model": "SeasonalNaive12", "Split": "test_outer", **metrics_dict(seas_df["y_true_count"], seas_df["y_pred_count"])})

    summary_df = pd.DataFrame(summary_rows)
    per_h_df = pd.DataFrame(per_h_rows)
    row_audit_df = pd.DataFrame(row_audit_rows)
    split_manifest_df = pd.DataFrame(split_manifest_rows)

    # outputs
    summary_df.to_csv(out_dir / "xgb_summary.csv", index=False)
    per_h_df.to_csv(out_dir / "xgb_per_horizon.csv", index=False)
    row_audit_df.to_csv(out_dir / "xgb_row_audit.csv", index=False)
    split_manifest_df.to_csv(out_dir / "xgb_split_manifest.csv", index=False)

    if train_pred_rows:
        pd.concat(train_pred_rows, ignore_index=True).to_csv(out_dir / "xgb_train_predictions_long.csv", index=False)
    if val_pred_rows:
        pd.concat(val_pred_rows, ignore_index=True).to_csv(out_dir / "xgb_val_predictions_long.csv", index=False)
    if test_pred_rows:
        pd.concat(test_pred_rows, ignore_index=True).to_csv(out_dir / "xgb_test_predictions_long.csv", index=False)

    # Feature importance (mean gain across horizons)
    if models:
        gain_df = pd.DataFrame(index=feature_names)
        for h, model in models.items():
            booster = model.get_booster()
            gain = pd.Series(booster.get_score(importance_type="gain"), dtype=float)
            fmap = {f"f{i}": name for i, name in enumerate(feature_names)}
            gain_named = gain.rename(index=fmap).reindex(feature_names).fillna(0.0)
            gain_df[f"h{h}"] = gain_named.values
        gain_df["mean_gain"] = gain_df.mean(axis=1)
        gain_df.sort_values("mean_gain", ascending=False).to_csv(out_dir / "xgb_feature_importance_gain.csv")

    # run summary
    run_summary = {
        "input_path": str(cfg.input_path),
        "n_rows": int(len(df)),
        "n_features": int(len(feature_names)),
        "feature_names": feature_names,
        "target_col": cfg.target_col,
        "horizon": int(cfg.horizon),
        "test_frac": float(cfg.test_frac),
        "purge_gap_rows": int(cfg.purge_gap),
        "val_frac_within_train": float(cfg.val_frac_within_train),
        "metadata_dropped_from_features": sorted(list(META_COLS & set(df.columns))),
        "note": "Legacy manuscript-style direct multi-horizon XGBoost adapted to current input.",
    }
    (out_dir / "run_summary.txt").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Legacy manuscript-style XGBoost pipeline on current input file.")
    parser.add_argument("--input", type=str, default=Config.input_path)
    parser.add_argument("--output_dir", type=str, default=Config.output_dir)
    parser.add_argument("--target_col", type=str, default=Config.target_col)
    parser.add_argument("--horizon", type=int, default=Config.horizon)
    parser.add_argument("--test_frac", type=float, default=Config.test_frac)
    parser.add_argument("--purge_gap", type=int, default=Config.purge_gap)
    parser.add_argument("--val_frac_within_train", type=float, default=Config.val_frac_within_train)
    parser.add_argument("--random_state", type=int, default=Config.random_state)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    cfg.input_path = args.input
    cfg.output_dir = args.output_dir
    cfg.target_col = args.target_col
    cfg.horizon = args.horizon
    cfg.test_frac = args.test_frac
    cfg.purge_gap = args.purge_gap
    cfg.val_frac_within_train = args.val_frac_within_train
    cfg.random_state = args.random_state
    cfg.xgb_params["random_state"] = cfg.random_state

    out_dir = Path(cfg.output_dir)
    ensure_dir(out_dir)

    df = load_and_prepare_dataframe(cfg)
    feat_df = build_feature_dataframe(df, cfg)
    run_pipeline(df, feat_df, cfg, out_dir)

    archive = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
    print(f"Saved output folder: {out_dir}")
    print(f"Saved zip archive : {archive}")


if __name__ == "__main__":
    main()
