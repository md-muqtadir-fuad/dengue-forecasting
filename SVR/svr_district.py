#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct multi-horizon SVR for the dengue district-panel pipeline.

Expected input:
    A preprocessed CSV produced by the latest preprocessing pipeline, with:
    - one row per District-Date
    - selected numeric features
    - columns: District, Date, Month-year, split, Log_NoOfDenguePatients
    - split labels already assigned in preprocessing: train / val / purge / test

Important:
    The input file MUST keep purge rows. Purge rows are not used as train/val/test
    targets, but they are necessary as source-context rows for higher-horizon targets.
    If purge rows are removed before modeling, horizon semantics become invalid.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import re

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pandas.tseries.offsets import DateOffset
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


# ---------------------------------------------------------------------
# Pretty names for tables / plots
# ---------------------------------------------------------------------
column_renaming = {
    "denv4": "DENV-4",
    "Year": "Year",
    "AvgTemp_lag_3": "Avg. Temp (Lag 3)",
    "PopulationDensity": "Pop. Density",
    "MonthlyPrevailingWindDir_ENE": "Wind Dir. (ENE)",
    "Rainfall_lag_2": "Rainfall (Lag 2)",
    "Rainfall_lag_3": "Rainfall (Lag 3)",
    "MonthlyAvgSunshineHours_lag_1": "Sunshine Hrs (Lag 1)",
    "denv1_lag_1": "DENV-1 (Lag 1)",
    "Humidity_lag_1": "Humidity (Lag 1)",
    "Log_NoOfDenguePatients": "Log(Dengue Cases)",
    "Month_sin": "Month (Sin)",
    "Month_cos": "Month (Cos)",
    "District": "District",
    "Date": "Date",
    "Month-year": "Month-Year",
}


def pretty_column_name(col: str) -> str:
    if col in column_renaming:
        return column_renaming[col]

    c = str(col).strip()

    lag_match = re.match(r"^(.*)_lag_(\d+)$", c)
    if lag_match:
        base, lag_num = lag_match.groups()
        return f"{pretty_column_name(base)} (Lag {lag_num})"

    exact_map = {
        "NoOfDenguePatients": "Dengue Cases",
        "PopulationDensity": "Pop. Density",
        "MonthlyAvgSunshineHours": "Sunshine Hrs",
        "MonthlyPrevailingWindDir": "Wind Direction",
        "MonthlyPrevailingWindSpeed": "Wind Speed",
        "MonthlyAvgSeaLevelPressure": "Sea Level Pressure",
        "MonthlyAvgVisibility": "Visibility",
        "AvgTemp": "Avg. Temp",
        "MinTemp": "Min. Temp",
        "MaxTemp": "Max. Temp",
        "Rainfall": "Rainfall",
        "Humidity": "Humidity",
        "dominant": "Dominant Serotype",
        "Month_sin": "Month (Sin)",
        "Month_cos": "Month (Cos)",
    }
    if c in exact_map:
        return exact_map[c]

    if re.fullmatch(r"denv\d+", c.lower()):
        num = re.findall(r"\d+", c)[0]
        return f"DENV-{num}"

    if c.startswith("MonthlyPrevailingWindDir_"):
        suffix = c.replace("MonthlyPrevailingWindDir_", "").strip()
        return f"Wind Dir. ({suffix})"

    c = c.replace("_", " ")
    c = re.sub(r"\s+", " ", c).strip()
    c = re.sub(r"([a-z])([A-Z])", r"\1 \2", c)
    c = c.replace("Avg ", "Avg. ")
    c = c.replace("Min ", "Min. ")
    c = c.replace("Max ", "Max. ")
    c = c.replace("Pop Density", "Pop. Density")
    return c


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class Config:
    input_path: str
    output_dir: str
    target_col: str = "Log_NoOfDenguePatients"
    district_col: str = "District"
    date_col: str = "Date"
    monthyear_col: str = "Month-year"
    split_col: str = "split"
    use_district_ohe: bool = True
    horizon: int = 6
    use_tuning: bool = False
    tuning_iter: int = 24
    random_state: int = 42
    require_purge_rows: bool = True
    shap_max_background: int = 80
    shap_max_samples: int = 120
    shap_nsamples: int = 100
    base_params: Dict | None = None

    def __post_init__(self):
        if self.base_params is None:
            self.base_params = {
                "kernel": "rbf",
                "C": 10.0,
                "epsilon": 0.05,
                "gamma": "scale",
                "cache_size": 1000,
                "shrinking": True,
                "tol": 1e-3,
                "max_iter": -1,
            }


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred))) if len(y_true) else np.nan


def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    m = denom > 0
    if m.sum() == 0:
        return np.nan
    return float(np.mean(2 * np.abs(y_pred[m] - y_true[m]) / denom[m]) * 100.0)


def cvrmse(y_true, y_pred):
    mu = float(np.mean(y_true)) if len(y_true) else np.nan
    if not len(y_true) or np.isclose(mu, 0.0):
        return np.nan
    return rmse(y_true, y_pred) / mu


def nrmse(y_true, y_pred):
    if not len(y_true):
        return np.nan
    r = float(np.max(y_true) - np.min(y_true))
    if np.isclose(r, 0.0):
        return np.nan
    return rmse(y_true, y_pred) / r


def compute_metrics(y_true, y_pred, naive_denom=np.nan):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return {k: np.nan for k in ["RMSE", "MAE", "R2", "CVRMSE", "NRMSE", "sMAPE", "MBE", "MedAE", "MASE"]} | {"n": 0}
    r2 = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan
    mae = float(mean_absolute_error(y_true, y_pred))
    mase = mae / naive_denom if pd.notna(naive_denom) and not np.isclose(naive_denom, 0.0) else np.nan
    return {
        "n": int(len(y_true)),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae,
        "R2": float(r2) if pd.notna(r2) else np.nan,
        "CVRMSE": cvrmse(y_true, y_pred),
        "NRMSE": nrmse(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "MBE": float(np.mean(y_pred - y_true)),
        "MedAE": float(np.median(np.abs(y_pred - y_true))),
        "MASE": mase,
    }


def safe_classification_metrics(y_true_bin, y_pred_score, threshold):
    y_true_bin = np.asarray(y_true_bin).astype(int)
    y_pred_score = np.asarray(y_pred_score)
    y_pred_bin = (y_pred_score >= threshold).astype(int)
    out = {
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "specificity": np.nan,
        "roc_auc": np.nan,
        "pr_auc": np.nan,
    }
    if len(y_true_bin) == 0:
        return out
    out["precision"] = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    out["recall"] = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    out["f1"] = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    if len(np.unique(y_true_bin)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
        out["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        try:
            out["roc_auc"] = roc_auc_score(y_true_bin, y_pred_score)
        except Exception:
            pass
        try:
            out["pr_auc"] = average_precision_score(y_true_bin, y_pred_score)
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------
def load_dataset(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.input_path)
    df.columns = [c.strip() for c in df.columns]

    required = [cfg.target_col, cfg.district_col, cfg.date_col, cfg.split_col]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")

    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="raise")
    df[cfg.district_col] = df[cfg.district_col].astype(str).str.strip()
    df[cfg.split_col] = df[cfg.split_col].astype(str).str.strip().str.lower()
    df = df.sort_values([cfg.district_col, cfg.date_col]).reset_index(drop=True)
    return df


def validate_panel_structure(df: pd.DataFrame, cfg: Config) -> None:
    district_dates = df.groupby(cfg.district_col)[cfg.date_col].apply(lambda s: tuple(sorted(s.unique())))
    first = district_dates.iloc[0]
    inconsistent = [d for d, dates in district_dates.items() if dates != first]
    if inconsistent:
        raise ValueError(
            "All districts must share the same monthly date grid. "
            f"Inconsistent districts found: {inconsistent[:5]}"
        )

    all_dates = pd.Series(sorted(pd.to_datetime(df[cfg.date_col].unique())))
    month_index = all_dates.dt.year * 12 + all_dates.dt.month
    month_steps = month_index.diff().dropna()
    bad_gaps = month_steps[month_steps != 1]
    if not bad_gaps.empty:
        gap_positions = bad_gaps.index.tolist()
        examples = []
        for idx in gap_positions[:3]:
            examples.append((str(all_dates.iloc[idx - 1].date()), str(all_dates.iloc[idx].date())))
        raise ValueError(
            "Monthly continuity check failed. The modeling input has missing calendar months. "
            "This usually means purge rows were removed before modeling. "
            f"Gap examples: {examples}. "
            "Export a modeling file that keeps train, val, purge, and test rows."
        )


def build_date_split_map(df: pd.DataFrame, cfg: Config) -> Dict[pd.Timestamp, str]:
    valid_splits = {"train", "val", "purge", "test"}
    bad = sorted(set(df[cfg.split_col].unique()) - valid_splits)
    if bad:
        raise ValueError(f"Unexpected split labels: {bad}")

    per_date = df.groupby(cfg.date_col)[cfg.split_col].nunique()
    if (per_date > 1).any():
        clash_dates = per_date[per_date > 1].index.tolist()[:5]
        raise ValueError(f"A calendar date maps to multiple splits: {clash_dates}")

    date_to_split = (
        df[[cfg.date_col, cfg.split_col]]
        .drop_duplicates()
        .sort_values(cfg.date_col)
        .set_index(cfg.date_col)[cfg.split_col]
        .to_dict()
    )

    split_order = [date_to_split[d] for d in sorted(date_to_split)]
    order_map = {"train": 0, "val": 1, "purge": 2, "test": 3}
    numeric_order = [order_map[s] for s in split_order]
    if numeric_order != sorted(numeric_order):
        raise ValueError("Split order is not chronological. Expected train -> val -> purge -> test.")

    if cfg.require_purge_rows and "purge" not in set(split_order):
        raise ValueError(
            "No purge rows found in the modeling file. "
            "For direct multi-horizon forecasting, purge rows must remain in the dataset "
            "as source-context rows, even though they are excluded from evaluation."
        )
    return date_to_split


def split_windows_from_map(date_to_split: Dict[pd.Timestamp, str]) -> Dict[str, List[pd.Timestamp]]:
    windows = {"train": [], "val": [], "purge": [], "test": [], "all": sorted(date_to_split)}
    for dt in sorted(date_to_split):
        windows[date_to_split[dt]].append(dt)
    return windows


def get_feature_columns(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, List[str]]:
    refs = {cfg.district_col, cfg.date_col, cfg.target_col, cfg.split_col}
    if cfg.monthyear_col in df.columns:
        refs.add(cfg.monthyear_col)
    feature_cols = [c for c in df.columns if c not in refs and pd.api.types.is_numeric_dtype(df[c])]
    if cfg.use_district_ohe:
        dummies = pd.get_dummies(df[cfg.district_col], prefix="District", dtype=int)
        df = pd.concat([df, dummies], axis=1)
        feature_cols.extend(dummies.columns.tolist())
    return df, feature_cols


def build_lookup(df: pd.DataFrame, cfg: Config) -> Dict[Tuple[str, pd.Timestamp], float]:
    y_count = np.clip(np.expm1(df[cfg.target_col].astype(float).values), a_min=0, a_max=None)
    return {(d, t): float(y) for d, t, y in zip(df[cfg.district_col], df[cfg.date_col], y_count)}


def compute_panel_mase_denom(df: pd.DataFrame, cfg: Config) -> float:
    train = df[df[cfg.split_col] == "train"].copy().sort_values([cfg.district_col, cfg.date_col])
    train["y_count"] = np.clip(np.expm1(train[cfg.target_col].astype(float)), a_min=0, a_max=None)
    diffs = train.groupby(cfg.district_col)["y_count"].diff().abs().dropna()
    if len(diffs) == 0:
        return np.nan
    return float(diffs.mean())


# ---------------------------------------------------------------------
# Horizon sample construction
# ---------------------------------------------------------------------
def build_horizon_sample(
    df: pd.DataFrame,
    features: List[str],
    cfg: Config,
    h: int,
    date_to_split: Dict[pd.Timestamp, str],
) -> pd.DataFrame:
    base = df[[cfg.district_col, cfg.date_col, cfg.split_col, cfg.target_col] + features].copy()
    base = base.sort_values([cfg.district_col, cfg.date_col]).reset_index(drop=True)
    base = base.rename(columns={cfg.split_col: "source_split", cfg.target_col: "y_current_log"})
    base["TargetDate"] = base[cfg.date_col] + DateOffset(months=h)

    future = df[[cfg.district_col, cfg.date_col, cfg.target_col]].copy()
    future = future.rename(columns={cfg.date_col: "TargetDate", cfg.target_col: "y_future_log"})

    sample = base.merge(future, on=[cfg.district_col, "TargetDate"], how="left", validate="many_to_one")
    sample = sample.dropna(subset=["y_future_log"]).copy()

    sample["target_split"] = sample["TargetDate"].map(date_to_split)
    sample["y_current_count"] = np.clip(np.expm1(sample["y_current_log"].astype(float)), a_min=0, a_max=None)
    sample["y_future_count"] = np.clip(np.expm1(sample["y_future_log"].astype(float)), a_min=0, a_max=None)

    lookup = build_lookup(df, cfg)
    sample["naive_last_count"] = sample["y_current_count"]
    sample["seasonal12_count"] = sample.apply(
        lambda r: lookup.get((r[cfg.district_col], pd.Timestamp(r["TargetDate"]) - DateOffset(months=12)), np.nan),
        axis=1,
    )
    return sample


# ---------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------
def param_candidates(cfg: Config):
    grid = {
        "C": np.logspace(-1, 2.2, 25).tolist(),
        "epsilon": [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30],
        "gamma": ["scale", "auto", 0.005, 0.01, 0.03, 0.05, 0.10, 0.20],
    }
    return list(ParameterSampler(grid, n_iter=cfg.tuning_iter, random_state=cfg.random_state))


def make_pipeline(cfg: Config, params: Dict | None = None) -> Pipeline:
    p = dict(cfg.base_params)
    if params:
        p.update(params)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(**p)),
    ])
    return model


def fit_one_horizon(sample: pd.DataFrame, features: List[str], cfg: Config, h: int, out_dir: Path, mase_denom: float) -> Dict:
    train_df = sample[sample["target_split"] == "train"].copy()
    val_df = sample[sample["target_split"] == "val"].copy()
    test_df = sample[sample["target_split"] == "test"].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(f"Horizon {h}: one of train/val/test is empty after target-date assignment.")

    X_train, y_train_log = train_df[features], train_df["y_future_log"].astype(float).values
    X_val, y_val_log = val_df[features], val_df["y_future_log"].astype(float).values
    X_test = test_df[features]

    y_train_count = train_df["y_future_count"].values
    threshold = float(np.quantile(y_train_count, 0.90)) if len(y_train_count) else 0.0
    w_train = np.where(y_train_count >= threshold, 3.0, 1.0)

    best_params = None
    tuning_rows = []
    if cfg.use_tuning:
        best_score = np.inf
        for params in param_candidates(cfg):
            pipeline = make_pipeline(cfg, params)
            pipeline.fit(X_train, y_train_log, svr__sample_weight=w_train)
            pred_val = np.clip(np.expm1(pipeline.predict(X_val)), a_min=0, a_max=None)
            true_val = np.clip(np.expm1(y_val_log), a_min=0, a_max=None)
            score = mean_absolute_error(true_val, pred_val)
            row = dict(params)
            row.update({
                "horizon": h,
                "val_mae_count": score,
                "n_support_vectors": int(np.sum(pipeline.named_steps["svr"].n_support_)),
            })
            tuning_rows.append(row)
            if score < best_score:
                best_score, best_params = score, params
        pd.DataFrame(tuning_rows).sort_values("val_mae_count").to_csv(out_dir / f"h{h}_tuning_results.csv", index=False)

    pipeline = make_pipeline(cfg, best_params)
    pipeline.fit(X_train, y_train_log, svr__sample_weight=w_train)
    joblib.dump(pipeline, out_dir / f"svr_model_h{h}.joblib")

    def pred_block(df_block: pd.DataFrame, split_name: str) -> pd.DataFrame:
        y_pred_log = pipeline.predict(df_block[features])
        y_pred_count = np.clip(np.expm1(y_pred_log), a_min=0, a_max=None)
        out = df_block[[cfg.district_col, cfg.date_col, "TargetDate", "source_split", "target_split"]].copy()
        out["split"] = split_name
        out["horizon"] = h
        out["y_true_log"] = df_block["y_future_log"].values
        out["y_pred_log"] = y_pred_log
        out["y_true_count"] = df_block["y_future_count"].values
        out["y_pred_count"] = y_pred_count
        out["naive_last_count"] = df_block["naive_last_count"].values
        out["seasonal12_count"] = df_block["seasonal12_count"].values
        out["residual_count"] = out["y_pred_count"] - out["y_true_count"]
        out["abs_error"] = np.abs(out["residual_count"])
        out["sq_error"] = out["residual_count"] ** 2
        out["outbreak_threshold"] = threshold
        out["regime"] = np.where(out["y_true_count"] >= threshold, "Outbreak", "Normal")
        out["model"] = "SVR"
        return out

    pred_train = pred_block(train_df, "train")
    pred_val = pred_block(val_df, "val")
    pred_test = pred_block(test_df, "test")

    m_train = compute_metrics(pred_train["y_true_count"], pred_train["y_pred_count"], mase_denom)
    m_val = compute_metrics(pred_val["y_true_count"], pred_val["y_pred_count"], mase_denom)
    m_test = compute_metrics(pred_test["y_true_count"], pred_test["y_pred_count"], mase_denom)

    mask_naive = pred_test["naive_last_count"].notna()
    mask_seas = pred_test["seasonal12_count"].notna()
    b_naive = compute_metrics(pred_test.loc[mask_naive, "y_true_count"], pred_test.loc[mask_naive, "naive_last_count"], mase_denom)
    b_seas = compute_metrics(pred_test.loc[mask_seas, "y_true_count"], pred_test.loc[mask_seas, "seasonal12_count"], mase_denom)

    m_test["Skill_RMSE_vs_NaiveLast_pct"] = 100 * (1 - m_test["RMSE"] / b_naive["RMSE"]) if pd.notna(b_naive["RMSE"]) and not np.isclose(b_naive["RMSE"], 0.0) else np.nan
    m_test["Skill_RMSE_vs_Seasonal12_pct"] = 100 * (1 - m_test["RMSE"] / b_seas["RMSE"]) if pd.notna(b_seas["RMSE"]) and not np.isclose(b_seas["RMSE"], 0.0) else np.nan
    m_test["Skill_MAE_vs_NaiveLast_pct"] = 100 * (1 - m_test["MAE"] / b_naive["MAE"]) if pd.notna(b_naive["MAE"]) and not np.isclose(b_naive["MAE"], 0.0) else np.nan
    m_test["Skill_MAE_vs_Seasonal12_pct"] = 100 * (1 - m_test["MAE"] / b_seas["MAE"]) if pd.notna(b_seas["MAE"]) and not np.isclose(b_seas["MAE"], 0.0) else np.nan

    cls = safe_classification_metrics((pred_test["y_true_count"] >= threshold).astype(int), pred_test["y_pred_count"], threshold)

    perm = permutation_importance(
        pipeline,
        X_test,
        y_true := test_df["y_future_log"].astype(float).values,
        n_repeats=20,
        random_state=cfg.random_state,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    perm_imp = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False)
    perm_sd = pd.Series(perm.importances_std, index=features).reindex(perm_imp.index)

    # SHAP using a small model-agnostic sample for runtime control
    background = X_train.copy()
    if len(background) > cfg.shap_max_background:
        idx = np.linspace(0, len(background) - 1, cfg.shap_max_background, dtype=int)
        background = background.iloc[idx].copy()

    shap_X = X_test.copy()
    if len(shap_X) > cfg.shap_max_samples:
        idx = np.linspace(0, len(shap_X) - 1, cfg.shap_max_samples, dtype=int)
        shap_X = shap_X.iloc[idx].copy()

    def predict_fn(x):
        if isinstance(x, pd.DataFrame):
            arr = x[features]
        else:
            arr = pd.DataFrame(x, columns=features)
        return pipeline.predict(arr)

    explainer = shap.KernelExplainer(predict_fn, background, link="identity")
    shap_values = explainer.shap_values(shap_X, nsamples=cfg.shap_nsamples)
    shap_imp = pd.DataFrame(np.asarray(shap_values), columns=features).abs().mean().sort_values(ascending=False)

    support_count = int(np.sum(pipeline.named_steps["svr"].n_support_))
    support_ratio = support_count / len(X_train) if len(X_train) else np.nan

    return {
        "horizon": h,
        "pipeline": pipeline,
        "feature_cols": features,
        "params": {**cfg.base_params, **(best_params or {})},
        "threshold": threshold,
        "pred_train": pred_train,
        "pred_val": pred_val,
        "pred_test": pred_test,
        "metrics_train": m_train,
        "metrics_val": m_val,
        "metrics_test": m_test,
        "baseline_test_naive": b_naive,
        "baseline_test_seasonal12": b_seas,
        "classification_test": cls,
        "permutation_importance": perm_imp,
        "permutation_importance_sd": perm_sd,
        "shap_values": np.asarray(shap_values),
        "shap_test_X": shap_X,
        "shap_importance": shap_imp,
        "support_count": support_count,
        "support_ratio": support_ratio,
    }


# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------
def aggregate_predictions(results, split_name):
    return pd.concat([r[f"pred_{split_name}"] for r in results], ignore_index=True)


def save_row_audit(results, out_dir: Path):
    rows = []
    for r in results:
        for split in ["train", "val", "test"]:
            df_ = r[f"pred_{split}"]
            if df_.empty:
                rows.append({
                    "horizon": r["horizon"],
                    "split": split,
                    "n_rows": 0,
                    "n_source_dates": 0,
                    "n_target_dates": 0,
                    "n_districts": 0,
                    "source_date_start": pd.NaT,
                    "source_date_end": pd.NaT,
                    "target_date_start": pd.NaT,
                    "target_date_end": pd.NaT,
                })
            else:
                rows.append({
                    "horizon": r["horizon"],
                    "split": split,
                    "n_rows": int(len(df_)),
                    "n_source_dates": int(df_["Date"].nunique()),
                    "n_target_dates": int(df_["TargetDate"].nunique()),
                    "n_districts": int(df_["District"].nunique()),
                    "source_date_start": df_["Date"].min(),
                    "source_date_end": df_["Date"].max(),
                    "target_date_start": df_["TargetDate"].min(),
                    "target_date_end": df_["TargetDate"].max(),
                })
    pd.DataFrame(rows).sort_values(["horizon", "split"]).to_csv(out_dir / "svr_row_audit.csv", index=False)


def save_prediction_tables(results, out_dir: Path):
    pred_train = aggregate_predictions(results, "train")
    pred_val = aggregate_predictions(results, "val")
    pred_test = aggregate_predictions(results, "test")
    pred_train.to_csv(out_dir / "svr_train_predictions_long.csv", index=False)
    pred_val.to_csv(out_dir / "svr_val_predictions_long.csv", index=False)
    pred_test.to_csv(out_dir / "svr_test_predictions_long.csv", index=False)
    pred_test.to_csv(out_dir / "svr_test_residuals_long.csv", index=False)
    return {"train": pred_train, "val": pred_val, "test": pred_test}


def save_split_manifest(windows, results, out_dir: Path):
    rows = []
    for r in results:
        h = r["horizon"]
        for split in ["train", "val", "test"]:
            df_ = r[f"pred_{split}"]
            rows.append({
                "horizon": h,
                "split": split,
                "source_date_start": df_["Date"].min() if len(df_) else pd.NaT,
                "source_date_end": df_["Date"].max() if len(df_) else pd.NaT,
                "target_date_start": df_["TargetDate"].min() if len(df_) else pd.NaT,
                "target_date_end": df_["TargetDate"].max() if len(df_) else pd.NaT,
                "n_rows": int(len(df_)),
                "n_districts": int(df_["District"].nunique()) if len(df_) else 0,
            })
        rows.append({
            "horizon": h,
            "split": "purge_dates_available_in_input",
            "source_date_start": min(windows["purge"]) if windows["purge"] else pd.NaT,
            "source_date_end": max(windows["purge"]) if windows["purge"] else pd.NaT,
            "target_date_start": pd.NaT,
            "target_date_end": pd.NaT,
            "n_rows": np.nan,
            "n_districts": np.nan,
        })
    pd.DataFrame(rows).to_csv(out_dir / "svr_split_manifest.csv", index=False)


def save_metrics(results, preds, out_dir: Path, mase_denom: float):
    rows = []
    for split, df_ in preds.items():
        m = compute_metrics(df_["y_true_count"], df_["y_pred_count"], mase_denom)
        m.update({"Model": "SVR", "Split": split})
        rows.append(m)

    base1 = compute_metrics(preds["test"]["y_true_count"], preds["test"]["naive_last_count"], mase_denom)
    base1.update({"Model": "NaiveLast", "Split": "test"})
    rows.append(base1)

    mask_seas = preds["test"]["seasonal12_count"].notna()
    base2 = compute_metrics(preds["test"].loc[mask_seas, "y_true_count"], preds["test"].loc[mask_seas, "seasonal12_count"], mase_denom)
    base2.update({"Model": "SeasonalNaive12", "Split": "test"})
    rows.append(base2)

    pd.DataFrame(rows)[["Model", "Split", "n", "RMSE", "MAE", "R2", "CVRMSE", "NRMSE", "sMAPE", "MBE", "MedAE", "MASE"]].to_csv(out_dir / "svr_summary.csv", index=False)

    per_h = []
    for r in results:
        row = {
            "horizon": r["horizon"],
            "OutbreakThreshold": r["threshold"],
            "SupportVectors": r["support_count"],
            "SupportVectorRatio": r["support_ratio"],
        }
        row.update(r["metrics_test"])
        row.update(r["classification_test"])
        per_h.append(row)
    pd.DataFrame(per_h).sort_values("horizon").to_csv(out_dir / "svr_per_horizon.csv", index=False)

    regime_rows = []
    for h, g_h in preds["test"].groupby("horizon"):
        for regime, g in g_h.groupby("regime"):
            m = compute_metrics(g["y_true_count"], g["y_pred_count"], mase_denom)
            m.update({"horizon": h, "regime": regime})
            regime_rows.append(m)
    pd.DataFrame(regime_rows).sort_values(["horizon", "regime"]).to_csv(out_dir / "svr_regimewise_metrics.csv", index=False)

    district_rows = []
    for district, g in preds["test"].groupby("District"):
        m = compute_metrics(g["y_true_count"], g["y_pred_count"], mase_denom)
        m.update({"District": district})
        district_rows.append(m)
    pd.DataFrame(district_rows).sort_values("District").to_csv(out_dir / "svr_metrics_by_district.csv", index=False)

    dh_rows = []
    for (district, h), g in preds["test"].groupby(["District", "horizon"]):
        m = compute_metrics(g["y_true_count"], g["y_pred_count"], mase_denom)
        m.update({"District": district, "horizon": h})
        dh_rows.append(m)
    pd.DataFrame(dh_rows).sort_values(["District", "horizon"]).to_csv(out_dir / "svr_metrics_by_district_horizon.csv", index=False)

    cls_rows = []
    for r in results:
        pred = r["pred_test"]
        cls = safe_classification_metrics((pred["y_true_count"] >= r["threshold"]).astype(int), pred["y_pred_count"], r["threshold"])
        cls.update({"horizon": r["horizon"], "outbreak_threshold": r["threshold"], "n": len(pred)})
        cls_rows.append(cls)
    pd.DataFrame(cls_rows).sort_values("horizon").to_csv(out_dir / "svr_outbreak_classification_metrics.csv", index=False)

    sv_rows = []
    for r in results:
        sv_rows.append({
            "horizon": r["horizon"],
            "support_vectors": r["support_count"],
            "support_vector_ratio": r["support_ratio"],
        })
    pd.DataFrame(sv_rows).sort_values("horizon").to_csv(out_dir / "svr_support_vectors.csv", index=False)


def save_importance(results, out_dir: Path):
    features = results[0]["feature_cols"]
    perm_df = pd.DataFrame(index=features)
    perm_sd_df = pd.DataFrame(index=features)
    shap_df = pd.DataFrame(index=features)
    for r in results:
        perm_df[f"h{r['horizon']}"] = r["permutation_importance"].reindex(features).fillna(0.0).values
        perm_sd_df[f"h{r['horizon']}"] = r["permutation_importance_sd"].reindex(features).fillna(0.0).values
        shap_df[f"h{r['horizon']}"] = r["shap_importance"].reindex(features).fillna(0.0).values
    perm_df["mean_importance"] = perm_df.mean(axis=1)
    perm_sd_df["mean_sd"] = perm_sd_df.mean(axis=1)
    shap_df["mean_abs_shap"] = shap_df.mean(axis=1)
    perm_df.sort_values("mean_importance", ascending=False).to_csv(out_dir / "svr_feature_importance_permutation.csv")
    perm_sd_df.sort_values("mean_sd", ascending=False).to_csv(out_dir / "svr_feature_importance_permutation_sd.csv")
    shap_df.sort_values("mean_abs_shap", ascending=False).to_csv(out_dir / "svr_shap_aggregate_importance.csv")



def make_figures(results, preds, out_dir: Path):
    test = preds["test"]

    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        data=test,
        x="y_true_count",
        y="y_pred_count",
        hue="regime",
        palette={"Normal": "#1f77b4", "Outbreak": "#d62728"},
        alpha=0.7,
        edgecolor=None,
        s=60,
    )
    m_max = max(test["y_true_count"].max(), test["y_pred_count"].max()) * 1.5
    plt.plot([0, m_max], [0, m_max], linestyle="--", color="gray", label="Perfect Forecast")
    plt.xscale("symlog", linthresh=10)
    plt.yscale("symlog", linthresh=10)
    plt.xlabel("True Dengue Cases (SymLog Scale)")
    plt.ylabel("Predicted Dengue Cases (SymLog Scale)")
    plt.title("SVR Test: True vs Predicted")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_test.pdf", dpi=300)
    plt.close()

    per_h = pd.read_csv(out_dir / "svr_per_horizon.csv")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=per_h, x="horizon", y="RMSE")
    plt.title("Test RMSE by Horizon")
    plt.xlabel("Forecast Horizon (Months)")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(out_dir / "per_h_rmse.pdf", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=per_h, x="horizon", y="MAE")
    plt.title("Test MAE by Horizon")
    plt.xlabel("Forecast Horizon (Months)")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(out_dir / "per_h_mae.pdf", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=per_h, x="horizon", y="SupportVectorRatio")
    plt.title("Support Vector Ratio by Horizon")
    plt.xlabel("Forecast Horizon (Months)")
    plt.ylabel("Support Vector Ratio")
    plt.tight_layout()
    plt.savefig(out_dir / "support_vector_ratio_by_horizon.pdf", dpi=300)
    plt.close()

    val_df = preds["val"]
    h1_val = val_df[val_df["horizon"] == 1]
    h1_test = test[test["horizon"] == 1]
    if not h1_val.empty and not h1_test.empty:
        v_line = h1_val.groupby("TargetDate", as_index=False)[["y_true_count", "y_pred_count"]].mean()
        t_line = h1_test.groupby("TargetDate", as_index=False)[["y_true_count", "y_pred_count"]].mean()
        plt.figure(figsize=(14, 5))
        plt.plot(v_line["TargetDate"], v_line["y_true_count"], marker="o", linewidth=2, label="True Cases")
        plt.plot(v_line["TargetDate"], v_line["y_pred_count"], marker="X", linewidth=2, label="Predicted Cases")
        plt.plot(t_line["TargetDate"], t_line["y_true_count"], marker="o", linewidth=2)
        plt.plot(t_line["TargetDate"], t_line["y_pred_count"], marker="X", linewidth=2)
        split_date = t_line["TargetDate"].min()
        plt.axvline(x=split_date, color="black", linestyle="--", linewidth=1.5, label="Val/Test Split")
        plt.title("Comparative Timeline: Validation vs Test (Mean over Districts)")
        plt.xlabel("Target Date")
        plt.ylabel("Dengue Cases")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(out_dir / "h1_comparative_timeline.pdf", dpi=300)
        plt.close()

    if not h1_test.empty:
        g = sns.FacetGrid(h1_test, col="District", col_wrap=4, sharey=False, height=3.5, aspect=1.2)
        g.map_dataframe(sns.lineplot, x="TargetDate", y="y_true_count", marker="o", label="True")
        g.map_dataframe(sns.lineplot, x="TargetDate", y="y_pred_count", marker="X", label="Predicted")
        g.set_titles(col_template="{col_name}")
        g.set_axis_labels("Target Date", "Cases")
        for ax in g.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(45)
        g.add_legend(title="")
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle("Horizon 1 Forecast by District (Test Window)")
        g.savefig(out_dir / "h1_district_grid.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    err = test.groupby(["District", "horizon"], as_index=False)["abs_error"].mean().pivot(index="District", columns="horizon", values="abs_error")
    plt.figure(figsize=(10, 8))
    sns.heatmap(err, annot=True, fmt=".1f", cmap="YlOrRd", annot_kws={"size": 11})
    plt.title("Mean Absolute Error by District and Horizon")
    plt.xlabel("Horizon")
    plt.ylabel("District")
    plt.tight_layout()
    plt.savefig(out_dir / "error_heatmap.pdf", dpi=300)
    plt.close()

    edge_h = sorted({1, max(r["horizon"] for r in results)})
    for h in edge_h:
        r = [x for x in results if x["horizon"] == h][0]
        shap_X_pretty = r["shap_test_X"].rename(columns=pretty_column_name)

        plt.figure()
        shap.summary_plot(r["shap_values"], shap_X_pretty, show=False)
        plt.title(f"SHAP Summary (Horizon {h})")
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_beeswarm_h{h}.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure()
        shap.summary_plot(r["shap_values"], shap_X_pretty, plot_type="bar", show=False)
        plt.title(f"Mean Absolute SHAP Importance (Horizon {h})")
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_importance_bar_h{h}.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    perm = pd.read_csv(out_dir / "svr_feature_importance_permutation.csv", index_col=0)["mean_importance"].sort_values(ascending=False).head(20).iloc[::-1]
    perm.index = perm.index.map(pretty_column_name)
    plt.figure(figsize=(10, 8))
    perm.plot(kind="barh")
    plt.title("Top 20 Mean Permutation Importance (All Horizons)")
    plt.xlabel("Mean Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance_permutation_bar.pdf", dpi=300)
    plt.close()

    sagg = pd.read_csv(out_dir / "svr_shap_aggregate_importance.csv", index_col=0)["mean_abs_shap"].sort_values(ascending=False).head(20).iloc[::-1]
    sagg.index = sagg.index.map(pretty_column_name)
    plt.figure(figsize=(10, 8))
    sagg.plot(kind="barh")
    plt.title("Top 20 Mean Absolute SHAP Importance (All Horizons)")
    plt.xlabel("Mean |SHAP|")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_importance_bar.pdf", dpi=300)
    plt.close()


def write_run_summary(cfg, windows, features, out_dir: Path):
    lines = [
        "SVR run summary",
        "=" * 80,
        f"Input file: {cfg.input_path}",
        f"Target column: {cfg.target_col}",
        f"Feature count: {len(features)}",
        f"Horizon: {cfg.horizon}",
        f"Use district one-hot: {cfg.use_district_ohe}",
        f"Use tuning: {cfg.use_tuning}",
        f"Require purge rows: {cfg.require_purge_rows}",
        f"SHAP background cap: {cfg.shap_max_background}",
        f"SHAP sample cap: {cfg.shap_max_samples}",
        f"SHAP nsamples: {cfg.shap_nsamples}",
        "",
        "Date windows from preprocessing split column",
        "-" * 80,
    ]
    for key in ["train", "val", "purge", "test"]:
        dates = windows[key]
        if dates:
            lines.append(f"{key}: {min(dates)} -> {max(dates)} ({len(dates)} months)")
        else:
            lines.append(f"{key}: <empty>")
    lines += ["", "Features used", "-" * 80] + features
    (out_dir / "run_summary.txt").write_text("\n".join(map(str, lines)), encoding="utf-8")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Direct multi-horizon SVR for the dengue district-panel pipeline.")
    p.add_argument("--input", type=str, default="./data/raw/prime_dataset_model_input_with_purge.csv")
    p.add_argument("--output_dir", type=str, default="SVR/outputs")
    p.add_argument("--target_col", type=str, default="Log_NoOfDenguePatients")
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--use_district_ohe", dest="use_district_ohe", action="store_true")
    p.add_argument("--no_district_ohe", dest="use_district_ohe", action="store_false")
    p.set_defaults(use_district_ohe=True)
    p.add_argument("--use_tuning", action="store_true")
    p.add_argument("--tuning_iter", type=int, default=24)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--shap_max_background", type=int, default=80)
    p.add_argument("--shap_max_samples", type=int, default=120)
    p.add_argument("--shap_nsamples", type=int, default=100)
    a = p.parse_args()
    return Config(
        input_path=a.input,
        output_dir=a.output_dir,
        target_col=a.target_col,
        horizon=a.horizon,
        use_district_ohe=a.use_district_ohe,
        use_tuning=a.use_tuning,
        tuning_iter=a.tuning_iter,
        random_state=a.random_state,
        shap_max_background=a.shap_max_background,
        shap_max_samples=a.shap_max_samples,
        shap_nsamples=a.shap_nsamples,
    )


def main():
    cfg = parse_args()
    out_dir = Path(cfg.output_dir)
    ensure_dir(out_dir)
    save_json(cfg.__dict__, out_dir / "run_config.json")

    df = load_dataset(cfg)
    validate_panel_structure(df, cfg)
    date_to_split = build_date_split_map(df, cfg)
    windows = split_windows_from_map(date_to_split)
    df, features = get_feature_columns(df, cfg)
    if not features:
        raise ValueError("No usable numeric predictor columns found after explicit dropping.")

    mase_denom = compute_panel_mase_denom(df, cfg)
    write_run_summary(cfg, windows, features, out_dir)

    results = []
    best_params = {}
    for h in range(1, cfg.horizon + 1):
        sample = build_horizon_sample(df, features, cfg, h, date_to_split)
        res = fit_one_horizon(sample, features, cfg, h, out_dir, mase_denom)
        results.append(res)
        best_params[f"h{h}"] = {
            "params": res["params"],
            "outbreak_threshold": res["threshold"],
            "support_vectors": res["support_count"],
            "support_vector_ratio": res["support_ratio"],
        }

    save_json(best_params, out_dir / "svr_best_params.json")
    save_split_manifest(windows, results, out_dir)
    save_row_audit(results, out_dir)
    preds = save_prediction_tables(results, out_dir)
    save_metrics(results, preds, out_dir, mase_denom)
    save_importance(results, out_dir)
    make_figures(results, preds, out_dir)
    archive = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
    print(f"Saved output folder: {out_dir}")
    print(f"Saved zip archive : {archive}")


if __name__ == "__main__":
    main()
