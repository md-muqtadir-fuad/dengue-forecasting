#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct multi-horizon CatBoost for a grouped district-level monthly dengue panel.

Main idea
---------
- One row = one District-Date.
- Preprocessing already created lagged predictors and assigned split labels.
- The model is fit on the pooled district-month panel.
- District identity is included directly as a categorical feature.
- National forecasts are produced by summing district forecasts by target month.

Expected input columns
----------------------
Required:
    District, Date, split, Log_NoOfDenguePatients
Optional but supported:
    Month-year and any number of predictors already prepared in preprocessing.

Important
---------
- The input file MUST contain purge rows. Purge rows are not evaluation targets, but
  they are necessary as source-context rows for future horizons.
- This script assumes the panel has a common monthly grid across districts.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor, Pool
from pandas.tseries.offsets import DateOffset
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

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


# ---------------------------------------------------------------------
# Pretty names for tables / plots
# ---------------------------------------------------------------------
COLUMN_RENAMING = {
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
    if col in COLUMN_RENAMING:
        return COLUMN_RENAMING[col]

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
    use_district_cat: bool = True
    keep_object_features: bool = True
    horizon: int = 6
    use_tuning: bool = False
    tuning_iter: int = 16
    random_state: int = 42
    require_purge_rows: bool = True
    blocked_features: List[str] = field(default_factory=lambda: ["Year", "Month_sin", "Month_cos"])
    outbreak_q: float = 0.90
    extreme_outbreak_q: float = 0.97
    outbreak_weight: float = 5.0
    extreme_outbreak_weight: float = 12.0
    base_params: Optional[Dict] = None

    def __post_init__(self):
        if self.base_params is None:
            self.base_params = {
                "loss_function": "RMSE",
                "eval_metric": "RMSE",
                "iterations": 3000,
                "learning_rate": 0.03,
                "depth": 6,
                "l2_leaf_reg": 5.0,
                "random_strength": 1.0,
                "bootstrap_type": "Bayesian",
                "bagging_temperature": 1.0,
                "random_seed": self.random_state,
                "allow_writing_files": False,
            }


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred))) if len(y_true) else np.nan


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100.0)


def cvrmse(y_true, y_pred) -> float:
    mu = float(np.mean(y_true)) if len(y_true) else np.nan
    if not len(y_true) or np.isclose(mu, 0.0):
        return np.nan
    return rmse(y_true, y_pred) / mu


def nrmse(y_true, y_pred) -> float:
    if not len(y_true):
        return np.nan
    value_range = float(np.max(y_true) - np.min(y_true))
    if np.isclose(value_range, 0.0):
        return np.nan
    return rmse(y_true, y_pred) / value_range


def compute_metrics(y_true, y_pred, naive_denom=np.nan) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        base = {k: np.nan for k in ["RMSE", "MAE", "R2", "CVRMSE", "NRMSE", "sMAPE", "MBE", "MedAE", "MASE"]}
        base["n"] = 0
        return base

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


def safe_classification_metrics(y_true_bin, y_pred_score, threshold) -> Dict[str, float]:
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

    all_dates = pd.Series(sorted(df[cfg.date_col].unique()))
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


def get_feature_columns(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, List[str], List[str]]:
    refs = {cfg.date_col, cfg.target_col, cfg.split_col}
    if cfg.monthyear_col in df.columns:
        refs.add(cfg.monthyear_col)

    blocked = set(cfg.blocked_features)
    feature_cols: List[str] = []
    cat_feature_cols: List[str] = []

    for c in df.columns:
        if c in refs or c in blocked:
            continue
        if c == cfg.district_col:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            feature_cols.append(c)
        elif cfg.keep_object_features:
            df[c] = df[c].astype(str).fillna("MISSING")
            feature_cols.append(c)
            cat_feature_cols.append(c)

    if cfg.use_district_cat:
        df[cfg.district_col] = df[cfg.district_col].astype(str).fillna("MISSING")
        feature_cols.append(cfg.district_col)
        cat_feature_cols.append(cfg.district_col)

    # stable de-duplication while preserving order
    feature_cols = list(dict.fromkeys(feature_cols))
    cat_feature_cols = [c for c in dict.fromkeys(cat_feature_cols) if c in feature_cols]
    return df, feature_cols, cat_feature_cols


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
    base_cols = [cfg.district_col, cfg.date_col, cfg.split_col, cfg.target_col]
    base_feature_cols = [c for c in features if c not in base_cols]
    base = df[base_cols + base_feature_cols].copy()
    base = base.sort_values([cfg.district_col, cfg.date_col]).reset_index(drop=True)
    rows_before_merge = len(base)
    base = base.rename(columns={cfg.split_col: "source_split", cfg.target_col: "y_current_log"})
    base["TargetDate"] = base[cfg.date_col] + DateOffset(months=h)

    future = df[[cfg.district_col, cfg.date_col, cfg.target_col]].copy()
    future = future.rename(columns={cfg.date_col: "TargetDate", cfg.target_col: "y_future_log"})

    sample = base.merge(future, on=[cfg.district_col, "TargetDate"], how="left", validate="many_to_one")
    rows_after_merge = len(sample)
    rows_without_future_target = int(sample["y_future_log"].isna().sum())
    sample = sample.dropna(subset=["y_future_log"]).copy()

    sample["target_split"] = sample["TargetDate"].map(date_to_split)
    sample["y_current_count"] = np.clip(np.expm1(sample["y_current_log"].astype(float)), a_min=0, a_max=None)
    sample["y_future_count"] = np.clip(np.expm1(sample["y_future_log"].astype(float)), a_min=0, a_max=None)

    # Use TARGET month seasonality for direct h-step forecasting.
    sample["TargetMonth"] = pd.to_datetime(sample["TargetDate"]).dt.month.astype(int)
    sample["TargetMonth_sin"] = np.sin(2 * np.pi * sample["TargetMonth"] / 12.0)
    sample["TargetMonth_cos"] = np.cos(2 * np.pi * sample["TargetMonth"] / 12.0)

    lookup = build_lookup(df, cfg)
    sample["naive_last_count"] = sample["y_current_count"]
    sample["seasonal12_count"] = sample.apply(
        lambda r: lookup.get((r[cfg.district_col], pd.Timestamp(r["TargetDate"]) - DateOffset(months=12)), np.nan),
        axis=1,
    )

    sample.attrs["row_audit"] = {
        "horizon": h,
        "rows_before_merge": int(rows_before_merge),
        "rows_after_merge": int(rows_after_merge),
        "rows_without_future_target": int(rows_without_future_target),
        "rows_kept": int(len(sample)),
        "train_rows": int((sample["target_split"] == "train").sum()),
        "val_rows": int((sample["target_split"] == "val").sum()),
        "test_rows": int((sample["target_split"] == "test").sum()),
    }
    return sample


# ---------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------
def param_candidates(cfg: Config):
    grid = {
        "iterations": [800, 1200, 1800, 2500, 3500],
        "learning_rate": [0.01, 0.02, 0.03, 0.05],
        "depth": [4, 5, 6, 7, 8],
        "l2_leaf_reg": [3.0, 5.0, 7.0, 10.0, 15.0],
        "random_strength": [0.5, 1.0, 2.0, 4.0],
        "bagging_temperature": [0.0, 0.5, 1.0, 2.0, 4.0],
    }
    return list(ParameterSampler(grid, n_iter=cfg.tuning_iter, random_state=cfg.random_state))


def make_model(cfg: Config, params: Optional[Dict] = None) -> CatBoostRegressor:
    p = dict(cfg.base_params)
    if params:
        p.update(params)
    p["random_seed"] = cfg.random_state
    return CatBoostRegressor(**p)


def make_pool(df_block: pd.DataFrame, features: List[str], cat_feature_cols: List[str], label=None, weight=None) -> Pool:
    cat_idx = [features.index(c) for c in cat_feature_cols if c in features]
    return Pool(
        data=df_block[features],
        label=label,
        weight=weight,
        cat_features=cat_idx,
    )


def fit_one_horizon(
    sample: pd.DataFrame,
    features: List[str],
    cat_feature_cols: List[str],
    cfg: Config,
    h: int,
    out_dir: Path,
    mase_denom: float,
) -> Dict:
    train_df = sample[sample["target_split"] == "train"].copy()
    val_df = sample[sample["target_split"] == "val"].copy()
    test_df = sample[sample["target_split"] == "test"].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(f"Horizon {h}: one of train/val/test is empty after target-date assignment.")

    horizon_features = [f for f in features if f not in {"Month_sin", "Month_cos"}]
    horizon_features += ["TargetMonth_sin", "TargetMonth_cos"]
    horizon_features = list(dict.fromkeys(horizon_features))
    horizon_cat_feature_cols = [c for c in cat_feature_cols if c in horizon_features]

    y_train_log = train_df["y_future_log"].astype(float).values
    y_val_log = val_df["y_future_log"].astype(float).values

    y_train_count = train_df["y_future_count"].values
    q90 = float(np.quantile(y_train_count, cfg.outbreak_q)) if len(y_train_count) else 0.0
    q97 = float(np.quantile(y_train_count, cfg.extreme_outbreak_q)) if len(y_train_count) else 0.0
    threshold = q90
    w_train = np.where(
        y_train_count >= q97, cfg.extreme_outbreak_weight,
        np.where(y_train_count >= q90, cfg.outbreak_weight, 1.0)
    )

    train_pool = make_pool(train_df, horizon_features, horizon_cat_feature_cols, label=y_train_log, weight=w_train)
    val_pool = make_pool(val_df, horizon_features, horizon_cat_feature_cols, label=y_val_log)
    test_pool = make_pool(test_df, horizon_features, horizon_cat_feature_cols, label=test_df["y_future_log"].astype(float).values)

    best_params = None
    tuning_rows = []
    if cfg.use_tuning:
        best_score = np.inf
        for params in param_candidates(cfg):
            model = make_model(cfg, params)
            model.fit(
                train_pool,
                eval_set=val_pool,
                use_best_model=True,
                early_stopping_rounds=100,
                verbose=False,
            )
            pred_val = np.clip(np.expm1(model.predict(val_pool)), a_min=0, a_max=None)
            true_val = np.clip(np.expm1(y_val_log), a_min=0, a_max=None)
            score = mean_absolute_error(true_val, pred_val)
            row = dict(params)
            row.update({
                "horizon": h,
                "val_mae_count": score,
                "best_iteration": int(getattr(model, "best_iteration_", model.tree_count_)),
            })
            tuning_rows.append(row)
            if score < best_score:
                best_score, best_params = score, params
        pd.DataFrame(tuning_rows).sort_values("val_mae_count").to_csv(out_dir / f"h{h}_catboost_tuning_results.csv", index=False)

    model = make_model(cfg, best_params)
    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        early_stopping_rounds=100,
        verbose=False,
    )
    model.save_model(str(out_dir / f"catboost_model_h{h}.cbm"))

    def pred_block(df_block: pd.DataFrame, pool_block: Pool, split_name: str) -> pd.DataFrame:
        y_pred_log = model.predict(pool_block)
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
        out["model"] = "CatBoost"
        return out

    pred_train = pred_block(train_df, train_pool, "train")
    pred_val = pred_block(val_df, val_pool, "val")
    pred_test = pred_block(test_df, test_pool, "test")

    m_train = compute_metrics(pred_train["y_true_count"], pred_train["y_pred_count"], mase_denom)
    m_val = compute_metrics(pred_val["y_true_count"], pred_val["y_pred_count"], mase_denom)
    m_test = compute_metrics(pred_test["y_true_count"], pred_test["y_pred_count"], mase_denom)

    mask_naive = pred_test["naive_last_count"].notna()
    mask_seas = pred_test["seasonal12_count"].notna()
    b_naive = compute_metrics(pred_test.loc[mask_naive, "y_true_count"], pred_test.loc[mask_naive, "naive_last_count"], mase_denom)
    b_seas = compute_metrics(pred_test.loc[mask_seas, "y_true_count"], pred_test.loc[mask_seas, "seasonal12_count"], mase_denom)

    m_test["Skill_RMSE_vs_NaiveLast_pct"] = (
        100 * (1 - m_test["RMSE"] / b_naive["RMSE"]) if pd.notna(b_naive["RMSE"]) and not np.isclose(b_naive["RMSE"], 0.0) else np.nan
    )
    m_test["Skill_RMSE_vs_Seasonal12_pct"] = (
        100 * (1 - m_test["RMSE"] / b_seas["RMSE"]) if pd.notna(b_seas["RMSE"]) and not np.isclose(b_seas["RMSE"], 0.0) else np.nan
    )
    m_test["Skill_MAE_vs_NaiveLast_pct"] = (
        100 * (1 - m_test["MAE"] / b_naive["MAE"]) if pd.notna(b_naive["MAE"]) and not np.isclose(b_naive["MAE"], 0.0) else np.nan
    )
    m_test["Skill_MAE_vs_Seasonal12_pct"] = (
        100 * (1 - m_test["MAE"] / b_seas["MAE"]) if pd.notna(b_seas["MAE"]) and not np.isclose(b_seas["MAE"], 0.0) else np.nan
    )

    cls = safe_classification_metrics((pred_test["y_true_count"] >= threshold).astype(int), pred_test["y_pred_count"], threshold)

    pred_importance = pd.Series(model.get_feature_importance(type="PredictionValuesChange"), index=horizon_features).sort_values(ascending=False)

    shap_df = test_df[horizon_features].copy()
    if len(shap_df) > 300:
        idx = np.linspace(0, len(shap_df) - 1, 300, dtype=int)
        shap_df = shap_df.iloc[idx].copy()
    shap_pool = make_pool(shap_df.assign(**{}), horizon_features, horizon_cat_feature_cols)
    shap_values_full = model.get_feature_importance(shap_pool, type="ShapValues")
    shap_values = shap_values_full[:, :-1]
    shap_imp = pd.Series(np.abs(shap_values).mean(axis=0), index=horizon_features).sort_values(ascending=False)

    return {
        "horizon": h,
        "model": model,
        "feature_cols": horizon_features,
        "cat_feature_cols": horizon_cat_feature_cols,
        "params": model.get_params(),
        "best_iteration": int(getattr(model, "best_iteration_", model.tree_count_)),
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
        "pred_importance": pred_importance,
        "shap_values": shap_values,
        "shap_test_X": shap_df,
        "shap_importance": shap_imp,
        "row_audit": sample.attrs.get("row_audit", {}),
    }


# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------
def aggregate_predictions(results: Sequence[Dict], split_name: str) -> pd.DataFrame:
    return pd.concat([r[f"pred_{split_name}"] for r in results], ignore_index=True)


def aggregate_national_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        pred_df.groupby(["split", "horizon", "TargetDate"], as_index=False)
        [["y_true_count", "y_pred_count", "naive_last_count", "seasonal12_count"]]
        .sum()
    )
    grouped["residual_count"] = grouped["y_pred_count"] - grouped["y_true_count"]
    grouped["abs_error"] = np.abs(grouped["residual_count"])
    grouped["sq_error"] = grouped["residual_count"] ** 2
    return grouped


def save_prediction_tables(results: Sequence[Dict], out_dir: Path) -> Dict[str, pd.DataFrame]:
    pred_train = aggregate_predictions(results, "train")
    pred_val = aggregate_predictions(results, "val")
    pred_test = aggregate_predictions(results, "test")

    pred_train.to_csv(out_dir / "catboost_train_predictions_long.csv", index=False)
    pred_val.to_csv(out_dir / "catboost_val_predictions_long.csv", index=False)
    pred_test.to_csv(out_dir / "catboost_test_predictions_long.csv", index=False)
    pred_test.to_csv(out_dir / "catboost_test_residuals_long.csv", index=False)

    national_train = aggregate_national_predictions(pred_train)
    national_val = aggregate_national_predictions(pred_val)
    national_test = aggregate_national_predictions(pred_test)
    national_train.to_csv(out_dir / "catboost_train_predictions_national.csv", index=False)
    national_val.to_csv(out_dir / "catboost_val_predictions_national.csv", index=False)
    national_test.to_csv(out_dir / "catboost_test_predictions_national.csv", index=False)

    return {
        "train": pred_train,
        "val": pred_val,
        "test": pred_test,
        "national_train": national_train,
        "national_val": national_val,
        "national_test": national_test,
    }


def save_row_audit(results: Sequence[Dict], out_dir: Path) -> None:
    rows = []
    for r in results:
        rows.append(r.get("row_audit", {}))
    if rows:
        pd.DataFrame(rows).sort_values("horizon").to_csv(out_dir / "catboost_row_audit.csv", index=False)


def save_split_manifest(windows: Dict[str, List[pd.Timestamp]], results: Sequence[Dict], out_dir: Path) -> None:
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
    pd.DataFrame(rows).to_csv(out_dir / "catboost_split_manifest.csv", index=False)


def save_metrics(results: Sequence[Dict], preds: Dict[str, pd.DataFrame], out_dir: Path, mase_denom: float) -> None:
    rows = []
    for split in ["train", "val", "test"]:
        m = compute_metrics(preds[split]["y_true_count"], preds[split]["y_pred_count"], mase_denom)
        m.update({"Model": "CatBoost", "Split": split})
        rows.append(m)

    base1 = compute_metrics(preds["test"]["y_true_count"], preds["test"]["naive_last_count"], mase_denom)
    base1.update({"Model": "NaiveLast", "Split": "test"})
    rows.append(base1)

    mask_seas = preds["test"]["seasonal12_count"].notna()
    base2 = compute_metrics(preds["test"].loc[mask_seas, "y_true_count"], preds["test"].loc[mask_seas, "seasonal12_count"], mase_denom)
    base2.update({"Model": "SeasonalNaive12", "Split": "test"})
    rows.append(base2)

    pd.DataFrame(rows)[["Model", "Split", "n", "RMSE", "MAE", "R2", "CVRMSE", "NRMSE", "sMAPE", "MBE", "MedAE", "MASE"]].to_csv(
        out_dir / "catboost_summary.csv", index=False
    )

    per_h = []
    for r in results:
        row = {"horizon": r["horizon"], "OutbreakThreshold": r["threshold"], "BestIteration": r["best_iteration"]}
        row.update(r["metrics_test"])
        row.update(r["classification_test"])
        per_h.append(row)
    pd.DataFrame(per_h).sort_values("horizon").to_csv(out_dir / "catboost_per_horizon.csv", index=False)

    regime_rows = []
    for h, g_h in preds["test"].groupby("horizon"):
        for regime, g in g_h.groupby("regime"):
            m = compute_metrics(g["y_true_count"], g["y_pred_count"], mase_denom)
            m.update({"horizon": h, "regime": regime})
            regime_rows.append(m)
    pd.DataFrame(regime_rows).sort_values(["horizon", "regime"]).to_csv(out_dir / "catboost_regimewise_metrics.csv", index=False)

    district_rows = []
    for district, g in preds["test"].groupby("District"):
        m = compute_metrics(g["y_true_count"], g["y_pred_count"], mase_denom)
        m.update({"District": district})
        district_rows.append(m)
    pd.DataFrame(district_rows).sort_values("District").to_csv(out_dir / "catboost_metrics_by_district.csv", index=False)

    national_rows = []
    for split in ["national_train", "national_val", "national_test"]:
        m = compute_metrics(preds[split]["y_true_count"], preds[split]["y_pred_count"], naive_denom=np.nan)
        m.update({"Model": "CatBoost_from_district_sum", "Split": split.replace("national_", "")})
        national_rows.append(m)
    base1n = compute_metrics(preds["national_test"]["y_true_count"], preds["national_test"]["naive_last_count"], naive_denom=np.nan)
    base1n.update({"Model": "NaiveLast", "Split": "test"})
    national_rows.append(base1n)
    mask_seas_n = preds["national_test"]["seasonal12_count"].notna()
    base2n = compute_metrics(
        preds["national_test"].loc[mask_seas_n, "y_true_count"],
        preds["national_test"].loc[mask_seas_n, "seasonal12_count"],
        naive_denom=np.nan,
    )
    base2n.update({"Model": "SeasonalNaive12", "Split": "test"})
    national_rows.append(base2n)
    pd.DataFrame(national_rows)[["Model", "Split", "n", "RMSE", "MAE", "R2", "CVRMSE", "NRMSE", "sMAPE", "MBE", "MedAE", "MASE"]].to_csv(
        out_dir / "catboost_summary_national_from_district_sum.csv", index=False
    )


def save_importance(results: Sequence[Dict], out_dir: Path) -> None:
    features = results[0]["feature_cols"]
    pred_imp_df = pd.DataFrame(index=features)
    shap_df = pd.DataFrame(index=features)
    for r in results:
        pred_imp_df[f"h{r['horizon']}"] = r["pred_importance"].reindex(features).fillna(0.0).values
        shap_df[f"h{r['horizon']}"] = r["shap_importance"].reindex(features).fillna(0.0).values
    pred_imp_df["mean_importance"] = pred_imp_df.mean(axis=1)
    shap_df["mean_abs_shap"] = shap_df.mean(axis=1)
    pred_imp_df.sort_values("mean_importance", ascending=False).to_csv(out_dir / "catboost_feature_importance_predictionvalueschange.csv")
    shap_df.sort_values("mean_abs_shap", ascending=False).to_csv(out_dir / "catboost_shap_aggregate_importance.csv")


def make_figures(results: Sequence[Dict], preds: Dict[str, pd.DataFrame], out_dir: Path) -> None:
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
    max_val = max(test["y_true_count"].max(), test["y_pred_count"].max()) * 1.5
    plt.plot([0, max_val], [0, max_val], linestyle="--", color="gray", label="Perfect Forecast")
    plt.xscale("symlog", linthresh=10)
    plt.yscale("symlog", linthresh=10)
    plt.xlabel("True Dengue Cases (SymLog Scale)")
    plt.ylabel("Predicted Dengue Cases (SymLog Scale)")
    plt.title("CatBoost Test: True vs Predicted")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "catboost_scatter_test.pdf", dpi=300)
    plt.close()

    per_h = pd.read_csv(out_dir / "catboost_per_horizon.csv")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=per_h, x="horizon", y="RMSE")
    plt.title("CatBoost Test RMSE by Horizon")
    plt.xlabel("Forecast Horizon (Months)")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(out_dir / "catboost_per_h_rmse.pdf", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=per_h, x="horizon", y="MAE")
    plt.title("CatBoost Test MAE by Horizon")
    plt.xlabel("Forecast Horizon (Months)")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(out_dir / "catboost_per_h_mae.pdf", dpi=300)
    plt.close()

    h1_test = test[test["horizon"] == 1].copy()
    if not h1_test.empty:
        grid = sns.FacetGrid(h1_test, col="District", col_wrap=4, sharey=False, height=3.5, aspect=1.2)
        grid.map_dataframe(sns.lineplot, x="TargetDate", y="y_true_count", marker="o", label="True")
        grid.map_dataframe(sns.lineplot, x="TargetDate", y="y_pred_count", marker="X", label="Predicted")
        grid.set_titles(col_template="{col_name}")
        grid.set_axis_labels("Target Date", "Cases")
        for ax in grid.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(45)
        grid.add_legend(title="")
        grid.fig.subplots_adjust(top=0.9)
        grid.fig.suptitle("CatBoost Horizon 1 Forecast by District (Test Window)")
        grid.savefig(out_dir / "catboost_h1_district_grid.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    err = test.groupby(["District", "horizon"], as_index=False)["abs_error"].mean().pivot(index="District", columns="horizon", values="abs_error")
    plt.figure(figsize=(10, 8))
    sns.heatmap(err, annot=True, fmt=".1f", cmap="YlOrRd", annot_kws={"size": 11})
    plt.title("CatBoost Mean Absolute Error by District and Horizon")
    plt.xlabel("Horizon")
    plt.ylabel("District")
    plt.tight_layout()
    plt.savefig(out_dir / "catboost_error_heatmap.pdf", dpi=300)
    plt.close()

    national_h1 = preds["national_test"][preds["national_test"]["horizon"] == 1].copy()
    if not national_h1.empty:
        plt.figure(figsize=(14, 5))
        plt.plot(national_h1["TargetDate"], national_h1["y_true_count"], marker="o", linewidth=2, label="National true")
        plt.plot(national_h1["TargetDate"], national_h1["y_pred_count"], marker="X", linewidth=2, label="National predicted")
        plt.title("National Total from Summed District CatBoost Forecasts (Horizon 1, Test)")
        plt.xlabel("Target Date")
        plt.ylabel("Dengue Cases")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(out_dir / "catboost_national_h1_timeline_from_district_sum.pdf", dpi=300)
        plt.close()

    top_pred = pd.read_csv(out_dir / "catboost_feature_importance_predictionvalueschange.csv", index_col=0)["mean_importance"].sort_values(ascending=False).head(20).iloc[::-1]
    top_pred.index = top_pred.index.map(pretty_column_name)
    plt.figure(figsize=(10, 8))
    top_pred.plot(kind="barh")
    plt.title("Top 20 CatBoost PredictionValuesChange Importance")
    plt.xlabel("Mean Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_dir / "catboost_feature_importance_bar.pdf", dpi=300)
    plt.close()

    top_shap = pd.read_csv(out_dir / "catboost_shap_aggregate_importance.csv", index_col=0)["mean_abs_shap"].sort_values(ascending=False).head(20).iloc[::-1]
    top_shap.index = top_shap.index.map(pretty_column_name)
    plt.figure(figsize=(10, 8))
    top_shap.plot(kind="barh")
    plt.title("Top 20 CatBoost Mean Absolute SHAP Importance")
    plt.xlabel("Mean |SHAP|")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_dir / "catboost_shap_importance_bar.pdf", dpi=300)
    plt.close()


def write_run_summary(cfg: Config, windows: Dict[str, List[pd.Timestamp]], features: List[str], cat_feature_cols: List[str], out_dir: Path) -> None:
    lines = [
        "CatBoost run summary",
        "=" * 80,
        f"Input file: {cfg.input_path}",
        f"Target column: {cfg.target_col}",
        f"Feature count: {len(features)}",
        f"Categorical feature count: {len(cat_feature_cols)}",
        f"Horizon: {cfg.horizon}",
        f"Use district categorical feature: {cfg.use_district_cat}",
        f"Keep object features: {cfg.keep_object_features}",
        f"Blocked features: {cfg.blocked_features}",
        f"Use tuning: {cfg.use_tuning}",
        f"Outbreak quantile: {cfg.outbreak_q}",
        f"Extreme outbreak quantile: {cfg.extreme_outbreak_q}",
        f"Outbreak weight: {cfg.outbreak_weight}",
        f"Extreme outbreak weight: {cfg.extreme_outbreak_weight}",
        "",
        "Date windows from preprocessing split column",
        "-" * 80,
    ]
    for key in ["train", "val", "purge", "test"]:
        dates = windows[key]
        lines.append(f"{key}: {min(dates)} -> {max(dates)} ({len(dates)} months)")
    lines += ["", "Features used", "-" * 80] + features + ["", "Categorical features", "-" * 80] + cat_feature_cols
    (out_dir / "run_summary.txt").write_text("\n".join(map(str, lines)), encoding="utf-8")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Direct multi-horizon CatBoost for the grouped district-level dengue panel.")
    parser.add_argument("--input", type=str, default="./data/raw/prime_dataset_model_input_with_purge.csv")
    parser.add_argument("--output_dir", type=str, default="CatBoost/outputs")
    parser.add_argument("--target_col", type=str, default="Log_NoOfDenguePatients")
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--use_tuning", action="store_true")
    parser.add_argument("--tuning_iter", type=int, default=16)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--no_district_cat", dest="use_district_cat", action="store_false")
    parser.set_defaults(use_district_cat=True)
    parser.add_argument("--keep_object_features", action="store_true", default=True)
    parser.add_argument("--block_feature", action="append", default=["Year", "Month_sin", "Month_cos"], help="Feature name to exclude. Repeatable.")
    parser.add_argument("--outbreak_q", type=float, default=0.90)
    parser.add_argument("--extreme_outbreak_q", type=float, default=0.97)
    parser.add_argument("--outbreak_weight", type=float, default=5.0)
    parser.add_argument("--extreme_outbreak_weight", type=float, default=12.0)
    args = parser.parse_args()

    return Config(
        input_path=args.input,
        output_dir=args.output_dir,
        target_col=args.target_col,
        horizon=args.horizon,
        use_district_cat=args.use_district_cat,
        keep_object_features=args.keep_object_features,
        use_tuning=args.use_tuning,
        tuning_iter=args.tuning_iter,
        random_state=args.random_state,
        blocked_features=list(dict.fromkeys(args.block_feature)),
        outbreak_q=args.outbreak_q,
        extreme_outbreak_q=args.extreme_outbreak_q,
        outbreak_weight=args.outbreak_weight,
        extreme_outbreak_weight=args.extreme_outbreak_weight,
    )


def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.output_dir)
    ensure_dir(out_dir)
    save_json(cfg.__dict__, out_dir / "run_config.json")

    df = load_dataset(cfg)
    validate_panel_structure(df, cfg)
    date_to_split = build_date_split_map(df, cfg)
    windows = split_windows_from_map(date_to_split)
    df, features, cat_feature_cols = get_feature_columns(df, cfg)
    mase_denom = compute_panel_mase_denom(df, cfg)
    write_run_summary(cfg, windows, features, cat_feature_cols, out_dir)

    results = []
    best_params = {}
    for h in range(1, cfg.horizon + 1):
        sample = build_horizon_sample(df, features, cfg, h, date_to_split)
        res = fit_one_horizon(sample, features, cat_feature_cols, cfg, h, out_dir, mase_denom)
        results.append(res)
        best_params[f"h{h}"] = {
            "best_iteration": res["best_iteration"],
            "params": res["params"],
            "outbreak_threshold": res["threshold"],
        }

    save_json(best_params, out_dir / "catboost_best_params.json")
    save_row_audit(results, out_dir)
    save_split_manifest(windows, results, out_dir)
    preds = save_prediction_tables(results, out_dir)
    save_metrics(results, preds, out_dir, mase_denom)
    save_importance(results, out_dir)
    make_figures(results, preds, out_dir)
    archive = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
    print(f"Saved output folder: {out_dir}")
    print(f"Saved zip archive : {archive}")


if __name__ == "__main__":
    main()