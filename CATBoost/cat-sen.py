#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity analysis driver for the district-panel CatBoost model.

Pipeline
--------
- Uses the same preprocessed panel file with split labels: train / val / purge / test.
- Uses the same direct multi-horizon target construction as the main CatBoost model.
- Uses fixed main-model CatBoost params from a per-horizon JSON when provided.
- Does NOT retune when a fixed-config JSON is supplied.
- Changes one factor at a time through feature-group ablation.
- Keeps one seed by default, or multiple seeds only if seed robustness is desired.

Typical experiments
-------------------
- full
- no_climate
- no_serotype
- no_temporal
- no_population_density
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
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
    "TargetMonth_sin": "Target Month (Sin)",
    "TargetMonth_cos": "Target Month (Cos)",
    "District": "District",
    "Date": "Date",
    "Month-year": "Month-Year",
}


CLIMATE_FEATURES = {
    "AvgTemp_lag_3",
    "MonthlyPrevailingWindDir_ENE",
    "Rainfall_lag_2",
    "Rainfall_lag_3",
    "MonthlyAvgSunshineHours_lag_1",
    "Humidity_lag_1",
}
SEROTYPE_FEATURES = {"denv4", "denv1_lag_1"}
TEMPORAL_SOURCE_FEATURES = {"Year", "Month_sin", "Month_cos"}
TEMPORAL_TARGET_FEATURES = {"TargetMonth_sin", "TargetMonth_cos"}
POP_DENSITY_FEATURES = {"PopulationDensity"}


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
    fixed_config_json: Optional[str] = None
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
    ablation_names: List[str] = field(default_factory=lambda: ["full", "no_climate", "no_serotype", "no_temporal", "no_population_density"])
    seed_list: List[int] = field(default_factory=lambda: [42])
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
            f"Gap examples: {examples}. Export a modeling file that keeps train, val, purge, and test rows."
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
        df[[cfg.date_col, cfg.split_col]].drop_duplicates().sort_values(cfg.date_col).set_index(cfg.date_col)[cfg.split_col].to_dict()
    )
    split_order = [date_to_split[d] for d in sorted(date_to_split)]
    order_map = {"train": 0, "val": 1, "purge": 2, "test": 3}
    numeric_order = [order_map[s] for s in split_order]
    if numeric_order != sorted(numeric_order):
        raise ValueError("Split order is not chronological. Expected train -> val -> purge -> test.")
    if cfg.require_purge_rows and "purge" not in set(split_order):
        raise ValueError(
            "No purge rows found in the modeling file. For direct multi-horizon forecasting, purge rows must remain "
            "in the dataset as source-context rows, even though they are excluded from evaluation."
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
    return float(diffs.mean()) if len(diffs) else np.nan


# ---------------------------------------------------------------------
# Horizon sample construction
# ---------------------------------------------------------------------
def build_horizon_sample(df: pd.DataFrame, features: List[str], cfg: Config, h: int, date_to_split: Dict[pd.Timestamp, str]) -> pd.DataFrame:
    base_cols = [cfg.district_col, cfg.date_col, cfg.split_col, cfg.target_col]
    base_feature_cols = [c for c in features if c not in base_cols]
    base = df[base_cols + base_feature_cols].copy().sort_values([cfg.district_col, cfg.date_col]).reset_index(drop=True)
    rows_before_merge = len(base)
    base = base.rename(columns={cfg.split_col: "source_split", cfg.target_col: "y_current_log"})
    base["TargetDate"] = base[cfg.date_col] + DateOffset(months=h)

    future = df[[cfg.district_col, cfg.date_col, cfg.target_col]].copy().rename(columns={cfg.date_col: "TargetDate", cfg.target_col: "y_future_log"})
    sample = base.merge(future, on=[cfg.district_col, "TargetDate"], how="left", validate="many_to_one")
    rows_after_merge = len(sample)
    rows_without_future_target = int(sample["y_future_log"].isna().sum())
    sample = sample.dropna(subset=["y_future_log"]).copy()

    sample["target_split"] = sample["TargetDate"].map(date_to_split)
    sample["y_current_count"] = np.clip(np.expm1(sample["y_current_log"].astype(float)), a_min=0, a_max=None)
    sample["y_future_count"] = np.clip(np.expm1(sample["y_future_log"].astype(float)), a_min=0, a_max=None)
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
# Sensitivity helpers
# ---------------------------------------------------------------------
def parse_list_arg(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def parse_seed_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def load_fixed_config(path: Optional[str]) -> Dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_fixed_config(fixed_cfg: Dict, horizon: int) -> None:
    if not fixed_cfg:
        return
    missing = [f"h{h}" for h in range(1, horizon + 1) if f"h{h}" not in fixed_cfg]
    if missing:
        raise ValueError(f"Fixed config JSON is missing horizons: {missing}")


def apply_ablation(ablation_name: str, base_features: List[str], base_cat_feature_cols: List[str]) -> Tuple[List[str], List[str], bool, List[str]]:
    features = list(base_features)
    cat_feature_cols = [c for c in base_cat_feature_cols if c in features]
    add_target_temporal = True
    removed: List[str] = []

    if ablation_name == "full":
        pass
    elif ablation_name == "no_climate":
        removed = [c for c in features if c in CLIMATE_FEATURES]
        features = [c for c in features if c not in CLIMATE_FEATURES]
    elif ablation_name == "no_serotype":
        removed = [c for c in features if c in SEROTYPE_FEATURES]
        features = [c for c in features if c not in SEROTYPE_FEATURES]
    elif ablation_name == "no_population_density":
        removed = [c for c in features if c in POP_DENSITY_FEATURES]
        features = [c for c in features if c not in POP_DENSITY_FEATURES]
    elif ablation_name == "no_temporal":
        removed = [c for c in features if c in TEMPORAL_SOURCE_FEATURES]
        features = [c for c in features if c not in TEMPORAL_SOURCE_FEATURES]
        add_target_temporal = False
        removed += sorted(TEMPORAL_TARGET_FEATURES)
    else:
        raise ValueError(f"Unknown ablation name: {ablation_name}")

    features = list(dict.fromkeys(features))
    cat_feature_cols = [c for c in cat_feature_cols if c in features]
    return features, cat_feature_cols, add_target_temporal, removed


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


def make_model(cfg: Config, seed: int, params: Optional[Dict] = None) -> CatBoostRegressor:
    p = dict(cfg.base_params)
    if params:
        p.update(params)
    p["random_seed"] = seed
    return CatBoostRegressor(**p)


def make_pool(df_block: pd.DataFrame, features: List[str], cat_feature_cols: List[str], label=None, weight=None) -> Pool:
    cat_idx = [features.index(c) for c in cat_feature_cols if c in features]
    return Pool(data=df_block[features], label=label, weight=weight, cat_features=cat_idx)


def fit_one_horizon(
    sample: pd.DataFrame,
    features: List[str],
    cat_feature_cols: List[str],
    cfg: Config,
    h: int,
    seed: int,
    out_dir: Path,
    mase_denom: float,
    fixed_hcfg: Optional[Dict] = None,
    add_target_temporal: bool = True,
) -> Dict:
    train_df = sample[sample["target_split"] == "train"].copy()
    val_df = sample[sample["target_split"] == "val"].copy()
    test_df = sample[sample["target_split"] == "test"].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(f"Horizon {h}: one of train/val/test is empty after target-date assignment.")

    horizon_features = list(features)
    if add_target_temporal:
        horizon_features += ["TargetMonth_sin", "TargetMonth_cos"]
    horizon_features = list(dict.fromkeys(horizon_features))
    horizon_cat_feature_cols = [c for c in cat_feature_cols if c in horizon_features]

    y_train_log = train_df["y_future_log"].astype(float).values
    y_val_log = val_df["y_future_log"].astype(float).values
    y_train_count = train_df["y_future_count"].values

    fixed_params = None
    fixed_threshold = None
    if fixed_hcfg:
        fixed_params = fixed_hcfg.get("params")
        fixed_threshold = fixed_hcfg.get("outbreak_threshold")

    q90 = float(np.quantile(y_train_count, cfg.outbreak_q)) if len(y_train_count) else 0.0
    q97 = float(np.quantile(y_train_count, cfg.extreme_outbreak_q)) if len(y_train_count) else 0.0
    threshold = float(fixed_threshold) if fixed_threshold is not None else q90
    extreme_threshold = max(q97, threshold)
    w_train = np.where(
        y_train_count >= extreme_threshold, cfg.extreme_outbreak_weight,
        np.where(y_train_count >= threshold, cfg.outbreak_weight, 1.0)
    )

    train_pool = make_pool(train_df, horizon_features, horizon_cat_feature_cols, label=y_train_log, weight=w_train)
    val_pool = make_pool(val_df, horizon_features, horizon_cat_feature_cols, label=y_val_log)
    test_pool = make_pool(test_df, horizon_features, horizon_cat_feature_cols, label=test_df["y_future_log"].astype(float).values)

    use_tuning_here = bool(cfg.use_tuning and not fixed_hcfg)
    best_params = None
    tuning_rows = []
    if use_tuning_here:
        best_score = np.inf
        for params in param_candidates(cfg):
            model = make_model(cfg, seed, params)
            model.fit(train_pool, eval_set=val_pool, use_best_model=True, early_stopping_rounds=100, verbose=False)
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

    model_params = fixed_params if fixed_params is not None else best_params
    model = make_model(cfg, seed, model_params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True, early_stopping_rounds=100, verbose=False)
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

    m_test["Skill_RMSE_vs_NaiveLast_pct"] = 100 * (1 - m_test["RMSE"] / b_naive["RMSE"]) if pd.notna(b_naive["RMSE"]) and not np.isclose(b_naive["RMSE"], 0.0) else np.nan
    m_test["Skill_RMSE_vs_Seasonal12_pct"] = 100 * (1 - m_test["RMSE"] / b_seas["RMSE"]) if pd.notna(b_seas["RMSE"]) and not np.isclose(b_seas["RMSE"], 0.0) else np.nan
    m_test["Skill_MAE_vs_NaiveLast_pct"] = 100 * (1 - m_test["MAE"] / b_naive["MAE"]) if pd.notna(b_naive["MAE"]) and not np.isclose(b_naive["MAE"], 0.0) else np.nan
    m_test["Skill_MAE_vs_Seasonal12_pct"] = 100 * (1 - m_test["MAE"] / b_seas["MAE"]) if pd.notna(b_seas["MAE"]) and not np.isclose(b_seas["MAE"], 0.0) else np.nan

    cls = safe_classification_metrics((pred_test["y_true_count"] >= threshold).astype(int), pred_test["y_pred_count"], threshold)

    return {
        "horizon": h,
        "feature_cols": horizon_features,
        "cat_feature_cols": horizon_cat_feature_cols,
        "params": model.get_params(),
        "best_iteration": int(getattr(model, "best_iteration_", model.tree_count_)),
        "threshold": threshold,
        "extreme_threshold": extreme_threshold,
        "pred_train": pred_train,
        "pred_val": pred_val,
        "pred_test": pred_test,
        "metrics_train": m_train,
        "metrics_val": m_val,
        "metrics_test": m_test,
        "baseline_test_naive": b_naive,
        "baseline_test_seasonal12": b_seas,
        "classification_test": cls,
        "row_audit": sample.attrs.get("row_audit", {}),
        "used_fixed_config": bool(fixed_hcfg),
    }


# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------
def aggregate_predictions(results: Sequence[Dict], split_name: str) -> pd.DataFrame:
    return pd.concat([r[f"pred_{split_name}"] for r in results], ignore_index=True)


def aggregate_national_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        pred_df.groupby(["split", "horizon", "TargetDate"], as_index=False)[["y_true_count", "y_pred_count", "naive_last_count", "seasonal12_count"]].sum()
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
    pd.DataFrame([r["row_audit"] for r in results]).sort_values("horizon").to_csv(out_dir / "catboost_row_audit.csv", index=False)


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


def save_metrics(results: Sequence[Dict], preds: Dict[str, pd.DataFrame], out_dir: Path, mase_denom: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    summary_df = pd.DataFrame(rows)
    summary_df[["Model", "Split", "n", "RMSE", "MAE", "R2", "CVRMSE", "NRMSE", "sMAPE", "MBE", "MedAE", "MASE"]].to_csv(out_dir / "catboost_summary.csv", index=False)

    per_h = []
    for r in results:
        row = {"horizon": r["horizon"], "OutbreakThreshold": r["threshold"], "BestIteration": r["best_iteration"]}
        row.update(r["metrics_test"])
        row.update(r["classification_test"])
        per_h.append(row)
    per_h_df = pd.DataFrame(per_h).sort_values("horizon")
    per_h_df.to_csv(out_dir / "catboost_per_horizon.csv", index=False)

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
    national_summary_df = pd.DataFrame(national_rows)
    national_summary_df[["Model", "Split", "n", "RMSE", "MAE", "R2", "CVRMSE", "NRMSE", "sMAPE", "MBE", "MedAE", "MASE"]].to_csv(
        out_dir / "catboost_summary_national_from_district_sum.csv", index=False
    )
    return summary_df, per_h_df


def write_run_summary(
    cfg: Config,
    windows: Dict[str, List[pd.Timestamp]],
    features: List[str],
    cat_feature_cols: List[str],
    out_dir: Path,
    ablation_name: str,
    seed: int,
    removed_features: List[str],
    fixed_config_json: Optional[str],
    add_target_temporal: bool,
) -> None:
    lines = [
        "CatBoost sensitivity run summary",
        "=" * 80,
        f"Input file: {cfg.input_path}",
        f"Fixed config JSON: {fixed_config_json}",
        f"Ablation: {ablation_name}",
        f"Seed: {seed}",
        f"Feature count: {len(features)}",
        f"Categorical feature count: {len(cat_feature_cols)}",
        f"Horizon: {cfg.horizon}",
        f"Use district categorical feature: {cfg.use_district_cat}",
        f"Keep object features: {cfg.keep_object_features}",
        f"Blocked features: {cfg.blocked_features}",
        f"Use tuning: {cfg.use_tuning}",
        f"Target temporal branch enabled: {add_target_temporal}",
        f"Removed features: {removed_features}",
        "",
        "Date windows from preprocessing split column",
        "-" * 80,
    ]
    for key in ["train", "val", "purge", "test"]:
        dates = windows[key]
        lines.append(f"{key}: {min(dates)} -> {max(dates)} ({len(dates)} months)")
    lines += ["", "Features used", "-" * 80] + features + ["", "Categorical features", "-" * 80] + cat_feature_cols
    (out_dir / "run_summary.txt").write_text("\n".join(map(str, lines)), encoding="utf-8")


# ---------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------
def run_experiment(
    df: pd.DataFrame,
    cfg: Config,
    windows: Dict[str, List[pd.Timestamp]],
    date_to_split: Dict[pd.Timestamp, str],
    base_features: List[str],
    base_cat_feature_cols: List[str],
    fixed_cfg: Dict,
    ablation_name: str,
    seed: int,
    root_out_dir: Path,
    mase_denom: float,
) -> Dict:
    exp_dir = root_out_dir / f"{ablation_name}__seed_{seed}"
    ensure_dir(exp_dir)

    features, cat_feature_cols, add_target_temporal, removed_features = apply_ablation(ablation_name, base_features, base_cat_feature_cols)
    write_run_summary(cfg, windows, features, cat_feature_cols, exp_dir, ablation_name, seed, removed_features, cfg.fixed_config_json, add_target_temporal)

    results = []
    fixed_params_used = {}
    for h in range(1, cfg.horizon + 1):
        sample = build_horizon_sample(df, features, cfg, h, date_to_split)
        hcfg = fixed_cfg.get(f"h{h}", {}) if fixed_cfg else {}
        res = fit_one_horizon(
            sample=sample,
            features=features,
            cat_feature_cols=cat_feature_cols,
            cfg=cfg,
            h=h,
            seed=seed,
            out_dir=exp_dir,
            mase_denom=mase_denom,
            fixed_hcfg=hcfg,
            add_target_temporal=add_target_temporal,
        )
        results.append(res)
        fixed_params_used[f"h{h}"] = {
            "used_fixed_config": bool(hcfg),
            "params": res["params"],
            "best_iteration": res["best_iteration"],
            "outbreak_threshold": res["threshold"],
            "extreme_threshold": res["extreme_threshold"],
            "feature_cols": res["feature_cols"],
            "cat_feature_cols": res["cat_feature_cols"],
        }

    save_json(fixed_params_used, exp_dir / "catboost_sensitivity_params_used.json")
    save_row_audit(results, exp_dir)
    save_split_manifest(windows, results, exp_dir)
    preds = save_prediction_tables(results, exp_dir)
    summary_df, per_h_df = save_metrics(results, preds, exp_dir, mase_denom)

    test_row = summary_df[(summary_df["Model"] == "CatBoost") & (summary_df["Split"] == "test")].iloc[0].to_dict()
    national_test_df = pd.read_csv(exp_dir / "catboost_summary_national_from_district_sum.csv")
    national_test_row = national_test_df[(national_test_df["Model"] == "CatBoost_from_district_sum") & (national_test_df["Split"] == "test")].iloc[0].to_dict()

    return {
        "experiment": f"{ablation_name}__seed_{seed}",
        "ablation_name": ablation_name,
        "seed": seed,
        "feature_count": len(features),
        "categorical_feature_count": len(cat_feature_cols),
        "removed_features": ",".join(removed_features),
        "target_temporal_branch_enabled": add_target_temporal,
        "test_RMSE": test_row.get("RMSE"),
        "test_MAE": test_row.get("MAE"),
        "test_MASE": test_row.get("MASE"),
        "test_R2": test_row.get("R2"),
        "test_sMAPE": test_row.get("sMAPE"),
        "national_test_RMSE": national_test_row.get("RMSE"),
        "national_test_MAE": national_test_row.get("MAE"),
        "national_test_R2": national_test_row.get("R2"),
        "out_dir": str(exp_dir),
        "per_horizon_df": per_h_df.assign(experiment=f"{ablation_name}__seed_{seed}", ablation_name=ablation_name, seed=seed),
    }


# ---------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------
def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Sensitivity analysis driver for the district-panel CatBoost model.")
    parser.add_argument("--input", type=str, default="./data/raw/prime_dataset_model_input_with_purge.csv")
    parser.add_argument("--output_dir", type=str, default="./CatBoost/outputs/sensitivity")
    parser.add_argument("--fixed_config_json", type=str, default=None)
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
    parser.add_argument("--ablation_names", type=str, default="full,no_climate,no_serotype,no_temporal,no_population_density")
    parser.add_argument("--seed_list", type=str, default="42")
    args = parser.parse_args()

    return Config(
        input_path=args.input,
        output_dir=args.output_dir,
        fixed_config_json=args.fixed_config_json,
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
        ablation_names=parse_list_arg(args.ablation_names),
        seed_list=parse_seed_list(args.seed_list),
    )


def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.output_dir)
    ensure_dir(out_dir)

    fixed_cfg = load_fixed_config(cfg.fixed_config_json)
    validate_fixed_config(fixed_cfg, cfg.horizon)
    if fixed_cfg:
        cfg.use_tuning = False

    save_json(cfg.__dict__, out_dir / "run_config.json")
    if fixed_cfg:
        save_json(fixed_cfg, out_dir / "fixed_config_snapshot.json")

    df = load_dataset(cfg)
    validate_panel_structure(df, cfg)
    date_to_split = build_date_split_map(df, cfg)
    windows = split_windows_from_map(date_to_split)
    df, base_features, base_cat_feature_cols = get_feature_columns(df, cfg)
    mase_denom = compute_panel_mase_denom(df, cfg)

    experiment_rows = []
    per_horizon_long_parts = []
    for seed in cfg.seed_list:
        for ablation_name in cfg.ablation_names:
            out = run_experiment(
                df=df,
                cfg=cfg,
                windows=windows,
                date_to_split=date_to_split,
                base_features=base_features,
                base_cat_feature_cols=base_cat_feature_cols,
                fixed_cfg=fixed_cfg,
                ablation_name=ablation_name,
                seed=seed,
                root_out_dir=out_dir,
                mase_denom=mase_denom,
            )
            experiment_rows.append({k: v for k, v in out.items() if k != "per_horizon_df"})
            per_horizon_long_parts.append(out["per_horizon_df"])

    experiments_df = pd.DataFrame(experiment_rows)
    experiments_df.to_csv(out_dir / "catboost_sensitivity_experiments.csv", index=False)

    per_horizon_long = pd.concat(per_horizon_long_parts, ignore_index=True)
    per_horizon_long.to_csv(out_dir / "catboost_sensitivity_per_horizon_long.csv", index=False)

    summary_cols = [c for c in ["RMSE", "MAE", "MASE", "R2", "sMAPE", "precision", "recall", "f1", "roc_auc", "pr_auc"] if c in per_horizon_long.columns]
    if summary_cols:
        per_horizon_summary = per_horizon_long.groupby(["ablation_name", "horizon"], as_index=False)[summary_cols].agg(["mean", "std"])
        per_horizon_summary.columns = ["_".join([str(x) for x in col if str(x) != ""]).strip("_") for col in per_horizon_summary.columns.to_flat_index()]
        per_horizon_summary.to_csv(out_dir / "catboost_sensitivity_per_horizon_summary.csv", index=False)

    ablation_summary = experiments_df.groupby("ablation_name", as_index=False)[
        [c for c in ["test_RMSE", "test_MAE", "test_MASE", "test_R2", "test_sMAPE", "national_test_RMSE", "national_test_MAE", "national_test_R2"] if c in experiments_df.columns]
    ].agg(["mean", "std"])
    ablation_summary.columns = ["_".join([str(x) for x in col if str(x) != ""]).strip("_") for col in ablation_summary.columns.to_flat_index()]
    ablation_summary.to_csv(out_dir / "catboost_sensitivity_summary_by_ablation.csv", index=False)

    archive = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
    print(f"Saved output folder: {out_dir}")
    print(f"Saved zip archive : {archive}")


if __name__ == "__main__":
    main()