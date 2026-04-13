#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct multi-horizon multiple linear regression (MLR) for the dengue district-panel pipeline.

Expected input:
    A preprocessed CSV produced by the latest preprocessing pipeline, with:
    - one row per District-Date
    - selected numeric features
    - columns: District, Date, Month-year, split, Log_NoOfDenguePatients
    - split labels already assigned in preprocessing: train / val / purge / test

Notes:
    - This script uses target-date assignment for direct multi-horizon forecasting.
    - If purge rows are missing from the modeling CSV, the script can still run, but
      some early test targets will be unavailable for larger horizons. Coverage is
      explicitly logged in the exported coverage manifest.
    - By default, district one-hot encoding is OFF for MLR because static district
      covariates such as PopulationDensity become perfectly collinear with district
      fixed effects. If district OHE is enabled, PopulationDensity is dropped
      automatically to avoid exact multicollinearity.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
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
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.graphics.gofplots import qqplot

warnings.filterwarnings("ignore", category=RuntimeWarning)
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
    use_district_ohe: bool = False
    horizon: int = 6
    random_state: int = 42
    robust_cov_type: str = "HC3"
    outbreak_quantile: float = 0.90


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


def audit_panel_structure(df: pd.DataFrame, cfg: Config) -> Dict:
    audit = {"warnings": []}

    # same date grid?
    district_dates = df.groupby(cfg.district_col)[cfg.date_col].apply(lambda s: tuple(sorted(s.unique())))
    first = district_dates.iloc[0]
    inconsistent = [d for d, dates in district_dates.items() if dates != first]
    audit["same_date_grid_across_districts"] = len(inconsistent) == 0
    audit["inconsistent_districts"] = inconsistent

    # global date order / gaps
    all_dates = pd.Series(sorted(df[cfg.date_col].unique()))
    gap_examples = []
    if len(all_dates) > 1:
        period_idx = all_dates.dt.to_period("M")
        month_steps = period_idx.map(lambda p: p.year * 12 + p.month).diff().dropna()
        bad_idx = month_steps[month_steps != 1].index.tolist()
        for idx in bad_idx[:10]:
            gap_examples.append({
                "previous_date": str(all_dates.iloc[idx - 1].date()),
                "next_date": str(all_dates.iloc[idx].date()),
                "missing_months_between": int(month_steps.loc[idx] - 1),
            })
    audit["date_gap_examples"] = gap_examples

    # split by date
    valid_splits = {"train", "val", "purge", "test"}
    bad_splits = sorted(set(df[cfg.split_col].unique()) - valid_splits)
    if bad_splits:
        raise ValueError(f"Unexpected split labels found: {bad_splits}")

    per_date = df.groupby(cfg.date_col)[cfg.split_col].nunique()
    if (per_date > 1).any():
        clash_dates = [str(d.date()) for d in per_date[per_date > 1].index[:10]]
        raise ValueError(f"A calendar date maps to multiple split labels: {clash_dates}")

    date_to_split = (
        df[[cfg.date_col, cfg.split_col]]
        .drop_duplicates()
        .sort_values(cfg.date_col)
        .set_index(cfg.date_col)[cfg.split_col]
        .to_dict()
    )
    audit["available_splits"] = sorted(set(date_to_split.values()))

    if "purge" not in set(date_to_split.values()):
        audit["warnings"].append(
            "No purge rows were found in the modeling CSV. The script can still run, but early test coverage will shrink as horizon increases."
        )
    if gap_examples:
        audit["warnings"].append(
            "The modeling CSV has calendar gaps. This is usually caused by removing purge rows before modeling. Coverage loss by horizon is logged in mlr_target_coverage.csv."
        )
    if inconsistent:
        audit["warnings"].append(
            "District date grids are not perfectly aligned. Calendar-merge horizon construction will still be used, but coverage may differ by district."
        )
    return audit


def build_date_split_map(df: pd.DataFrame, cfg: Config) -> Dict[pd.Timestamp, str]:
    return (
        df[[cfg.date_col, cfg.split_col]]
        .drop_duplicates()
        .sort_values(cfg.date_col)
        .set_index(cfg.date_col)[cfg.split_col]
        .to_dict()
    )


def split_windows_from_map(date_to_split: Dict[pd.Timestamp, str]) -> Dict[str, List[pd.Timestamp]]:
    windows = {"train": [], "val": [], "purge": [], "test": [], "all": sorted(date_to_split)}
    for dt in sorted(date_to_split):
        windows[date_to_split[dt]].append(dt)
    return windows


def get_feature_columns(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, List[str], List[str]]:
    refs = {cfg.district_col, cfg.date_col, cfg.target_col, cfg.split_col}
    if cfg.monthyear_col in df.columns:
        refs.add(cfg.monthyear_col)
    feature_cols = [c for c in df.columns if c not in refs and pd.api.types.is_numeric_dtype(df[c])]
    notes = []

    if cfg.use_district_ohe:
        if "PopulationDensity" in feature_cols:
            feature_cols.remove("PopulationDensity")
            notes.append(
                "PopulationDensity was dropped because district one-hot encoding was enabled, and PopulationDensity is static within district, causing exact multicollinearity with district fixed effects."
            )
        dummies = pd.get_dummies(df[cfg.district_col], prefix="District", drop_first=True, dtype=int)
        df = pd.concat([df, dummies], axis=1)
        feature_cols.extend(dummies.columns.tolist())

    return df, feature_cols, notes


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


def build_coverage_manifest(sample: pd.DataFrame, windows: Dict[str, List[pd.Timestamp]], h: int) -> pd.DataFrame:
    rows = []
    for split in ["train", "val", "test"]:
        target_dates_all = windows[split]
        target_dates_hit = sorted(sample.loc[sample["target_split"] == split, "TargetDate"].drop_duplicates())
        rows.append({
            "horizon": h,
            "split": split,
            "possible_target_months": len(target_dates_all),
            "covered_target_months": len(target_dates_hit),
            "coverage_pct": 100 * len(target_dates_hit) / len(target_dates_all) if target_dates_all else np.nan,
            "target_start": min(target_dates_hit) if target_dates_hit else pd.NaT,
            "target_end": max(target_dates_hit) if target_dates_hit else pd.NaT,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# MLR fitting
# ---------------------------------------------------------------------
def add_constant_design(df_X: pd.DataFrame) -> pd.DataFrame:
    return sm.add_constant(df_X, has_constant="add")


def drop_constant_columns(train_X: pd.DataFrame, other_Xs: List[pd.DataFrame]) -> Tuple[pd.DataFrame, List[pd.DataFrame], List[str]]:
    constant_cols = [c for c in train_X.columns if train_X[c].nunique(dropna=False) <= 1]
    if constant_cols:
        train_X = train_X.drop(columns=constant_cols)
        other_Xs = [x.drop(columns=constant_cols, errors="ignore") for x in other_Xs]
    return train_X, other_Xs, constant_cols


def compute_vif_table(X: pd.DataFrame) -> pd.DataFrame:
    if X.shape[1] <= 1:
        return pd.DataFrame({"Feature": X.columns, "VIF": [np.nan] * X.shape[1]})
    arr = X.astype(float).values
    vals = []
    for i, col in enumerate(X.columns):
        try:
            vals.append(variance_inflation_factor(arr, i))
        except Exception:
            vals.append(np.nan)
    return pd.DataFrame({"Feature": X.columns, "VIF": vals}).sort_values("VIF", ascending=False)


def fit_one_horizon(sample: pd.DataFrame, features: List[str], cfg: Config, h: int, out_dir: Path, mase_denom: float) -> Dict:
    train_df = sample[sample["target_split"] == "train"].copy()
    val_df = sample[sample["target_split"] == "val"].copy()
    test_df = sample[sample["target_split"] == "test"].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(f"Horizon {h}: one of train/val/test is empty after target-date assignment.")

    X_train = train_df[features].copy()
    X_val = val_df[features].copy()
    X_test = test_df[features].copy()
    y_train_log = train_df["y_future_log"].astype(float)
    y_val_log = val_df["y_future_log"].astype(float)
    y_test_log = test_df["y_future_log"].astype(float)

    X_train, [X_val, X_test], constant_cols = drop_constant_columns(X_train, [X_val, X_test])
    fitted_feature_cols = X_train.columns.tolist()

    X_train_sm = add_constant_design(X_train)
    X_val_sm = add_constant_design(X_val)
    X_test_sm = add_constant_design(X_test)

    ols = sm.OLS(y_train_log, X_train_sm).fit(cov_type=cfg.robust_cov_type)
    ols_nonrobust = sm.OLS(y_train_log, X_train_sm).fit()

    pred_train_log = ols.predict(X_train_sm)
    pred_val_log = ols.predict(X_val_sm)
    pred_test_log = ols.predict(X_test_sm)

    pred_train_count = np.clip(np.expm1(pred_train_log), a_min=0, a_max=None)
    pred_val_count = np.clip(np.expm1(pred_val_log), a_min=0, a_max=None)
    pred_test_count = np.clip(np.expm1(pred_test_log), a_min=0, a_max=None)

    y_train_count = train_df["y_future_count"].values
    y_val_count = val_df["y_future_count"].values
    y_test_count = test_df["y_future_count"].values
    threshold = float(np.quantile(y_train_count, cfg.outbreak_quantile)) if len(y_train_count) else 0.0

    def pred_block(df_block: pd.DataFrame, split_name: str, y_pred_log, y_pred_count) -> pd.DataFrame:
        out = df_block[[cfg.district_col, cfg.date_col, "TargetDate", "source_split", "target_split"]].copy()
        out["split"] = split_name
        out["horizon"] = h
        out["y_true_log"] = df_block["y_future_log"].values
        out["y_pred_log"] = np.asarray(y_pred_log)
        out["y_true_count"] = df_block["y_future_count"].values
        out["y_pred_count"] = np.asarray(y_pred_count)
        out["naive_last_count"] = df_block["naive_last_count"].values
        out["seasonal12_count"] = df_block["seasonal12_count"].values
        out["residual_count"] = out["y_pred_count"] - out["y_true_count"]
        out["residual_log"] = out["y_pred_log"] - out["y_true_log"]
        out["abs_error"] = np.abs(out["residual_count"])
        out["sq_error"] = out["residual_count"] ** 2
        out["outbreak_threshold"] = threshold
        out["regime"] = np.where(out["y_true_count"] >= threshold, "Outbreak", "Normal")
        out["model"] = "MLR"
        return out

    pred_train = pred_block(train_df, "train", pred_train_log, pred_train_count)
    pred_val = pred_block(val_df, "val", pred_val_log, pred_val_count)
    pred_test = pred_block(test_df, "test", pred_test_log, pred_test_count)

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

    # coefficient table from robust fit
    params = ols.params
    conf = ols.conf_int()
    coef_table = pd.DataFrame({
        "Feature": params.index,
        "Coefficient": params.values,
        "RobustSE": ols.bse.values,
        "z_or_t": ols.tvalues.values,
        "p_value": ols.pvalues.values,
        "CI_lower": conf[0].values,
        "CI_upper": conf[1].values,
    })

    # standardized betas for comparability (exclude intercept)
    y_sd = float(y_train_log.std(ddof=0)) if len(y_train_log) else np.nan
    std_rows = []
    for feat in fitted_feature_cols:
        x_sd = float(X_train[feat].std(ddof=0)) if len(X_train) else np.nan
        beta_std = params.get(feat, np.nan) * x_sd / y_sd if pd.notna(x_sd) and pd.notna(y_sd) and not np.isclose(y_sd, 0.0) else np.nan
        std_rows.append({"Feature": feat, "StdBeta": beta_std})
    std_beta = pd.DataFrame(std_rows).sort_values("StdBeta", key=lambda s: s.abs(), ascending=False)

    # training diagnostics
    resid_train = ols_nonrobust.resid
    fitted_train = ols_nonrobust.fittedvalues
    try:
        jb_stat, jb_pvalue, skew, kurt = jarque_bera(resid_train)
    except Exception:
        jb_stat, jb_pvalue, skew, kurt = [np.nan] * 4
    try:
        bp_stat, bp_pvalue, _, _ = het_breuschpagan(resid_train, X_train_sm)
    except Exception:
        bp_stat, bp_pvalue = [np.nan] * 2

    vif_table = compute_vif_table(X_train)
    max_vif = float(vif_table["VIF"].dropna().max()) if not vif_table.empty and vif_table["VIF"].notna().any() else np.nan

    diagnostics = {
        "horizon": h,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "n_features": int(len(fitted_feature_cols)),
        "adj_R2_train_log": float(ols_nonrobust.rsquared_adj) if hasattr(ols_nonrobust, "rsquared_adj") else np.nan,
        "AIC_train_log": float(ols_nonrobust.aic),
        "BIC_train_log": float(ols_nonrobust.bic),
        "condition_number": float(ols_nonrobust.condition_number),
        "durbin_watson_train": float(durbin_watson(resid_train)),
        "jarque_bera_stat": float(jb_stat) if pd.notna(jb_stat) else np.nan,
        "jarque_bera_pvalue": float(jb_pvalue) if pd.notna(jb_pvalue) else np.nan,
        "breusch_pagan_stat": float(bp_stat) if pd.notna(bp_stat) else np.nan,
        "breusch_pagan_pvalue": float(bp_pvalue) if pd.notna(bp_pvalue) else np.nan,
        "max_vif_train": max_vif,
        "dropped_constant_cols": ", ".join(constant_cols) if constant_cols else "",
    }

    # save per-horizon text and coefficient artifacts
    (out_dir / f"mlr_h{h}_summary.txt").write_text(ols.summary().as_text(), encoding="utf-8")
    coef_table.to_csv(out_dir / f"mlr_h{h}_coefficients.csv", index=False)
    std_beta.to_csv(out_dir / f"mlr_h{h}_standardized_coefficients.csv", index=False)
    vif_table.to_csv(out_dir / f"mlr_h{h}_vif.csv", index=False)

    return {
        "horizon": h,
        "feature_cols": fitted_feature_cols,
        "model": ols,
        "model_nonrobust": ols_nonrobust,
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
        "diagnostics": diagnostics,
        "coef_table": coef_table,
        "std_beta": std_beta,
        "vif_table": vif_table,
        "fitted_train_log": fitted_train,
        "resid_train_log": resid_train,
        "X_train_cols": X_train.columns.tolist(),
    }


# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------
def aggregate_predictions(results, split_name):
    return pd.concat([r[f"pred_{split_name}"] for r in results], ignore_index=True)


def save_prediction_tables(results, out_dir: Path):
    pred_train = aggregate_predictions(results, "train")
    pred_val = aggregate_predictions(results, "val")
    pred_test = aggregate_predictions(results, "test")
    pred_train.to_csv(out_dir / "mlr_train_predictions_long.csv", index=False)
    pred_val.to_csv(out_dir / "mlr_val_predictions_long.csv", index=False)
    pred_test.to_csv(out_dir / "mlr_test_predictions_long.csv", index=False)
    pred_test.to_csv(out_dir / "mlr_test_residuals_long.csv", index=False)
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
    pd.DataFrame(rows).to_csv(out_dir / "mlr_split_manifest.csv", index=False)


def save_metrics(results, preds, out_dir: Path, mase_denom: float):
    rows = []
    for split, df_ in preds.items():
        m = compute_metrics(df_["y_true_count"], df_["y_pred_count"], mase_denom)
        m.update({"Model": "MLR", "Split": split})
        rows.append(m)

    base1 = compute_metrics(preds["test"]["y_true_count"], preds["test"]["naive_last_count"], mase_denom)
    base1.update({"Model": "NaiveLast", "Split": "test"})
    rows.append(base1)

    mask_seas = preds["test"]["seasonal12_count"].notna()
    base2 = compute_metrics(preds["test"].loc[mask_seas, "y_true_count"], preds["test"].loc[mask_seas, "seasonal12_count"], mase_denom)
    base2.update({"Model": "SeasonalNaive12", "Split": "test"})
    rows.append(base2)

    pd.DataFrame(rows)[["Model", "Split", "n", "RMSE", "MAE", "R2", "CVRMSE", "NRMSE", "sMAPE", "MBE", "MedAE", "MASE"]].to_csv(out_dir / "mlr_summary.csv", index=False)

    per_h = []
    for r in results:
        row = {"horizon": r["horizon"], "OutbreakThreshold": r["threshold"]}
        row.update(r["metrics_test"])
        row.update(r["classification_test"])
        row.update(r["diagnostics"])
        per_h.append(row)
    pd.DataFrame(per_h).sort_values("horizon").to_csv(out_dir / "mlr_per_horizon.csv", index=False)

    regime_rows = []
    for h, g_h in preds["test"].groupby("horizon"):
        for regime, g in g_h.groupby("regime"):
            m = compute_metrics(g["y_true_count"], g["y_pred_count"], mase_denom)
            m.update({"horizon": h, "regime": regime})
            regime_rows.append(m)
    pd.DataFrame(regime_rows).sort_values(["horizon", "regime"]).to_csv(out_dir / "mlr_regimewise_metrics.csv", index=False)

    district_rows = []
    for district, g in preds["test"].groupby("District"):
        m = compute_metrics(g["y_true_count"], g["y_pred_count"], mase_denom)
        m.update({"District": district})
        district_rows.append(m)
    pd.DataFrame(district_rows).sort_values("District").to_csv(out_dir / "mlr_metrics_by_district.csv", index=False)

    dh_rows = []
    for (district, h), g in preds["test"].groupby(["District", "horizon"]):
        m = compute_metrics(g["y_true_count"], g["y_pred_count"], mase_denom)
        m.update({"District": district, "horizon": h})
        dh_rows.append(m)
    pd.DataFrame(dh_rows).sort_values(["District", "horizon"]).to_csv(out_dir / "mlr_metrics_by_district_horizon.csv", index=False)

    cls_rows = []
    for r in results:
        pred = r["pred_test"]
        cls = safe_classification_metrics((pred["y_true_count"] >= r["threshold"]).astype(int), pred["y_pred_count"], r["threshold"])
        cls.update({"horizon": r["horizon"], "outbreak_threshold": r["threshold"], "n": len(pred)})
        cls_rows.append(cls)
    pd.DataFrame(cls_rows).sort_values("horizon").to_csv(out_dir / "mlr_outbreak_classification_metrics.csv", index=False)


def save_coefficients(results, out_dir: Path):
    std_tables = []
    coef_tables = []
    for r in results:
        h = r["horizon"]
        c = r["coef_table"].copy()
        c["horizon"] = h
        coef_tables.append(c)

        s = r["std_beta"].copy()
        s["horizon"] = h
        std_tables.append(s)

    coef_all = pd.concat(coef_tables, ignore_index=True)
    std_all = pd.concat(std_tables, ignore_index=True)
    coef_all.to_csv(out_dir / "mlr_all_coefficients_long.csv", index=False)
    std_all.to_csv(out_dir / "mlr_all_standardized_coefficients_long.csv", index=False)

    agg = (
        std_all.groupby("Feature", as_index=False)["StdBeta"]
        .agg(mean_std_beta=lambda s: s.mean(), mean_abs_std_beta=lambda s: s.abs().mean())
        .sort_values("mean_abs_std_beta", ascending=False)
    )
    agg.to_csv(out_dir / "mlr_standardized_coefficients_aggregate.csv", index=False)


def save_diagnostics(results, out_dir: Path):
    pd.DataFrame([r["diagnostics"] for r in results]).sort_values("horizon").to_csv(out_dir / "mlr_diagnostics_per_horizon.csv", index=False)


def make_figures(results, preds, out_dir: Path):
    test = preds["test"]

    # 1. Scatter plot
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        data=test,
        x="y_true_count",
        y="y_pred_count",
        hue="regime",
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
    plt.title("MLR Test: True vs Predicted")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_test.pdf", dpi=300)
    plt.close()

    # 2. RMSE / MAE by horizon
    per_h = pd.read_csv(out_dir / "mlr_per_horizon.csv")
    plt.figure(figsize=(10, 5))
    plt.bar(per_h["horizon"], per_h["RMSE"])
    plt.title("Test RMSE by Horizon")
    plt.xlabel("Forecast Horizon (Months)")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(out_dir / "per_h_rmse.pdf", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(per_h["horizon"], per_h["MAE"])
    plt.title("Test MAE by Horizon")
    plt.xlabel("Forecast Horizon (Months)")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(out_dir / "per_h_mae.pdf", dpi=300)
    plt.close()

    # 3. Validation vs test timeline for h1
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

    # 4. H1 district grid
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

    # 5. Error heatmap
    err = test.groupby(["District", "horizon"], as_index=False)["abs_error"].mean().pivot(index="District", columns="horizon", values="abs_error")
    plt.figure(figsize=(10, 8))
    sns.heatmap(err, annot=True, fmt=".1f", cmap="YlOrRd", annot_kws={"size": 11})
    plt.title("Mean Absolute Error by District and Horizon")
    plt.xlabel("Horizon")
    plt.ylabel("District")
    plt.tight_layout()
    plt.savefig(out_dir / "error_heatmap.pdf", dpi=300)
    plt.close()

    # 6. Coefficient bar plots (edge horizons)
    edge_h = sorted({1, max(r["horizon"] for r in results)})
    for h in edge_h:
        r = [x for x in results if x["horizon"] == h][0]
        coef = r["std_beta"].copy().dropna()
        if not coef.empty:
            coef = coef.sort_values("StdBeta", key=lambda s: s.abs(), ascending=False).head(20).iloc[::-1]
            coef["FeaturePretty"] = coef["Feature"].map(pretty_column_name)
            plt.figure(figsize=(10, 8))
            plt.barh(coef["FeaturePretty"], coef["StdBeta"])
            plt.title(f"Top Standardized Coefficients (Horizon {h})")
            plt.xlabel("Standardized Coefficient")
            plt.ylabel("Feature")
            plt.tight_layout()
            plt.savefig(out_dir / f"coef_bar_h{h}.pdf", dpi=300)
            plt.close()

    # 7. Aggregate standardized coefficients
    agg = pd.read_csv(out_dir / "mlr_standardized_coefficients_aggregate.csv")
    if not agg.empty:
        top = agg.head(20).iloc[::-1].copy()
        top["FeaturePretty"] = top["Feature"].map(pretty_column_name)
        plt.figure(figsize=(10, 8))
        plt.barh(top["FeaturePretty"], top["mean_abs_std_beta"])
        plt.title("Top Mean Absolute Standardized Coefficients (All Horizons)")
        plt.xlabel("Mean Absolute Standardized Coefficient")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(out_dir / "coef_bar_aggregate.pdf", dpi=300)
        plt.close()

    # 8. Residual diagnostics for edge horizons
    for h in edge_h:
        r = [x for x in results if x["horizon"] == h][0]
        resid = pd.Series(r["resid_train_log"])
        fitted = pd.Series(r["fitted_train_log"])

        plt.figure(figsize=(8, 5))
        plt.hist(resid.dropna(), bins=30)
        plt.title(f"Training Residual Distribution (Log Scale) - Horizon {h}")
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(out_dir / f"residual_hist_h{h}.pdf", dpi=300)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.scatter(fitted, resid, alpha=0.7)
        plt.axhline(0, linestyle="--", color="gray")
        plt.title(f"Residuals vs Fitted (Log Scale) - Horizon {h}")
        plt.xlabel("Fitted")
        plt.ylabel("Residual")
        plt.tight_layout()
        plt.savefig(out_dir / f"residuals_vs_fitted_h{h}.pdf", dpi=300)
        plt.close()

        fig = qqplot(resid, line="45", fit=True)
        plt.title(f"Q-Q Plot of Training Residuals - Horizon {h}")
        plt.tight_layout()
        plt.savefig(out_dir / f"qqplot_h{h}.pdf", dpi=300)
        plt.close(fig)


def write_run_summary(cfg, windows, features, out_dir: Path, audit: Dict, notes: List[str]):
    lines = [
        "MLR run summary",
        "=" * 80,
        f"Input file: {cfg.input_path}",
        f"Target column: {cfg.target_col}",
        f"Feature count: {len(features)}",
        f"Horizon: {cfg.horizon}",
        f"Use district one-hot: {cfg.use_district_ohe}",
        f"Robust covariance type: {cfg.robust_cov_type}",
        "",
        "Date windows from preprocessing split column",
        "-" * 80,
    ]
    for key in ["train", "val", "purge", "test"]:
        dates = windows[key]
        if dates:
            lines.append(f"{key}: {min(dates)} -> {max(dates)} ({len(dates)} months)")
        else:
            lines.append(f"{key}: none")
    lines += ["", "Features used", "-" * 80] + features

    if notes:
        lines += ["", "Feature notes", "-" * 80] + notes
    if audit.get("warnings"):
        lines += ["", "Data audit warnings", "-" * 80] + audit["warnings"]

    (out_dir / "run_summary.txt").write_text("\n".join(map(str, lines)), encoding="utf-8")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Direct multi-horizon MLR for the dengue district-panel pipeline.")
    p.add_argument("--input", type=str, default="./data/raw/prime_dataset_model_input_with_purge.csv")
    p.add_argument("--output_dir", type=str, default="MLR/outputs")
    p.add_argument("--target_col", type=str, default="Log_NoOfDenguePatients")
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--use_district_ohe", action="store_true")
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--robust_cov_type", type=str, default="HC3")
    p.add_argument("--outbreak_quantile", type=float, default=0.90)
    a = p.parse_args()
    return Config(
        input_path=a.input,
        output_dir=a.output_dir,
        target_col=a.target_col,
        horizon=a.horizon,
        use_district_ohe=a.use_district_ohe,
        random_state=a.random_state,
        robust_cov_type=a.robust_cov_type,
        outbreak_quantile=a.outbreak_quantile,
    )


def main():
    cfg = parse_args()
    out_dir = Path(cfg.output_dir)
    ensure_dir(out_dir)
    save_json(cfg.__dict__, out_dir / "run_config.json")

    df = load_dataset(cfg)
    audit = audit_panel_structure(df, cfg)
    save_json(audit, out_dir / "mlr_data_audit.json")

    date_to_split = build_date_split_map(df, cfg)
    windows = split_windows_from_map(date_to_split)
    df, features, notes = get_feature_columns(df, cfg)
    if not features:
        raise ValueError("No usable numeric predictor columns found after explicit dropping.")

    mase_denom = compute_panel_mase_denom(df, cfg)
    write_run_summary(cfg, windows, features, out_dir, audit, notes)

    results = []
    coverage_frames = []
    for h in range(1, cfg.horizon + 1):
        sample = build_horizon_sample(df, features, cfg, h, date_to_split)
        coverage_frames.append(build_coverage_manifest(sample, windows, h))
        res = fit_one_horizon(sample, features, cfg, h, out_dir, mase_denom)
        results.append(res)

    pd.concat(coverage_frames, ignore_index=True).to_csv(out_dir / "mlr_target_coverage.csv", index=False)
    save_split_manifest(windows, results, out_dir)
    preds = save_prediction_tables(results, out_dir)
    save_metrics(results, preds, out_dir, mase_denom)
    save_coefficients(results, out_dir)
    save_diagnostics(results, out_dir)
    make_figures(results, preds, out_dir)
    archive = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
    print(f"Saved output folder: {out_dir}")
    print(f"Saved zip archive : {archive}")


if __name__ == "__main__":
    main()