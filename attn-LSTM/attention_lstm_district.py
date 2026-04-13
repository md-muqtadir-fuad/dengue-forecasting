#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct multi-horizon attention-based LSTM for the dengue district-panel pipeline.

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

Modeling choice:
    - pooled district-level panel
    - direct multi-horizon setup (one model per horizon)
    - attention-based stacked LSTM on source sequences ending at source date
    - horizon-specific target-month cyclical features are added as a static branch
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Input, Lambda, LSTM, Multiply, Softmax
from tensorflow.keras.losses import Huber
from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

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
    "TargetMonth_sin": "Target Month (Sin)",
    "TargetMonth_cos": "Target Month (Cos)",
    "AR_LogCases": "AR Log(Dengue Cases)",
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
        "TargetMonth_sin": "Target Month (Sin)",
        "TargetMonth_cos": "Target Month (Cos)",
        "AR_LogCases": "AR Log(Dengue Cases)",
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
    lookback: int = 6
    epochs: int = 300
    batch_size: int = 32
    random_state: int = 42
    require_purge_rows: bool = True
    use_tuning: bool = False
    tuning_iter: int = 12
    permutation_sample_size: int = 256
    loss_name: str = "mse"
    include_autoregressive_target: bool = True
    outbreak_quantile_high: float = 0.90
    outbreak_quantile_extreme: float = 0.97
    outbreak_weight_high: float = 5.0
    outbreak_weight_extreme: float = 12.0
    early_stopping_start_epoch: int = 25
    early_stopping_patience: int = 40
    base_model_config: Dict | None = None

    def __post_init__(self):
        if self.base_model_config is None:
            self.base_model_config = {
                "lstm_units_1": 64,
                "lstm_units_2": 32,
                "dense_units": 32,
                "dropout": 0.20,
                "learning_rate": 0.001,
            }


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


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


def get_feature_columns(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, List[str]]:
    refs = {cfg.district_col, cfg.date_col, cfg.target_col, cfg.split_col}
    if cfg.monthyear_col in df.columns:
        refs.add(cfg.monthyear_col)

    # Intentional: source-month seasonality and Year are replaced by target-month seasonality.
    blocked = {"Year", "Month_sin", "Month_cos"}

    feature_cols = [
        c for c in df.columns
        if c not in refs
        and c not in blocked
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if cfg.use_district_ohe:
        dummies = pd.get_dummies(df[cfg.district_col], prefix="District", dtype=int)
        df = pd.concat([df, dummies], axis=1)
        feature_cols.extend(dummies.columns.tolist())

    feature_cols = list(dict.fromkeys(feature_cols))
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
# Horizon sample construction for sequences
# ---------------------------------------------------------------------
def build_horizon_sequence_sample(
    df: pd.DataFrame,
    features: List[str],
    cfg: Config,
    h: int,
    date_to_split: Dict[pd.Timestamp, str],
) -> Dict[str, np.ndarray | pd.DataFrame | List[str]]:
    rows_meta = []
    seq_list = []
    static_list = []
    target_log_list = []
    target_count_list = []
    y_current_log_list = []
    y_current_count_list = []
    naive_last_list = []
    seasonal12_list = []

    lookup = build_lookup(df, cfg)

    work = df[[cfg.district_col, cfg.date_col, cfg.split_col, cfg.target_col] + features].copy()
    work = work.sort_values([cfg.district_col, cfg.date_col]).reset_index(drop=True)

    sequence_features = list(features)
    if cfg.include_autoregressive_target:
        sequence_features.append("AR_LogCases")

    for district, g in work.groupby(cfg.district_col, sort=False):
        g = g.sort_values(cfg.date_col).reset_index(drop=True)
        base_values = g[features].astype(float).values
        dates = pd.to_datetime(g[cfg.date_col]).tolist()
        source_splits = g[cfg.split_col].tolist()
        y_log = g[cfg.target_col].astype(float).values
        y_count = np.clip(np.expm1(y_log), a_min=0, a_max=None)
        if cfg.include_autoregressive_target:
            values = np.column_stack([base_values, y_log.reshape(-1, 1)])
        else:
            values = base_values
        date_to_idx = {pd.Timestamp(dt): i for i, dt in enumerate(dates)}

        for end_idx in range(cfg.lookback - 1, len(g)):
            source_date = pd.Timestamp(dates[end_idx])
            target_date = source_date + DateOffset(months=h)
            target_idx = date_to_idx.get(pd.Timestamp(target_date))
            if target_idx is None:
                continue

            seq_start = end_idx - cfg.lookback + 1
            X_seq = values[seq_start:end_idx + 1]
            if X_seq.shape[0] != cfg.lookback:
                continue

            target_split = date_to_split.get(pd.Timestamp(target_date), np.nan)
            target_month = int(pd.Timestamp(target_date).month)
            target_month_sin = np.sin(2 * np.pi * target_month / 12.0)
            target_month_cos = np.cos(2 * np.pi * target_month / 12.0)

            seq_list.append(X_seq)
            static_list.append([target_month_sin, target_month_cos])
            target_log_list.append(float(y_log[target_idx]))
            target_count_list.append(float(y_count[target_idx]))
            y_current_log_list.append(float(y_log[end_idx]))
            y_current_count_list.append(float(y_count[end_idx]))
            naive_last_list.append(float(y_count[end_idx]))
            seasonal12_list.append(lookup.get((district, pd.Timestamp(target_date) - DateOffset(months=12)), np.nan))

            rows_meta.append({
                cfg.district_col: district,
                cfg.date_col: source_date,
                "TargetDate": pd.Timestamp(target_date),
                "source_split": source_splits[end_idx],
                "target_split": target_split,
            })

    if not rows_meta:
        raise ValueError(f"No usable sequence samples were created for horizon {h}.")

    meta = pd.DataFrame(rows_meta)
    sample = {
        "meta": meta,
        "X_seq": np.asarray(seq_list, dtype=np.float32),
        "X_static": np.asarray(static_list, dtype=np.float32),
        "y_future_log": np.asarray(target_log_list, dtype=np.float32),
        "y_future_count": np.asarray(target_count_list, dtype=np.float32),
        "y_current_log": np.asarray(y_current_log_list, dtype=np.float32),
        "y_current_count": np.asarray(y_current_count_list, dtype=np.float32),
        "naive_last_count": np.asarray(naive_last_list, dtype=np.float32),
        "seasonal12_count": np.asarray(seasonal12_list, dtype=np.float32),
        "sequence_features": sequence_features,
        "static_features": ["TargetMonth_sin", "TargetMonth_cos"],
    }
    return sample


def subset_sample(sample: Dict, split_name: str) -> Dict:
    mask = sample["meta"]["target_split"].eq(split_name).values
    return {
        "meta": sample["meta"].loc[mask].reset_index(drop=True),
        "X_seq": sample["X_seq"][mask],
        "X_static": sample["X_static"][mask],
        "y_future_log": sample["y_future_log"][mask],
        "y_future_count": sample["y_future_count"][mask],
        "y_current_log": sample["y_current_log"][mask],
        "y_current_count": sample["y_current_count"][mask],
        "naive_last_count": sample["naive_last_count"][mask],
        "seasonal12_count": sample["seasonal12_count"][mask],
        "sequence_features": sample["sequence_features"],
        "static_features": sample["static_features"],
    }


def fit_scalers(train_sample: Dict) -> Tuple[StandardScaler, StandardScaler]:
    n_train, lookback, n_feat = train_sample["X_seq"].shape
    seq_scaler = StandardScaler()
    seq_scaler.fit(train_sample["X_seq"].reshape(n_train * lookback, n_feat))

    static_scaler = StandardScaler()
    static_scaler.fit(train_sample["X_static"])
    return seq_scaler, static_scaler


def apply_scalers(sample: Dict, seq_scaler: StandardScaler, static_scaler: StandardScaler) -> Dict:
    X_seq = sample["X_seq"].copy()
    n, lookback, n_feat = X_seq.shape
    X_seq_scaled = seq_scaler.transform(X_seq.reshape(n * lookback, n_feat)).reshape(n, lookback, n_feat)
    X_static_scaled = static_scaler.transform(sample["X_static"])

    out = dict(sample)
    out["X_seq"] = X_seq_scaled.astype(np.float32)
    out["X_static"] = X_static_scaled.astype(np.float32)
    return out


def save_scaler_artifacts(seq_scaler: StandardScaler, static_scaler: StandardScaler, sequence_features: List[str], static_features: List[str], out_dir: Path, h: int) -> None:
    payload = {
        "sequence_scaler": seq_scaler,
        "static_scaler": static_scaler,
        "sequence_features": sequence_features,
        "static_features": static_features,
    }
    with open(out_dir / f"h{h}_scalers.pkl", "wb") as f:
        pickle.dump(payload, f)

    scaler_summary = {
        "sequence_features": sequence_features,
        "static_features": static_features,
        "sequence_scaler_mean": seq_scaler.mean_.tolist(),
        "sequence_scaler_scale": seq_scaler.scale_.tolist(),
        "static_scaler_mean": static_scaler.mean_.tolist(),
        "static_scaler_scale": static_scaler.scale_.tolist(),
    }
    save_json(scaler_summary, out_dir / f"h{h}_scaler_summary.json")


def compute_best_epoch_from_history(history_df: pd.DataFrame) -> int:
    if history_df.empty:
        return 0
    if "val_loss" in history_df.columns:
        return int(history_df["val_loss"].astype(float).idxmin() + 1)
    if "loss" in history_df.columns:
        return int(history_df["loss"].astype(float).idxmin() + 1)
    return int(len(history_df))


def save_training_artifacts(model: Model, history_df: pd.DataFrame, out_dir: Path, h: int) -> None:
    model.save(out_dir / f"attention_lstm_model_h{h}.keras")
    history_df.to_csv(out_dir / f"h{h}_history.csv", index=False)
    history_df.to_csv(out_dir / f"h{h}_training_log.csv", index=False)


def plot_training_history(history_df: pd.DataFrame, out_dir: Path, h: int) -> None:
    if history_df.empty:
        return

    epochs = np.arange(1, len(history_df) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    if "loss" in history_df.columns:
        ax.plot(epochs, history_df["loss"], label="Train Loss")
    if "val_loss" in history_df.columns:
        ax.plot(epochs, history_df["val_loss"], label="Val Loss")
    ax.set_title(f"Horizon {h}: Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    ax = axes[1]
    metric_pairs = [("mae", "val_mae", "MAE"), ("rmse", "val_rmse", "RMSE")]
    plotted = False
    for train_col, val_col, label in metric_pairs:
        if train_col in history_df.columns:
            ax.plot(epochs, history_df[train_col], label=f"Train {label}")
            plotted = True
        if val_col in history_df.columns:
            ax.plot(epochs, history_df[val_col], label=f"Val {label}")
            plotted = True
    if plotted:
        ax.legend()
    ax.set_title(f"Horizon {h}: Metric Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")

    fig.tight_layout()
    fig.savefig(out_dir / f"h{h}_loss_curve.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------
def sampled_attention_configs(cfg: Config):
    grid = {
        "lstm_units_1": [32, 48, 64, 96],
        "lstm_units_2": [16, 24, 32, 48],
        "dense_units": [16, 32, 48, 64],
        "dropout": [0.10, 0.20, 0.30],
        "learning_rate": [0.0005, 0.0010, 0.0020],
        "batch_size": [16, 32, 64],
    }
    return list(ParameterSampler(grid, n_iter=cfg.tuning_iter, random_state=cfg.random_state))


def build_attention_lstm_model(seq_shape: Tuple[int, int], static_shape: Tuple[int], cfg: Config, model_cfg: Dict) -> Model:
    seq_input = Input(shape=seq_shape, name="sequence_input")
    static_input = Input(shape=static_shape, name="static_input")

    x = LSTM(model_cfg["lstm_units_1"], return_sequences=True, name="lstm_1")(seq_input)
    x = Dropout(model_cfg["dropout"], name="dropout_1")(x)
    x = LSTM(model_cfg["lstm_units_2"], return_sequences=True, name="lstm_2")(x)
    x = Dropout(model_cfg["dropout"], name="dropout_2")(x)

    attn_score = Dense(1, activation="tanh", name="attention_score")(x)
    attn_weights = Softmax(axis=1, name="attention_weights")(attn_score)
    weighted_seq = Multiply(name="attention_apply")([x, attn_weights])
    context = Lambda(lambda t: tf.reduce_sum(t, axis=1), name="attention_context")(weighted_seq)

    x = Concatenate(name="concat_static")([context, static_input])
    x = Dense(model_cfg["dense_units"], activation="relu", name="dense_1")(x)
    x = Dropout(model_cfg["dropout"], name="dropout_3")(x)
    output = Dense(1, activation="linear", name="forecast_log_cases")(x)

    model = Model(inputs=[seq_input, static_input], outputs=output, name="attention_lstm_direct")

    if cfg.loss_name.lower() == "huber":
        loss = Huber(delta=1.0)
    else:
        loss = "mse"

    model.compile(
        optimizer=Adam(learning_rate=float(model_cfg["learning_rate"])),
        loss=loss,
        metrics=[RootMeanSquaredError(name="rmse"), MeanAbsoluteError(name="mae")],
    )
    return model


def predict_count(model: Model, sample: Dict) -> Tuple[np.ndarray, np.ndarray]:
    pred_log = model.predict([sample["X_seq"], sample["X_static"]], verbose=0).reshape(-1)
    pred_count = np.clip(np.expm1(pred_log), a_min=0, a_max=None)
    return pred_log.astype(np.float32), pred_count.astype(np.float32)


def extract_attention_weights(model: Model, sample: Dict) -> np.ndarray:
    attn_model = Model(inputs=model.inputs, outputs=model.get_layer("attention_weights").output)
    weights = attn_model.predict([sample["X_seq"], sample["X_static"]], verbose=0)
    return np.squeeze(weights, axis=-1).astype(np.float32)


def compute_outbreak_weights(y_count: np.ndarray, cfg: Config, thresholds: Tuple[float, float] | None = None):
    y_count = np.asarray(y_count, dtype=float)
    if thresholds is None:
        q_high = float(np.quantile(y_count, cfg.outbreak_quantile_high)) if len(y_count) else 0.0
        q_extreme = float(np.quantile(y_count, cfg.outbreak_quantile_extreme)) if len(y_count) else q_high
    else:
        q_high, q_extreme = thresholds
    w = np.ones(len(y_count), dtype=np.float32)
    w = np.where(y_count >= q_high, cfg.outbreak_weight_high, w).astype(np.float32)
    w = np.where(y_count >= q_extreme, cfg.outbreak_weight_extreme, w).astype(np.float32)
    return w, (float(q_high), float(q_extreme))


def weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    weights = np.asarray(weights, dtype=float)
    denom = weights.sum()
    if np.isclose(denom, 0.0):
        return float(np.mean(np.abs(y_true - y_pred)))
    return float(np.sum(weights * np.abs(y_true - y_pred)) / denom)


def train_one_model(train_sample: Dict, val_sample: Dict, cfg: Config, h: int, out_dir: Path, model_cfg: Dict, log_path: Path | None = None):
    set_global_seed(cfg.random_state + h)
    model = build_attention_lstm_model(
        seq_shape=(train_sample["X_seq"].shape[1], train_sample["X_seq"].shape[2]),
        static_shape=(train_sample["X_static"].shape[1],),
        cfg=cfg,
        model_cfg=model_cfg,
    )

    y_train_count = train_sample["y_future_count"]
    w_train, (threshold_high, threshold_extreme) = compute_outbreak_weights(y_train_count, cfg)
    w_val, _ = compute_outbreak_weights(val_sample["y_future_count"], cfg, thresholds=(threshold_high, threshold_extreme))

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=cfg.early_stopping_patience,
            start_from_epoch=cfg.early_stopping_start_epoch,
            restore_best_weights=True,
            verbose=0,
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=12, min_lr=1e-5, verbose=0),
    ]
    if log_path is not None:
        callbacks.append(CSVLogger(str(log_path), append=False))

    history = model.fit(
        x=[train_sample["X_seq"], train_sample["X_static"]],
        y=train_sample["y_future_log"],
        validation_data=([val_sample["X_seq"], val_sample["X_static"]], val_sample["y_future_log"], w_val),
        sample_weight=w_train,
        epochs=cfg.epochs,
        batch_size=int(model_cfg.get("batch_size", cfg.batch_size)),
        verbose=0,
        callbacks=callbacks,
    )
    return model, history, threshold_high, threshold_extreme


def fit_one_horizon(sample: Dict, cfg: Config, h: int, out_dir: Path, mase_denom: float) -> Dict:
    train_sample = subset_sample(sample, "train")
    val_sample = subset_sample(sample, "val")
    test_sample = subset_sample(sample, "test")

    if len(train_sample["meta"]) == 0 or len(val_sample["meta"]) == 0 or len(test_sample["meta"]) == 0:
        raise ValueError(f"Horizon {h}: one of train/val/test is empty after target-date assignment.")

    seq_scaler, static_scaler = fit_scalers(train_sample)
    train_scaled = apply_scalers(train_sample, seq_scaler, static_scaler)
    val_scaled = apply_scalers(val_sample, seq_scaler, static_scaler)
    test_scaled = apply_scalers(test_sample, seq_scaler, static_scaler)
    save_scaler_artifacts(seq_scaler, static_scaler, sample["sequence_features"], sample["static_features"], out_dir, h)

    best_model_cfg = dict(cfg.base_model_config)
    best_model_cfg["batch_size"] = cfg.batch_size
    tuning_rows = []
    selected_history_df = pd.DataFrame()

    if cfg.use_tuning:
        best_score = np.inf
        best_candidate_model = None
        best_threshold = None
        best_threshold_extreme = None
        best_history_df = None

        for i, candidate in enumerate(sampled_attention_configs(cfg), start=1):
            model_cfg = dict(cfg.base_model_config)
            model_cfg.update(candidate)
            model, history, threshold, threshold_extreme = train_one_model(train_scaled, val_scaled, cfg, h, out_dir, model_cfg, log_path=None)
            history_df = pd.DataFrame(history.history).copy()
            _, pred_val_count = predict_count(model, val_scaled)
            val_weights, _ = compute_outbreak_weights(val_scaled["y_future_count"], cfg, thresholds=(threshold, threshold_extreme))
            score = weighted_mae(val_scaled["y_future_count"], pred_val_count, val_weights)
            tuning_rows.append({
                **model_cfg,
                "horizon": h,
                "candidate": i,
                "val_mae_count": float(score),
                "epochs_trained": int(len(history_df.get("loss", []))),
                "best_epoch": compute_best_epoch_from_history(history_df),
            })
            if score < best_score:
                best_score = score
                best_model_cfg = dict(model_cfg)
                best_candidate_model = model
                best_threshold = threshold
                best_threshold_extreme = threshold_extreme
                best_history_df = history_df

        pd.DataFrame(tuning_rows).sort_values("val_mae_count").to_csv(out_dir / f"h{h}_tuning_results.csv", index=False)
        if best_candidate_model is not None and best_history_df is not None:
            model = best_candidate_model
            threshold = float(best_threshold)
            threshold_extreme = float(best_threshold_extreme)
            selected_history_df = best_history_df.copy()
        else:
            model, history, threshold, threshold_extreme = train_one_model(train_scaled, val_scaled, cfg, h, out_dir, best_model_cfg, log_path=None)
            selected_history_df = pd.DataFrame(history.history).copy()
    else:
        model, history, threshold, threshold_extreme = train_one_model(train_scaled, val_scaled, cfg, h, out_dir, best_model_cfg, log_path=None)
        selected_history_df = pd.DataFrame(history.history).copy()

    save_training_artifacts(model, selected_history_df, out_dir, h)
    plot_training_history(selected_history_df, out_dir, h)
    best_epoch = compute_best_epoch_from_history(selected_history_df)

    def pred_block(raw_sample: Dict, scaled_sample: Dict, split_name: str) -> pd.DataFrame:
        y_pred_log, y_pred_count = predict_count(model, scaled_sample)
        out = raw_sample["meta"][[cfg.district_col, cfg.date_col, "TargetDate", "source_split", "target_split"]].copy()
        out["split"] = split_name
        out["horizon"] = h
        out["y_true_log"] = raw_sample["y_future_log"]
        out["y_pred_log"] = y_pred_log
        out["y_true_count"] = raw_sample["y_future_count"]
        out["y_pred_count"] = y_pred_count
        out["naive_last_count"] = raw_sample["naive_last_count"]
        out["seasonal12_count"] = raw_sample["seasonal12_count"]
        out["residual_count"] = out["y_pred_count"] - out["y_true_count"]
        out["abs_error"] = np.abs(out["residual_count"])
        out["sq_error"] = out["residual_count"] ** 2
        out["outbreak_threshold"] = threshold
        out["regime"] = np.where(out["y_true_count"] >= threshold, "Outbreak", "Normal")
        out["model"] = "AttentionLSTM"
        return out

    pred_train = pred_block(train_sample, train_scaled, "train")
    pred_val = pred_block(val_sample, val_scaled, "val")
    pred_test = pred_block(test_sample, test_scaled, "test")

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

    perm = compute_permutation_importance(
        model=model,
        test_sample_scaled=test_scaled,
        feature_names=sample["sequence_features"],
        static_feature_names=sample["static_features"],
        sample_size=cfg.permutation_sample_size,
        random_state=cfg.random_state + h,
    )

    attn_test = extract_attention_weights(model, test_scaled)
    attention_profile = pd.Series(attn_test.mean(axis=0), index=[f"lag_{cfg.lookback - i - 1}" for i in range(cfg.lookback)], dtype=float)

    return {
        "horizon": h,
        "model": model,
        "sequence_features": sample["sequence_features"],
        "static_features": sample["static_features"],
        "best_model_config": best_model_cfg,
        "best_epoch": best_epoch,
        "threshold": threshold,
        "threshold_extreme": threshold_extreme,
        "pred_train": pred_train,
        "pred_val": pred_val,
        "pred_test": pred_test,
        "metrics_train": m_train,
        "metrics_val": m_val,
        "metrics_test": m_test,
        "baseline_test_naive": b_naive,
        "baseline_test_seasonal12": b_seas,
        "classification_test": cls,
        "permutation_importance": perm,
        "attention_test": attn_test,
        "attention_profile": attention_profile,
    }


def compute_permutation_importance(
    model: Model,
    test_sample_scaled: Dict,
    feature_names: List[str],
    static_feature_names: List[str],
    sample_size: int,
    random_state: int,
) -> pd.Series:
    rng = np.random.default_rng(random_state)
    X_seq = test_sample_scaled["X_seq"]
    X_static = test_sample_scaled["X_static"]
    y_true = test_sample_scaled["y_future_count"]
    n = len(y_true)
    if n == 0:
        return pd.Series(dtype=float)

    if n > sample_size:
        idx = np.linspace(0, n - 1, sample_size, dtype=int)
        X_seq = X_seq[idx]
        X_static = X_static[idx]
        y_true = y_true[idx]

    baseline_pred = np.clip(np.expm1(model.predict([X_seq, X_static], verbose=0).reshape(-1)), a_min=0, a_max=None)
    baseline_mae = mean_absolute_error(y_true, baseline_pred)
    scores = {}

    for j, name in enumerate(feature_names):
        X_perm = X_seq.copy()
        perm_idx = rng.permutation(len(X_perm))
        X_perm[:, :, j] = X_perm[perm_idx, :, j]
        pred = np.clip(np.expm1(model.predict([X_perm, X_static], verbose=0).reshape(-1)), a_min=0, a_max=None)
        scores[name] = float(mean_absolute_error(y_true, pred) - baseline_mae)

    for j, name in enumerate(static_feature_names):
        Xs_perm = X_static.copy()
        perm_idx = rng.permutation(len(Xs_perm))
        Xs_perm[:, j] = Xs_perm[perm_idx, j]
        pred = np.clip(np.expm1(model.predict([X_seq, Xs_perm], verbose=0).reshape(-1)), a_min=0, a_max=None)
        scores[name] = float(mean_absolute_error(y_true, pred) - baseline_mae)

    return pd.Series(scores).sort_values(ascending=False)


# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------
def aggregate_predictions(results, split_name):
    return pd.concat([r[f"pred_{split_name}"] for r in results], ignore_index=True)


def save_prediction_tables(results, out_dir: Path):
    pred_train = aggregate_predictions(results, "train")
    pred_val = aggregate_predictions(results, "val")
    pred_test = aggregate_predictions(results, "test")
    pred_train.to_csv(out_dir / "attention_lstm_train_predictions_long.csv", index=False)
    pred_val.to_csv(out_dir / "attention_lstm_val_predictions_long.csv", index=False)
    pred_test.to_csv(out_dir / "attention_lstm_test_predictions_long.csv", index=False)
    pred_test[[
        "District", "Date", "TargetDate", "horizon", "y_true_count", "y_pred_count",
        "residual_count", "abs_error", "sq_error", "regime"
    ]].to_csv(out_dir / "attention_lstm_test_residuals_long.csv", index=False)
    return {"train": pred_train, "val": pred_val, "test": pred_test}


def save_row_audit(results, out_dir: Path):
    rows = []
    for r in results:
        for split in ["train", "val", "test"]:
            pred = r[f"pred_{split}"]
            rows.append({
                "horizon": r["horizon"],
                "split": split,
                "n_rows": int(len(pred)),
                "n_districts": int(pred["District"].nunique()) if len(pred) else 0,
                "source_date_start": pred["Date"].min() if len(pred) else pd.NaT,
                "source_date_end": pred["Date"].max() if len(pred) else pd.NaT,
                "target_date_start": pred["TargetDate"].min() if len(pred) else pd.NaT,
                "target_date_end": pred["TargetDate"].max() if len(pred) else pd.NaT,
            })
    pd.DataFrame(rows).sort_values(["horizon", "split"]).to_csv(out_dir / "attention_lstm_row_audit.csv", index=False)


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
    pd.DataFrame(rows).to_csv(out_dir / "attention_lstm_split_manifest.csv", index=False)


def save_attention_outputs(results, out_dir: Path, cfg: Config):
    heatmap_rows = []
    for r in results:
        h = r["horizon"]
        attn = r["attention_test"]
        pred_test = r["pred_test"][["District", "Date", "TargetDate", "horizon"]].reset_index(drop=True)
        attn_cols = [f"lag_{cfg.lookback - i - 1}" for i in range(cfg.lookback)]
        attn_df = pd.concat([pred_test, pd.DataFrame(attn, columns=attn_cols)], axis=1)
        attn_df.to_csv(out_dir / f"attention_weights_h{h}_test.csv", index=False)

        profile = r["attention_profile"].rename("mean_attention_weight").reset_index().rename(columns={"index": "lag"})
        profile["horizon"] = h
        profile.to_csv(out_dir / f"attention_profile_h{h}.csv", index=False)

        row = {"horizon": h}
        row.update(r["attention_profile"].to_dict())
        heatmap_rows.append(row)

        plt.figure(figsize=(7, 4))
        prof = r["attention_profile"].copy()
        prof.index = [x.replace("lag_", "t-") for x in prof.index]
        prof.plot(kind="bar")
        plt.title(f"Mean Attention Profile on Test Set (Horizon {h})")
        plt.xlabel("Sequence Position")
        plt.ylabel("Mean Attention Weight")
        plt.tight_layout()
        plt.savefig(out_dir / f"attention_profile_h{h}.pdf", dpi=300)
        plt.close()

    heatmap_df = pd.DataFrame(heatmap_rows).sort_values("horizon")
    heatmap_df.to_csv(out_dir / "attention_profile_by_horizon.csv", index=False)


def save_metrics(results, preds, out_dir: Path, mase_denom: float):
    rows = []
    for split, df_ in preds.items():
        m = compute_metrics(df_["y_true_count"], df_["y_pred_count"], mase_denom)
        m.update({"Model": "AttentionLSTM", "Split": split})
        rows.append(m)

    base1 = compute_metrics(preds["test"]["y_true_count"], preds["test"]["naive_last_count"], mase_denom)
    base1.update({"Model": "NaiveLast", "Split": "test"})
    rows.append(base1)

    mask_seas = preds["test"]["seasonal12_count"].notna()
    base2 = compute_metrics(preds["test"].loc[mask_seas, "y_true_count"], preds["test"].loc[mask_seas, "seasonal12_count"], mase_denom)
    base2.update({"Model": "SeasonalNaive12", "Split": "test"})
    rows.append(base2)

    pd.DataFrame(rows)[["Model", "Split", "n", "RMSE", "MAE", "R2", "CVRMSE", "NRMSE", "sMAPE", "MBE", "MedAE", "MASE"]].to_csv(out_dir / "attention_lstm_summary.csv", index=False)

    per_h = []
    for r in results:
        row = {"horizon": r["horizon"], "OutbreakThreshold": r["threshold"], "BestEpoch": r["best_epoch"]}
        row.update(r["metrics_test"])
        row.update(r["classification_test"])
        per_h.append(row)
    pd.DataFrame(per_h).sort_values("horizon").to_csv(out_dir / "attention_lstm_per_horizon.csv", index=False)

    regime_rows = []
    for h, g_h in preds["test"].groupby("horizon"):
        for regime, g in g_h.groupby("regime"):
            m = compute_metrics(g["y_true_count"], g["y_pred_count"], mase_denom)
            m.update({"horizon": h, "regime": regime})
            regime_rows.append(m)
    pd.DataFrame(regime_rows).sort_values(["horizon", "regime"]).to_csv(out_dir / "attention_lstm_regimewise_metrics.csv", index=False)

    district_rows = []
    for district, g in preds["test"].groupby("District"):
        m = compute_metrics(g["y_true_count"], g["y_pred_count"], mase_denom)
        m.update({"District": district})
        district_rows.append(m)
    pd.DataFrame(district_rows).sort_values("District").to_csv(out_dir / "attention_lstm_metrics_by_district.csv", index=False)

    dh_rows = []
    for (district, h), g in preds["test"].groupby(["District", "horizon"]):
        m = compute_metrics(g["y_true_count"], g["y_pred_count"], mase_denom)
        m.update({"District": district, "horizon": h})
        dh_rows.append(m)
    pd.DataFrame(dh_rows).sort_values(["District", "horizon"]).to_csv(out_dir / "attention_lstm_metrics_by_district_horizon.csv", index=False)

    cls_rows = []
    for r in results:
        pred = r["pred_test"]
        cls = safe_classification_metrics((pred["y_true_count"] >= r["threshold"]).astype(int), pred["y_pred_count"], r["threshold"])
        cls.update({"horizon": r["horizon"], "outbreak_threshold": r["threshold"], "n": len(pred)})
        cls_rows.append(cls)
    pd.DataFrame(cls_rows).sort_values("horizon").to_csv(out_dir / "attention_lstm_outbreak_classification_metrics.csv", index=False)


def save_importance(results, out_dir: Path):
    feat_union = list(dict.fromkeys(results[0]["sequence_features"] + results[0]["static_features"]))
    perm_df = pd.DataFrame(index=feat_union)
    for r in results:
        perm_df[f"h{r['horizon']}"] = r["permutation_importance"].reindex(feat_union).fillna(0.0).values
    perm_df["mean_delta_mae"] = perm_df.mean(axis=1)
    perm_df.sort_values("mean_delta_mae", ascending=False).to_csv(out_dir / "attention_lstm_permutation_importance.csv")

    for r in results:
        r["permutation_importance"].rename_axis("feature").reset_index(name="delta_mae").to_csv(
            out_dir / f"attention_lstm_permutation_importance_h{r['horizon']}.csv", index=False
        )


def make_figures(results, preds, out_dir: Path, cfg: Config):
    test = preds["test"]

    # 1. Scatter plot
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
    plt.title("Attention-Based LSTM Test: True vs Predicted")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_test.pdf", dpi=300)
    plt.close()

    # 2. RMSE/MAE by horizon
    per_h = pd.read_csv(out_dir / "attention_lstm_per_horizon.csv")
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

    # 3. Horizon 1 comparative val vs test timeline
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

    # 6. Aggregate permutation importance
    perm = pd.read_csv(out_dir / "attention_lstm_permutation_importance.csv", index_col=0)["mean_delta_mae"].sort_values(ascending=False).head(20).iloc[::-1]
    perm.index = perm.index.map(pretty_column_name)
    plt.figure(figsize=(10, 8))
    perm.plot(kind="barh")
    plt.title("Top 20 Mean Permutation Importance (All Horizons)")
    plt.xlabel("Increase in MAE after permutation")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_dir / "permutation_importance_bar.pdf", dpi=300)
    plt.close()

    # 7. Attention heatmap by horizon
    attention_df = pd.read_csv(out_dir / "attention_profile_by_horizon.csv")
    lag_cols = [f"lag_{cfg.lookback - i - 1}" for i in range(cfg.lookback)]
    attn_mat = attention_df.set_index("horizon")[lag_cols]
    attn_mat.columns = [c.replace("lag_", "t-") for c in attn_mat.columns]
    plt.figure(figsize=(10, 6))
    sns.heatmap(attn_mat, annot=True, fmt=".3f", cmap="Blues")
    plt.title("Mean Attention Weight by Horizon and Sequence Position")
    plt.xlabel("Sequence Position")
    plt.ylabel("Horizon")
    plt.tight_layout()
    plt.savefig(out_dir / "attention_heatmap_by_horizon.pdf", dpi=300)
    plt.close()


def write_run_summary(cfg, windows, features, out_dir: Path):
    summary_features = list(dict.fromkeys(list(features) + (["AR_LogCases"] if cfg.include_autoregressive_target else []) + ["TargetMonth_sin", "TargetMonth_cos"]))

    lines = [
        "Attention-Based LSTM run summary",
        "=" * 80,
        f"Input file: {cfg.input_path}",
        f"Target column: {cfg.target_col}",
        f"Feature count: {len(summary_features)}",
        f"Horizon: {cfg.horizon}",
        f"Lookback: {cfg.lookback}",
        f"Use district one-hot: {cfg.use_district_ohe}",
        f"Use tuning: {cfg.use_tuning}",
        "",
        "Date windows from preprocessing split column",
        "-" * 80,
    ]

    for key in ["train", "val", "purge", "test"]:
        dates = windows[key]
        lines.append(f"{key}: {min(dates)} -> {max(dates)} ({len(dates)} months)")

    lines += [
        "",
        "Features used",
        "-" * 80,
        *summary_features,
        "",
        "Note",
        "-" * 80,
        "TargetMonth_sin and TargetMonth_cos are horizon-specific seasonal features derived from TargetDate and added as a static branch inside each direct multi-horizon model.",
        "AR_LogCases is added as an autoregressive sequence feature built from the historical Log_NoOfDenguePatients values of each district.",
        "Year, Month_sin, and Month_cos from the source row are intentionally excluded.",
    ]

    (out_dir / "run_summary.txt").write_text("\n".join(map(str, lines)), encoding="utf-8")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Direct multi-horizon attention-based LSTM for the dengue district-panel pipeline.")
    p.add_argument("--input", type=str, default="./data/raw/prime_dataset_model_input_with_purge.csv")
    p.add_argument("--output_dir", type=str, default="./attn-LSTM/outputs")
    p.add_argument("--target_col", type=str, default="Log_NoOfDenguePatients")
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--lookback", type=int, default=6)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--use_district_ohe", dest="use_district_ohe", action="store_true")
    p.add_argument("--no_district_ohe", dest="use_district_ohe", action="store_false")
    p.set_defaults(use_district_ohe=True)
    p.add_argument("--use_tuning", action="store_true")
    p.add_argument("--tuning_iter", type=int, default=12)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--loss_name", type=str, default="mse", choices=["huber", "mse"])
    p.add_argument("--include_autoregressive_target", dest="include_autoregressive_target", action="store_true")
    p.add_argument("--no_autoregressive_target", dest="include_autoregressive_target", action="store_false")
    p.set_defaults(include_autoregressive_target=True)
    p.add_argument("--outbreak_quantile_high", type=float, default=0.90)
    p.add_argument("--outbreak_quantile_extreme", type=float, default=0.97)
    p.add_argument("--outbreak_weight_high", type=float, default=5.0)
    p.add_argument("--outbreak_weight_extreme", type=float, default=12.0)
    p.add_argument("--early_stopping_start_epoch", type=int, default=25)
    p.add_argument("--early_stopping_patience", type=int, default=40)
    a = p.parse_args()
    return Config(
        input_path=a.input,
        output_dir=a.output_dir,
        target_col=a.target_col,
        horizon=a.horizon,
        lookback=a.lookback,
        epochs=a.epochs,
        batch_size=a.batch_size,
        use_district_ohe=a.use_district_ohe,
        use_tuning=a.use_tuning,
        tuning_iter=a.tuning_iter,
        random_state=a.random_state,
        loss_name=a.loss_name,
        include_autoregressive_target=a.include_autoregressive_target,
        outbreak_quantile_high=a.outbreak_quantile_high,
        outbreak_quantile_extreme=a.outbreak_quantile_extreme,
        outbreak_weight_high=a.outbreak_weight_high,
        outbreak_weight_extreme=a.outbreak_weight_extreme,
        early_stopping_start_epoch=a.early_stopping_start_epoch,
        early_stopping_patience=a.early_stopping_patience,
    )


def main():
    cfg = parse_args()
    set_global_seed(cfg.random_state)
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
    best_configs = {}
    for h in range(1, cfg.horizon + 1):
        sample = build_horizon_sequence_sample(df, features, cfg, h, date_to_split)
        res = fit_one_horizon(sample, cfg, h, out_dir, mase_denom)
        results.append(res)
        best_configs[f"h{h}"] = {
            "best_epoch": res["best_epoch"],
            "model_config": res["best_model_config"],
            "outbreak_threshold": res["threshold"],
            "extreme_outbreak_threshold": res["threshold_extreme"],
            "sequence_features": res["sequence_features"],
            "static_features": res["static_features"],
        }

    save_json(best_configs, out_dir / "attention_lstm_best_configs.json")
    save_row_audit(results, out_dir)
    save_split_manifest(windows, results, out_dir)
    save_attention_outputs(results, out_dir, cfg)
    preds = save_prediction_tables(results, out_dir)
    save_metrics(results, preds, out_dir, mase_denom)
    save_importance(results, out_dir)
    make_figures(results, preds, out_dir, cfg)
    archive = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
    print(f"Saved output folder: {out_dir}")
    print(f"Saved zip archive : {archive}")


if __name__ == "__main__":
    main()