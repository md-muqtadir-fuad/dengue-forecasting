#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporal Fusion Transformer (TFT) for the dengue district-panel pipeline.

Expected input:
    A preprocessed CSV produced by the latest preprocessing pipeline, with:
    - one row per District-Date
    - selected numeric features
    - columns: District, Date, Month-year, split, Log_NoOfDenguePatients
    - split labels already assigned in preprocessing: train / val / purge / test

Important:
    The input file MUST keep purge rows. Purge rows are not used as train/val/test
    targets, but they are necessary as source-context rows for later test windows.

Modeling choice:
    - pooled district-level panel
    - one TFT model predicts all horizons up to max_prediction_length
    - target-month cyclical features are included as known future covariates
    - source-row Year / Month_sin / Month_cos are intentionally excluded
    - prediction intervals are produced via quantile loss
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

import torch

try:
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
except Exception:  # pragma: no cover - compatibility fallback
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

try:
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
except Exception as e:  # pragma: no cover - import guidance
    raise ImportError(
        "This script requires pytorch-forecasting, torch, and lightning/pytorch-lightning. "
        "Install them first, then rerun."
    ) from e

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
    "District": "District",
    "Date": "Date",
    "Month-year": "Month-Year",
    "time_idx": "Time Index",
}


def pretty_column_name(col: str) -> str:
    if col in column_renaming:
        return column_renaming[col]

    c = str(col).strip()
    lag_match = re.match(r"^(.*)_lag_(\d+)$", c)
    if lag_match:
        base, lag_num = lag_match.groups()
        return f"{pretty_column_name(base)} (Lag {lag_num})"

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
    time_idx_col: str = "time_idx"
    max_prediction_length: int = 6
    max_encoder_length: int = 12
    batch_size: int = 64
    max_epochs: int = 150
    random_state: int = 42
    require_purge_rows: bool = True
    use_tuning: bool = False
    tuning_iter: int = 10
    accelerator: str = "auto"
    devices: int = 1
    gradient_clip_val: float = 0.1
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)
    base_model_config: Dict | None = None

    def __post_init__(self):
        if self.base_model_config is None:
            self.base_model_config = {
                "hidden_size": 16,
                "attention_head_size": 4,
                "dropout": 0.10,
                "hidden_continuous_size": 8,
                "learning_rate": 0.03,
            }


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def get_lightning_root(cfg: Config) -> Path:
    return Path(cfg.output_dir).resolve().parent / "lightning_logs"


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


def pinball_loss(y_true, y_pred, q: float):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    e = y_true - y_pred
    return float(np.mean(np.maximum(q * e, (q - 1) * e))) if len(y_true) else np.nan


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


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def save_versions(out_dir: Path):
    versions = {
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "torch": torch.__version__,
        "pytorch_forecasting": __import__("pytorch_forecasting").__version__,
    }
    try:
        import lightning
        versions["lightning"] = lightning.__version__
    except Exception:
        try:
            import pytorch_lightning
            versions["pytorch_lightning"] = pytorch_lightning.__version__
        except Exception:
            pass
    save_json(versions, out_dir / "versions.json")


# ---------------------------------------------------------------------
# Data preparation
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
            "For TFT evaluation, purge rows must remain in the dataset as context rows, even though they are excluded from evaluation."
        )
    return date_to_split


def split_windows_from_map(date_to_split: Dict[pd.Timestamp, str]) -> Dict[str, List[pd.Timestamp]]:
    windows = {"train": [], "val": [], "purge": [], "test": [], "all": sorted(date_to_split)}
    for dt in sorted(date_to_split):
        windows[date_to_split[dt]].append(dt)
    return windows


def prepare_tft_frame(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict[int, pd.Timestamp], Dict[Tuple[str, int], float], Dict[Tuple[str, int], float], Dict[str, List[str]]]:
    work = df.copy()
    base_year = work[cfg.date_col].dt.year.min()
    base_month = work[cfg.date_col].dt.month.min()
    work[cfg.time_idx_col] = (
        (work[cfg.date_col].dt.year - base_year) * 12
        + (work[cfg.date_col].dt.month - base_month)
    ).astype(int)

    month = work[cfg.date_col].dt.month.astype(int)
    work["TargetMonth_sin"] = np.sin(2 * np.pi * month / 12.0)
    work["TargetMonth_cos"] = np.cos(2 * np.pi * month / 12.0)

    refs = {cfg.district_col, cfg.date_col, cfg.monthyear_col, cfg.target_col, cfg.split_col, cfg.time_idx_col}
    blocked = {"Year", "Month_sin", "Month_cos"}
    district_ohe_cols = {c for c in work.columns if c.startswith(f"{cfg.district_col}_")}

    numeric_candidates = [
        c for c in work.columns
        if pd.api.types.is_numeric_dtype(work[c])
        and c not in refs
        and c not in blocked
        and c not in district_ohe_cols
    ]

    static_reals = [c for c in ["PopulationDensity"] if c in numeric_candidates]
    known_reals = [cfg.time_idx_col, "TargetMonth_sin", "TargetMonth_cos"]
    unknown_reals = [cfg.target_col] + [c for c in numeric_candidates if c not in static_reals and c not in known_reals]

    time_idx_to_date = (
        work[[cfg.time_idx_col, cfg.date_col]]
        .drop_duplicates()
        .sort_values(cfg.time_idx_col)
        .set_index(cfg.time_idx_col)[cfg.date_col]
        .to_dict()
    )
    actual_log_lookup = {(d, int(t)): float(y) for d, t, y in zip(work[cfg.district_col], work[cfg.time_idx_col], work[cfg.target_col])}
    actual_count_lookup = {(d, int(t)): float(np.expm1(y)) for d, t, y in zip(work[cfg.district_col], work[cfg.time_idx_col], work[cfg.target_col])}

    schema = {
        "static_categoricals": [cfg.district_col],
        "static_reals": static_reals,
        "time_varying_known_reals": known_reals,
        "time_varying_unknown_reals": unknown_reals,
    }

    work = append_future_stub_rows(work, cfg, schema)
    return work, time_idx_to_date, actual_log_lookup, actual_count_lookup, schema



def append_future_stub_rows(work: pd.DataFrame, cfg: Config, schema: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Extend each district by (max_prediction_length - 1) future months so late test
    targets can still be scored for all horizons.

    PyTorch Forecasting does not allow NA values in the target column (or other
    real-valued variables passed into the dataset), even on future placeholder rows.
    Therefore, future stub rows are populated with:
    - static reals copied from the district's last observed row
    - known future calendar covariates computed from the stub date
    - unknown reals (including the target) forward-filled from the district's last
      observed row, with a final 0.0 fallback only if a district-level value is missing

    These filled values are used only to make dataset construction possible; final
    evaluation still filters strictly to observed target dates via prediction_to_long().
    """
    n_future = max(int(cfg.max_prediction_length) - 1, 0)
    if n_future <= 0:
        return work

    observed = work.copy()
    last_date = pd.Timestamp(observed[cfg.date_col].max())
    base_year = observed[cfg.date_col].dt.year.min()
    base_month = observed[cfg.date_col].dt.month.min()
    districts = observed[cfg.district_col].drop_duplicates().tolist()

    static_cols = list(schema.get("static_reals", []))
    unknown_fill_cols = list(dict.fromkeys(schema.get("time_varying_unknown_reals", [])))

    tail = (
        observed.sort_values([cfg.district_col, cfg.date_col])
        .groupby(cfg.district_col, as_index=False)
        .tail(1)
        .set_index(cfg.district_col)
    )

    stub_rows = []
    for district in districts:
        tail_vals = tail.loc[district].to_dict() if district in tail.index else {}
        for step in range(1, n_future + 1):
            future_date = last_date + pd.DateOffset(months=step)
            row = {c: np.nan for c in observed.columns}

            row[cfg.district_col] = district
            row[cfg.date_col] = future_date
            if cfg.monthyear_col in observed.columns:
                row[cfg.monthyear_col] = future_date.strftime("%y-%b")
            row[cfg.split_col] = "future"
            row[cfg.time_idx_col] = int((future_date.year - base_year) * 12 + (future_date.month - base_month))

            # keep blocked calendar fields finite too, even though they are not used by TFT
            if "Year" in observed.columns:
                row["Year"] = int(future_date.year)
            if "Month_sin" in observed.columns:
                row["Month_sin"] = float(np.sin(2 * np.pi * future_date.month / 12.0))
            if "Month_cos" in observed.columns:
                row["Month_cos"] = float(np.cos(2 * np.pi * future_date.month / 12.0))

            row["TargetMonth_sin"] = float(np.sin(2 * np.pi * future_date.month / 12.0))
            row["TargetMonth_cos"] = float(np.cos(2 * np.pi * future_date.month / 12.0))

            for c in static_cols:
                row[c] = tail_vals.get(c, row.get(c, np.nan))

            # PyTorch Forecasting requires finite reals even on future placeholder rows.
            for c in unknown_fill_cols:
                if c in observed.columns:
                    v = tail_vals.get(c, np.nan)
                    if pd.isna(v):
                        v = 0.0
                    row[c] = float(v)

            stub_rows.append(row)

    if not stub_rows:
        return observed

    stub_df = pd.DataFrame(stub_rows, columns=observed.columns)

    # Final safety net: no NaN in columns consumed by TimeSeriesDataSet.
    fill_cols = list(dict.fromkeys(static_cols + schema.get("time_varying_known_reals", []) + unknown_fill_cols))
    for col in fill_cols:
        if col in stub_df.columns and pd.api.types.is_numeric_dtype(observed[col]):
            if col in {"TargetMonth_sin", "TargetMonth_cos"}:
                continue
            stub_df[col] = stub_df.groupby(cfg.district_col)[col].transform(lambda s: s.ffill().bfill())
            if stub_df[col].isna().any():
                stub_df[col] = stub_df[col].fillna(0.0)

    out = pd.concat([observed, stub_df], axis=0, ignore_index=True)
    out = out.sort_values([cfg.district_col, cfg.date_col]).reset_index(drop=True)
    return out

def compute_panel_mase_denom(df: pd.DataFrame, cfg: Config) -> float:
    train = df[df[cfg.split_col] == "train"].copy().sort_values([cfg.district_col, cfg.date_col])
    train["y_count"] = np.clip(np.expm1(train[cfg.target_col].astype(float)), a_min=0, a_max=None)
    diffs = train.groupby(cfg.district_col)["y_count"].diff().abs().dropna()
    if len(diffs) == 0:
        return np.nan
    return float(diffs.mean())


def build_datasets(df: pd.DataFrame, cfg: Config, schema: Dict[str, List[str]], out_dir: Path):
    min_time_idx = int(df[cfg.time_idx_col].min())
    train_end_idx = int(df.loc[df[cfg.split_col] == "train", cfg.time_idx_col].max())
    val_start_idx = int(df.loc[df[cfg.split_col] == "val", cfg.time_idx_col].min())
    val_end_idx = int(df.loc[df[cfg.split_col] == "val", cfg.time_idx_col].max())
    test_start_idx = int(df.loc[df[cfg.split_col] == "test", cfg.time_idx_col].min())
    test_end_idx = int(df.loc[df[cfg.split_col] == "test", cfg.time_idx_col].max())

    max_available_idx = int(df[cfg.time_idx_col].max())
    train_eval_end_idx = min(train_end_idx + cfg.max_prediction_length - 1, max_available_idx)
    val_eval_end_idx = min(val_end_idx + cfg.max_prediction_length - 1, max_available_idx)
    test_eval_end_idx = min(test_end_idx + cfg.max_prediction_length - 1, max_available_idx)

    train_frame = df[df[cfg.time_idx_col] <= train_end_idx].copy()
    train_eval_frame = df[df[cfg.time_idx_col] <= train_eval_end_idx].copy()
    val_frame = df[df[cfg.time_idx_col] <= val_eval_end_idx].copy()
    test_frame = df[df[cfg.time_idx_col] <= test_eval_end_idx].copy()

    # Decoder starts must begin BEFORE a split starts so that horizon-h rows can still
    # land inside that split after expansion.
    # Example: with horizon=6, a September target needs an April decoder start.
    first_eval_decoder_idx = max(min_time_idx + 1, int(df[cfg.time_idx_col].min()) + 1)
    val_first_decoder_idx = max(min_time_idx + 1, val_start_idx - cfg.max_prediction_length + 1)
    test_first_decoder_idx = max(min_time_idx + 1, test_start_idx - cfg.max_prediction_length + 1)

    training = TimeSeriesDataSet(
        train_frame,
        time_idx=cfg.time_idx_col,
        target=cfg.target_col,
        group_ids=[cfg.district_col],
        min_encoder_length=cfg.max_encoder_length,
        max_encoder_length=cfg.max_encoder_length,
        min_prediction_length=cfg.max_prediction_length,
        max_prediction_length=cfg.max_prediction_length,
        static_categoricals=schema["static_categoricals"],
        static_reals=schema["static_reals"],
        time_varying_known_categoricals=[],
        time_varying_known_reals=schema["time_varying_known_reals"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=schema["time_varying_unknown_reals"],
        target_normalizer=GroupNormalizer(groups=[cfg.district_col], transformation=None),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
        randomize_length=False,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        val_frame,
        min_prediction_idx=val_first_decoder_idx,
        stop_randomization=True,
        predict=False,
    )

    test = TimeSeriesDataSet.from_dataset(
        training,
        test_frame,
        min_prediction_idx=test_first_decoder_idx,
        stop_randomization=True,
        predict=False,
    )

    train_eval = TimeSeriesDataSet.from_dataset(
        training,
        train_eval_frame,
        min_prediction_idx=first_eval_decoder_idx,
        stop_randomization=True,
        predict=False,
    )

    with open(out_dir / "tft_training_dataset_parameters.pkl", "wb") as f:
        pickle.dump(training.get_parameters(), f)

    return training, validation, test, train_eval


# ---------------------------------------------------------------------
# TFT model fit / tuning
# ---------------------------------------------------------------------
def make_dataloaders(training, validation, test, train_eval, cfg: Config):
    train_loader = training.to_dataloader(train=True, batch_size=cfg.batch_size, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=max(cfg.batch_size * 4, 64), num_workers=0)
    test_loader = test.to_dataloader(train=False, batch_size=max(cfg.batch_size * 4, 64), num_workers=0)
    train_eval_loader = train_eval.to_dataloader(train=False, batch_size=max(cfg.batch_size * 4, 64), num_workers=0)
    return train_loader, val_loader, test_loader, train_eval_loader


def sampled_tft_configs(cfg: Config):
    grid = {
        "hidden_size": [8, 16, 24, 32],
        "attention_head_size": [1, 2, 4],
        "dropout": [0.05, 0.10, 0.20, 0.30],
        "hidden_continuous_size": [4, 8, 16],
        "learning_rate": [0.01, 0.02, 0.03, 0.05],
        "batch_size": [32, 64, 128],
        "gradient_clip_val": [0.01, 0.05, 0.10, 0.50],
    }
    return list(ParameterSampler(grid, n_iter=cfg.tuning_iter, random_state=cfg.random_state))


def create_tft_from_dataset(training, cfg: Config, model_cfg: Dict[str, Any]) -> TemporalFusionTransformer:
    return TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=float(model_cfg["learning_rate"]),
        hidden_size=int(model_cfg["hidden_size"]),
        attention_head_size=int(model_cfg["attention_head_size"]),
        dropout=float(model_cfg["dropout"]),
        hidden_continuous_size=int(model_cfg["hidden_continuous_size"]),
        output_size=len(cfg.quantiles),
        loss=QuantileLoss(list(cfg.quantiles)),
        log_interval=-1,
        reduce_on_plateau_patience=4,
    )


def fit_tft_candidate(training, train_loader, val_loader, cfg: Config, model_cfg: Dict[str, Any], candidate_dir: Path, candidate_name: str):
    seed_everything(cfg.random_state, workers=True)
    ensure_dir(candidate_dir)

    lightning_root = get_lightning_root(cfg)
    ensure_dir(lightning_root)
    csv_logger = CSVLogger(save_dir=str(lightning_root), name=candidate_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(candidate_dir / "checkpoints"),
        filename=f"{candidate_name}" + "-{epoch:03d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=20, verbose=False, mode="min")
    lr_logger = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        gradient_clip_val=float(model_cfg.get("gradient_clip_val", cfg.gradient_clip_val)),
        callbacks=[checkpoint_callback, early_stop_callback, lr_logger],
        logger=csv_logger,
        enable_model_summary=True,
        log_every_n_steps=1,
        enable_progress_bar=True,
    )

    tft = create_tft_from_dataset(training, cfg, model_cfg)
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_path = checkpoint_callback.best_model_path
    if not best_path:
        raise RuntimeError(f"No checkpoint was saved for candidate {candidate_name}")

    best_model = TemporalFusionTransformer.load_from_checkpoint(best_path)
    metrics_path = Path(csv_logger.log_dir) / "metrics.csv"
    return best_model, trainer, Path(best_path), metrics_path


# ---------------------------------------------------------------------
# Prediction extraction and evaluation
# ---------------------------------------------------------------------
def unpack_prediction(pred_obj):
    output = getattr(pred_obj, "output", None)
    index = getattr(pred_obj, "index", None)
    x = getattr(pred_obj, "x", None)
    y = getattr(pred_obj, "y", None)
    if output is None and isinstance(pred_obj, tuple):
        output = pred_obj[0]
    return output, index, x, y


def extract_prediction_array(output):
    if isinstance(output, dict):
        out = output.get("prediction", output)
    elif hasattr(output, "prediction"):
        out = output.prediction
    else:
        out = output
    arr = to_numpy(out)
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr


def predict_quantiles(model, loader, cfg: Config):
    pred = model.predict(
        loader,
        mode="quantiles",
        return_index=True,
        trainer_kwargs={"accelerator": cfg.accelerator, "devices": cfg.devices},
    )
    output, index, _, _ = unpack_prediction(pred)
    return extract_prediction_array(output), index.reset_index(drop=True)


def predict_raw(model, loader, cfg: Config):
    pred = model.predict(
        loader,
        mode="raw",
        return_x=True,
        trainer_kwargs={"accelerator": cfg.accelerator, "devices": cfg.devices},
    )
    return pred


def quantile_indices(quantiles: Tuple[float, ...]) -> Tuple[int, int, int]:
    qs = np.asarray(list(quantiles), dtype=float)
    lo = int(np.argmin(np.abs(qs - 0.1)))
    med = int(np.argmin(np.abs(qs - 0.5)))
    hi = int(np.argmin(np.abs(qs - 0.9)))
    return lo, med, hi


def prediction_to_long(
    pred_quantiles: np.ndarray,
    pred_index: pd.DataFrame,
    split_name: str,
    cfg: Config,
    time_idx_to_date: Dict[int, pd.Timestamp],
    actual_log_lookup: Dict[Tuple[str, int], float],
    actual_count_lookup: Dict[Tuple[str, int], float],
    date_to_split: Dict[pd.Timestamp, str],
) -> pd.DataFrame:
    lo_idx, med_idx, hi_idx = quantile_indices(cfg.quantiles)
    rows = []

    if cfg.district_col not in pred_index.columns or cfg.time_idx_col not in pred_index.columns:
        raise ValueError(
            f"Prediction index is missing required columns. Found: {pred_index.columns.tolist()}"
        )

    n_samples, pred_len, _ = pred_quantiles.shape
    for i in range(n_samples):
        district = str(pred_index.loc[i, cfg.district_col])
        first_target_idx = int(pred_index.loc[i, cfg.time_idx_col])
        for h in range(pred_len):
            horizon = h + 1
            target_idx = first_target_idx + h
            source_idx = target_idx - horizon
            if target_idx not in time_idx_to_date or source_idx not in time_idx_to_date:
                continue
            target_date = pd.Timestamp(time_idx_to_date[target_idx])
            source_date = pd.Timestamp(time_idx_to_date[source_idx])
            target_split = date_to_split.get(target_date, None)
            if target_split != split_name:
                continue
            if (district, target_idx) not in actual_log_lookup or (district, target_idx) not in actual_count_lookup:
                continue

            y_true_log = actual_log_lookup[(district, target_idx)]
            y_true_count = actual_count_lookup[(district, target_idx)]
            y_pred_log_p10 = float(pred_quantiles[i, h, lo_idx])
            y_pred_log_p50 = float(pred_quantiles[i, h, med_idx])
            y_pred_log_p90 = float(pred_quantiles[i, h, hi_idx])

            rows.append({
                cfg.district_col: district,
                cfg.date_col: source_date,
                "TargetDate": target_date,
                cfg.time_idx_col: target_idx,
                "split": split_name,
                "source_split": date_to_split.get(source_date, "unknown"),
                "target_split": target_split,
                "horizon": horizon,
                "y_true_log": y_true_log,
                "y_true_count": y_true_count,
                "y_pred_log_p10": y_pred_log_p10,
                "y_pred_log_p50": y_pred_log_p50,
                "y_pred_log_p90": y_pred_log_p90,
                "y_pred_count_p10": float(np.expm1(y_pred_log_p10)),
                "y_pred_count_p50": float(np.expm1(y_pred_log_p50)),
                "y_pred_count_p90": float(np.expm1(y_pred_log_p90)),
                "naive_last_count": actual_count_lookup.get((district, source_idx), np.nan),
                "seasonal12_count": actual_count_lookup.get((district, target_idx - 12), np.nan),
                "model": "TFT",
            })

    out = pd.DataFrame(rows)
    # Rows are already filtered on target_split inside the loop.
    out["residual_count"] = out["y_pred_count_p50"] - out["y_true_count"]
    out["abs_error"] = np.abs(out["residual_count"])
    out["sq_error"] = out["residual_count"] ** 2
    out["interval_width_80"] = out["y_pred_count_p90"] - out["y_pred_count_p10"]
    out["covered_80"] = ((out["y_true_count"] >= out["y_pred_count_p10"]) & (out["y_true_count"] <= out["y_pred_count_p90"])).astype(int)
    out = out.sort_values([cfg.district_col, "TargetDate", "horizon", cfg.date_col]).reset_index(drop=True)
    return out


def add_outbreak_thresholds(pred_train: pd.DataFrame, pred_val: pd.DataFrame, pred_test: pd.DataFrame):
    thresholds = pred_train.groupby("horizon")["y_true_count"].quantile(0.90).to_dict()
    for df_ in [pred_train, pred_val, pred_test]:
        df_["outbreak_threshold"] = df_["horizon"].map(thresholds)
        df_["regime"] = np.where(df_["y_true_count"] >= df_["outbreak_threshold"], "Outbreak", "Normal")
    return thresholds


def load_metrics_csv(metrics_path: Path) -> pd.DataFrame:
    if not metrics_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(metrics_path)
    if "epoch" not in df.columns:
        return df
    cols = [c for c in df.columns if c != "step"]
    metrics = df[cols].groupby("epoch", as_index=False).max(numeric_only=False)
    return metrics


# ---------------------------------------------------------------------
# Training curves and interpretation
# ---------------------------------------------------------------------
def plot_training_history(metrics_df: pd.DataFrame, out_dir: Path):
    if metrics_df.empty or "epoch" not in metrics_df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = metrics_df["epoch"]

    ax = axes[0]
    if "train_loss_epoch" in metrics_df.columns:
        ax.plot(x, metrics_df["train_loss_epoch"], label="Train Loss")
    elif "train_loss_step" in metrics_df.columns:
        ax.plot(x, metrics_df["train_loss_step"], label="Train Loss")
    if "val_loss" in metrics_df.columns:
        ax.plot(x, metrics_df["val_loss"], label="Val Loss")
    ax.set_title("TFT Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    ax = axes[1]
    metric_cols = [c for c in metrics_df.columns if any(k in c.lower() for k in ["mae", "smape", "rmse"]) and c != "epoch"]
    plotted = False
    for c in metric_cols[:6]:
        ax.plot(x, metrics_df[c], label=c)
        plotted = True
    ax.set_title("Logged Metrics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    if plotted:
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "training_curve.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_plot_predictions(best_model, raw_pred, out_dir: Path, n_examples: int = 6):
    try:
        output, _, x, _ = unpack_prediction(raw_pred)
        for idx in range(min(n_examples, len(x["decoder_lengths"]) if isinstance(x, dict) and "decoder_lengths" in x else n_examples)):
            fig = best_model.plot_prediction(x, output, idx=idx, add_loss_to_title=True)
            if hasattr(fig, "savefig"):
                fig.savefig(out_dir / f"sample_prediction_{idx}.pdf", dpi=300, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.gcf().savefig(out_dir / f"sample_prediction_{idx}.pdf", dpi=300, bbox_inches="tight")
                plt.close()
    except Exception:
        pass


def save_interpretation(best_model, raw_pred, out_dir: Path):
    try:
        output, _, _, _ = unpack_prediction(raw_pred)
        interpretation = best_model.interpret_output(output, reduction="sum")
        figs = best_model.plot_interpretation(interpretation)
        if isinstance(figs, dict):
            for name, fig in figs.items():
                safe_name = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(name))
                if hasattr(fig, "savefig"):
                    fig.savefig(out_dir / f"interpretation_{safe_name}.pdf", dpi=300, bbox_inches="tight")
                    plt.close(fig)
        # save arrays when possible
        for key, value in interpretation.items():
            try:
                arr = to_numpy(value)
                np.save(out_dir / f"interpretation_{key}.npy", arr)
                if arr.ndim == 1:
                    pd.DataFrame({"feature": np.arange(len(arr)), key: arr}).to_csv(out_dir / f"interpretation_{key}.csv", index=False)
            except Exception:
                pass
    except Exception:
        pass


# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------
def save_prediction_tables(pred_train, pred_val, pred_test, out_dir: Path):
    pred_train.to_csv(out_dir / "tft_train_predictions_long.csv", index=False)
    pred_val.to_csv(out_dir / "tft_val_predictions_long.csv", index=False)
    pred_test.to_csv(out_dir / "tft_test_predictions_long.csv", index=False)
    pred_test[[
        "District", "Date", "TargetDate", "source_split", "target_split", "horizon", "y_true_count", "y_pred_count_p50",
        "y_pred_count_p10", "y_pred_count_p90", "residual_count", "abs_error", "sq_error",
        "interval_width_80", "covered_80", "regime"
    ]].to_csv(out_dir / "tft_test_residuals_long.csv", index=False)
    return {"train": pred_train, "val": pred_val, "test": pred_test}


def save_row_audit(pred_train, pred_val, pred_test, out_dir: Path):
    rows = []
    for split_name, df_ in [("train", pred_train), ("val", pred_val), ("test", pred_test)]:
        if df_.empty:
            continue
        for h, g in df_.groupby("horizon"):
            rows.append({
                "split": split_name,
                "horizon": int(h),
                "n_rows": int(len(g)),
                "n_districts": int(g["District"].nunique()),
                "n_source_dates": int(g["Date"].nunique()),
                "n_target_dates": int(g["TargetDate"].nunique()),
                "source_date_start": g["Date"].min(),
                "source_date_end": g["Date"].max(),
                "target_date_start": g["TargetDate"].min(),
                "target_date_end": g["TargetDate"].max(),
            })
    pd.DataFrame(rows).sort_values(["split", "horizon"]).to_csv(out_dir / "tft_row_audit.csv", index=False)


def save_split_manifest(pred_train, pred_val, pred_test, windows, out_dir: Path):
    rows = []
    for split_name, df_ in [("train", pred_train), ("val", pred_val), ("test", pred_test)]:
        rows.append({
            "split": split_name,
            "source_date_start": df_["Date"].min() if len(df_) else pd.NaT,
            "source_date_end": df_["Date"].max() if len(df_) else pd.NaT,
            "target_date_start": df_["TargetDate"].min() if len(df_) else pd.NaT,
            "target_date_end": df_["TargetDate"].max() if len(df_) else pd.NaT,
            "n_rows": int(len(df_)),
            "n_districts": int(df_["District"].nunique()) if len(df_) else 0,
        })
    rows.append({
        "split": "purge_dates_available_in_input",
        "source_date_start": min(windows["purge"]) if windows["purge"] else pd.NaT,
        "source_date_end": max(windows["purge"]) if windows["purge"] else pd.NaT,
        "target_date_start": pd.NaT,
        "target_date_end": pd.NaT,
        "n_rows": np.nan,
        "n_districts": np.nan,
    })
    pd.DataFrame(rows).to_csv(out_dir / "tft_split_manifest.csv", index=False)


def save_metrics(preds: Dict[str, pd.DataFrame], out_dir: Path, mase_denom: float, quantiles: Tuple[float, ...]):
    rows = []
    for split, df_ in preds.items():
        m = compute_metrics(df_["y_true_count"], df_["y_pred_count_p50"], mase_denom)
        m.update({"Model": "TFT", "Split": split})
        rows.append(m)

    base1 = compute_metrics(preds["test"]["y_true_count"], preds["test"]["naive_last_count"], mase_denom)
    base1.update({"Model": "NaiveLast", "Split": "test"})
    rows.append(base1)

    mask_seas = preds["test"]["seasonal12_count"].notna()
    base2 = compute_metrics(preds["test"].loc[mask_seas, "y_true_count"], preds["test"].loc[mask_seas, "seasonal12_count"], mase_denom)
    base2.update({"Model": "SeasonalNaive12", "Split": "test"})
    rows.append(base2)

    pd.DataFrame(rows)[["Model", "Split", "n", "RMSE", "MAE", "R2", "CVRMSE", "NRMSE", "sMAPE", "MBE", "MedAE", "MASE"]].to_csv(out_dir / "tft_summary.csv", index=False)

    per_h = []
    lower_q, median_q, upper_q = quantiles[0], quantiles[len(quantiles)//2], quantiles[-1]
    for h, g in preds["test"].groupby("horizon"):
        m = compute_metrics(g["y_true_count"], g["y_pred_count_p50"], mase_denom)
        cls = safe_classification_metrics((g["y_true_count"] >= g["outbreak_threshold"]).astype(int), g["y_pred_count_p50"], g["outbreak_threshold"].iloc[0])
        m.update(cls)
        m.update({
            "horizon": h,
            "OutbreakThreshold": float(g["outbreak_threshold"].iloc[0]),
            "Coverage80": float(g["covered_80"].mean()),
            "MeanIntervalWidth80": float(g["interval_width_80"].mean()),
            f"Pinball_q{int(lower_q*100):02d}": pinball_loss(g["y_true_count"], g["y_pred_count_p10"], lower_q),
            f"Pinball_q{int(median_q*100):02d}": pinball_loss(g["y_true_count"], g["y_pred_count_p50"], median_q),
            f"Pinball_q{int(upper_q*100):02d}": pinball_loss(g["y_true_count"], g["y_pred_count_p90"], upper_q),
        })
        per_h.append(m)
    pd.DataFrame(per_h).sort_values("horizon").to_csv(out_dir / "tft_per_horizon.csv", index=False)

    regime_rows = []
    for (h, regime), g in preds["test"].groupby(["horizon", "regime"]):
        m = compute_metrics(g["y_true_count"], g["y_pred_count_p50"], mase_denom)
        m.update({"horizon": h, "regime": regime, "Coverage80": float(g["covered_80"].mean()), "MeanIntervalWidth80": float(g["interval_width_80"].mean())})
        regime_rows.append(m)
    pd.DataFrame(regime_rows).sort_values(["horizon", "regime"]).to_csv(out_dir / "tft_regimewise_metrics.csv", index=False)

    district_rows = []
    for district, g in preds["test"].groupby("District"):
        m = compute_metrics(g["y_true_count"], g["y_pred_count_p50"], mase_denom)
        m.update({"District": district, "Coverage80": float(g["covered_80"].mean()), "MeanIntervalWidth80": float(g["interval_width_80"].mean())})
        district_rows.append(m)
    pd.DataFrame(district_rows).sort_values("District").to_csv(out_dir / "tft_metrics_by_district.csv", index=False)

    dh_rows = []
    for (district, h), g in preds["test"].groupby(["District", "horizon"]):
        m = compute_metrics(g["y_true_count"], g["y_pred_count_p50"], mase_denom)
        m.update({"District": district, "horizon": h, "Coverage80": float(g["covered_80"].mean()), "MeanIntervalWidth80": float(g["interval_width_80"].mean())})
        dh_rows.append(m)
    pd.DataFrame(dh_rows).sort_values(["District", "horizon"]).to_csv(out_dir / "tft_metrics_by_district_horizon.csv", index=False)

    cls_rows = []
    for h, g in preds["test"].groupby("horizon"):
        cls = safe_classification_metrics((g["y_true_count"] >= g["outbreak_threshold"]).astype(int), g["y_pred_count_p50"], g["outbreak_threshold"].iloc[0])
        cls.update({"horizon": h, "outbreak_threshold": float(g["outbreak_threshold"].iloc[0]), "n": len(g)})
        cls_rows.append(cls)
    pd.DataFrame(cls_rows).sort_values("horizon").to_csv(out_dir / "tft_outbreak_classification_metrics.csv", index=False)

    interval_rows = []
    for h, g in preds["test"].groupby("horizon"):
        interval_rows.append({
            "horizon": h,
            "coverage_80": float(g["covered_80"].mean()),
            "mean_interval_width_80": float(g["interval_width_80"].mean()),
        })
    pd.DataFrame(interval_rows).sort_values("horizon").to_csv(out_dir / "tft_interval_metrics.csv", index=False)


def save_variable_importance(best_model, out_dir: Path):
    try:
        imp = best_model.interpret_output(torch.tensor([]), reduction="sum")  # likely fails, fallback below
        del imp
    except Exception:
        pass
    # rely on hparams if available; interpretation arrays are saved separately when possible
    hparams = getattr(best_model, "hparams", None)
    if hparams is not None:
        try:
            save_json(dict(hparams), out_dir / "tft_hparams.json")
        except Exception:
            pass


def make_figures(preds: Dict[str, pd.DataFrame], out_dir: Path):
    test = preds["test"]

    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        data=test,
        x="y_true_count",
        y="y_pred_count_p50",
        hue="regime",
        palette={"Normal": "#1f77b4", "Outbreak": "#d62728"},
        alpha=0.7,
        edgecolor=None,
        s=60,
    )
    m_max = max(test["y_true_count"].max(), test["y_pred_count_p50"].max()) * 1.5
    plt.plot([0, m_max], [0, m_max], linestyle="--", color="gray", label="Perfect Forecast")
    plt.xscale("symlog", linthresh=10)
    plt.yscale("symlog", linthresh=10)
    plt.xlabel("True Dengue Cases (SymLog Scale)")
    plt.ylabel("Predicted Dengue Cases (Median, SymLog Scale)")
    plt.title("TFT Test: True vs Predicted")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_test.pdf", dpi=300)
    plt.close()

    per_h = pd.read_csv(out_dir / "tft_per_horizon.csv")
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
    sns.barplot(data=per_h, x="horizon", y="coverage_80" if "coverage_80" in per_h.columns else "Coverage80")
    plt.title("80% Interval Coverage by Horizon")
    plt.xlabel("Forecast Horizon (Months)")
    plt.ylabel("Coverage")
    plt.tight_layout()
    plt.savefig(out_dir / "per_h_coverage80.pdf", dpi=300)
    plt.close()

    width_col = "mean_interval_width_80" if "mean_interval_width_80" in per_h.columns else "MeanIntervalWidth80"
    plt.figure(figsize=(10, 5))
    sns.barplot(data=per_h, x="horizon", y=width_col)
    plt.title("Mean 80% Interval Width by Horizon")
    plt.xlabel("Forecast Horizon (Months)")
    plt.ylabel("Interval Width")
    plt.tight_layout()
    plt.savefig(out_dir / "per_h_interval_width80.pdf", dpi=300)
    plt.close()

    val_df = preds["val"]
    h1_val = val_df[val_df["horizon"] == 1]
    h1_test = test[test["horizon"] == 1]
    if not h1_val.empty and not h1_test.empty:
        v_line = h1_val.groupby("TargetDate", as_index=False)[["y_true_count", "y_pred_count_p50"]].mean()
        t_line = h1_test.groupby("TargetDate", as_index=False)[["y_true_count", "y_pred_count_p50"]].mean()
        plt.figure(figsize=(14, 5))
        plt.plot(v_line["TargetDate"], v_line["y_true_count"], marker="o", linewidth=2, label="True Cases")
        plt.plot(v_line["TargetDate"], v_line["y_pred_count_p50"], marker="X", linewidth=2, label="Predicted Cases")
        plt.plot(t_line["TargetDate"], t_line["y_true_count"], marker="o", linewidth=2)
        plt.plot(t_line["TargetDate"], t_line["y_pred_count_p50"], marker="X", linewidth=2)
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
        g.map_dataframe(sns.lineplot, x="TargetDate", y="y_pred_count_p50", marker="X", label="Predicted")
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


def write_run_summary(cfg: Config, windows: Dict[str, List[pd.Timestamp]], schema: Dict[str, List[str]], out_dir: Path):
    features = schema["static_reals"] + schema["time_varying_known_reals"] + schema["time_varying_unknown_reals"]
    lines = [
        "Temporal Fusion Transformer run summary",
        "=" * 80,
        f"Input file: {cfg.input_path}",
        f"Target column: {cfg.target_col}",
        f"Encoder length: {cfg.max_encoder_length}",
        f"Prediction length: {cfg.max_prediction_length}",
        f"Batch size: {cfg.batch_size}",
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
        "Variable schema",
        "-" * 80,
        f"static_categoricals: {schema['static_categoricals']}",
        f"static_reals: {schema['static_reals']}",
        f"time_varying_known_reals: {schema['time_varying_known_reals']}",
        f"time_varying_unknown_reals: {schema['time_varying_unknown_reals']}",
        "",
        "Features used",
        "-" * 80,
        *features,
        "",
        "Note",
        "-" * 80,
        "TargetMonth_sin and TargetMonth_cos are known future covariates derived from each row's calendar month.",
        "Year, Month_sin, and Month_cos from the source row are intentionally excluded.",
        "Validation and test evaluation are filtered by TARGET split after rolling decoder expansion, so all eligible target months are retained.",
    ]
    (out_dir / "run_summary.txt").write_text("\n".join(map(str, lines)), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def parse_args() -> Config:
    p = argparse.ArgumentParser(description="TFT for the dengue district-panel pipeline.")
    p.add_argument("--input", type=str, default="data/raw/prime_dataset_model_input_with_purge.csv")
    p.add_argument("--output_dir", type=str, default="TFT/outputs")
    p.add_argument("--target_col", type=str, default="Log_NoOfDenguePatients")
    p.add_argument("--max_prediction_length", type=int, default=6)
    p.add_argument("--max_encoder_length", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_epochs", type=int, default=150)
    p.add_argument("--use_tuning", action="store_true")
    p.add_argument("--tuning_iter", type=int, default=10)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--gradient_clip_val", type=float, default=0.1)
    a = p.parse_args()
    return Config(
        input_path=a.input,
        output_dir=a.output_dir,
        target_col=a.target_col,
        max_prediction_length=a.max_prediction_length,
        max_encoder_length=a.max_encoder_length,
        batch_size=a.batch_size,
        max_epochs=a.max_epochs,
        use_tuning=a.use_tuning,
        tuning_iter=a.tuning_iter,
        random_state=a.random_state,
        accelerator=a.accelerator,
        devices=a.devices,
        gradient_clip_val=a.gradient_clip_val,
    )


def main():
    cfg = parse_args()
    seed_everything(cfg.random_state, workers=True)

    out_dir = Path(cfg.output_dir)
    ensure_dir(out_dir)
    save_json(cfg.__dict__, out_dir / "run_config.json")
    save_versions(out_dir)

    df = load_dataset(cfg)
    validate_panel_structure(df, cfg)
    date_to_split = build_date_split_map(df, cfg)
    windows = split_windows_from_map(date_to_split)
    df_tft, time_idx_to_date, actual_log_lookup, actual_count_lookup, schema = prepare_tft_frame(df, cfg)
    write_run_summary(cfg, windows, schema, out_dir)
    mase_denom = compute_panel_mase_denom(df_tft, cfg)

    training, validation, test, train_eval = build_datasets(df_tft, cfg, schema, out_dir)
    train_loader, val_loader, test_loader, train_eval_loader = make_dataloaders(training, validation, test, train_eval, cfg)

    # tuning on validation MAE in count space
    if cfg.use_tuning:
        tuning_rows = []
        best_score = np.inf
        best_cfg = dict(cfg.base_model_config)
        best_cfg["batch_size"] = cfg.batch_size

        for i, cand in enumerate(sampled_tft_configs(cfg), start=1):
            model_cfg = dict(cfg.base_model_config)
            model_cfg.update(cand)
            candidate_dir = out_dir / f"tuning_candidate_{i}"
            cand_cfg = Config(**{**cfg.__dict__})
            cand_cfg.batch_size = int(model_cfg.get("batch_size", cfg.batch_size))
            cand_cfg.gradient_clip_val = float(model_cfg.get("gradient_clip_val", cfg.gradient_clip_val))
            cand_train_loader, cand_val_loader, _, _ = make_dataloaders(training, validation, test, train_eval, cand_cfg)
            best_model, trainer, ckpt_path, metrics_path = fit_tft_candidate(training, cand_train_loader, cand_val_loader, cand_cfg, model_cfg, candidate_dir, f"candidate_{i}")
            val_q, val_index = predict_quantiles(best_model, cand_val_loader, cand_cfg)
            pred_val = prediction_to_long(val_q, val_index, "val", cand_cfg, time_idx_to_date, actual_log_lookup, actual_count_lookup, date_to_split)
            score = mean_absolute_error(pred_val["y_true_count"], pred_val["y_pred_count_p50"])
            tuning_rows.append({
                **model_cfg,
                "candidate": i,
                "val_mae_count": float(score),
                "checkpoint": str(ckpt_path),
            })
            if score < best_score:
                best_score = score
                best_cfg = dict(model_cfg)
                best_cfg["batch_size"] = int(model_cfg.get("batch_size", cfg.batch_size))
                best_cfg["gradient_clip_val"] = float(model_cfg.get("gradient_clip_val", cfg.gradient_clip_val))

        pd.DataFrame(tuning_rows).sort_values("val_mae_count").to_csv(out_dir / "tft_tuning_results.csv", index=False)
    else:
        best_cfg = dict(cfg.base_model_config)
        best_cfg["batch_size"] = cfg.batch_size
        best_cfg["gradient_clip_val"] = cfg.gradient_clip_val

    # final canonical fit with selected config
    final_cfg = Config(**{**cfg.__dict__})
    final_cfg.batch_size = int(best_cfg.get("batch_size", cfg.batch_size))
    final_cfg.gradient_clip_val = float(best_cfg.get("gradient_clip_val", cfg.gradient_clip_val))
    train_loader, val_loader, test_loader, train_eval_loader = make_dataloaders(training, validation, test, train_eval, final_cfg)
    final_model, trainer, best_ckpt, metrics_path = fit_tft_candidate(training, train_loader, val_loader, final_cfg, best_cfg, out_dir / "final_fit", "tft_final")
    canonical_ckpt = out_dir / "tft_model.ckpt"
    shutil.copy2(best_ckpt, canonical_ckpt)
    metrics_df = load_metrics_csv(metrics_path)
    metrics_df.to_csv(out_dir / "training_metrics.csv", index=False)
    plot_training_history(metrics_df, out_dir)
    save_json(best_cfg, out_dir / "tft_best_config.json")
    save_variable_importance(final_model, out_dir)

    # predictions
    train_q, train_index = predict_quantiles(final_model, train_eval_loader, final_cfg)
    val_q, val_index = predict_quantiles(final_model, val_loader, final_cfg)
    test_q, test_index = predict_quantiles(final_model, test_loader, final_cfg)

    pred_train = prediction_to_long(train_q, train_index, "train", final_cfg, time_idx_to_date, actual_log_lookup, actual_count_lookup, date_to_split)
    pred_val = prediction_to_long(val_q, val_index, "val", final_cfg, time_idx_to_date, actual_log_lookup, actual_count_lookup, date_to_split)
    pred_test = prediction_to_long(test_q, test_index, "test", final_cfg, time_idx_to_date, actual_log_lookup, actual_count_lookup, date_to_split)

    add_outbreak_thresholds(pred_train, pred_val, pred_test)
    preds = save_prediction_tables(pred_train, pred_val, pred_test, out_dir)
    save_split_manifest(pred_train, pred_val, pred_test, windows, out_dir)
    save_row_audit(pred_train, pred_val, pred_test, out_dir)
    save_metrics(preds, out_dir, mase_denom, final_cfg.quantiles)

    # raw predictions and interpretation/sample figures
    raw_test = predict_raw(final_model, test_loader, final_cfg)
    save_plot_predictions(final_model, raw_test, out_dir, n_examples=6)
    save_interpretation(final_model, raw_test, out_dir)

    make_figures(preds, out_dir)

    archive = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
    print(f"Saved output folder: {out_dir}")
    print(f"Saved zip archive : {archive}")


if __name__ == "__main__":
    main()