#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prophet sensitivity runner for grouped district-level monthly dengue forecasting.

Design
- One Prophet model per district.
- Rolling-origin monthly forecasting up to horizon H.
- Train / val / purge / test semantics come from the preprocessing `split` column.
- District forecasts are also summed to national totals for aggregate evaluation.
- Sensitivity keeps the same Prophet pipeline and fixed tuned district parameters from the
  main-model tuning table, then changes one honest Prophet-specific factor at a time.
"""

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

try:
    from prophet import Prophet
    from prophet.utilities import regressor_coefficients
except ImportError:  # pragma: no cover
    from fbprophet import Prophet  # type: ignore
    from fbprophet.utilities import regressor_coefficients  # type: ignore

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


# ---------------------------------------------------------------------
# Pretty names
# ---------------------------------------------------------------------
column_renaming = {
    "Log_NoOfDenguePatients": "Log(Dengue Cases)",
    "District": "District",
    "Date": "Date",
    "Month-year": "Month-Year",
    "yhat": "Predicted Log(Dengue Cases)",
    "yhat_lower": "Predicted Lower Log(Dengue Cases)",
    "yhat_upper": "Predicted Upper Log(Dengue Cases)",
}


def pretty_column_name(col: str) -> str:
    return column_renaming.get(col, col)


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
    horizon: int = 6
    min_history_months: int = 12
    interval_width: float = 0.95
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0
    seasonality_mode: str = "additive"
    use_month_dummies: bool = True
    use_tuning: bool = False
    random_state: int = 42
    fixed_tuning_csv: Optional[str] = None
    ablation_names: str = "full,no_month_dummies"


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


MONTH_DUMMY_COLS = [
    "is_jan", "is_feb", "is_mar", "is_apr", "is_may", "is_jun",
    "is_jul", "is_aug", "is_sep", "is_oct", "is_nov", "is_dec",
]


def add_month_dummies(df: pd.DataFrame, ds_col: str = "ds") -> pd.DataFrame:
    out = df.copy()
    month = pd.to_datetime(out[ds_col]).dt.month
    mapping = {
        1: "is_jan", 2: "is_feb", 3: "is_mar", 4: "is_apr", 5: "is_may", 6: "is_jun",
        7: "is_jul", 8: "is_aug", 9: "is_sep", 10: "is_oct", 11: "is_nov", 12: "is_dec",
    }
    for m, col in mapping.items():
        out[col] = (month == m).astype(int)
    return out


def build_active_month_regressors(hist_df: pd.DataFrame) -> List[str]:
    """Return active month-dummy regressors, dropping one reference month to avoid perfect collinearity."""
    active = [c for c in MONTH_DUMMY_COLS if c in hist_df.columns and hist_df[c].nunique() > 1]
    if len(active) <= 1:
        return active
    # Drop the last active month as reference category.
    return active[:-1]


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

    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="raise").dt.to_period("M").dt.to_timestamp()
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
            f"Gap examples: {examples}."
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
    return date_to_split


def split_windows_from_map(date_to_split: Dict[pd.Timestamp, str]) -> Dict[str, List[pd.Timestamp]]:
    windows = {"train": [], "val": [], "purge": [], "test": [], "all": sorted(date_to_split)}
    for dt in sorted(date_to_split):
        windows[date_to_split[dt]].append(dt)
    return windows


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

def load_fixed_tuning_params(csv_path: Optional[str]) -> Tuple[Dict[str, Dict], pd.DataFrame]:
    if not csv_path:
        return {}, pd.DataFrame()

    df = pd.read_csv(csv_path)
    required = {"District", "changepoint_prior_scale", "seasonality_prior_scale", "seasonality_mode", "selected"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Fixed tuning CSV is missing required columns: {missing}")

    sel = df[df["selected"].astype(bool)].copy()
    if sel.empty:
        raise ValueError("Fixed tuning CSV has no rows marked selected=True.")
    if sel["District"].duplicated().any():
        dups = sel.loc[sel["District"].duplicated(), "District"].tolist()
        raise ValueError(f"Fixed tuning CSV has duplicate selected rows for districts: {dups}")

    mapping: Dict[str, Dict] = {}
    for _, row in sel.iterrows():
        mapping[str(row["District"]).strip()] = {
            "changepoint_prior_scale": float(row["changepoint_prior_scale"]),
            "seasonality_prior_scale": float(row["seasonality_prior_scale"]),
            "seasonality_mode": str(row["seasonality_mode"]),
        }
    return mapping, sel.copy()


def validate_ablations(ablation_names: str) -> List[str]:
    allowed = {"full", "no_month_dummies"}
    names = [x.strip() for x in str(ablation_names).split(",") if x.strip()]
    if not names:
        raise ValueError("No ablation names were provided.")
    bad = sorted(set(names) - allowed)
    if bad:
        raise ValueError(
            f"Unsupported Prophet sensitivity experiments: {bad}. "
            f"Allowed experiments are: {sorted(allowed)}"
        )
    return names


def apply_experiment_to_cfg(cfg: Config, experiment: str) -> Config:
    exp_cfg = Config(**cfg.__dict__)
    exp_cfg.use_tuning = False
    if experiment == "full":
        exp_cfg.use_month_dummies = True
    elif experiment == "no_month_dummies":
        exp_cfg.use_month_dummies = False
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    return exp_cfg


def run_one_experiment(
    df: pd.DataFrame,
    base_cfg: Config,
    date_to_split: Dict[pd.Timestamp, str],
    windows: Dict[str, List[pd.Timestamp]],
    y_lookup: Dict[Tuple[str, pd.Timestamp], float],
    mase_denom: float,
    experiment: str,
    out_dir: Path,
    fixed_params_by_district: Dict[str, Dict],
    selected_params_table: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    exp_cfg = apply_experiment_to_cfg(base_cfg, experiment)
    ensure_dir(out_dir)

    run_cfg = dict(exp_cfg.__dict__)
    run_cfg["experiment"] = experiment
    save_json(run_cfg, out_dir / "run_config.json")
    write_run_summary(exp_cfg, windows, out_dir)

    if len(selected_params_table):
        tbl = selected_params_table.copy()
        tbl["experiment"] = experiment
        tbl.to_csv(out_dir / "prophet_fixed_selected_params.csv", index=False)

    district_pred_frames = []
    district_audit_frames = []
    selected_params_by_district: Dict[str, Dict] = {}

    for district, g in df.groupby(exp_cfg.district_col):
        district_key = str(district)
        selected_params = fixed_params_by_district.get(district_key, {
            "changepoint_prior_scale": exp_cfg.changepoint_prior_scale,
            "seasonality_prior_scale": exp_cfg.seasonality_prior_scale,
            "seasonality_mode": exp_cfg.seasonality_mode,
        })
        selected_params_by_district[district_key] = selected_params
        pred_df, audit_df = rolling_forecasts_for_district(g.copy(), exp_cfg, date_to_split, y_lookup, selected_params=selected_params)
        district_pred_frames.append(pred_df)
        district_audit_frames.append(audit_df)

    pred_all = pd.concat(district_pred_frames, ignore_index=True) if district_pred_frames else pd.DataFrame()
    audit_all = pd.concat(district_audit_frames, ignore_index=True) if district_audit_frames else pd.DataFrame()

    save_audit_tables(audit_all, out_dir)
    preds = save_prediction_tables(pred_all, out_dir)
    save_split_manifest(preds, out_dir)
    save_metrics(preds, out_dir, mase_denom)
    save_national_summary_from_district_sum(preds, out_dir, mase_denom)
    save_final_train_diagnostics(df, exp_cfg, out_dir, selected_params_by_district)
    make_figures(preds, out_dir)

    tuning_manifest = selected_params_table.copy() if len(selected_params_table) else pd.DataFrame([
        {"District": d, **p, "status": "fixed_default", "selected": True} for d, p in selected_params_by_district.items()
    ])
    if len(tuning_manifest):
        tuning_manifest["experiment"] = experiment
        tuning_manifest.to_csv(out_dir / "prophet_tuning_results.csv", index=False)

    archive = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
    print(f"[{experiment}] Saved output folder: {out_dir}")
    print(f"[{experiment}] Saved zip archive : {archive}")
    return preds


def save_sensitivity_root_tables(root_dir: Path, experiment_results: Dict[str, Dict[str, pd.DataFrame]], mase_denom: float) -> None:
    rows = []
    per_h_rows = []

    for experiment, preds in experiment_results.items():
        test_df = preds["test"]
        m = compute_metrics(test_df["y_true_count"], test_df["y_pred_count"], mase_denom)
        m.update({
            "experiment": experiment,
            "Coverage": float(test_df["coverage"].mean()) if len(test_df) else np.nan,
            "MeanIntervalWidth": float(test_df["interval_width_count"].mean()) if len(test_df) else np.nan,
            "month_dummies_enabled": experiment != "no_month_dummies",
        })
        rows.append(m)

        for h, g_h in test_df.groupby("horizon"):
            mh = compute_metrics(g_h["y_true_count"], g_h["y_pred_count"], mase_denom)
            mh.update({
                "experiment": experiment,
                "horizon": h,
                "Coverage": float(g_h["coverage"].mean()) if len(g_h) else np.nan,
                "MeanIntervalWidth": float(g_h["interval_width_count"].mean()) if len(g_h) else np.nan,
                "month_dummies_enabled": experiment != "no_month_dummies",
            })
            per_h_rows.append(mh)

    summary_df = pd.DataFrame(rows).sort_values("experiment")
    summary_df.to_csv(root_dir / "prophet_sensitivity_experiments.csv", index=False)

    per_h_df = pd.DataFrame(per_h_rows).sort_values(["experiment", "horizon"])
    per_h_df.to_csv(root_dir / "prophet_sensitivity_per_horizon_long.csv", index=False)

    if len(per_h_df):
        agg = per_h_df.groupby("experiment", as_index=False)[["RMSE", "MAE", "R2", "CVRMSE", "NRMSE", "sMAPE", "MBE", "MedAE", "MASE", "Coverage", "MeanIntervalWidth"]].mean()
        agg["month_dummies_enabled"] = agg["experiment"].ne("no_month_dummies")
        agg.to_csv(root_dir / "prophet_sensitivity_per_horizon_summary.csv", index=False)

    manifest = pd.DataFrame([
        {"experiment": "full", "month_dummies_enabled": True, "notes": "Fixed tuned district Prophet params; month dummy regressors enabled."},
        {"experiment": "no_month_dummies", "month_dummies_enabled": False, "notes": "Same fixed tuned district Prophet params; month dummy regressors disabled."},
    ])
    manifest.to_csv(root_dir / "prophet_sensitivity_feature_manifest.csv", index=False)


# ---------------------------------------------------------------------
# Prophet modeling
# ---------------------------------------------------------------------
def prepare_prophet_history(hist_df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, List[str]]:
    p_df = hist_df[[cfg.date_col, cfg.target_col]].copy()
    p_df = p_df.rename(columns={cfg.date_col: "ds", cfg.target_col: "y"})
    p_df = add_month_dummies(p_df, ds_col="ds")
    active_regs = build_active_month_regressors(p_df) if cfg.use_month_dummies else []
    return p_df, active_regs


def prophet_param_candidates(cfg: Config) -> List[Dict]:
    cps_grid = [0.01, 0.05]
    sps_grid = [5.0, 10.0]
    mode_grid = ["additive", "multiplicative"]
    candidates = []
    for cps in cps_grid:
        for sps in sps_grid:
            for mode in mode_grid:
                candidates.append({
                    "changepoint_prior_scale": cps,
                    "seasonality_prior_scale": sps,
                    "seasonality_mode": mode,
                })
    return candidates


def make_prophet_model(cfg: Config, active_regs: Sequence[str], params: Optional[Dict] = None) -> Prophet:
    params = params or {}
    cps = float(params.get("changepoint_prior_scale", cfg.changepoint_prior_scale))
    sps = float(params.get("seasonality_prior_scale", cfg.seasonality_prior_scale))
    smode = str(params.get("seasonality_mode", cfg.seasonality_mode))
    m = Prophet(
        growth="linear",
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=cps,
        seasonality_prior_scale=sps,
        seasonality_mode=smode,
        interval_width=cfg.interval_width,
    )
    for reg in active_regs:
        m.add_regressor(reg, mode="additive", standardize=False)
    return m


def fit_prophet_one_origin(hist_df: pd.DataFrame, cfg: Config, params: Optional[Dict] = None) -> Tuple[Optional[Prophet], List[str], Optional[str]]:
    if len(hist_df) < cfg.min_history_months:
        return None, [], "insufficient_history"

    p_hist, active_regs = prepare_prophet_history(hist_df, cfg)
    model = make_prophet_model(cfg, active_regs, params)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(p_hist)
        return model, active_regs, None
    except Exception as e:  # pragma: no cover
        return None, active_regs, str(e)


def forecast_from_origin(model: Prophet, source_date: pd.Timestamp, horizon: int, active_regs: Sequence[str]) -> pd.DataFrame:
    future = model.make_future_dataframe(periods=horizon, freq="MS", include_history=False)
    future = add_month_dummies(future, ds_col="ds")
    keep_cols = ["ds"] + list(active_regs)
    future = future[keep_cols]
    fcst = model.predict(future)
    return fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()


def rolling_forecasts_for_district(
    g: pd.DataFrame,
    cfg: Config,
    date_to_split: Dict[pd.Timestamp, str],
    y_lookup: Dict[Tuple[str, pd.Timestamp], float],
    selected_params: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    g = g.sort_values(cfg.date_col).reset_index(drop=True)
    district = str(g[cfg.district_col].iloc[0])
    pred_rows: List[Dict] = []
    audit_rows: List[Dict] = []
    dates = list(g[cfg.date_col])
    y_series = g.set_index(cfg.date_col)[cfg.target_col].astype(float).to_dict()

    for source_date in dates:
        hist_df = g[g[cfg.date_col] <= source_date].copy()
        max_target = source_date + DateOffset(months=cfg.horizon)
        if max_target < min(date_to_split):
            continue

        model, active_regs, fit_error = fit_prophet_one_origin(hist_df, cfg, selected_params)
        if model is None:
            audit_rows.append({
                "District": district,
                "source_date": source_date,
                "history_rows": len(hist_df),
                "fit_ok": False,
                "fit_error": fit_error,
                "active_regressors": ",".join(active_regs),
                "selected_params": json.dumps(selected_params or {}, sort_keys=True),
                "predictions_emitted": 0,
            })
            continue

        try:
            fcst = forecast_from_origin(model, source_date, cfg.horizon, active_regs)
        except Exception as e:  # pragma: no cover
            audit_rows.append({
                "District": district,
                "source_date": source_date,
                "history_rows": len(hist_df),
                "fit_ok": False,
                "fit_error": f"predict_error: {e}",
                "active_regressors": ",".join(active_regs),
                "selected_params": json.dumps(selected_params or {}, sort_keys=True),
                "predictions_emitted": 0,
            })
            continue

        emitted = 0
        source_log = float(y_series[source_date])
        source_count = float(np.clip(np.expm1(source_log), a_min=0, a_max=None))

        for _, row in fcst.iterrows():
            target_date = pd.Timestamp(row["ds"]).to_period("M").to_timestamp()
            split = date_to_split.get(target_date)
            if split not in {"train", "val", "test"}:
                continue
            key = (district, target_date)
            if key not in y_lookup or target_date not in y_series:
                continue
            true_log = float(y_series[target_date])
            true_count = float(np.clip(np.expm1(true_log), a_min=0, a_max=None))
            pred_log = float(row["yhat"])
            pred_lower_log = float(row["yhat_lower"])
            pred_upper_log = float(row["yhat_upper"])
            pred_count = float(np.clip(np.expm1(pred_log), a_min=0, a_max=None))
            pred_lower_count = float(np.clip(np.expm1(pred_lower_log), a_min=0, a_max=None))
            pred_upper_count = float(np.clip(np.expm1(pred_upper_log), a_min=0, a_max=None))
            horizon = (target_date.to_period("M") - source_date.to_period("M")).n
            if horizon < 1 or horizon > cfg.horizon:
                continue

            seasonal_key = (district, target_date - DateOffset(months=12))
            seasonal12_count = y_lookup.get(seasonal_key, np.nan)
            pred_rows.append({
                "District": district,
                "Date": source_date,
                "TargetDate": target_date,
                "source_split": g.loc[g[cfg.date_col] == source_date, cfg.split_col].iloc[0],
                "target_split": split,
                "split": split,
                "horizon": horizon,
                "y_true_log": true_log,
                "y_pred_log": pred_log,
                "y_pred_lower_log": pred_lower_log,
                "y_pred_upper_log": pred_upper_log,
                "y_true_count": true_count,
                "y_pred_count": pred_count,
                "y_pred_lower_count": pred_lower_count,
                "y_pred_upper_count": pred_upper_count,
                "naive_last_count": source_count,
                "seasonal12_count": seasonal12_count,
                "coverage": float(pred_lower_count <= true_count <= pred_upper_count),
                "interval_width_count": float(max(pred_upper_count - pred_lower_count, 0.0)),
                "residual_count": pred_count - true_count,
                "abs_error": abs(pred_count - true_count),
                "sq_error": (pred_count - true_count) ** 2,
                "active_regressors": ",".join(active_regs),
                "changepoint_prior_scale": (selected_params or {}).get("changepoint_prior_scale", cfg.changepoint_prior_scale),
                "seasonality_prior_scale": (selected_params or {}).get("seasonality_prior_scale", cfg.seasonality_prior_scale),
                "seasonality_mode": (selected_params or {}).get("seasonality_mode", cfg.seasonality_mode),
                "model": "Prophet",
            })
            emitted += 1

        audit_rows.append({
            "District": district,
            "source_date": source_date,
            "history_rows": len(hist_df),
            "fit_ok": True,
            "fit_error": "",
            "active_regressors": ",".join(active_regs),
            "selected_params": json.dumps(selected_params or {}, sort_keys=True),
            "predictions_emitted": emitted,
        })

    return pd.DataFrame(pred_rows), pd.DataFrame(audit_rows)


def tune_prophet_for_district(
    g: pd.DataFrame,
    cfg: Config,
    date_to_split: Dict[pd.Timestamp, str],
    y_lookup: Dict[Tuple[str, pd.Timestamp], float],
) -> Tuple[Optional[Dict], pd.DataFrame]:
    district = str(g[cfg.district_col].iloc[0])
    val_dates = sorted([d for d, s in date_to_split.items() if s == "val"])
    if not cfg.use_tuning or not val_dates:
        selected = {
            "changepoint_prior_scale": cfg.changepoint_prior_scale,
            "seasonality_prior_scale": cfg.seasonality_prior_scale,
            "seasonality_mode": cfg.seasonality_mode,
        }
        return selected, pd.DataFrame([{
            "District": district,
            **selected,
            "val_rows": np.nan,
            "val_mae_count": np.nan,
            "val_rmse_count": np.nan,
            "status": "default_no_tuning",
            "selected": True,
        }])

    results = []
    best_params = None
    best_score = np.inf
    candidates = prophet_param_candidates(cfg)

    for params in candidates:
        pred_df, audit_df = rolling_forecasts_for_district(g.copy(), cfg, date_to_split, y_lookup, selected_params=params)
        val_pred = pred_df[pred_df["split"] == "val"].copy()
        fit_fail_rate = np.nan
        if len(audit_df):
            fit_fail_rate = float((~audit_df["fit_ok"]).mean())
        if val_pred.empty:
            results.append({
                "District": district, **params, "val_rows": 0, "val_mae_count": np.nan, "val_rmse_count": np.nan,
                "fit_fail_rate": fit_fail_rate, "status": "no_val_predictions", "selected": False,
            })
            continue
        mae_val = float(mean_absolute_error(val_pred["y_true_count"], val_pred["y_pred_count"]))
        rmse_val = rmse(val_pred["y_true_count"], val_pred["y_pred_count"])
        row = {
            "District": district, **params, "val_rows": int(len(val_pred)),
            "val_mae_count": mae_val, "val_rmse_count": rmse_val, "fit_fail_rate": fit_fail_rate,
            "status": "ok", "selected": False,
        }
        results.append(row)
        if np.isfinite(mae_val) and mae_val < best_score:
            best_score = mae_val
            best_params = params

    if best_params is None:
        best_params = {
            "changepoint_prior_scale": cfg.changepoint_prior_scale,
            "seasonality_prior_scale": cfg.seasonality_prior_scale,
            "seasonality_mode": cfg.seasonality_mode,
        }

    for r in results:
        if all(r.get(k) == best_params.get(k) for k in ["changepoint_prior_scale", "seasonality_prior_scale", "seasonality_mode"]):
            r["selected"] = True

    return best_params, pd.DataFrame(results)


def fit_final_train_model_for_district(
    g: pd.DataFrame,
    cfg: Config,
    diagnostics_dir: Path,
    selected_params: Optional[Dict] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict]]:
    """Fit one final train-only Prophet model per district for diagnostics, components, and coefficients."""
    district = str(g[cfg.district_col].iloc[0])
    train_df = g[g[cfg.split_col] == "train"].copy().sort_values(cfg.date_col)
    if len(train_df) < cfg.min_history_months:
        return None, None, {
            "District": district,
            "train_rows": len(train_df),
            "fit_ok": False,
            "fit_error": "insufficient_history",
            "active_regressors": "",
            "changepoints_count": np.nan,
            "sigma_obs": np.nan,
            "changepoint_prior_scale": (selected_params or {}).get("changepoint_prior_scale", cfg.changepoint_prior_scale),
            "seasonality_prior_scale": (selected_params or {}).get("seasonality_prior_scale", cfg.seasonality_prior_scale),
            "seasonality_mode": (selected_params or {}).get("seasonality_mode", cfg.seasonality_mode),
        }

    model, active_regs, fit_error = fit_prophet_one_origin(train_df, cfg, selected_params)
    if model is None:
        return None, None, {
            "District": district,
            "train_rows": len(train_df),
            "fit_ok": False,
            "fit_error": fit_error,
            "active_regressors": ",".join(active_regs),
            "changepoints_count": np.nan,
            "sigma_obs": np.nan,
            "changepoint_prior_scale": (selected_params or {}).get("changepoint_prior_scale", cfg.changepoint_prior_scale),
            "seasonality_prior_scale": (selected_params or {}).get("seasonality_prior_scale", cfg.seasonality_prior_scale),
            "seasonality_mode": (selected_params or {}).get("seasonality_mode", cfg.seasonality_mode),
        }

    p_hist, _ = prepare_prophet_history(train_df, cfg)
    hist_pred = model.predict(p_hist[["ds"] + active_regs])
    train_diag = pd.DataFrame({
        "ds": p_hist["ds"].values,
        "y_true_log": p_hist["y"].values,
        "y_pred_log": hist_pred["yhat"].values,
        "y_pred_lower_log": hist_pred["yhat_lower"].values,
        "y_pred_upper_log": hist_pred["yhat_upper"].values,
    })
    train_diag["y_true_count"] = np.clip(np.expm1(train_diag["y_true_log"]), a_min=0, a_max=None)
    train_diag["y_pred_count"] = np.clip(np.expm1(train_diag["y_pred_log"]), a_min=0, a_max=None)
    train_diag["residual_log"] = train_diag["y_true_log"] - train_diag["y_pred_log"]
    train_diag["residual_count"] = train_diag["y_true_count"] - train_diag["y_pred_count"]
    train_diag["District"] = district

    # Forecast/fit plot over train + test horizon
    full_dates = pd.DataFrame({"ds": pd.date_range(train_df[cfg.date_col].min(), g[cfg.date_col].max(), freq="MS")})
    full_dates = add_month_dummies(full_dates, ds_col="ds")
    future_input = full_dates[["ds"] + active_regs]
    full_fcst = model.predict(future_input)

    fig = model.plot(full_fcst)
    ax = fig.axes[0]
    ax.scatter(train_df[cfg.date_col], np.expm1(train_df[cfg.target_col].astype(float)), s=18, color="black", label="Train true")
    plt.title(f"Prophet fit and forecast - {district}")
    plt.tight_layout()
    fig.savefig(diagnostics_dir / f"{district}_forecast_plot.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig2 = model.plot_components(full_fcst)
    fig2.savefig(diagnostics_dir / f"{district}_components.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # Residual ACF/PACF
    resid = train_diag["residual_log"].dropna().values
    if len(resid) >= 8:
        fig3, axes = plt.subplots(2, 1, figsize=(10, 8))
        plot_acf(resid, ax=axes[0], lags=min(24, len(resid) - 1), zero=False)
        axes[0].set_title(f"Residual ACF - {district}")
        plot_pacf(resid, ax=axes[1], lags=min(24, max(1, len(resid) // 2 - 1)), zero=False, method="ywm")
        axes[1].set_title(f"Residual PACF - {district}")
        plt.tight_layout()
        fig3.savefig(diagnostics_dir / f"{district}_residual_acf_pacf.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig3)

    # Regressor coefficients, if any.
    coef_df = None
    if active_regs:
        try:
            coef_df = regressor_coefficients(model)
            coef_df["District"] = district
        except Exception:
            coef_df = None

    sigma_obs = np.nan
    try:
        sigma_obs = float(np.nanmean(model.params.get("sigma_obs", np.array([np.nan]))))
    except Exception:
        pass

    # Residual diagnostics summary.
    lb_p = np.nan
    if len(resid) >= 8:
        try:
            lb_p = float(acorr_ljungbox(resid, lags=[min(12, len(resid) - 1)], return_df=True)["lb_pvalue"].iloc[0])
        except Exception:
            pass

    diag_summary = {
        "District": district,
        "train_rows": len(train_df),
        "fit_ok": True,
        "fit_error": "",
        "active_regressors": ",".join(active_regs),
        "changepoints_count": int(len(getattr(model, "changepoints", []))),
        "sigma_obs": sigma_obs,
        "train_rmse_log": rmse(train_diag["y_true_log"], train_diag["y_pred_log"]),
        "train_mae_log": float(mean_absolute_error(train_diag["y_true_log"], train_diag["y_pred_log"])),
        "train_rmse_count": rmse(train_diag["y_true_count"], train_diag["y_pred_count"]),
        "train_mae_count": float(mean_absolute_error(train_diag["y_true_count"], train_diag["y_pred_count"])),
        "residual_ljungbox_pvalue": lb_p,
        "changepoint_prior_scale": (selected_params or {}).get("changepoint_prior_scale", cfg.changepoint_prior_scale),
        "seasonality_prior_scale": (selected_params or {}).get("seasonality_prior_scale", cfg.seasonality_prior_scale),
        "seasonality_mode": (selected_params or {}).get("seasonality_mode", cfg.seasonality_mode),
    }

    return train_diag, coef_df, diag_summary


# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------
def aggregate_predictions(pred_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    return pred_df[pred_df["split"] == split_name].copy().reset_index(drop=True)


def save_prediction_tables(pred_df: pd.DataFrame, out_dir: Path):
    pred_train = aggregate_predictions(pred_df, "train")
    pred_val = aggregate_predictions(pred_df, "val")
    pred_test = aggregate_predictions(pred_df, "test")
    pred_train.to_csv(out_dir / "prophet_train_predictions_long.csv", index=False)
    pred_val.to_csv(out_dir / "prophet_val_predictions_long.csv", index=False)
    pred_test.to_csv(out_dir / "prophet_test_predictions_long.csv", index=False)
    return {"train": pred_train, "val": pred_val, "test": pred_test}


def save_audit_tables(audit_df: pd.DataFrame, out_dir: Path):
    audit_df.to_csv(out_dir / "prophet_row_audit.csv", index=False)


def save_tuning_tables(tuning_df: pd.DataFrame, out_dir: Path):
    if tuning_df is not None and len(tuning_df):
        tuning_df.to_csv(out_dir / "prophet_tuning_results.csv", index=False)
        sel = tuning_df[tuning_df["selected"] == True].copy()
        if len(sel):
            sel.to_csv(out_dir / "prophet_selected_params.csv", index=False)


def save_split_manifest(preds, out_dir: Path):
    rows = []
    concat = pd.concat(preds.values(), ignore_index=True)
    for h, g_h in concat.groupby("horizon"):
        for split, g in g_h.groupby("split"):
            rows.append({
                "horizon": h,
                "split": split,
                "source_date_start": g["Date"].min() if len(g) else pd.NaT,
                "source_date_end": g["Date"].max() if len(g) else pd.NaT,
                "target_date_start": g["TargetDate"].min() if len(g) else pd.NaT,
                "target_date_end": g["TargetDate"].max() if len(g) else pd.NaT,
                "n_rows": int(len(g)),
                "n_districts": int(g["District"].nunique()) if len(g) else 0,
            })
    pd.DataFrame(rows).sort_values(["horizon", "split"]).to_csv(out_dir / "prophet_split_manifest.csv", index=False)


def save_metrics(preds, out_dir: Path, mase_denom: float):
    rows = []
    for split, df_ in preds.items():
        m = compute_metrics(df_["y_true_count"], df_["y_pred_count"], mase_denom)
        m.update({
            "Model": "Prophet",
            "Split": split,
            "Coverage": float(df_["coverage"].mean()) if len(df_) else np.nan,
            "MeanIntervalWidth": float(df_["interval_width_count"].mean()) if len(df_) else np.nan,
        })
        rows.append(m)

    base1 = compute_metrics(preds["test"]["y_true_count"], preds["test"]["naive_last_count"], mase_denom)
    base1.update({"Model": "NaiveLast", "Split": "test", "Coverage": np.nan, "MeanIntervalWidth": np.nan})
    rows.append(base1)

    mask_seas = preds["test"]["seasonal12_count"].notna()
    base2 = compute_metrics(preds["test"].loc[mask_seas, "y_true_count"], preds["test"].loc[mask_seas, "seasonal12_count"], mase_denom)
    base2.update({"Model": "SeasonalNaive12", "Split": "test", "Coverage": np.nan, "MeanIntervalWidth": np.nan})
    rows.append(base2)
    pd.DataFrame(rows)[[
        "Model", "Split", "n", "RMSE", "MAE", "R2", "CVRMSE", "NRMSE", "sMAPE", "MBE", "MedAE", "MASE", "Coverage", "MeanIntervalWidth"
    ]].to_csv(out_dir / "prophet_summary.csv", index=False)

    per_h = []
    for h, g_h in preds["test"].groupby("horizon"):
        m = compute_metrics(g_h["y_true_count"], g_h["y_pred_count"], mase_denom)
        train_h = preds["train"][preds["train"]["horizon"] == h]
        if len(train_h):
            threshold = float(np.quantile(train_h["y_true_count"], 0.90))
            cls = safe_classification_metrics((g_h["y_true_count"] >= threshold).astype(int), g_h["y_pred_count"], threshold)
        else:
            threshold = np.nan
            cls = safe_classification_metrics([], [], np.nan)
        m.update({
            "horizon": h,
            "OutbreakThreshold": threshold,
            "Coverage": float(g_h["coverage"].mean()) if len(g_h) else np.nan,
            "MeanIntervalWidth": float(g_h["interval_width_count"].mean()) if len(g_h) else np.nan,
        })
        m.update(cls)
        per_h.append(m)
    pd.DataFrame(per_h).sort_values("horizon").to_csv(out_dir / "prophet_per_horizon.csv", index=False)

    district_rows = []
    for district, g in preds["test"].groupby("District"):
        m = compute_metrics(g["y_true_count"], g["y_pred_count"], naive_denom=np.nan)
        m.update({"District": district, "Coverage": float(g["coverage"].mean()), "MeanIntervalWidth": float(g["interval_width_count"].mean())})
        district_rows.append(m)
    pd.DataFrame(district_rows).sort_values("District").to_csv(out_dir / "prophet_metrics_by_district.csv", index=False)

    dh_rows = []
    for (district, h), g in preds["test"].groupby(["District", "horizon"]):
        m = compute_metrics(g["y_true_count"], g["y_pred_count"], naive_denom=np.nan)
        m.update({"District": district, "horizon": h, "Coverage": float(g["coverage"].mean()), "MeanIntervalWidth": float(g["interval_width_count"].mean())})
        dh_rows.append(m)
    pd.DataFrame(dh_rows).sort_values(["District", "horizon"]).to_csv(out_dir / "prophet_metrics_by_district_horizon.csv", index=False)


def save_national_summary_from_district_sum(preds, out_dir: Path, mase_denom: float):
    rows = []
    for split, df_ in preds.items():
        agg = df_.groupby(["TargetDate", "horizon"], as_index=False)[[
            "y_true_count", "y_pred_count", "naive_last_count", "seasonal12_count", "coverage", "interval_width_count"
        ]].sum()
        # Coverage should be mean, not sum.
        cov_mean = float(df_["coverage"].mean()) if len(df_) else np.nan
        int_w = float(df_["interval_width_count"].mean()) if len(df_) else np.nan
        m = compute_metrics(agg["y_true_count"], agg["y_pred_count"], mase_denom)
        m.update({"Model": "Prophet", "Split": split, "Coverage": cov_mean, "MeanIntervalWidth": int_w})
        rows.append(m)
        agg.to_csv(out_dir / f"prophet_{split}_predictions_national.csv", index=False)

    test_agg = preds["test"].groupby(["TargetDate", "horizon"], as_index=False)[[
        "y_true_count", "y_pred_count", "naive_last_count", "seasonal12_count"
    ]].sum()
    base1 = compute_metrics(test_agg["y_true_count"], test_agg["naive_last_count"], mase_denom)
    base1.update({"Model": "NaiveLast", "Split": "test", "Coverage": np.nan, "MeanIntervalWidth": np.nan})
    rows.append(base1)
    mask = test_agg["seasonal12_count"].notna()
    base2 = compute_metrics(test_agg.loc[mask, "y_true_count"], test_agg.loc[mask, "seasonal12_count"], mase_denom)
    base2.update({"Model": "SeasonalNaive12", "Split": "test", "Coverage": np.nan, "MeanIntervalWidth": np.nan})
    rows.append(base2)
    pd.DataFrame(rows)[[
        "Model", "Split", "n", "RMSE", "MAE", "R2", "CVRMSE", "NRMSE", "sMAPE", "MBE", "MedAE", "MASE", "Coverage", "MeanIntervalWidth"
    ]].to_csv(out_dir / "prophet_summary_national_from_district_sum.csv", index=False)


def save_final_train_diagnostics(df: pd.DataFrame, cfg: Config, out_dir: Path, selected_params_by_district: Optional[Dict[str, Dict]] = None):
    diagnostics_dir = out_dir / "diagnostics"
    ensure_dir(diagnostics_dir)

    model_rows: List[Dict] = []
    coef_frames: List[pd.DataFrame] = []
    residual_frames: List[pd.DataFrame] = []

    for district, g in df.groupby(cfg.district_col):
        selected_params = (selected_params_by_district or {}).get(str(district), None)
        train_diag, coef_df, diag_summary = fit_final_train_model_for_district(g.copy(), cfg, diagnostics_dir, selected_params)
        if diag_summary is not None:
            model_rows.append(diag_summary)
        if coef_df is not None and len(coef_df):
            coef_frames.append(coef_df)
        if train_diag is not None and len(train_diag):
            residual_frames.append(train_diag)

    if model_rows:
        pd.DataFrame(model_rows).sort_values("District").to_csv(out_dir / "prophet_model_diagnostics.csv", index=False)
    if coef_frames:
        pd.concat(coef_frames, ignore_index=True).to_csv(out_dir / "prophet_regressor_coefficients.csv", index=False)
    if residual_frames:
        pd.concat(residual_frames, ignore_index=True).to_csv(out_dir / "prophet_train_fitted_residuals.csv", index=False)


def make_figures(preds, out_dir: Path):
    test = preds["test"]
    if test.empty:
        return

    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=test, x="y_true_count", y="y_pred_count", alpha=0.7, s=60)
    m_max = max(test["y_true_count"].max(), test["y_pred_count"].max()) * 1.5
    plt.plot([0, m_max], [0, m_max], linestyle="--", color="gray", label="Perfect Forecast")
    plt.xscale("symlog", linthresh=10)
    plt.yscale("symlog", linthresh=10)
    plt.xlabel("True Dengue Cases (SymLog Scale)")
    plt.ylabel("Predicted Dengue Cases (SymLog Scale)")
    plt.title("Prophet Test: True vs Predicted")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_test.pdf", dpi=300)
    plt.close()

    per_h = pd.read_csv(out_dir / "prophet_per_horizon.csv")
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
    sns.barplot(data=per_h, x="horizon", y="Coverage")
    plt.title("Prediction Interval Coverage by Horizon")
    plt.xlabel("Forecast Horizon (Months)")
    plt.ylabel("Coverage")
    plt.tight_layout()
    plt.savefig(out_dir / "per_h_coverage.pdf", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=per_h, x="horizon", y="MeanIntervalWidth")
    plt.title("Prediction Interval Width by Horizon")
    plt.xlabel("Forecast Horizon (Months)")
    plt.ylabel("Mean interval width (count scale)")
    plt.tight_layout()
    plt.savefig(out_dir / "per_h_interval_width.pdf", dpi=300)
    plt.close()

    h1_test = test[test["horizon"] == 1]
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

        nat = h1_test.groupby("TargetDate", as_index=False)[["y_true_count", "y_pred_count"]].sum()
        plt.figure(figsize=(14, 5))
        plt.plot(nat["TargetDate"], nat["y_true_count"], marker="o", linewidth=2, label="True National Total")
        plt.plot(nat["TargetDate"], nat["y_pred_count"], marker="X", linewidth=2, label="Predicted National Total")
        plt.title("National Total from Summed District Forecasts (Horizon 1, Test)")
        plt.xlabel("Target Date")
        plt.ylabel("Dengue Cases")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(out_dir / "national_h1_timeline_from_district_sum.pdf", dpi=300)
        plt.close()


def write_run_summary(cfg: Config, windows, out_dir: Path):
    lines = [
        "Prophet grouped-district run summary",
        "=" * 80,
        f"Input file: {cfg.input_path}",
        f"Target column: {cfg.target_col}",
        f"Horizon: {cfg.horizon}",
        f"Min history months: {cfg.min_history_months}",
        f"Use month dummies: {cfg.use_month_dummies}",
        f"Seasonality mode: {cfg.seasonality_mode}",
        f"Changepoint prior scale: {cfg.changepoint_prior_scale}",
        f"Seasonality prior scale: {cfg.seasonality_prior_scale}",
        f"Use tuning: {cfg.use_tuning}",
        f"Interval width: {cfg.interval_width}",
        f"Fixed tuning CSV: {cfg.fixed_tuning_csv}",
        f"Ablation names: {cfg.ablation_names}",
        "",
        "Date windows from preprocessing split column",
        "-" * 80,
    ]
    for key in ["train", "val", "purge", "test"]:
        dates = windows[key]
        lines.append(f"{key}: {min(dates)} -> {max(dates)} ({len(dates)} months)")
    lines += [
        "",
        "Model notes",
        "-" * 80,
        "One Prophet model is refit per district and source date (rolling-origin evaluation).",
        "Monthly forecasts use freq='MS'.",
        "Built-in yearly seasonality is disabled and month dummy regressors are used instead for monthly data.",
        "A reference month dummy is dropped to avoid perfect multicollinearity.",
        "Per-district final-train diagnostics, coefficients, and component plots are exported separately.",
        "For sensitivity runs, district-level tuned Prophet parameters can be frozen from a prior tuning table.",
        "The honest Prophet-specific sensitivity factor here is whether month-dummy regressors are enabled.",
    ]
    (out_dir / "run_summary.txt").write_text("\n".join(map(str, lines)), encoding="utf-8")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Grouped district-level Prophet benchmark for dengue forecasting.")
    p.add_argument("--input", type=str, default="./data/raw/prime_dataset_model_input_with_purge.csv")
    p.add_argument("--output_dir", type=str, default="Prophet/outputs/sensitivity")
    p.add_argument("--target_col", type=str, default="Log_NoOfDenguePatients")
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--min_history_months", type=int, default=12)
    p.add_argument("--interval_width", type=float, default=0.95)
    p.add_argument("--changepoint_prior_scale", type=float, default=0.05)
    p.add_argument("--seasonality_prior_scale", type=float, default=10.0)
    p.add_argument("--seasonality_mode", type=str, default="additive", choices=["additive", "multiplicative"])
    p.add_argument("--no_month_dummies", dest="use_month_dummies", action="store_false")
    p.add_argument("--use_tuning", action="store_true")
    p.add_argument("--fixed_tuning_csv", type=str, default="")
    p.add_argument("--ablation_names", type=str, default="full,no_month_dummies")
    p.set_defaults(use_month_dummies=True)
    a = p.parse_args()
    return Config(
        input_path=a.input,
        output_dir=a.output_dir,
        target_col=a.target_col,
        horizon=a.horizon,
        min_history_months=a.min_history_months,
        interval_width=a.interval_width,
        changepoint_prior_scale=a.changepoint_prior_scale,
        seasonality_prior_scale=a.seasonality_prior_scale,
        seasonality_mode=a.seasonality_mode,
        use_month_dummies=a.use_month_dummies,
        use_tuning=a.use_tuning,
        fixed_tuning_csv=(a.fixed_tuning_csv or None),
        ablation_names=a.ablation_names,
    )


def main():
    cfg = parse_args()
    root_out_dir = Path(cfg.output_dir)
    ensure_dir(root_out_dir)
    save_json(cfg.__dict__, root_out_dir / "run_config_root.json")

    df = load_dataset(cfg)
    validate_panel_structure(df, cfg)
    date_to_split = build_date_split_map(df, cfg)
    windows = split_windows_from_map(date_to_split)
    y_lookup = build_lookup(df, cfg)
    mase_denom = compute_panel_mase_denom(df, cfg)

    fixed_params_by_district, selected_params_table = load_fixed_tuning_params(cfg.fixed_tuning_csv)
    experiments = validate_ablations(cfg.ablation_names)

    if cfg.use_tuning:
        print("Sensitivity mode detected: --use_tuning is ignored. Fixed params/default params are used instead.")

    experiment_results: Dict[str, Dict[str, pd.DataFrame]] = {}
    for experiment in experiments:
        exp_out_dir = root_out_dir / experiment
        preds = run_one_experiment(
            df=df,
            base_cfg=cfg,
            date_to_split=date_to_split,
            windows=windows,
            y_lookup=y_lookup,
            mase_denom=mase_denom,
            experiment=experiment,
            out_dir=exp_out_dir,
            fixed_params_by_district=fixed_params_by_district,
            selected_params_table=selected_params_table,
        )
        experiment_results[experiment] = preds

    save_sensitivity_root_tables(root_out_dir, experiment_results, mase_denom)
    archive = shutil.make_archive(str(root_out_dir), "zip", root_dir=root_out_dir)
    print(f"Saved Prophet sensitivity root folder: {root_out_dir}")
    print(f"Saved Prophet sensitivity root zip : {archive}")


if __name__ == "__main__":
    main()