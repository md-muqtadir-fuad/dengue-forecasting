#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grouped district-level SARIMAX benchmark for dengue forecasting.

Design:
- One SARIMAX model per district
- Uses the existing preprocessing split column: train / val / purge / test
- Keeps purge rows in the input as source-context months
- Forecasts on the log target column (default: Log_NoOfDenguePatients)
- Uses differencing by default through order=(1,1,1) and seasonal_order=(0,1,1,12)
- Evaluates district-level predictions and national totals obtained by summing district forecasts
- Adds per-district diagnostics, ACF/PACF residual plots, parameter tables, and forecast intervals

Expected input columns:
- District
- Date
- split
- Log_NoOfDenguePatients
- Optional Month-year

Notes:
- No exogenous regressors are used by default to avoid future-exog leakage in multi-step forecasting.
"""

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


@dataclass
class Config:
    input_path: str
    output_dir: str
    target_col: str = "Log_NoOfDenguePatients"
    district_col: str = "District"
    date_col: str = "Date"
    split_col: str = "split"
    monthyear_col: str = "Month-year"
    horizon: int = 6
    random_state: int = 42
    order: Tuple[int, int, int] = (1, 1, 1)
    seasonal_order: Tuple[int, int, int, int] = (0, 1, 1, 12)
    min_train_points: int = 24
    maxiter: int = 200
    max_pred_log_clip: float = 12.0
    diagnostics_lags: int = 24
    use_order_search: bool = True
    adf_alpha: float = 0.05


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def safe_inverse_log1p(value: float, max_log_clip: float = 12.0) -> float:
    if not np.isfinite(value):
        return np.nan
    value = float(np.clip(value, a_min=-20.0, a_max=max_log_clip))
    return float(np.expm1(value))


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


def _format_p_for_display(p: float) -> str:
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _run_adf(series: pd.Series, alpha: float) -> Dict:
    s = pd.Series(series, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 8:
        return {
            "adf_statistic": np.nan,
            "p_value": np.nan,
            "p_value_display": "NA",
            "used_lag": np.nan,
            "nobs": int(len(s)),
            "critical_value_1pct": np.nan,
            "critical_value_5pct": np.nan,
            "critical_value_10pct": np.nan,
            "icbest": np.nan,
            "conclusion": "Insufficient Data",
            "fit_ok": False,
            "error": "Series too short for ADF.",
        }
    if s.nunique() <= 1:
        return {
            "adf_statistic": np.nan,
            "p_value": np.nan,
            "p_value_display": "NA",
            "used_lag": np.nan,
            "nobs": int(len(s)),
            "critical_value_1pct": np.nan,
            "critical_value_5pct": np.nan,
            "critical_value_10pct": np.nan,
            "icbest": np.nan,
            "conclusion": "Constant Series",
            "fit_ok": False,
            "error": "Series has no variation.",
        }
    try:
        stat, pval, usedlag, nobs, crit, icbest = adfuller(s, autolag="AIC")
        return {
            "adf_statistic": float(stat),
            "p_value": float(pval),
            "p_value_display": _format_p_for_display(float(pval)),
            "used_lag": int(usedlag),
            "nobs": int(nobs),
            "critical_value_1pct": float(crit.get("1%", np.nan)),
            "critical_value_5pct": float(crit.get("5%", np.nan)),
            "critical_value_10pct": float(crit.get("10%", np.nan)),
            "icbest": float(icbest) if np.isfinite(icbest) else np.nan,
            "conclusion": "Stationary" if float(pval) < alpha else "Non-Stationary",
            "fit_ok": True,
            "error": "",
        }
    except Exception as e:
        return {
            "adf_statistic": np.nan,
            "p_value": np.nan,
            "p_value_display": "NA",
            "used_lag": np.nan,
            "nobs": int(len(s)),
            "critical_value_1pct": np.nan,
            "critical_value_5pct": np.nan,
            "critical_value_10pct": np.nan,
            "icbest": np.nan,
            "conclusion": "ADF Failed",
            "fit_ok": False,
            "error": str(e),
        }


def save_adf_outputs(df: pd.DataFrame, cfg: Config, out_dir: Path) -> None:
    diagnostics_dir = out_dir / "diagnostics"
    adf_dir = diagnostics_dir / "adf"
    ensure_dir(diagnostics_dir)
    ensure_dir(adf_dir)

    long_rows = []
    wide_rows = []

    for district, g in df.groupby(cfg.district_col):
        g = g.sort_values(cfg.date_col).reset_index(drop=True)
        hist = g[g[cfg.split_col] != "test"].copy()
        y_log = hist[cfg.target_col].astype(float)
        y_diff1 = y_log.diff().dropna()

        series_map = {
            "Log1p Transformed": y_log,
            "First-Order Differenced": y_diff1,
        }

        district_simple_rows = []
        district_wide = {
            "district": district,
            "split_scope": "non-test history",
            "source_start_date": hist[cfg.date_col].min() if len(hist) else pd.NaT,
            "source_end_date": hist[cfg.date_col].max() if len(hist) else pd.NaT,
            "alpha": cfg.adf_alpha,
        }

        for series_name, series_values in series_map.items():
            adf_row = _run_adf(series_values, cfg.adf_alpha)
            out_row = {
                "district": district,
                "split_scope": "non-test history",
                "source_start_date": hist[cfg.date_col].min() if len(hist) else pd.NaT,
                "source_end_date": hist[cfg.date_col].max() if len(hist) else pd.NaT,
                "Time Series": series_name,
                "ADF Statistic": adf_row["adf_statistic"],
                "p-value": adf_row["p_value"],
                "p_value_display": adf_row["p_value_display"],
                "Conclusion (at α = 0.05)": adf_row["conclusion"],
                "used_lag": adf_row["used_lag"],
                "nobs": adf_row["nobs"],
                "critical_value_1pct": adf_row["critical_value_1pct"],
                "critical_value_5pct": adf_row["critical_value_5pct"],
                "critical_value_10pct": adf_row["critical_value_10pct"],
                "icbest": adf_row["icbest"],
                "fit_ok": adf_row["fit_ok"],
                "error": adf_row["error"],
            }
            long_rows.append(out_row)
            district_simple_rows.append({
                "Time Series": series_name,
                "ADF Statistic": adf_row["adf_statistic"],
                "p-value": adf_row["p_value_display"],
                "Conclusion (at α = 0.05)": adf_row["conclusion"],
            })

            prefix = "log1p" if series_name == "Log1p Transformed" else "diff1"
            district_wide[f"{prefix}_adf_statistic"] = adf_row["adf_statistic"]
            district_wide[f"{prefix}_p_value"] = adf_row["p_value"]
            district_wide[f"{prefix}_p_value_display"] = adf_row["p_value_display"]
            district_wide[f"{prefix}_conclusion"] = adf_row["conclusion"]
            district_wide[f"{prefix}_used_lag"] = adf_row["used_lag"]
            district_wide[f"{prefix}_nobs"] = adf_row["nobs"]

        wide_rows.append(district_wide)
        pd.DataFrame(district_simple_rows).to_csv(adf_dir / f"{district}_adf_table.csv", index=False)

    if long_rows:
        long_df = pd.DataFrame(long_rows).sort_values(["district", "Time Series"])
        long_df.to_csv(out_dir / "sarimax_adf_results_long.csv", index=False)
        long_df[["district", "Time Series", "ADF Statistic", "p-value", "Conclusion (at α = 0.05)"]].to_csv(
            out_dir / "sarimax_adf_results_for_manuscript.csv", index=False
        )
    if wide_rows:
        wide_df = pd.DataFrame(wide_rows).sort_values("district")
        wide_df.to_csv(out_dir / "sarimax_adf_summary_by_district.csv", index=False)
        overall = pd.DataFrame([
            {
                "series": "Log1p Transformed",
                "n_districts": int(len(wide_df)),
                "stationary_count": int((wide_df["log1p_conclusion"] == "Stationary").sum()),
                "non_stationary_count": int((wide_df["log1p_conclusion"] == "Non-Stationary").sum()),
            },
            {
                "series": "First-Order Differenced",
                "n_districts": int(len(wide_df)),
                "stationary_count": int((wide_df["diff1_conclusion"] == "Stationary").sum()),
                "non_stationary_count": int((wide_df["diff1_conclusion"] == "Non-Stationary").sum()),
            },
        ])
        overall.to_csv(out_dir / "sarimax_adf_overall_summary.csv", index=False)


def make_candidate_specs(cfg: Config) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]]:
    specs = [(cfg.order, cfg.seasonal_order)]
    if cfg.use_order_search:
        specs.extend([
            ((1, 1, 0), (0, 1, 1, 12)),
            ((0, 1, 1), (0, 1, 1, 12)),
            ((1, 1, 1), (1, 1, 0, 12)),
            ((0, 1, 1), (1, 1, 0, 12)),
            ((1, 1, 0), (0, 1, 0, 12)),
            ((0, 1, 1), (0, 1, 0, 12)),
        ])
    # de-duplicate while preserving order
    seen = set()
    out = []
    for spec in specs:
        if spec not in seen:
            seen.add(spec)
            out.append(spec)
    return out


def try_fit_sarimax(y_train: pd.Series, cfg: Config):
    best = None
    candidate_rows = []
    last_exc = None
    for order, seas in make_candidate_specs(cfg):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    y_train,
                    order=order,
                    seasonal_order=seas,
                    trend="n",
                    simple_differencing=False,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = model.fit(disp=False, maxiter=cfg.maxiter)
            converged = bool(getattr(res, "mle_retvals", {}).get("converged", True))
            aic = float(getattr(res, "aic", np.nan))
            bic = float(getattr(res, "bic", np.nan))
            score = (np.inf if not np.isfinite(aic) else aic, np.inf if not converged else 0.0)
            candidate_rows.append({
                "order_used": str(order),
                "seasonal_order_used": str(seas),
                "aic": aic,
                "bic": bic,
                "converged": converged,
                "fit_failed": False,
                "error": "",
            })
            if best is None or score < best["score"]:
                best = {"res": res, "order": order, "seas": seas, "score": score}
        except Exception as e:
            last_exc = e
            candidate_rows.append({
                "order_used": str(order),
                "seasonal_order_used": str(seas),
                "aic": np.nan,
                "bic": np.nan,
                "converged": False,
                "fit_failed": True,
                "error": str(e),
            })
    if best is None:
        raise RuntimeError(f"All SARIMAX candidate specifications failed. Last error: {last_exc}")
    return best["res"], best["order"], best["seas"], candidate_rows


def _extract_lb_pvalue(res, lags: int = 12) -> float:
    try:
        arr = np.asarray(res.test_serial_correlation(method="ljungbox", lags=[lags]))
        if arr.ndim == 3:
            return float(arr[0, 1, -1])
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return float(arr[0, 1])
        flat = arr.ravel()
        return float(flat[-1]) if len(flat) else np.nan
    except Exception:
        return np.nan


def _extract_jb_pvalue(res) -> float:
    try:
        arr = np.asarray(res.test_normality(method="jarquebera"))
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return float(arr[0, 1])
        flat = arr.ravel()
        return float(flat[1]) if len(flat) > 1 else np.nan
    except Exception:
        return np.nan


def _extract_het_pvalue(res) -> float:
    try:
        arr = np.asarray(res.test_heteroskedasticity(method="breakvar"))
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return float(arr[0, 1])
        flat = arr.ravel()
        return float(flat[1]) if len(flat) > 1 else np.nan
    except Exception:
        return np.nan


def fit_reference_model_for_district(g: pd.DataFrame, cfg: Config) -> Dict:
    district = str(g[cfg.district_col].iloc[0])
    g = g.sort_values(cfg.date_col).reset_index(drop=True)
    # Use all non-test months as the reference in-sample fit for diagnostics.
    hist = g[g[cfg.split_col] != "test"].copy()
    y_train = hist[cfg.target_col].astype(float).reset_index(drop=True)
    if len(y_train) < cfg.min_train_points:
        return {
            "district": district,
            "fit_ok": False,
            "reason": f"insufficient history: {len(y_train)}",
        }
    try:
        res, order_used, seas_used, cand = try_fit_sarimax(y_train, cfg)
        return {
            "district": district,
            "fit_ok": True,
            "res": res,
            "order_used": order_used,
            "seasonal_order_used": seas_used,
            "train_points_used": int(len(y_train)),
            "candidate_rows": cand,
            "start_date": hist[cfg.date_col].min(),
            "end_date": hist[cfg.date_col].max(),
        }
    except Exception as e:
        return {
            "district": district,
            "fit_ok": False,
            "reason": str(e),
        }


def rolling_forecasts_for_district(
    g: pd.DataFrame,
    cfg: Config,
    h: int,
    date_to_split: Dict[pd.Timestamp, str],
    y_lookup: Dict[Tuple[str, pd.Timestamp], float],
) -> Tuple[pd.DataFrame, List[Dict]]:
    district = str(g[cfg.district_col].iloc[0])
    g = g.sort_values(cfg.date_col).reset_index(drop=True)
    all_dates = g[cfg.date_col].tolist()
    y_all = g[cfg.target_col].astype(float).reset_index(drop=True)
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    pred_rows: List[Dict] = []
    audit_rows: List[Dict] = []

    for split_name in ["train", "val", "test"]:
        target_dates = [d for d, s in date_to_split.items() if s == split_name]
        attempted = 0
        usable = 0
        skipped_no_origin = 0
        skipped_short_history = 0
        fit_failures = 0

        for target_date in target_dates:
            attempted += 1
            origin_date = pd.Timestamp(target_date) - DateOffset(months=h)
            if origin_date not in date_to_idx:
                skipped_no_origin += 1
                continue
            origin_idx = date_to_idx[origin_date]
            y_train = y_all.iloc[: origin_idx + 1].copy()
            if len(y_train) < cfg.min_train_points:
                skipped_short_history += 1
                continue
            try:
                res, order_used, seas_used, _cand = try_fit_sarimax(y_train, cfg)
                fc = res.get_forecast(steps=h)
                y_pred_log = float(fc.predicted_mean.iloc[-1])
                ci = fc.conf_int(alpha=0.05)
                lower_log = float(ci.iloc[-1, 0]) if ci.shape[1] >= 1 else np.nan
                upper_log = float(ci.iloc[-1, 1]) if ci.shape[1] >= 2 else np.nan
                converged = bool(getattr(res, "mle_retvals", {}).get("converged", True))
                aic = float(getattr(res, "aic", np.nan))
                bic = float(getattr(res, "bic", np.nan))
            except Exception:
                fit_failures += 1
                continue

            true_idx = date_to_idx.get(pd.Timestamp(target_date), None)
            if true_idx is None:
                continue
            y_true_log = float(y_all.iloc[true_idx])
            y_true_count = safe_inverse_log1p(y_true_log, cfg.max_pred_log_clip)
            y_pred_count = safe_inverse_log1p(y_pred_log, cfg.max_pred_log_clip)
            y_pred_lower_count = safe_inverse_log1p(lower_log, cfg.max_pred_log_clip)
            y_pred_upper_count = safe_inverse_log1p(upper_log, cfg.max_pred_log_clip)
            source_count = safe_inverse_log1p(float(y_all.iloc[origin_idx]), cfg.max_pred_log_clip)
            seasonal12_count = y_lookup.get((district, pd.Timestamp(target_date) - DateOffset(months=12)), np.nan)

            pred_rows.append({
                cfg.district_col: district,
                cfg.date_col: pd.Timestamp(origin_date),
                "TargetDate": pd.Timestamp(target_date),
                "source_split": date_to_split.get(pd.Timestamp(origin_date), "unknown"),
                "target_split": split_name,
                "split": split_name,
                "horizon": h,
                "y_true_log": y_true_log,
                "y_pred_log": y_pred_log,
                "y_pred_lower_log": lower_log,
                "y_pred_upper_log": upper_log,
                "y_true_count": y_true_count,
                "y_pred_count": y_pred_count,
                "y_pred_lower_count": y_pred_lower_count,
                "y_pred_upper_count": y_pred_upper_count,
                "naive_last_count": source_count,
                "seasonal12_count": seasonal12_count,
                "residual_count": y_pred_count - y_true_count if np.isfinite(y_pred_count) and np.isfinite(y_true_count) else np.nan,
                "abs_error": abs(y_pred_count - y_true_count) if np.isfinite(y_pred_count) and np.isfinite(y_true_count) else np.nan,
                "sq_error": float((y_pred_count - y_true_count) ** 2) if np.isfinite(y_pred_count) and np.isfinite(y_true_count) else np.nan,
                "order_used": str(order_used),
                "seasonal_order_used": str(seas_used),
                "train_points_used": int(len(y_train)),
                "converged": converged,
                "aic": aic,
                "bic": bic,
                "model": "SARIMAX",
            })
            usable += 1

        audit_rows.append({
            "district": district,
            "horizon": h,
            "split": split_name,
            "attempted_target_dates": attempted,
            "usable_predictions": usable,
            "skipped_no_origin": skipped_no_origin,
            "skipped_short_history": skipped_short_history,
            "fit_failures": fit_failures,
        })

    return pd.DataFrame(pred_rows), audit_rows


def assign_threshold_and_regime(df_pred: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    train_counts = df_pred.loc[df_pred["split"] == "train", "y_true_count"].values
    threshold = float(np.quantile(train_counts, 0.90)) if len(train_counts) else 0.0
    out = df_pred.copy()
    out["outbreak_threshold"] = threshold
    out["regime"] = np.where(out["y_true_count"] >= threshold, "Outbreak", "Normal")
    return out, threshold


def aggregate_predictions(results, split_name):
    return pd.concat([r[f"pred_{split_name}"] for r in results], ignore_index=True)


def save_prediction_tables(results, out_dir: Path):
    pred_train = aggregate_predictions(results, "train")
    pred_val = aggregate_predictions(results, "val")
    pred_test = aggregate_predictions(results, "test")
    pred_train.to_csv(out_dir / "sarimax_train_predictions_long.csv", index=False)
    pred_val.to_csv(out_dir / "sarimax_val_predictions_long.csv", index=False)
    pred_test.to_csv(out_dir / "sarimax_test_predictions_long.csv", index=False)
    pred_test[[
        "District", "Date", "TargetDate", "horizon", "y_true_count", "y_pred_count",
        "y_pred_lower_count", "y_pred_upper_count", "residual_count", "abs_error", "sq_error"
    ]].to_csv(out_dir / "sarimax_test_residuals_long.csv", index=False)
    return {"train": pred_train, "val": pred_val, "test": pred_test}


def save_row_audit(all_audits: List[Dict], out_dir: Path):
    if all_audits:
        pd.DataFrame(all_audits).sort_values(["district", "horizon", "split"]).to_csv(out_dir / "sarimax_row_audit.csv", index=False)


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
    pd.DataFrame(rows).to_csv(out_dir / "sarimax_split_manifest.csv", index=False)


def aggregate_national_from_district(df_pred: pd.DataFrame) -> pd.DataFrame:
    nat = (
        df_pred.groupby(["split", "horizon", "TargetDate"], as_index=False)[
            ["y_true_count", "y_pred_count", "y_pred_lower_count", "y_pred_upper_count", "naive_last_count", "seasonal12_count"]
        ]
        .sum()
        .sort_values(["split", "horizon", "TargetDate"])
        .reset_index(drop=True)
    )
    nat["model"] = "SARIMAX_from_district_sum"
    return nat


def save_national_predictions(preds: Dict[str, pd.DataFrame], out_dir: Path):
    nat_train = aggregate_national_from_district(preds["train"])
    nat_val = aggregate_national_from_district(preds["val"])
    nat_test = aggregate_national_from_district(preds["test"])
    nat_train.to_csv(out_dir / "sarimax_train_predictions_national.csv", index=False)
    nat_val.to_csv(out_dir / "sarimax_val_predictions_national.csv", index=False)
    nat_test.to_csv(out_dir / "sarimax_test_predictions_national.csv", index=False)
    return {"train": nat_train, "val": nat_val, "test": nat_test}


def save_metrics(results, preds, nat_preds, out_dir: Path, mase_denom: float):
    rows = []
    for split, df_ in preds.items():
        m = compute_metrics(df_["y_true_count"], df_["y_pred_count"], naive_denom=mase_denom)
        m.update({"Model": "SARIMAX", "Split": split})
        rows.append(m)

    base1 = compute_metrics(preds["test"]["y_true_count"], preds["test"]["naive_last_count"], naive_denom=mase_denom)
    base1.update({"Model": "NaiveLast", "Split": "test"})
    rows.append(base1)

    mask_seas = preds["test"]["seasonal12_count"].notna()
    base2 = compute_metrics(preds["test"].loc[mask_seas, "y_true_count"], preds["test"].loc[mask_seas, "seasonal12_count"], naive_denom=mase_denom)
    base2.update({"Model": "SeasonalNaive12", "Split": "test"})
    rows.append(base2)

    pd.DataFrame(rows)[["Model", "Split", "n", "RMSE", "MAE", "R2", "CVRMSE", "NRMSE", "sMAPE", "MBE", "MedAE", "MASE"]].to_csv(out_dir / "sarimax_summary.csv", index=False)

    nat_rows = []
    for split, df_ in nat_preds.items():
        m = compute_metrics(df_["y_true_count"], df_["y_pred_count"], naive_denom=np.nan)
        m.update({"Model": "SARIMAX_from_district_sum", "Split": split})
        nat_rows.append(m)

    nat_base1 = compute_metrics(nat_preds["test"]["y_true_count"], nat_preds["test"]["naive_last_count"], naive_denom=np.nan)
    nat_base1.update({"Model": "NaiveLast_from_district_sum", "Split": "test"})
    nat_rows.append(nat_base1)

    nat_mask_seas = nat_preds["test"]["seasonal12_count"].notna()
    nat_base2 = compute_metrics(nat_preds["test"].loc[nat_mask_seas, "y_true_count"], nat_preds["test"].loc[nat_mask_seas, "seasonal12_count"], naive_denom=np.nan)
    nat_base2.update({"Model": "SeasonalNaive12_from_district_sum", "Split": "test"})
    nat_rows.append(nat_base2)

    pd.DataFrame(nat_rows)[["Model", "Split", "n", "RMSE", "MAE", "R2", "CVRMSE", "NRMSE", "sMAPE", "MBE", "MedAE", "MASE"]].to_csv(out_dir / "sarimax_summary_national_from_district_sum.csv", index=False)

    per_h = []
    for r in results:
        row = {"horizon": r["horizon"], "OutbreakThreshold": r["threshold"]}
        row.update(r["metrics_test"])
        row.update(r["classification_test"])
        per_h.append(row)
    pd.DataFrame(per_h).sort_values("horizon").to_csv(out_dir / "sarimax_per_horizon.csv", index=False)

    regime_rows = []
    for h, g_h in preds["test"].groupby("horizon"):
        for regime, g in g_h.groupby("regime"):
            m = compute_metrics(g["y_true_count"], g["y_pred_count"], naive_denom=np.nan)
            m.update({"horizon": h, "regime": regime})
            regime_rows.append(m)
    pd.DataFrame(regime_rows).sort_values(["horizon", "regime"]).to_csv(out_dir / "sarimax_regimewise_metrics.csv", index=False)

    district_rows = []
    for district, g in preds["test"].groupby("District"):
        m = compute_metrics(g["y_true_count"], g["y_pred_count"], naive_denom=np.nan)
        m.update({"District": district})
        district_rows.append(m)
    pd.DataFrame(district_rows).sort_values("District").to_csv(out_dir / "sarimax_metrics_by_district.csv", index=False)

    dh_rows = []
    for (district, h), g in preds["test"].groupby(["District", "horizon"]):
        m = compute_metrics(g["y_true_count"], g["y_pred_count"], naive_denom=np.nan)
        m.update({"District": district, "horizon": h})
        dh_rows.append(m)
    pd.DataFrame(dh_rows).sort_values(["District", "horizon"]).to_csv(out_dir / "sarimax_metrics_by_district_horizon.csv", index=False)


def save_model_diagnostics(reference_models: List[Dict], out_dir: Path, cfg: Config) -> None:
    diagnostics_dir = out_dir / "diagnostics"
    ensure_dir(diagnostics_dir)
    selection_rows = []
    param_rows = []
    diag_rows = []
    candidate_rows = []

    for item in reference_models:
        district = item["district"]
        if not item.get("fit_ok", False):
            selection_rows.append({
                "district": district,
                "fit_ok": False,
                "reason": item.get("reason", ""),
                "order_used": "",
                "seasonal_order_used": "",
                "aic": np.nan,
                "bic": np.nan,
                "hqic": np.nan,
                "converged": False,
                "train_points_used": np.nan,
                "start_date": pd.NaT,
                "end_date": pd.NaT,
            })
            continue

        res = item["res"]
        converged = bool(getattr(res, "mle_retvals", {}).get("converged", True))
        selection_rows.append({
            "district": district,
            "fit_ok": True,
            "reason": "",
            "order_used": str(item["order_used"]),
            "seasonal_order_used": str(item["seasonal_order_used"]),
            "aic": float(getattr(res, "aic", np.nan)),
            "bic": float(getattr(res, "bic", np.nan)),
            "hqic": float(getattr(res, "hqic", np.nan)),
            "converged": converged,
            "train_points_used": item["train_points_used"],
            "start_date": item["start_date"],
            "end_date": item["end_date"],
        })

        for row in item.get("candidate_rows", []):
            candidate_rows.append({"district": district, **row})

        try:
            ci = res.conf_int()
            ci_df = ci if isinstance(ci, pd.DataFrame) else pd.DataFrame(ci)
            ci_df.columns = ["ci_lower", "ci_upper"]
        except Exception:
            ci_df = pd.DataFrame(index=res.param_names, data={"ci_lower": np.nan, "ci_upper": np.nan})

        for pname, est, se in zip(res.param_names, res.params, getattr(res, "bse", np.repeat(np.nan, len(res.params)))):
            low = ci_df.loc[pname, "ci_lower"] if pname in ci_df.index else np.nan
            high = ci_df.loc[pname, "ci_upper"] if pname in ci_df.index else np.nan
            param_rows.append({
                "district": district,
                "parameter": pname,
                "estimate": float(est) if np.isfinite(est) else np.nan,
                "std_error": float(se) if np.isfinite(se) else np.nan,
                "ci_lower": float(low) if np.isfinite(low) else np.nan,
                "ci_upper": float(high) if np.isfinite(high) else np.nan,
            })

        lb_p = _extract_lb_pvalue(res, lags=min(cfg.diagnostics_lags, max(1, item["train_points_used"] // 3)))
        jb_p = _extract_jb_pvalue(res)
        het_p = _extract_het_pvalue(res)
        diag_rows.append({
            "district": district,
            "ljungbox_pvalue": lb_p,
            "jarquebera_pvalue": jb_p,
            "heteroskedasticity_pvalue": het_p,
            "aic": float(getattr(res, "aic", np.nan)),
            "bic": float(getattr(res, "bic", np.nan)),
            "hqic": float(getattr(res, "hqic", np.nan)),
            "converged": converged,
        })

        # Standard diagnostic plot
        try:
            fig = res.plot_diagnostics(figsize=(12, 8), lags=min(cfg.diagnostics_lags, max(1, item["train_points_used"] // 3)))
            fig.suptitle(f"SARIMAX diagnostics - {district}", y=1.02)
            fig.tight_layout()
            fig.savefig(diagnostics_dir / f"{district}_plot_diagnostics.pdf", dpi=300, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            plt.close("all")

        # Residual ACF/PACF
        resid = pd.Series(np.asarray(res.resid)).dropna()
        if len(resid) >= 8:
            lags = min(cfg.diagnostics_lags, max(1, len(resid) // 2 - 1))
            try:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                plot_acf(resid, lags=lags, ax=axes[0])
                axes[0].set_title(f"Residual ACF - {district}")
                plot_pacf(resid, lags=lags, ax=axes[1], method="ywm")
                axes[1].set_title(f"Residual PACF - {district}")
                fig.tight_layout()
                fig.savefig(diagnostics_dir / f"{district}_residual_acf_pacf.pdf", dpi=300, bbox_inches="tight")
                plt.close(fig)
            except Exception:
                plt.close("all")

    pd.DataFrame(selection_rows).sort_values("district").to_csv(out_dir / "sarimax_model_selection.csv", index=False)
    if candidate_rows:
        pd.DataFrame(candidate_rows).sort_values(["district", "order_used", "seasonal_order_used"]).to_csv(out_dir / "sarimax_order_search_candidates.csv", index=False)
    if param_rows:
        pd.DataFrame(param_rows).sort_values(["district", "parameter"]).to_csv(out_dir / "sarimax_parameter_estimates.csv", index=False)
    if diag_rows:
        pd.DataFrame(diag_rows).sort_values("district").to_csv(out_dir / "sarimax_diagnostics.csv", index=False)


def make_figures(preds: Dict[str, pd.DataFrame], nat_preds: Dict[str, pd.DataFrame], out_dir: Path):
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
    plt.title("SARIMAX Test: True vs Predicted")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_test.pdf", dpi=300)
    plt.close()

    per_h = pd.read_csv(out_dir / "sarimax_per_horizon.csv")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=per_h, x="horizon", y="RMSE")
    plt.title("SARIMAX Test RMSE by Horizon")
    plt.xlabel("Forecast Horizon (Months)")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(out_dir / "per_h_rmse.pdf", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=per_h, x="horizon", y="MAE")
    plt.title("SARIMAX Test MAE by Horizon")
    plt.xlabel("Forecast Horizon (Months)")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(out_dir / "per_h_mae.pdf", dpi=300)
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

    nat_test = nat_preds["test"]
    h1_nat = nat_test[nat_test["horizon"] == 1]
    if not h1_nat.empty:
        plt.figure(figsize=(14, 5))
        plt.plot(h1_nat["TargetDate"], h1_nat["y_true_count"], marker="o", linewidth=2, label="True Cases")
        plt.plot(h1_nat["TargetDate"], h1_nat["y_pred_count"], marker="X", linewidth=2, label="Predicted Cases")
        if "y_pred_lower_count" in h1_nat.columns and "y_pred_upper_count" in h1_nat.columns:
            plt.fill_between(h1_nat["TargetDate"], h1_nat["y_pred_lower_count"], h1_nat["y_pred_upper_count"], alpha=0.2, label="95% PI")
        plt.title("National Horizon 1 Forecast from Summed District SARIMAX Predictions")
        plt.xlabel("Target Date")
        plt.ylabel("Dengue Cases")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(out_dir / "national_h1_timeline_from_district_sum.pdf", dpi=300)
        plt.close()


def write_run_summary(cfg: Config, windows: Dict[str, List[pd.Timestamp]], out_dir: Path) -> None:
    lines = [
        "SARIMAX grouped district run summary",
        "=" * 80,
        f"Input file: {cfg.input_path}",
        f"Target column: {cfg.target_col}",
        f"District column: {cfg.district_col}",
        f"Date column: {cfg.date_col}",
        f"Horizon: {cfg.horizon}",
        f"Order: {cfg.order}",
        f"Seasonal order: {cfg.seasonal_order}",
        f"Use order search: {cfg.use_order_search}",
        f"Min train points per district-origin: {cfg.min_train_points}",
        f"Max iterations per fit: {cfg.maxiter}",
        f"Diagnostics lags: {cfg.diagnostics_lags}",
        "",
        "Date windows from preprocessing split column",
        "-" * 80,
    ]
    for key in ["train", "val", "purge", "test"]:
        dates = windows[key]
        lines.append(f"{key}: {min(dates)} -> {max(dates)} ({len(dates)} months)")
    lines += [
        "",
        "Modeling note",
        "-" * 80,
        "One SARIMAX model is fitted per district with differencing enabled through the order and seasonal_order.",
        "No exogenous regressors are used in this benchmark to avoid future-exogenous leakage in multi-step forecasting.",
        "National forecasts are obtained by summing district forecasts.",
        "Additional outputs include model selection tables, parameter estimates, diagnostics tests, ADF stationarity tables, diagnostic plots, and residual ACF/PACF.",
    ]
    (out_dir / "run_summary.txt").write_text("\n".join(map(str, lines)), encoding="utf-8")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Grouped district SARIMAX benchmark for dengue forecasting.")
    p.add_argument("--input", type=str, default="./data/raw/prime_dataset_model_input_with_purge.csv")
    p.add_argument("--output_dir", type=str, default="SARIMAX/outputs")
    p.add_argument("--target_col", type=str, default="Log_NoOfDenguePatients")
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--p", type=int, default=1)
    p.add_argument("--d", type=int, default=1)
    p.add_argument("--q", type=int, default=1)
    p.add_argument("--P", type=int, default=0)
    p.add_argument("--D", type=int, default=1)
    p.add_argument("--Q", type=int, default=1)
    p.add_argument("--seasonal_period", type=int, default=12)
    p.add_argument("--min_train_points", type=int, default=24)
    p.add_argument("--maxiter", type=int, default=200)
    p.add_argument("--diagnostics_lags", type=int, default=24)
    p.add_argument("--adf_alpha", type=float, default=0.05)
    p.add_argument("--no_order_search", dest="use_order_search", action="store_false")
    p.set_defaults(use_order_search=True)
    a = p.parse_args()
    return Config(
        input_path=a.input,
        output_dir=a.output_dir,
        target_col=a.target_col,
        horizon=a.horizon,
        random_state=a.random_state,
        order=(a.p, a.d, a.q),
        seasonal_order=(a.P, a.D, a.Q, a.seasonal_period),
        min_train_points=a.min_train_points,
        maxiter=a.maxiter,
        diagnostics_lags=a.diagnostics_lags,
        use_order_search=a.use_order_search,
        adf_alpha=a.adf_alpha,
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
    y_lookup = build_lookup(df, cfg)
    mase_denom = compute_panel_mase_denom(df, cfg)
    write_run_summary(cfg, windows, out_dir)

    results = []
    all_audits: List[Dict] = []

    for h in range(1, cfg.horizon + 1):
        district_pred_frames = []
        for district, g in df.groupby(cfg.district_col):
            pred_df, audit_rows = rolling_forecasts_for_district(g, cfg, h, date_to_split, y_lookup)
            if not pred_df.empty:
                district_pred_frames.append(pred_df)
            all_audits.extend(audit_rows)

        pred_all = pd.concat(district_pred_frames, ignore_index=True) if district_pred_frames else pd.DataFrame()
        pred_all, threshold = assign_threshold_and_regime(pred_all)

        pred_train = pred_all[pred_all["split"] == "train"].copy()
        pred_val = pred_all[pred_all["split"] == "val"].copy()
        pred_test = pred_all[pred_all["split"] == "test"].copy()

        m_train = compute_metrics(pred_train["y_true_count"], pred_train["y_pred_count"], naive_denom=mase_denom)
        m_val = compute_metrics(pred_val["y_true_count"], pred_val["y_pred_count"], naive_denom=mase_denom)
        m_test = compute_metrics(pred_test["y_true_count"], pred_test["y_pred_count"], naive_denom=mase_denom)

        mask_naive = pred_test["naive_last_count"].notna()
        mask_seas = pred_test["seasonal12_count"].notna()
        b_naive = compute_metrics(pred_test.loc[mask_naive, "y_true_count"], pred_test.loc[mask_naive, "naive_last_count"], naive_denom=mase_denom)
        b_seas = compute_metrics(pred_test.loc[mask_seas, "y_true_count"], pred_test.loc[mask_seas, "seasonal12_count"], naive_denom=mase_denom)

        cls = safe_classification_metrics((pred_test["y_true_count"] >= threshold).astype(int), pred_test["y_pred_count"], threshold)

        results.append({
            "horizon": h,
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
        })

    # Per-district reference fits for diagnostics and coefficient reporting
    reference_models = [fit_reference_model_for_district(g, cfg) for _, g in df.groupby(cfg.district_col)]

    save_row_audit(all_audits, out_dir)
    save_split_manifest(windows, results, out_dir)
    preds = save_prediction_tables(results, out_dir)
    nat_preds = save_national_predictions(preds, out_dir)
    save_metrics(results, preds, nat_preds, out_dir, mase_denom)
    save_model_diagnostics(reference_models, out_dir, cfg)
    save_adf_outputs(df, cfg, out_dir)
    make_figures(preds, nat_preds, out_dir)
    archive = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
    print(f"Saved output folder: {out_dir}")
    print(f"Saved zip archive : {archive}")


if __name__ == "__main__":
    main()