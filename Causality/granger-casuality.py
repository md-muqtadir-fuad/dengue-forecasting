#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conditional linear Granger causality for monthly dengue data.

What this script does
---------------------
1. Loads a preprocessed CSV.
2. Builds either:
   - a national monthly series, or
   - district-wise monthly series.
3. Runs conditional linear Granger causality tests using nested OLS models.
4. Reports the strongest lag and p-value for each causal hypothesis.
5. Detects already-lagged input variables and reports effective total lag.
6. Exports manuscript-friendly summary tables.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues
import statsmodels.api as sm


# ---------------------------------------------------------------------
# USER CONFIG
# ---------------------------------------------------------------------
INPUT_PATH = "./data/raw/prime_dataset_model_input_with_purge.csv"
OUTPUT_DIR = "./Causality/outputs/GRANGER"

DATE_COL = "Date"
DISTRICT_COL = "District"
MODE = "district"  # "national" or "district"

TARGET_COL = "Log_NoOfDenguePatients"
TARGET_IS_LOG1P = True

PREDICTORS = ["AvgTemp_lag_3", "Rainfall_lag_2"]
CONTROLS = ["denv4","denv1_lag_1"]

MAX_LAG = 6
ALPHA = 0.05

NATIONAL_NUMERIC_AGG = "mean"            # "mean", "median", "sum"
NATIONAL_TARGET_AGG = "sum_from_log1p"   # "sum_raw_count", "sum_from_log1p", "mean"


# ---------------------------------------------------------------------
# Pretty names for tables / plots
# ---------------------------------------------------------------------
BASE_NAME_MAP = {
    # target / id
    "District": "District",
    "Date": "Date",
    "Month-year": "Month-Year",
    "Year": "Year",
    "Month": "Month",
    "Month_sin": "Month (Sin)",
    "Month_cos": "Month (Cos)",

    # dengue target
    "NoOfDenguePatients": "Dengue Cases",
    "Log_NoOfDenguePatients": "Log(Dengue Cases)",

    # climate
    "Rainfall": "Rainfall",
    "MinTemp": "Min. Temp",
    "AvgTemp": "Avg. Temp",
    "MaxTemp": "Max. Temp",
    "Humidity": "Humidity",
    "MonthlyAvgVisibility": "Visibility",
    "MonthlyAvgSeaLevelPressure": "Sea Level Pressure",
    "MonthlyAvgSunshineHours": "Sunshine Hrs",
    "MonthlyPrevailingWindSpeed": "Wind Speed",
    "MonthlyPrevailingWindDir": "Wind Direction",

    # static / categorical
    "PopulationDensity": "Pop. Density",
    "dominant": "Dominant Serotype",

    # serotypes
    "denv1": "DENV-1",
    "denv2": "DENV-2",
    "denv3": "DENV-3",
    "denv4": "DENV-4",
}

column_renaming = {
    "MonthlyPrevailingWindDir_ENE": "Wind Dir. (ENE)",
}


def _format_base_name(col: str) -> str:
    c = str(col).strip()

    if c in column_renaming:
        return column_renaming[c]

    if c in BASE_NAME_MAP:
        return BASE_NAME_MAP[c]

    m = re.fullmatch(r"denv(\d+)", c.lower())
    if m:
        return f"DENV-{m.group(1)}"

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


def pretty_column_name(col: str) -> str:
    c = str(col).strip()
    if c in column_renaming:
        return column_renaming[c]
    lag_match = re.match(r"^(.*)_lag_(\d+)$", c)
    if lag_match:
        base, lag_num = lag_match.groups()
        return f"{_format_base_name(base)} (Lag {lag_num})"
    return _format_base_name(c)


def parse_variable_lag(col: str) -> Tuple[str, int]:
    c = str(col).strip()
    lag_match = re.match(r"^(.*)_lag_(\d+)$", c)
    if lag_match:
        base, lag_num = lag_match.groups()
        return base, int(lag_num)
    return c, 0


def effective_total_lag(source_col: str, model_lag: int, target_col: str) -> int:
    _, source_offset = parse_variable_lag(source_col)
    _, target_offset = parse_variable_lag(target_col)
    return int(source_offset + int(model_lag) - target_offset)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class Config:
    input_path: str
    output_dir: str = "granger_outputs"
    date_col: str = "Date"
    district_col: str = "District"
    mode: str = "district"
    target_col: str = "Log_NoOfDenguePatients"
    target_is_log1p: bool = True
    predictor_cols: Tuple[str, ...] = ("AvgTemp_lag_3", "Rainfall_lag_2")
    control_cols: Tuple[str, ...] = tuple()
    max_lag: int = 6
    alpha: float = 0.05
    national_numeric_agg: str = "mean"
    national_target_agg: str = "sum_from_log1p"


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def month_str(n: int) -> str:
    return f"{n} Mo." if n == 1 else f"{n} Mo."


def conclusion_text(p: float, alpha: float) -> str:
    if pd.isna(p):
        return "N/A"
    return "Sig." if p < alpha else "Not Sig."


def write_variable_lag_metadata(cfg: Config, out_dir: Path) -> pd.DataFrame:
    rows = []
    for role, cols in [("target", [cfg.target_col]), ("predictor", list(cfg.predictor_cols)), ("control", list(cfg.control_cols))]:
        for col in cols:
            base, offset = parse_variable_lag(col)
            rows.append({
                "role": role,
                "variable": col,
                "pretty_name": pretty_column_name(col),
                "base_variable": base,
                "base_pretty_name": _format_base_name(base),
                "input_offset_months": offset,
                "already_lagged": bool(offset > 0),
            })
    meta = pd.DataFrame(rows)
    meta.to_csv(out_dir / "variable_lag_metadata.csv", index=False)

    lagged = meta[meta["already_lagged"]].copy()
    if lagged.empty:
        note = (
            "No already-lagged analysis variables were detected.\n"
            "Effective total lag equals the reported Granger lag.\n"
        )
    else:
        note = (
            "Already-lagged variables were detected.\n"
            "Interpretation rule:\n"
            "Effective Total Lag = Source Input Offset + Granger Lag - Target Input Offset\n\n"
            "Detected lagged variables:\n"
        )
        for _, r in lagged.iterrows():
            note += f"- {r['variable']} -> {r['base_variable']} with input offset {int(r['input_offset_months'])} month(s)\n"

    (out_dir / "lag_interpretation_note.txt").write_text(note, encoding="utf-8")
    return meta


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Conditional linear Granger causality for dengue time series.")
    p.add_argument("--input", type=str, default=INPUT_PATH)
    p.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    p.add_argument("--date_col", type=str, default=DATE_COL)
    p.add_argument("--district_col", type=str, default=DISTRICT_COL)
    p.add_argument("--mode", type=str, choices=["national", "district"], default=MODE)
    p.add_argument("--target_col", type=str, default=TARGET_COL)
    p.add_argument("--target_is_log1p", dest="target_is_log1p", action="store_true", default=TARGET_IS_LOG1P)
    p.add_argument("--no-target_is_log1p", dest="target_is_log1p", action="store_false")
    p.add_argument("--predictors", nargs="+", default=PREDICTORS)
    p.add_argument("--controls", nargs="*", default=CONTROLS)
    p.add_argument("--max_lag", type=int, default=MAX_LAG)
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--national_numeric_agg", type=str, default=NATIONAL_NUMERIC_AGG, choices=["mean", "median", "sum"])
    p.add_argument("--national_target_agg", type=str, default=NATIONAL_TARGET_AGG, choices=["sum_raw_count", "sum_from_log1p", "mean"])
    a = p.parse_args()
    return Config(
        input_path=a.input,
        output_dir=a.output_dir,
        date_col=a.date_col,
        district_col=a.district_col,
        mode=a.mode,
        target_col=a.target_col,
        target_is_log1p=a.target_is_log1p,
        predictor_cols=tuple(a.predictors),
        control_cols=tuple(a.controls),
        max_lag=a.max_lag,
        alpha=a.alpha,
        national_numeric_agg=a.national_numeric_agg,
        national_target_agg=a.national_target_agg,
    )


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_data(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.input_path)
    df.columns = [c.strip() for c in df.columns]

    required = {cfg.date_col, cfg.target_col, *cfg.predictor_cols, *cfg.control_cols}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "The requested columns are missing from the CSV: "
            f"{missing}.\n\n"
            "For manuscript-style Temp/Rain conditioned on DENV-2/DENV-3 analysis, "
            "you need a richer preprocessed dataset that actually contains those columns."
        )

    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="raise")
    if cfg.district_col in df.columns:
        df[cfg.district_col] = df[cfg.district_col].astype(str).str.strip()

    keep_cols = [cfg.date_col]
    if cfg.district_col in df.columns:
        keep_cols.append(cfg.district_col)
    keep_cols += [cfg.target_col, *cfg.predictor_cols, *cfg.control_cols]
    keep_cols = list(dict.fromkeys(keep_cols))
    df = df[keep_cols].copy()
    df = df.dropna().sort_values(keep_cols[:2] if cfg.district_col in df.columns else [cfg.date_col]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------
# Series construction
# ---------------------------------------------------------------------
def _aggregate_target_national(df: pd.DataFrame, cfg: Config) -> pd.Series:
    g = df.groupby(cfg.date_col)[cfg.target_col]
    if cfg.national_target_agg == "sum_raw_count":
        return g.sum()
    if cfg.national_target_agg == "sum_from_log1p":
        if not cfg.target_is_log1p:
            raise ValueError(
                "national_target_agg='sum_from_log1p' requires --target_is_log1p, "
                "because it back-transforms each district row with expm1 before summing."
            )
        return g.apply(lambda s: np.log1p(np.expm1(s.astype(float)).sum()))
    if cfg.national_target_agg == "mean":
        return g.mean()
    raise ValueError(f"Unsupported national_target_agg: {cfg.national_target_agg}")


def build_national_series(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    agg_map = {col: cfg.national_numeric_agg for col in [*cfg.predictor_cols, *cfg.control_cols]}
    out = df.groupby(cfg.date_col).agg(agg_map)
    out[cfg.target_col] = _aggregate_target_national(df, cfg)
    out = out.reset_index().sort_values(cfg.date_col).reset_index(drop=True)
    return out


def build_district_series(df: pd.DataFrame, cfg: Config) -> Dict[str, pd.DataFrame]:
    if cfg.district_col not in df.columns:
        raise ValueError("District mode requires a district column in the CSV.")

    series_dict = {}
    cols = [cfg.date_col, cfg.target_col, *cfg.predictor_cols, *cfg.control_cols]
    cols = list(dict.fromkeys(cols))  # deduplicate

    for district, g in df.groupby(cfg.district_col):
        g = g.sort_values(cfg.date_col).reset_index(drop=True)
        series_dict[district] = g[cols].copy()

    return series_dict


# ---------------------------------------------------------------------
# Conditional linear Granger
# ---------------------------------------------------------------------
def make_lagged_design(df: pd.DataFrame, y_col: str, x_col: str, control_cols: List[str], lag: int):
    work = df.copy()
    work["y_t"] = pd.to_numeric(work[y_col], errors="coerce")

    # remove duplicates and remove predictor from controls
    control_cols = [c for c in dict.fromkeys(control_cols) if c != x_col]

    y_lag_cols, x_lag_cols, control_lag_cols = [], [], []

    for k in range(1, lag + 1):
        yc = f"{y_col}_lag_{k}"
        xc = f"{x_col}_lag_{k}"

        work[yc] = pd.to_numeric(work[y_col], errors="coerce").shift(k)
        work[xc] = pd.to_numeric(work[x_col], errors="coerce").shift(k)

        y_lag_cols.append(yc)
        x_lag_cols.append(xc)

        for c in control_cols:
            cc = f"{c}_lag_{k}"
            work[cc] = pd.to_numeric(work[c], errors="coerce").shift(k)
            control_lag_cols.append(cc)

    needed = ["y_t", *y_lag_cols, *x_lag_cols, *control_lag_cols]
    design = work[needed].dropna().reset_index(drop=True)
    restricted_cols = [*y_lag_cols, *control_lag_cols]
    unrestricted_cols = [*restricted_cols, *x_lag_cols]
    return design, restricted_cols, unrestricted_cols


def conditional_granger_one_pair(df: pd.DataFrame, y_col: str, x_col: str, control_cols: List[str], max_lag: int) -> pd.DataFrame:
    rows = []
    x_base, x_input_offset = parse_variable_lag(x_col)
    y_base, y_input_offset = parse_variable_lag(y_col)

    for lag in range(1, max_lag + 1):
        design, restricted_cols, unrestricted_cols = make_lagged_design(df, y_col, x_col, control_cols, lag)
        eff_lag = effective_total_lag(x_col, lag, y_col)
        if len(design) <= (len(unrestricted_cols) + 2):
            rows.append({
                "predictor": x_col,
                "predictor_pretty": pretty_column_name(x_col),
                "predictor_base": x_base,
                "predictor_base_pretty": _format_base_name(x_base),
                "predictor_input_offset_months": x_input_offset,
                "target": y_col,
                "target_pretty": pretty_column_name(y_col),
                "target_base": y_base,
                "target_base_pretty": _format_base_name(y_base),
                "target_input_offset_months": y_input_offset,
                "lag": lag,
                "effective_total_lag_months": eff_lag,
                "n_obs": int(len(design)),
                "f_stat": np.nan,
                "p_value": np.nan,
                "r2_restricted": np.nan,
                "r2_unrestricted": np.nan,
                "status": "too_few_observations",
            })
            continue

        y = design["y_t"].astype(float)
        X_res = sm.add_constant(design[restricted_cols].astype(float), has_constant="add")
        X_unres = sm.add_constant(design[unrestricted_cols].astype(float), has_constant="add")
        model_res = sm.OLS(y, X_res).fit()
        model_unres = sm.OLS(y, X_unres).fit()

        x_lag_cols = [c for c in unrestricted_cols if c.startswith(f"{x_col}_lag_")]
        R = np.zeros((len(x_lag_cols), X_unres.shape[1]))
        for i, col in enumerate(x_lag_cols):
            j = list(X_unres.columns).index(col)
            R[i, j] = 1.0

        ftest = model_unres.f_test(R)
        rows.append({
            "predictor": x_col,
            "predictor_pretty": pretty_column_name(x_col),
            "predictor_base": x_base,
            "predictor_base_pretty": _format_base_name(x_base),
            "predictor_input_offset_months": x_input_offset,
            "target": y_col,
            "target_pretty": pretty_column_name(y_col),
            "target_base": y_base,
            "target_base_pretty": _format_base_name(y_base),
            "target_input_offset_months": y_input_offset,
            "lag": lag,
            "effective_total_lag_months": eff_lag,
            "n_obs": int(len(design)),
            "f_stat": float(np.squeeze(ftest.fvalue)) if hasattr(ftest, "fvalue") else np.nan,
            "p_value": float(np.squeeze(ftest.pvalue)) if hasattr(ftest, "pvalue") else np.nan,
            "r2_restricted": float(model_res.rsquared),
            "r2_unrestricted": float(model_unres.rsquared),
            "status": "ok",
        })

    out = pd.DataFrame(rows)
    if not out.empty and out["p_value"].notna().any():
        best_idx = out["p_value"].idxmin()
        out["is_strongest_lag"] = False
        out.loc[best_idx, "is_strongest_lag"] = True
    else:
        out["is_strongest_lag"] = False
    return out


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------
def summarize_single_series(test_df: pd.DataFrame, predictor: str, target_col: str, alpha: float) -> pd.DataFrame:
    x_base, x_offset = parse_variable_lag(predictor)
    if test_df["p_value"].notna().any():
        best = test_df.loc[test_df["p_value"].idxmin()]
        strongest_lag = int(best["lag"])
        effective_lag = int(best["effective_total_lag_months"])
        p_value = float(best["p_value"])
    else:
        strongest_lag = np.nan
        effective_lag = np.nan
        p_value = np.nan

    return pd.DataFrame([{
        "Causal Hypothesis": f"{_format_base_name(x_base)} -> Dengue",
        "Predictor": predictor,
        "Predictor Pretty": pretty_column_name(predictor),
        "Predictor Base": x_base,
        "Source Input Offset (Months)": x_offset,
        "Method": "Granger",
        "Strongest Granger Lag": month_str(strongest_lag) if pd.notna(strongest_lag) else "N/A",
        "Effective Total Lag": month_str(effective_lag) if pd.notna(effective_lag) else "N/A",
        "p-value": p_value,
        f"Conclusion (at α = {alpha:.2f})": conclusion_text(p_value, alpha),
    }])


def summarize_district_series(all_tests: Dict[str, pd.DataFrame], predictor: str, target_col: str, alpha: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x_base, x_offset = parse_variable_lag(predictor)
    district_rows, lag_pool_rows = [], []

    for district, test_df in all_tests.items():
        if test_df["p_value"].notna().any():
            best = test_df.loc[test_df["p_value"].idxmin()]
            district_rows.append({
                "district": district,
                "predictor": predictor,
                "predictor_pretty": pretty_column_name(predictor),
                "predictor_base": x_base,
                "predictor_base_pretty": _format_base_name(x_base),
                "predictor_input_offset_months": x_offset,
                "strongest_granger_lag_months": int(best["lag"]),
                "effective_total_lag_months": int(best["effective_total_lag_months"]),
                "p_value": float(best["p_value"]),
                "conclusion": conclusion_text(float(best["p_value"]), alpha),
            })
        else:
            district_rows.append({
                "district": district,
                "predictor": predictor,
                "predictor_pretty": pretty_column_name(predictor),
                "predictor_base": x_base,
                "predictor_base_pretty": _format_base_name(x_base),
                "predictor_input_offset_months": x_offset,
                "strongest_granger_lag_months": np.nan,
                "effective_total_lag_months": np.nan,
                "p_value": np.nan,
                "conclusion": "N/A",
            })

    district_summary = pd.DataFrame(district_rows).sort_values(["predictor", "district"]).reset_index(drop=True)

    lags = sorted({int(l) for df_ in all_tests.values() for l in df_["lag"].dropna().tolist()})
    for lag in lags:
        pvals = []
        for _, test_df in all_tests.items():
            row = test_df.loc[test_df["lag"] == lag]
            if not row.empty:
                p = row["p_value"].iloc[0]
                if pd.notna(p):
                    pvals.append(float(p))
        if pvals:
            fisher_stat, fisher_p = combine_pvalues(pvals, method="fisher")
            lag_pool_rows.append({
                "predictor": predictor,
                "predictor_pretty": pretty_column_name(predictor),
                "predictor_base": x_base,
                "predictor_base_pretty": _format_base_name(x_base),
                "predictor_input_offset_months": x_offset,
                "lag": lag,
                "effective_total_lag_months": effective_total_lag(predictor, lag, target_col),
                "n_districts_contributing": len(pvals),
                "fisher_stat": float(fisher_stat),
                "combined_p_value": float(fisher_p),
            })

    lag_pool = pd.DataFrame(lag_pool_rows)
    return district_summary, lag_pool


def write_manuscript_tables(cfg: Config, national_results: Optional[Dict[str, pd.DataFrame]], district_lag_pool: Optional[Dict[str, pd.DataFrame]], out_dir: Path) -> None:
    rows = []
    if cfg.mode == "national":
        for predictor, test_df in national_results.items():
            rows.append(summarize_single_series(test_df, predictor, cfg.target_col, cfg.alpha).iloc[0].to_dict())
    else:
        for predictor, lag_pool in district_lag_pool.items():
            x_base, x_offset = parse_variable_lag(predictor)
            if lag_pool.empty:
                rows.append({
                    "Causal Hypothesis": f"{_format_base_name(x_base)} -> Dengue",
                    "Predictor": predictor,
                    "Predictor Pretty": pretty_column_name(predictor),
                    "Predictor Base": x_base,
                    "Source Input Offset (Months)": x_offset,
                    "Method": "Granger",
                    "Strongest Granger Lag": "N/A",
                    "Effective Total Lag": "N/A",
                    "p-value": np.nan,
                    f"Conclusion (at α = {cfg.alpha:.2f})": "N/A",
                })
                continue
            best = lag_pool.loc[lag_pool["combined_p_value"].idxmin()]
            pval = float(best["combined_p_value"])
            rows.append({
                "Causal Hypothesis": f"{_format_base_name(x_base)} -> Dengue",
                "Predictor": predictor,
                "Predictor Pretty": pretty_column_name(predictor),
                "Predictor Base": x_base,
                "Source Input Offset (Months)": x_offset,
                "Method": "Granger",
                "Strongest Granger Lag": month_str(int(best["lag"])),
                "Effective Total Lag": month_str(int(best["effective_total_lag_months"])),
                "p-value": pval,
                f"Conclusion (at α = {cfg.alpha:.2f})": conclusion_text(pval, cfg.alpha),
            })

    manuscript = pd.DataFrame(rows)
    manuscript.to_csv(out_dir / "granger_summary_for_manuscript.csv", index=False)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    cfg = parse_args()
    print(f"Input file : {cfg.input_path}")
    print(f"Output dir : {cfg.output_dir}")
    out_dir = Path(cfg.output_dir)
    ensure_dir(out_dir)
    save_json(asdict(cfg), out_dir / "run_config.json")

    lag_meta = write_variable_lag_metadata(cfg, out_dir)
    df = load_data(cfg)

    run_log = {
        "input_path": cfg.input_path,
        "n_rows_after_dropna": int(len(df)),
        "date_min": str(df[cfg.date_col].min()),
        "date_max": str(df[cfg.date_col].max()),
        "mode": cfg.mode,
        "target_col": cfg.target_col,
        "target_is_log1p": cfg.target_is_log1p,
        "predictor_cols": list(cfg.predictor_cols),
        "control_cols": list(cfg.control_cols),
        "max_lag": cfg.max_lag,
        "alpha": cfg.alpha,
        "n_lagged_analysis_variables": int(lag_meta["already_lagged"].sum()),
    }
    if cfg.district_col in df.columns:
        run_log["n_districts"] = int(df[cfg.district_col].nunique())
    save_json(run_log, out_dir / "run_log.json")

    if cfg.mode == "national":
        ts = build_national_series(df, cfg)
        ts.to_csv(out_dir / "national_series_used.csv", index=False)
        national_results: Dict[str, pd.DataFrame] = {}
        for predictor in cfg.predictor_cols:
            test_df = conditional_granger_one_pair(
                df=ts,
                y_col=cfg.target_col,
                x_col=predictor,
                control_cols=[c for c in cfg.control_cols if c != predictor],
                max_lag=cfg.max_lag,
            )
            test_df.to_csv(out_dir / f"granger_national_{predictor}.csv", index=False)
            national_results[predictor] = test_df
        write_manuscript_tables(cfg, national_results=national_results, district_lag_pool=None, out_dir=out_dir)
    else:
        series_dict = build_district_series(df, cfg)
        district_dir = out_dir / "district_tests"
        ensure_dir(district_dir)

        district_level_summaries = []
        lag_pool_dict: Dict[str, pd.DataFrame] = {}
        for predictor in cfg.predictor_cols:
            all_tests = {}
            for district, ts in series_dict.items():
                test_df = conditional_granger_one_pair(
                    df=ts,
                    y_col=cfg.target_col,
                    x_col=predictor,
                    control_cols=[c for c in cfg.control_cols if c != predictor],
                    max_lag=cfg.max_lag,
                )
                test_df.insert(0, "district", district)
                test_df.to_csv(district_dir / f"granger_{district}_{predictor}.csv", index=False)
                all_tests[district] = test_df

            district_summary, lag_pool = summarize_district_series(all_tests, predictor, cfg.target_col, cfg.alpha)
            district_summary.to_csv(out_dir / f"granger_district_summary_{predictor}.csv", index=False)
            lag_pool.to_csv(out_dir / f"granger_fisher_by_lag_{predictor}.csv", index=False)
            lag_pool_dict[predictor] = lag_pool
            district_level_summaries.append(district_summary)

        if district_level_summaries:
            pd.concat(district_level_summaries, ignore_index=True).to_csv(out_dir / "granger_district_summary_all.csv", index=False)

        write_manuscript_tables(cfg, national_results=None, district_lag_pool=lag_pool_dict, out_dir=out_dir)

    readme = f"""
Outputs
=======
- run_config.json: exact configuration used
- run_log.json: basic run metadata
- variable_lag_metadata.csv: lag metadata for target / predictors / controls
- lag_interpretation_note.txt: how to interpret effective total lag
- granger_summary_for_manuscript.csv: manuscript-friendly summary table
- national_series_used.csv: national aggregated monthly series (national mode only)
- granger_national_<predictor>.csv: per-lag national results (national mode only)
- district_tests/granger_<DISTRICT>_<predictor>.csv: per-lag district results (district mode only)
- granger_district_summary_<predictor>.csv: best lag by district (district mode only)
- granger_fisher_by_lag_<predictor>.csv: Fisher-combined p-value by lag across districts (district mode only)
- granger_district_summary_all.csv: merged district summaries (district mode only)

Important note
==============
If already-lagged predictors such as AvgTemp_lag_3 are provided,
the script reports both:
- Strongest Granger Lag
- Effective Total Lag

Interpretation rule:
Effective Total Lag = Source Input Offset + Granger Lag - Target Input Offset
""".strip()
    (out_dir / "README.txt").write_text(readme, encoding="utf-8")

    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()